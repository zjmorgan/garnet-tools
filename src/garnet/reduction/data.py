import os
from collections import defaultdict

import numpy as np

from mantid.simpleapi import (
    Load,
    LoadNexus,
    LoadEventNexus,
    LoadParameterFile,
    LoadIsawDetCal,
    LoadNexusLogs,
    CreateSampleWorkspace,
    Rebin,
    ApplyCalibration,
    Multiply,
    PreprocessDetectorsToMD,
    ExtractMonitors,
    LoadMask,
    MaskDetectors,
    MaskDetectorsIf,
    ExtractMask,
    SetGoniometer,
    LoadWANDSCD,
    HB3AAdjustSampleNorm,
    CorelliCrossCorrelate,
    NormaliseByCurrent,
    GroupDetectors,
    LoadEmptyInstrument,
    SolidAngle,
    CopyInstrumentParameters,
    ConvertToMD,
    ConvertHFIRSCDtoMDE,
    ReplicateMD,
    BinMD,
    SliceMD,
    DivideMD,
    MDNorm,
    ConvertWANDSCDtoQ,
    ConvertUnits,
    CropWorkspaceForMDNorm,
    CropToComponent,
    RecalculateTrajectoriesExtents,
    ClearUB,
    LoadIsawUB,
    SaveIsawUB,
    CloneWorkspace,
    PlusMD,
    MinusMD,
    SaveMD,
    LoadMD,
    CreateMDHistoWorkspace,
    CreateSingleValuedWorkspace,
    AddSampleLog,
    RemoveLogs,
    CompressEvents,
    GenerateEventsFilter,
    FilterEvents,
    CopySample,
    DeleteWorkspace,
    DeleteWorkspaces,
    MergeMD,
    MergeMDFiles,
    mtd,
)

from mantid.dataobjects import EventWorkspace
from mantid.kernel import FloatTimeSeriesProperty

from mantid import config

config["Q.convention"] = "Crystallography"


def DataModel(instrument_config):
    if type(instrument_config["Wavelength"]) is list:
        return LaueData(instrument_config)
    else:
        return MonochromaticData(instrument_config)


class BaseDataModel:
    def __init__(self, instrument_config):
        self.elastic = None
        self.grouping = None
        self.custom_path = False
        self.det_IDs = None

        self.instrument_config = instrument_config
        self.instrument = self.instrument_config["FancyName"]

        facility = self.instrument_config["Facility"]
        name = self.instrument_config["Name"]
        iptspath = "IPTS-{}"
        rawfile = self.instrument_config["RawFile"]

        raw_file_path = os.path.join("/", facility, name, iptspath, rawfile)

        self.raw_file_path = raw_file_path

        self.gon_axis = 6 * [None]
        gon = self.instrument_config.get("Goniometer")
        gon_axis_names = self.instrument_config.get("GoniometerAxisNames")
        if gon_axis_names is None:
            gon_axis_names = list(gon.keys())
        axes = list(gon.items())

        gon_ind = 0
        for i, name in enumerate(gon_axis_names):
            axis = axes[i][1]
            if name is not None:
                self.gon_axis[gon_ind] = ",".join(5 * ["{}"]).format(
                    name, *axis
                )
                gon_ind += 1

        self.gon_axis_logs = gon_axis_names

        wl = instrument_config["Wavelength"]

        self.update_wavelength(wl)

        self.ref_inst = self.instrument_config["InstrumentName"]

        self.dt = np.deg2rad(instrument_config["DeltaTheta"])

        CreateSingleValuedWorkspace(
            OutputWorkspace="unity", DataValue=1, ErrorValue=1
        )

    def get_counting_rate(self):
        return 1

    def update_wavelength(self, wl):
        self.wavelength_band = (
            wl if type(wl) is list else [0.98 * wl, 1.02 * wl]
        )
        self.wavelength = np.mean(wl) if type(wl) is list else wl

        self.k_min = 2 * np.pi / np.max(self.wavelength_band)
        self.k_max = 2 * np.pi / np.min(self.wavelength_band)

    def workspace_exists(self, ws):
        return ws if mtd.doesExist(ws) else None

    def update_raw_path(self, plan):
        """
        Set additional parameters for data file.

        Parameters
        ----------
        plan : dict
            Reduction plan.

        """

        instrument = plan["Instrument"]

        wl = plan.get("Wavelength")
        if wl is not None:
            self.update_wavelength(wl)

        self.elastic = None
        self.time_offset = None

        IPTS = plan["IPTS"]

        if plan.get("RawFile") is not None:
            self.raw_file_path = plan["RawFile"]
            self.custom_path = True

        raw_path = os.path.dirname(self.raw_file_path)
        raw_file = os.path.basename(self.raw_file_path)

        if instrument == "DEMAND":
            exp = plan["Experiment"]
            raw_file = raw_file.format(exp, "{:04}")
        elif instrument == "CORELLI":
            self.elastic = plan.get("Elastic")
            self.time_offset = plan.get("TimeOffset")
            if self.elastic == True and self.time_offset is None:
                raw_path = raw_path.replace("nexus", "shared/autoreduce")
                raw_file = raw_file.replace(".nxs.h5", "_elastic.nxs")

        self.raw_file_path = os.path.join(raw_path, raw_file)

        files = self.get_file_name_list(IPTS, plan["Runs"])

        if not os.path.exists(files[0]):
            raw_path = raw_path.replace("nexus", "data")
            raw_file = raw_file.replace(".nxs.h5", "_event.nxs")
            self.raw_file_path = os.path.join(raw_path, raw_file)
            files = self.get_file_name_list(IPTS, plan["Runs"])

        if instrument != "DEMAND":
            if not self.elastic and not self.custom_path:
                LoadEventNexus(
                    Filename=files[0],
                    OutputWorkspace=self.instrument,
                    MetaDataOnly=True,
                    LoadLogs=False,
                )
            else:
                LoadNexus(Filename=files[0], OutputWorkspace=self.instrument)
        else:
            LoadEmptyInstrument(
                InstrumentName=self.ref_inst, OutputWorkspace=self.instrument
            )

        CloneWorkspace(
            InputWorkspace=self.instrument,
            OutputWorkspace="goniometer",
        )

    def combine_files(self, IPTS, process_map):
        """
        Combine runs together based on crystal orientation.

        """

        gon = self.instrument_config.get("Goniometer")
        gon_axis_names = self.instrument_config.get("GoniometerAxisNames")
        if gon_axis_names is None:
            gon_axis_names = list(gon.keys())

        gonio = ",".join(gon_axis_names)

        runs = list(process_map.keys())

        filenames = [self.file_names(IPTS, run) for run in runs]

        for filename in filenames:
            CreateSampleWorkspace(OutputWorkspace="logs")
            LoadNexusLogs(Workspace="logs", Filename=filename, AllowList=gonio)
            self.set_goniometer("logs")

    def load_clear_UB(self, filename, ws, run_number=None):
        """
        Load UB from file and replace.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.
        ws : str, optional
           Name of data.
        run_number : str, optional
            Run number to replace starred expression in filename.

        """

        ClearUB(Workspace=ws)
        LoadIsawUB(
            InputWorkspace=ws, Filename=filename.replace("*", str(run_number))
        )

    def save_UB(self, filename, ws):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.
        ws : str, optional
           Name of data.

        """

        SaveIsawUB(InputWorkspace=ws, Filename=filename)

    def get_file_name_list(self, IPTS, runs):
        """
        Complete list of file paths.

        Parameters
        ----------
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).

        Returns
        -------
        filenames : list
            List of filepaths.

        """

        if type(runs) is int:
            runs = [runs]

        filename = self.raw_file_path
        if not self.custom_path:
            return [filename.format(IPTS, run) for run in runs]
        else:
            return [filename.format(run) for run in runs]

    def file_names(self, IPTS, runs):
        """
        Complete file paths.

        Parameters
        ----------
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).

        Returns
        -------
        filenames : str
            Comma separated filepaths.

        """

        filenames = self.get_file_name_list(IPTS, runs)
        return ",".join(filenames)

    def get_min_max_values(self):
        """
        The minimum and maximum Q-values.

        Returns
        -------
        Q_min_vals: list
            Minumum Q.
        Q_max_vals: list
            Maximum Q.

        """

        return 3 * [-self.Q_max], 3 * [+self.Q_max]

    def set_goniometer(self, ws):
        """
        Set the goniomter motor angles

        Parameters
        ----------
        ws : str, optional
           Name of raw data.

        """

        SetGoniometer(
            Workspace=ws,
            Goniometers="None, Specify Individually",
            Axis0=self.gon_axis[0],
            Axis1=self.gon_axis[1],
            Axis2=self.gon_axis[2],
            Axis3=self.gon_axis[3],
            Axis4=self.gon_axis[4],
            Axis5=self.gon_axis[5],
            Average=self.laue,
        )

        if self.laue:
            self.R = mtd[ws].run().getGoniometer().getR().copy()

    def calculate_binning_from_bins(self, xmin, xmax, bins):
        """
        Determine the binning from the number of bins.

        Parameters
        ----------
        xmin : float
            Minimum bin center.
        xmax : float
            Maximum bin center.
        bins : int
            Number of bins.

        Returns
        -------
        min_edge : float
            Minimum bin edge.
        max_edge : float
            Maximum bin edge.
        step : float
            Bin step.

        """

        if bins > 1:
            step = (xmax - xmin) / (bins - 1)

            min_bin = xmin - 0.5 * step
            max_bin = xmax + 0.5 * step

            return min_bin, max_bin, step

        else:
            return xmin, xmax, xmax - xmin

    def calculate_binning_from_step(xmin, xmax, step):
        """
        Determine the binning from step size.

        Parameters
        ----------
        xmin : float
            Minimum bin center.
        xmax : float
            Maximum bin center.
        step : float
            Bin step.

        Returns
        -------
        min_edge : float
            Minimum bin edge.
        max_edge : float
            Maximum bin edge.
        bins : int
            Number of bins.

        """

        if step < xmax - xmin:
            bins = np.ceil((xmax - xmin) / step) + 1

            min_bin = xmin - 0.5 * step
            max_bin = xmax + 0.5 * step

            return min_bin, max_bin, bins

        else:
            return xmin, xmax, 1

    def extract_bin_info(self, ws):
        """
        Obtain the bin information from a histogram.

        Parameters
        ----------
        ws : str
            Name of histogram.

        Returns
        -------
        signal : array
            Data signal.
        error : array
            Data uncertanies.
        x0, x1, ... : array
            Dense bin center coordinates.

        """

        signal = mtd[ws].getSignalArray().copy()
        error = np.sqrt(mtd[ws].getErrorSquaredArray())

        dims = [mtd[ws].getDimension(i) for i in range(mtd[ws].getNumDims())]

        xs = [
            np.linspace(
                dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
            )
            for dim in dims
        ]

        xs = [0.5 * (x[1:] + x[:-1]) for x in xs]

        xs = np.meshgrid(*xs, indexing="ij")

        return signal, error, *xs

    def extract_counts(self, ws):
        """
        Obtain the bin counts from a histogram.

        Parameters
        ----------
        ws : str
            Name of histogram.

        Returns
        -------
        Counts : array
            Number of events.

        """

        return mtd[ws].getNumEventsArray().copy()

    def extract_axis_info(self, ws):
        """
        Obtain the axis information from a histogram.

        Parameters
        ----------
        ws : str
            Name of histogram.

        Returns
        -------
        UB : 3x3-matrix
            UB-matrix.
        W : 3x3-matrix
            Projection matrix.
        titles : list
            Axis names and units.
        x0, x1, ... : array
            Bin center coordinates.

        """

        ei = mtd[ws].getExperimentInfo(0)

        UB = ei.sample().getOrientedLattice().getUB().copy()

        if len(list(mtd[ws].getBasisVector(0))) == mtd[ws].getNumDims():
            W = np.column_stack([mtd[ws].getBasisVector(i) for i in range(3)])
        W = np.array(ei.run().getProperty("W_MATRIX").value).reshape(3, 3)

        dims = [mtd[ws].getDimension(i) for i in range(mtd[ws].getNumDims())]

        titles = [dim.name + " " + dim.getUnits() for dim in dims]

        xs = [
            np.linspace(
                dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
            )
            for dim in dims
        ]

        xs = [0.5 * (x[1:] + x[:-1]) for x in xs]

        return UB, W, titles, xs

    def check_volume_preservation(self, ws):
        UB, W, *_ = self.extract_axis_info(ws)

        det = np.linalg.det(2 * np.pi * UB @ W)

        assert np.isclose(det, 1)

    def delete_workspace(self, ws):
        """
        Delete workspace.

        Parameters
        ----------
        ws : str
            Workspace to remove.

        """

        if mtd.doesExist(ws):
            DeleteWorkspace(Workspace=ws)

    def combine_histograms(self, ws, merge):
        """
        Add two histogram workspaces together.

        Parameters
        ----------
        ws : str
            Name of histogram to be added.
        merge : str
            Name of histogram to be accumulated.

        """

        if not mtd.doesExist(merge):
            CloneWorkspace(InputWorkspace=ws, OutputWorkspace=merge)

        else:
            PlusMD(LHSWorkspace=merge, RHSWorkspace=ws, OutputWorkspace=merge)

            if mtd[ws].getNumExperimentInfo() > 0:
                for prop in mtd[ws].getExperimentInfo(0).run().getProperties():
                    if prop.type in ["string", "number"]:
                        log_type = (
                            "String" if prop.type == "string" else "Number"
                        )
                        AddSampleLog(
                            Workspace=merge,
                            LogName=prop.name,
                            LogText=str(prop.value),
                            LogUnit=str(prop.units),
                            LogType=log_type,
                        )

            DeleteWorkspace(Workspace=ws)

    def divide_histograms(self, ws, num, den):
        """
        Divide two histogram workspaces.

        Parameters
        ----------
        ws : str
            Name of resulting histogram.
        num : str
            Name of numerator histogram.
        den : str
            Name of denominator histogram.

        """

        DivideMD(LHSWorkspace=num, RHSWorkspace=den, OutputWorkspace=ws)

    def subtract_histograms(self, ws, ws1, ws2):
        """
        Difference between two histograms.

        Parameters
        ----------
        ws : str
            Name of resulting histogram.
        ws1 : str
            Name of first histogram.
        ws2 : str
            Name of second histogram.

        """

        MinusMD(LHSWorkspace=ws1, RHSWorkspace=ws2, OutputWorkspace=ws)

    def load_histograms(self, filename, ws):
        """
        Load histograms file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        ws : str
            Name of histogram to be added.

        """

        LoadMD(Filename=filename, OutputWorkspace=ws, LoadHistory=False)

    def save_histograms(self, filename, ws, sample_logs=False):
        """
        Save histograms file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        ws : str
            Name of histogram to be added.

        """

        SaveMD(
            Filename=filename,
            InputWorkspace=ws,
            SaveHistory=False,
            SaveInstrument=sample_logs,
            SaveSample=sample_logs,
            SaveLogs=sample_logs,
        )

    def merge_Q_sample(self, filenames, filename, merge):
        """
        Merge Q-sample files into one.

        Parameters
        ----------
        filenames : list
            Name of Q-sample filenames to be combined.
        filename: str
            Name of Q-sample filename to be saved.
        merge : str
            Name of Q-sample workspace to be accumulated.

        """

        MergeMDFiles(
            Filenames=filenames,
            OutputFilenames=filename,
            Parallel=True,
            OutputWorkspace=merge,
        )

    def combine_Q_sample(self, combine, merge):
        """
        Merge Q-sample workspaces into one.

        Parameters
        ----------
        combine : list
            Name of Q-sample workspaces to be accumulated.
        merge : str
            Name of combined Q-sample workspace.

        """

        MergeMD(InputWorkspaces=combine, OutputWorkspace=merge)

        DeleteWorkspaces(WorkspaceList=combine)

    def add_UBW(self, ws, ub_file, projections):
        """
        Attach sample UB and projection matrix to workspace

        Parameters
        ----------
        ws : str
            Name of histogram to be added.
        filename : str
            Name of UB file with extension .mat.
        projections : list
            Axis projections.

        """

        CreateSingleValuedWorkspace(OutputWorkspace="ubw")

        W = np.column_stack(projections)

        W_MATRIX = ",".join(9 * ["{}"]).format(*W.flatten())

        LoadIsawUB(InputWorkspace="ubw", Filename=ub_file)

        if mtd.doesExist(ws):
            AddSampleLog(
                Workspace=ws,
                LogName="W_MATRIX",
                LogText=W_MATRIX,
                LogType="String",
            )

            run = mtd[ws].getExperimentInfo(0).run()
            run.addProperty("W_MATRIX", list(W.flatten() * 1.0), True)

            CopySample(
                InputWorkspace="ubw",
                OutputWorkspace=ws,
                CopyName=False,
                CopyMaterial=False,
                CopyEnvironment=False,
                CopyLattice=True,
                CopyOrientationOnly=False,
            )

            logs, units = {}, None
            for prop in run.getProperties():
                if "log_" in prop.name and prop.type == "number":
                    num = int(prop.name.split("_")[1])
                    logs[num] = prop.value
                    units = prop.units

            keys = list(logs.keys())
            keys.sort()

            ref = logs[0] if units == "Seconds" else 0

            data = []
            for key in keys:
                data.append(logs[key] - ref)

            if len(data) > 0:
                AddSampleLog(
                    Workspace=ws, LogName="log", LogText="0", LogType="String"
                )
                run.addProperty("log", data, True)

    def get_resolution_in_Q(self, lamda, two_theta):
        """
        Obtain the wavelength and detector-dependent Q-resolution

        Parameters
        ----------
        lamda : float
            Incident wavelength.
        two_theta : float
            Detector scattering angle in degrees.

        Returns
        -------
        dQ : float
            Q-resolution.

        """

        dl = 1e-3 * lamda

        theta = 0.5 * np.deg2rad(two_theta)

        Q = 4 * np.pi / lamda * np.sin(theta)

        dQ = Q * np.sqrt((dl / lamda) ** 2 + (self.dt / np.tan(theta)) ** 2)

        return dQ

    def clear_norm(self, md):
        if mtd.doesExist(md + "_data"):
            DeleteWorkspace(Workspace=md + "_data")
        if mtd.doesExist(md + "_norm"):
            DeleteWorkspace(Workspace=md + "_norm")

        if mtd.doesExist(md + "_bkg_data"):
            DeleteWorkspace(Workspace=md + "_bkg_data")
        if mtd.doesExist(md + "_bkg_norm"):
            DeleteWorkspace(Workspace=md + "_bkg_norm")

    def bin_in_Q(self, md, extents, bins, projections):
        """
        Histogram data into Q-space.

        Parameters
        ----------
        md : str
            3D Q-space data.
        extents : list
            Min/max pairs for each dimension.
        bins : list
            Number of bins for each dimension.
        projections : list
            Direction of projection for each dimension.

        """

        if mtd.doesExist(md):
            extents = np.array(extents).flatten().tolist()

            u0, u1, u2 = projections

            bins[bins == 0] = 1

            BinMD(
                InputWorkspace=md,
                AxisAligned=False,
                BasisVector0="Q_x,Angstrom^-1,{},{},{}".format(*u0),
                BasisVector1="Q_y,Angstrom^-1,{},{},{}".format(*u1),
                BasisVector2="Q_z,Angstrom^-1,{},{},{}".format(*u2),
                OutputExtents=extents,
                OutputBins=bins,
                OutputWorkspace=md + "_bin",
            )

            y, e, x0, x1, x2 = self.extract_bin_info(md + "_bin")

            return y, e, x0, x1, x2

    def slice_extents(self, c, r):
        return [x for v in c for x in [v - 6 * r, v + 6 * r]]

    def slice_roi(self, md, extents):
        extents = np.array(extents).flatten().tolist()

        SliceMD(
            InputWorkspace=md,
            AxisAligned=False,
            BasisVector0="Q_sample_x,Angstrom^-1,1,0,0",
            BasisVector1="Q_sample_y,Angstrom^-1,0,1,0",
            BasisVector2="Q_sample_z,Angstrom^-1,0,0,1",
            OutputExtents=extents,
            OutputBins="1,1,1",
            OutputWorkspace=md + "_slice",
        )

    def update_logs_for_time(self, ws):
        log = mtd[ws].run().getProperty("duration")
        t_max = log.value
        t_units = log.units
        values = np.linspace(0, t_max, 10000)
        times = mtd[ws].run().startTime().totalNanoseconds() + values * 1e9
        log = FloatTimeSeriesProperty("time")
        for t, v in zip(times, values):
            log.addValue(t, v)
        mtd[ws].run()["time"] = log
        mtd[ws].run().getProperty("time").units = t_units

    def log_split_info(self, ws, log_name, log_limits, log_bins):
        """
        Generate split information for filtering events by log values.

        Parameters
        ----------
        ws : str
            Workspace with log time and values.
        log_name : str
            Name of the log.
        log_limits : list, float
            Min/max values of log (bin center).
        log_bins : int
            Number of equally space bins.

        """

        run = mtd[ws].run()
        log = run.getProperty(log_name) if run.hasProperty(log_name) else None

        if log is not None:
            if log_bins > 0:
                assert log is not None
                log_min, log_max = log_limits
                log_interval = (log_max - log_min) / (log_bins - 1)

                GenerateEventsFilter(
                    InputWorkspace=ws,
                    OutputWorkspace="split",
                    InformationWorkspace="info",
                    LogName=log_name,
                    MinimumLogValue=log_min,
                    MaximumLogValue=log_max,
                    LogValueInterval=log_interval,
                    LogBoundary="Left",
                )

                log_vals = np.linspace(
                    log_min,
                    log_max,
                    log_bins,
                )

            else:
                log_vals = log.timeAverageValue()

            log_units = log.units
        else:
            start_time = np.datetime64(run.getProperty("start_time").value)
            duration = run.getProperty("duration").value
            log_vals = float(start_time.astype(np.int64) * 1e-9 + duration / 2)
            log_units = "Seconds"

        return log_vals, log_units

    def combine_splits(self, md, log_name, log_vals, log_units, index, runs):
        """
        Remove all accumulation workspaces.

        Parameters
        ----------
        md : str
            Base name of histogram.

        """

        workspaces = ["_data", "_norm", "_bkg_data", "_bkg_norm"]

        ws = md + "_result"

        dims = [mtd[ws].getDimension(i) for i in range(mtd[ws].getNumDims())]

        extents, bins, names, units = [], [], [], []

        if type(log_vals) is float:
            extents.append(0.5)
            extents.append(0.5 + len(runs))
            bins.append(len(runs))
            names.append("bin")
            units.append("index")
        else:
            interval = np.diff(log_vals).mean()
            extents.append(log_vals[0] - 0.5 * interval)
            extents.append(log_vals[-1] + 0.5 * interval)
            bins.append(len(log_vals))
            names.append(log_name)
            units.append(log_units)

        for dim in dims:
            extents.append(dim.getMinimum())
            extents.append(dim.getMaximum())
            bins.append(dim.getNBins())
            names.append(dim.name)
            units.append(dim.getUnits())

        for ws in workspaces:
            if mtd.doesExist(md + ws):
                if not mtd.doesExist(md + ws + "_split"):
                    CreateMDHistoWorkspace(
                        SignalInput=np.zeros(np.prod(bins)),
                        ErrorInput=np.zeros(np.prod(bins)),
                        Dimensionality=len(dims) + 1,
                        Extents=extents,
                        NumberOfBins=bins,
                        Names=names,
                        Units=units,
                        OutputWorkspace=md + ws + "_split",
                    )
                    if type(log_vals) is float:
                        AddSampleLog(
                            Workspace=md + ws + "_split",
                            LogName="log_index",
                            LogText=str(log_name),
                            LogUnit=str(log_units),
                            LogType="String",
                        )

                if type(log_vals) is float:
                    AddSampleLog(
                        Workspace=md + ws + "_split",
                        LogName="log_{}".format(index),
                        LogText=str(log_vals),
                        LogUnit=str(log_units),
                        LogType="Number",
                    )

                signal = mtd[md + ws + "_split"].getSignalArray().copy()
                signal[index] += mtd[md + ws].getSignalArray()
                mtd[md + ws + "_split"].setSignalArray(signal)

                error_sq = (
                    mtd[md + ws + "_split"].getErrorSquaredArray().copy()
                )
                error_sq[index] += mtd[md + ws].getErrorSquaredArray()
                mtd[md + ws + "_split"].setErrorSquaredArray(error_sq)

                DeleteWorkspace(Workspace=md + ws)


class MonochromaticData(BaseDataModel):
    def __init__(self, instrument_config):
        super(MonochromaticData, self).__init__(instrument_config)

        self.laue = False

    def load_data(self, histo_name, IPTS, runs, grouping=None):
        """
        Load raw data into detector counts vs rotation index.

        Parameters
        ----------
        histo_name : str
            Name of raw histogram data.
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).
        grouping : str, optional
            Options for grouping pixels.

        """

        filenames = self.file_names(IPTS, runs)

        self.grouping = "None" if grouping is None else grouping

        if self.instrument == "DEMAND":
            HB3AAdjustSampleNorm(
                Filename=filenames,
                OutputType="Detector",
                NormaliseBy="None",
                Grouping=self.grouping,
                OutputWorkspace=histo_name,
            )
            run = mtd[histo_name].getExperimentInfo(0).run()
            self.scale = run.getProperty("time").value

        else:
            LoadWANDSCD(
                Filename=filenames,
                Grouping=grouping,
                OutputWorkspace=histo_name,
            )
            run = mtd[histo_name].getExperimentInfo(0).run()
            self.scale = run.getProperty("duration").value

        self.theta_max = 0.5 * np.max(run.getProperty("TwoTheta").value)

        if run.hasProperty("wavelength"):
            wl = float(run.getProperty("wavelength").value)
            self.wavelength = wl
            self.wavelength_band = [0.98 * wl, 1.02 * wl]

        self.Q_max = 4 * np.pi / self.wavelength * np.sin(self.theta_max)

        self.set_goniometer(histo_name)

    def convert_to_Q_sample(self, histo_name, md_name, lorentz_corr=False):
        """
        Convert raw data to Q-sample.

        Parameters
        ----------
        histo_name : str
            Name of raw histogram data.
        md_name : str
            Name of Q-sample workspace.
        lorentz_corr : bool, optional
            Apply Lorentz correction. The default is False.

        """

        if mtd.doesExist(histo_name):
            Q_min_vals, Q_max_vals = self.get_min_max_values()

            ConvertHFIRSCDtoMDE(
                InputWorkspace=histo_name,
                Wavelength=self.wavelength,
                LorentzCorrection=lorentz_corr,
                MinValues=Q_min_vals,
                MaxValues=Q_max_vals,
                OutputWorkspace=md_name,
                SplitInto=2,
            )

    def load_generate_normalization(self, filename, histo_name=None):
        """
        Load a vanadium file and generate normalization data.
        Provided a histogram workspace name, generate corresponding shape.

        Parameters
        ----------
        filename : str
            Vanadium file.
        histo_name : str, optional
            Name of raw histogram data.

        """

        if not mtd.doesExist("van"):
            if self.instrument == "DEMAND":
                HB3AAdjustSampleNorm(
                    Filename=filename,
                    OutputType="Detector",
                    NormaliseBy="None",
                    Grouping=self.grouping,
                    OutputWorkspace="van",
                )

            else:
                LoadWANDSCD(
                    Filename=filename,
                    Grouping=self.grouping,
                    OutputWorkspace="van",
                )

            if histo_name is not None:
                if mtd.doesExist(histo_name):
                    ws_name = "{}_van".format(histo_name)

                    ReplicateMD(
                        ShapeWorkspace=histo_name,
                        DataWorkspace="van",
                        OutputWorkspace=ws_name,
                    )

                    signal = mtd[ws_name].getSignalArray().copy()
                    mtd[ws_name].setSignalArray(signal * self.scale)

                    DivideMD(
                        LHSWorkspace=histo_name,
                        RHSWorkspace=ws_name,
                        OutputWorkspace=histo_name,
                    )

    def load_background(self, filename, histo_name=None):
        """
        Load a background file and scale to data.

        Parameters
        ----------
        filename : str
            Background file.
        histo_name : str
            Name of raw histogram data.

        """

        if not mtd.doesExist("bkg") and filename is not None:
            if self.instrument == "DEMAND":
                HB3AAdjustSampleNorm(
                    Filename=filename,
                    OutputType="Detector",
                    NormaliseBy="None",
                    Grouping=self.grouping,
                    OutputWorkspace="bkg",
                )
                run = mtd["bkg"].getExperimentInfo(0).run()
                scale = run().getProperty("time").value

            else:
                LoadWANDSCD(
                    Filename=filename,
                    Grouping=self.grouping,
                    OutputWorkspace="bkg",
                )
                run = mtd["bkg"].getExperimentInfo(0).run()
                scale = run.getProperty("duration").value

            if histo_name is not None:
                if mtd.doesExist(histo_name):
                    ws_name = "{}_bkg".format(histo_name)

                    if not mtd.doesExist(ws_name):
                        ReplicateMD(
                            ShapeWorkspace=histo_name,
                            DataWorkspace="bkg",
                            OutputWorkspace=ws_name,
                        )

                        self.set_goniometer(ws_name)

                        signal = mtd[ws_name].getSignalArray().copy()
                        mtd[ws_name].setSignalArray(
                            signal * self.scale / scale
                        )

    def normalize_to_hkl(self, ws, projections, extents, bins, symmetry=None):
        """
        Normalizae to binned hkl.

        Parameters
        ----------
        ws : str
            3D detector counts vs rotation index data.
        projections : list
            Projection axis vectors.
        extents : list
            Min/max pairs defining the bin center limits.
        bins : list
            Number of bins.
        symmetry : str, optional
            Laue point group. The default is None.

        """

        if mtd.doesExist(ws) and mtd.doesExist("van"):
            v0, v1, v2 = projections

            (Q0_min, Q0_max), (Q1_min, Q1_max), (Q2_min, Q2_max) = extents

            nQ0, nQ1, nQ2 = bins

            Q0_min, Q0_max, dQ0 = self.calculate_binning_from_bins(
                Q0_min, Q0_max, nQ0
            )

            Q1_min, Q1_max, dQ1 = self.calculate_binning_from_bins(
                Q1_min, Q1_max, nQ1
            )

            Q2_min, Q2_max, dQ2 = self.calculate_binning_from_bins(
                Q2_min, Q2_max, nQ2
            )

            bkg_ws = self.workspace_exists(ws + "_bkg")

            bkg_data = self.workspace_exists(ws + "_bkg_data")
            bkg_norm = self.workspace_exists(ws + "_bkg_norm")

            _data = self.workspace_exists(ws + "_data")
            _norm = self.workspace_exists(ws + "_norm")

            __data = self.workspace_exists(ws + "_bkg_data")
            __norm = self.workspace_exists(ws + "_bkg_norm")

            ConvertWANDSCDtoQ(
                InputWorkspace=ws,
                NormalisationWorkspace="van",
                UBWorkspace=ws,
                BackgroundWorkspace=bkg_ws,
                OutputWorkspace=ws + "_result",
                OutputDataWorkspace=ws + "_data",
                OutputNormalizationWorkspace=ws + "_norm",
                OutputBackgroundDataWorkspace=bkg_data,
                OutputBackgroundNormalizationWorkspace=bkg_norm,
                NormaliseBy="Time",
                Frame="HKL",
                Wavelength=self.wavelength,
                SymmetryOperations=symmetry,
                KeepTemporaryWorkspaces=True,
                TemporaryDataWorkspace=_data,
                TemporaryNormalizationWorkspace=_norm,
                TemporaryBackgroundDataWorkspace=__data,
                TemporaryBackgroundNormalizationWorkspace=__norm,
                Uproj=v0,
                Vproj=v1,
                Wproj=v2,
                BinningDim0=[Q0_min, Q0_max, nQ0],
                BinningDim1=[Q1_min, Q1_max, nQ1],
                BinningDim2=[Q2_min, Q2_max, nQ2],
            )


class LaueData(BaseDataModel):
    def __init__(self, instrument_config):
        super(LaueData, self).__init__(instrument_config)

        self.laue = True

        self.sa_cal = False
        self.flux_cal = False

    def load_data(self, event_name, IPTS, runs, grouping=None, time_cut=None):
        """
        Load raw data into time-of-flight vs counts.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).
        time_cut: float, optional
            Time cut off for faster loading.

        """

        filenames = self.file_names(IPTS, runs)

        if self.elastic and self.time_offset is None:
            LoadNexus(Filename=filenames, OutputWorkspace=event_name)
            x = mtd[event_name].extractX()[0]
            Rebin(
                InputWorkspace=event_name,
                Params=[x[0], x[-1], x[-1]],
                OutputWorkspace=event_name,
            )
        else:
            Load(
                Filename=filenames,
                OutputWorkspace=event_name,
                NumberOfBins=1,
                FilterByTimeStop=time_cut,
                FilterByTofMin=None,
                FilterByTofMax=None,
            )

        if type(runs) is list and mtd[event_name].isGroup():
            for run, ws in zip(runs, mtd[event_name].getNames()):
                mtd[ws].run()["run_number"] = run
        else:
            mtd[event_name].run()["run_number"] = runs

        if self.elastic == True and self.time_offset is not None:
            CopyInstrumentParameters(
                InputWorkspace=self.ref_inst, OutputWorkspace=event_name
            )

            CorelliCrossCorrelate(
                InputWorkspace=event_name,
                OutputWorkspace=event_name,
                TimingOffset=self.time_offset,
            )

        self.monitor = mtd[event_name].run().getProtonCharge()

        self.set_goniometer(event_name)

        if self.grouping is None and grouping is not None:
            self.preprocess_detectors(event_name)
            self.create_grouping(grouping)
            self.delete_workspace("detectors")

    def get_counting_rate(self, event_name):
        """
        Counting rate.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.

        Returns
        -------
        cntrt : float
            Counting rate in events per proton charge.

        """

        total_events = mtd[event_name].getNumberEvents()
        total_charge = mtd[event_name].run().getProperty("gd_prtn_chrg").value

        return total_events / total_charge

    def calculate_maximum_Q(self):
        """
        Update maximum Q.

        """

        lamda_min = np.min(self.wavelength_band)

        self.Q_max = 4 * np.pi / lamda_min * np.sin(self.theta_max)

    def apply_calibration(
        self,
        event_name,
        detector_calibration,
        tube_calibration=None,
        goniometer_calibration=None,
    ):
        """
        Apply detector calibration.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        detector_calibration : str
            Detector calibration as either .xml or .DetCal.
        tube_calibration : str, optional
            CORELLI-only tube calibration. The default is None.
        goniometer_calibration : str
            Goniometer calibration as .xml. The default is None.

        """

        if tube_calibration is not None:
            if not mtd.doesExist("tube_table"):
                LoadNexus(
                    Filename=tube_calibration, OutputWorkspace="tube_table"
                )

            ApplyCalibration(
                Workspace=event_name, CalibrationTable="tube_table"
            )

        if detector_calibration is not None:
            if os.path.splitext(detector_calibration)[1] == ".xml":
                LoadParameterFile(
                    Workspace=event_name, Filename=detector_calibration
                )

            else:
                LoadIsawDetCal(
                    InputWorkspace=event_name, Filename=detector_calibration
                )

        if goniometer_calibration is not None:
            if os.path.splitext(goniometer_calibration)[1] == ".xml":
                LoadParameterFile(
                    Workspace="goniometer", Filename=goniometer_calibration
                )

                inst = mtd["goniometer"].getInstrument()
                run = mtd[event_name].run()

                params = ["omega-offset", "chi-offset", "phi-offset"]

                for i, param in enumerate(params):
                    if inst.hasParameter(param):
                        val = inst.getNumberParameter(param)[0]
                        name = self.gon_axis_logs[i]
                        values = run.getProperty(name).value
                        times = run.getProperty(name).times
                        log = FloatTimeSeriesProperty(name)
                        for t, v in zip(times, values):
                            log.addValue(t, v + val)
                        run[name] = log

                self.set_goniometer(event_name)

                run = mtd[event_name].run()

                param = "goniometer-tilt"
                if inst.hasParameter(param):
                    v = inst.getStringParameter(param)[0]
                    G = np.array(v.split(",")).astype(float).reshape(3, 3)
                    gon = run.getGoniometer()
                    gon.setR(G @ gon.getR())

        if mtd.doesExist("sa") and not self.sa_cal:
            self.sa_cal = True
            self.apply_calibration(
                "sa", detector_calibration, tube_calibration
            )

        if mtd.doesExist("flux") and not self.flux_cal:
            self.flux_cal = True
            self.apply_calibration(
                "flux", detector_calibration, tube_calibration
            )

    def preprocess_detector_banks(self, bank):
        """
        Generate detector coordinates for each bank.

        Parameters
        ----------
        bank : str
            Bank-only event data.

        """

        if not mtd.doesExist(bank + "_dets"):
            PreprocessDetectorsToMD(
                InputWorkspace=bank, OutputWorkspace=bank + "_dets"
            )

    def preprocess_detectors(self, ws=None):
        """
        Generate detector coordinates.

        Parameters
        ----------
        event_name : str
            Workspace with instrument data.

        """

        if ws is None:
            ws = self.instrument

        if not mtd.doesExist("detectors") and mtd.doesExist(ws):
            ExtractMonitors(InputWorkspace=ws, DetectorWorkspace=ws)

            PreprocessDetectorsToMD(
                InputWorkspace=ws, OutputWorkspace="detectors"
            )

            self.det_IDs = mtd["detectors"].column("DetectorID")

            if not mtd.doesExist("solid_angle"):
                SolidAngle(InputWorkspace=ws, OutputWorkspace="solid_angle")

                inds = mtd["solid_angle"].getIndicesFromDetectorIDs(
                    self.det_IDs
                )

                self.solid_angle_y = mtd["solid_angle"].extractY().ravel()

                self.solid_angle_ind = {
                    det_ID: ind for det_ID, ind in zip(self.det_IDs, inds)
                }
                self.solid_angle_ind[-1] = 0

            two_theta = mtd["detectors"].column("TwoTheta")

            self.theta_max = 0.5 * np.max(two_theta)

            self.calculate_maximum_Q()

    def mask_to_bank(self, event_name, bank_name):
        """
        Mask to bank

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        bank_name : str
            Bank name.

        """

        CropToComponent(
            InputWorkspace=event_name,
            OutputWorkspace=bank_name,
            ComponentNames=bank_name,
        )

    def apply_mask(self, event_name, detector_mask):
        """
        Apply detector mask.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        detector_mask : str
            Detector mask as .xml.

        """

        if detector_mask is not None and not mtd.doesExist("mask"):
            LoadMask(
                Instrument=self.ref_inst,
                InputFile=detector_mask,
                RefWorkspace=event_name,
                OutputWorkspace="mask",
            )

        if mtd.doesExist("sa_mask"):
            MaskDetectors(Workspace=event_name, MaskedWorkspace="sa_mask")

        if mtd.doesExist("mask"):
            MaskDetectors(Workspace=event_name, MaskedWorkspace="mask")

    def create_grouping(self, grouping):
        """
        Generate grouping pattern.

        Parameters
        ----------
        gropuing : str
            Grouping pattern (rows)x(cols).

        """

        if grouping is None or grouping == "1x1":
            return

        c, r = [int(val) for val in grouping.split("x")]

        cols, rows = self.instrument_config["BankPixels"]

        det_map = np.array(mtd["detectors"].column(5)).reshape(-1, cols, rows)

        shape = det_map.shape
        i, j, k = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        keys = np.stack((i, j // c, k // r), axis=-1)
        keys_flat = keys.reshape(-1, keys.shape[-1])
        det_map_flat = det_map.ravel().astype(str)
        grouped_ids = defaultdict(list)
        for key, detector_id in zip(map(tuple, keys_flat), det_map_flat):
            grouped_ids[key].append(detector_id)
        self.grouping = ",".join(
            "+".join(group) for group in grouped_ids.values()
        )

    def group_pixels(self, ws):
        """
        Group pixels with grouping pattern.

        Parameters
        ----------
        ws : str
            Workspace name.

        """

        if self.grouping is not None:
            GroupDetectors(
                InputWorkspace=ws,
                GroupingPattern=self.grouping,
                OutputWorkspace=ws,
            )

        if type(mtd[ws]) is EventWorkspace:
            CompressEvents(
                InputWorkspace=ws,
                OutputWorkspace=ws,
                Tolerance=1e-3,
            )

    def convert_to_Q_sample(
        self, event_name, md_name, lorentz_corr=False, preproc_dets=None
    ):
        """
        Convert raw data to Q-sample.

        Parameters
        ----------
        event_name : str
            Name of raw event name data.
        md_name : str
            Name of Q-sample workspace.
        lorentz_corr : bool, optional
            Apply Lorentz correction. The default is False.
        preproc_dets: str, optional
            Preprocess detector information workspace.

        """

        self.preprocess_detectors(event_name)

        self.calculate_maximum_Q()

        if mtd.doesExist(event_name):
            Q_min_vals, Q_max_vals = self.get_min_max_values()

            if preproc_dets is None:
                preproc_dets = "detectors"

            ConvertToMD(
                InputWorkspace=event_name,
                QDimensions="Q3D",
                dEAnalysisMode="Elastic",
                Q3DFrames="Q_sample",
                LorentzCorrection=lorentz_corr,
                MinValues=Q_min_vals,
                MaxValues=Q_max_vals,
                OutputWorkspace=md_name,
                PreprocDetectorsWS=preproc_dets,
                SplitInto=2,
                MaxRecursionDepth=10,
            )

            RecalculateTrajectoriesExtents(
                InputWorkspace=md_name, OutputWorkspace=md_name
            )

    def convert_to_Q_lab(self, event_name, md_name, lorentz_corr=False):
        """
        Convert raw data to Q-lab.

        Parameters
        ----------
        event_name : str
            Name of raw event name data.
        md_name : str
            Name of Q-sample workspace.
        lorentz_corr : bool, optional
            Apply Lorentz correction. The default is False.

        """

        self.preprocess_detectors(event_name)

        self.calculate_maximum_Q()

        if mtd.doesExist(event_name):
            Q_min_vals, Q_max_vals = self.get_min_max_values()

            ConvertToMD(
                InputWorkspace=event_name,
                QDimensions="Q3D",
                dEAnalysisMode="Elastic",
                Q3DFrames="Q_lab",
                LorentzCorrection=lorentz_corr,
                MinValues=Q_min_vals,
                MaxValues=Q_max_vals,
                OutputWorkspace=md_name,
                PreprocDetectorsWS="detectors",
                SplitInto=2,
                MaxRecursionDepth=10,
            )

    def load_generate_normalization(self, vanadium_file, flux_file):
        """
        Load a vanadium file and generate normalization data.

        Parameters
        ----------
        vanadium_file : str
            Solid angle file.
        flux_file : str
            Flux file.

        """

        if not mtd.doesExist("sa"):
            LoadNexus(Filename=vanadium_file, OutputWorkspace="sa")

            # if self.grouping is not None:
            #     self.group_pixels("sa")

            RemoveLogs(Workspace="sa")

            MaskDetectorsIf(
                InputWorkspace="sa", Operator="LessEqual", OutputWorkspace="sa"
            )

            ExtractMask(InputWorkspace="sa", OutputWorkspace="sa_mask")

        if not mtd.doesExist("flux"):
            LoadNexus(Filename=flux_file, OutputWorkspace="flux")

            # NormaliseSpectra(InputWorkspace="flux", OutputWorkspace="flux")

            RemoveLogs(Workspace="flux")

            self.k_min = mtd["flux"].getXDimension().getMinimum()
            self.k_max = mtd["flux"].getXDimension().getMaximum()

            if self.det_IDs is not None:
                inds = mtd["sa"].getIndicesFromDetectorIDs(self.det_IDs)

                self.sa_y = mtd["sa"].extractY().ravel()

                self.sa_ind = {
                    det_ID: ind for det_ID, ind in zip(self.det_IDs, inds)
                }
                self.sa_ind[-1] = 0

                lamda_min = 2 * np.pi / self.k_max
                lamda_max = 2 * np.pi / self.k_min

                self.wavelength_band = [lamda_min, lamda_max]

                y = mtd["flux"].extractY()
                x = mtd["flux"].extractX()

                k = 0.5 * (x[:, 1:] + x[:, :-1])
                y = np.diff(y) * y.shape[1]

                x = 2 * np.pi / k
                z = 2 * np.pi * y / x**2
                y = z * ((x[:, 0] - x[:, -1]) / (k[:, -1] - k[:, 0]))[:, None]

                inds = mtd["flux"].getIndicesFromDetectorIDs(self.det_IDs)

                self.spec_ind = {
                    det_ID: ind for det_ID, ind in zip(self.det_IDs, inds)
                }
                self.spec_ind[-1] = 0

                self.lamda_x = x[:, ::-1].mean(axis=0)
                self.spect_y = y[:, ::-1]

    def get_volume_in_Q(self, lamda, two_theta, det_ID):
        L_inv = 2 * np.sin(0.5 * np.deg2rad(two_theta)) ** 2 / lamda**4

        solid_angle_ind = self.solid_angle_ind[det_ID]

        return L_inv * solid_angle_ind

    def approximate_norm(self, lamda, two_theta, det_ID):
        vol = self.get_volume_in_Q(lamda, two_theta, det_ID)

        i = np.searchsorted(self.lamda_x, lamda, side="right") - 1

        spec_ind = self.spec_ind[det_ID]
        sa_ind = self.sa_ind[det_ID]

        return self.spect_y[spec_ind][i] * self.sa_y[sa_ind] / vol

    def crop_for_normalization(self, event_name):
        """
        Convert units to momentum and crop to momentum band.

        event_name : str
            Name of raw event data.

        """

        if mtd.doesExist(event_name):
            ConvertUnits(
                InputWorkspace=event_name,
                OutputWorkspace=event_name,
                Target="Momentum",
            )

            CropWorkspaceForMDNorm(
                InputWorkspace=event_name,
                XMin=self.k_min,
                XMax=self.k_max,
                OutputWorkspace=event_name,
            )

    def load_background(self, filename, event_name):
        """
        Load a background file and scale to data.

        Parameters
        ----------
        filename : str
            Background file.
        event_name : str
            Name of raw event data.

        """

        if not mtd.doesExist("bkg_md") and filename is not None:
            if not mtd.doesExist("bkg"):
                Load(Filename=filename, OutputWorkspace="bkg")

                ConvertUnits(
                    InputWorkspace="bkg",
                    OutputWorkspace="bkg",
                    Target="Momentum",
                )

                CropWorkspaceForMDNorm(
                    InputWorkspace="bkg",
                    XMin=self.k_min,
                    XMax=self.k_max,
                    OutputWorkspace="bkg",
                )

                # if self.grouping is not None:
                #     self.group_pixels("bkg")

                Rebin(
                    InputWorkspace="bkg",
                    Params=[self.k_min, self.k_max, self.k_max],
                    OutputWorkspace="bkg",
                )

                MaskDetectorsIf(
                    InputWorkspace="bkg",
                    Operator="LessEqual",
                    OutputWorkspace="bkg",
                )

                if mtd.doesExist("sa_mask"):
                    MaskDetectors(Workspace="bkg", MaskedWorkspace="sa_mask")

                if mtd.doesExist("mask"):
                    MaskDetectors(Workspace="bkg", MaskedWorkspace="mask")

                if not mtd["bkg"].run().hasProperty("NormalizationFactor"):
                    NormaliseByCurrent(
                        InputWorkspace="bkg", OutputWorkspace="bkg"
                    )

            pc = mtd["bkg"].run().getProperty("gd_prtn_chrg").value

            CreateSingleValuedWorkspace(
                DataValue=pc, OutputWorkspace="pc_scale"
            )

            Multiply(
                LHSWorkspace="bkg",
                RHSWorkspace="pc_scale",
                OutputWorkspace="bkg",
            )

            self.convert_to_Q_lab("bkg", "bkg_md")

            DeleteWorkspace(Workspace="bkg")

    def normalize_to_hkl(self, md, projections, extents, bins, symmetry=None):
        """
        Normalize to binned hkl.

        Parameters
        ----------
        md : str
            3D Q-sample data.
        projections : list
            Projection axis vectors.
        extents : list
            Min/max pairs defining the bin center limits.
        bins : list
            Number of bins.
        symmetry : str, optional
            Laue point group. The default is None.

        """

        if mtd.doesExist(md) and mtd.doesExist("sa") and mtd.doesExist("flux"):
            v0, v1, v2 = projections

            (Q0_min, Q0_max), (Q1_min, Q1_max), (Q2_min, Q2_max) = extents

            nQ0, nQ1, nQ2 = bins

            Q0_min, Q0_max, dQ0 = self.calculate_binning_from_bins(
                Q0_min, Q0_max, nQ0
            )

            Q1_min, Q1_max, dQ1 = self.calculate_binning_from_bins(
                Q1_min, Q1_max, nQ1
            )

            Q2_min, Q2_max, dQ2 = self.calculate_binning_from_bins(
                Q2_min, Q2_max, nQ2
            )

            bkg_ws = self.workspace_exists("bkg_md")

            bkg_data = md + "_bkg_data" if mtd.doesExist("bkg_md") else None
            bkg_norm = md + "_bkg_norm" if mtd.doesExist("bkg_md") else None

            _data = self.workspace_exists(md + "_data")
            _norm = self.workspace_exists(md + "_norm")

            __data = self.workspace_exists(md + "_bkg_data")
            __norm = self.workspace_exists(md + "_bkg_norm")

            MDNorm(
                InputWorkspace=md,
                SolidAngleWorkspace="sa",
                FluxWorkspace="flux",
                BackgroundWorkspace=bkg_ws,
                QDimension0=v0,
                QDimension1=v1,
                QDimension2=v2,
                Dimension0Name="QDimension0",
                Dimension1Name="QDimension1",
                Dimension2Name="QDimension2",
                Dimension0Binning=[Q0_min, dQ0, Q0_max],
                Dimension1Binning=[Q1_min, dQ1, Q1_max],
                Dimension2Binning=[Q2_min, dQ2, Q2_max],
                SymmetryOperations=symmetry,
                TemporaryDataWorkspace=_data,
                TemporaryNormalizationWorkspace=_norm,
                TemporaryBackgroundDataWorkspace=__data,
                TemporaryBackgroundNormalizationWorkspace=__norm,
                OutputWorkspace=md + "_result",
                OutputDataWorkspace=md + "_data",
                OutputNormalizationWorkspace=md + "_norm",
                OutputBackgroundDataWorkspace=bkg_data,
                OutputBackgroundNormalizationWorkspace=bkg_norm,
            )

    def filter_events(self, ws, run, runs):
        """
        Split workspaces according to log.
        If no splitting, index according to run number.

        Parameters
        ----------
        ws : str
            Workspace to split.
        run : int
            Run number.
        runs : list
            All run numbers

        Returns
        -------
        indices : list, str
            Split index.
        workspaces : list, float
            Split workspace.

        """

        if mtd.doesExist("split"):
            if mtd["split"].rowCount() > 0:
                FilterEvents(
                    InputWorkspace=ws,
                    SplitterWorkspace="split",
                    InformationWorkspace="info",
                    OutputWorkspaceBaseName=ws + "_split",
                    GroupWorkspaces=True,
                    CorrectionToSample="Elastic",
                )

                workspaces = list(mtd[ws + "_split"].getNames())
                indices = [int(ws.split("_")[-1]) for ws in workspaces]

            else:
                return [], []

        else:
            workspaces = [ws]
            indices = [runs.index(run)]

        return indices, workspaces
