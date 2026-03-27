import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import yaml

import numpy as np

from mantid.simpleapi import (
    Load,
    LoadNexus,
    SaveNexus,
    LoadEmptyInstrument,
    ExtractMonitors,
    LoadIsawDetCal,
    LoadParameterFile,
    ApplyCalibration,
    ConvertUnits,
    CropWorkspace,
    SetGoniometer,
    CreateSingleValuedWorkspace,
    CopyInstrumentParameters,
    SetUB,
    SaveIsawUB,
    ReorientUnitCell,
    FindUBUsingLatticeParameters,
    OptimizeLatticeForCellType,
    ConvertToMD,
    PlusMD,
    SaveMD,
    LoadMD,
    ConvertQtoHKLMDHisto,
    PreprocessDetectorsToMD,
    IndexPeaks,
    FindPeaksMD,
    CloneWorkspace,
    CombinePeaksWorkspaces,
    mtd,
)
import multiprocessing


def _process_run(config, ipts, run, idx, tol):
    instrument = config["instrument"]
    file_folder = config["file_folder"]
    file_name = config["file_name"]
    output_folder = config["output_folder"]
    wavelength_band = config["wavelength_band"]
    gon_axis = config["gon_axis"]
    Q_max = config["Q_max"]
    Q_min = config["Q_min"]
    max_peaks = config["max_peaks"]
    strong_threshold = config["strong_threshold"]
    a = config["a"]
    b = config["b"]
    c = config["c"]
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    cell_type = config["cell_type"]
    crystal_system = config["crystal_system"]
    lattice_system = config["lattice_system"]
    tube_calibration = config["tube_calibration"]
    detector_calibration = config["detector_calibration"]

    file_to_load = os.path.join(
        file_folder.format(instrument, ipts), file_name.format(instrument, run)
    )

    data_ws = "data"
    md_ws = "md"
    strong_ws = "strong"
    combine_ws = "combine"

    Load(Filename=file_to_load, OutputWorkspace=data_ws)

    if tube_calibration is not None:
        LoadNexus(
            Filename=tube_calibration,
            OutputWorkspace="tube_table",
        )
        ApplyCalibration(Workspace=data_ws, CalibrationTable="tube_table")

    if detector_calibration is not None:
        ext = os.path.splitext(detector_calibration)[1]
        if ext == ".xml":
            LoadParameterFile(
                Workspace=data_ws,
                Filename=detector_calibration,
            )
        else:
            LoadIsawDetCal(
                InputWorkspace=data_ws,
                Filename=detector_calibration,
            )

    ConvertUnits(
        InputWorkspace=data_ws, OutputWorkspace=data_ws, Target="Wavelength"
    )

    CropWorkspace(
        InputWorkspace=data_ws,
        OutputWorkspace=data_ws,
        XMin=wavelength_band[0],
        XMax=wavelength_band[1],
    )

    SetGoniometer(
        Workspace=data_ws,
        Goniometers="None, Specify Individually",
        Axis0=gon_axis[0],
        Axis1=gon_axis[1],
        Axis2=gon_axis[2],
        Axis3=gon_axis[3],
        Axis4=gon_axis[4],
        Axis5=gon_axis[5],
        Average=True,
    )

    ConvertToMD(
        InputWorkspace=data_ws,
        QDimensions="Q3D",
        dEAnalysisMode="Elastic",
        Q3DFrames="Q_sample",
        LorentzCorrection=True,
        MinValues=[-Q_max] * 3,
        MaxValues=[+Q_max] * 3,
        OutputWorkspace=md_ws,
    )

    FindPeaksMD(
        InputWorkspace=md_ws,
        MaxPeaks=max_peaks,
        PeakDistanceThreshold=Q_min,
        DensityThresholdFactor=strong_threshold,
        OutputWorkspace=strong_ws,
    )

    FindUBUsingLatticeParameters(
        PeaksWorkspace=strong_ws,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        Tolerance=tol,
        Iterations=3,
        NumInitial=100,
        FixParameters=True,
    )

    IndexPeaks(PeaksWorkspace=strong_ws, Tolerance=tol, RoundHKLs=True)

    OptimizeLatticeForCellType(
        PeaksWorkspace=strong_ws,
        CellType=cell_type,
        Apply=True,
        Tolerance=tol,
        OutputDirectory=output_folder,
    )

    ReorientUnitCell(
        PeaksWorkspace=strong_ws,
        Tolerance=tol,
        CrystalSystem=crystal_system,
        LatticeSystem=lattice_system,
    )

    mat_file = os.path.join(output_folder, "{}.mat".format(run))

    SaveIsawUB(InputWorkspace=strong_ws, Filename=mat_file)

    peaks_file = os.path.join(output_folder, "peaks_{}.nxs".format(run))
    SaveNexus(InputWorkspace=strong_ws, Filename=peaks_file)

    extents = [
        -config["h_max"] / 2,
        +config["h_max"] / 2,
        -config["k_max"] / 2,
        +config["k_max"] / 2,
        -config["l_max"] / 2,
        +config["l_max"] / 2,
    ]

    ConvertQtoHKLMDHisto(
        InputWorkspace=md_ws,
        PeaksWorkspace=strong_ws,
        Extents=extents,
        OutputWorkspace=combine_ws,
    )

    md_filename = os.path.join(output_folder, "mdhkl_{}.nxs".format(run))
    SaveMD(
        InputWorkspace=combine_ws,
        Filename=md_filename,
        SaveHistory=False,
        SaveLogs=False,
        SaveInstrument=False,
    )


from garnet.config.instruments import beamlines


class Peaks:
    def __init__(self, config):
        defaults = {
            "Instrument": "TOPAZ",
            "InstrumentDefinition": None,
            "IPTS": 31856,
            "Runs": None,
            "PeaksTable": None,
            "OutputFolder": "",
            "UnitCellLengths": [5.431, 5.431, 5.431],
            "UnitCellAngles": [90, 90, 90],
            "CrystalSystem": "Cubic",
            "LatticeSystem": "Cubic",
            "MaxPeaks": 500,
            "PeakThreshold": 100,
        }
        defaults.update(config)

        self.instrument = defaults.get("Instrument")
        self.instrument_definition = defaults.get("InstrumentDefinition")

        self.ipts = defaults.get("IPTS")
        self.nos = defaults.get("Runs")

        self.output_folder = defaults.get("OutputFolder")

        self.detector_calibration = defaults.get("DetectorCalibration")
        self.tube_calibration = defaults.get("TubeCalibration")

        self.file_folder = "/SNS/{}/IPTS-{}/nexus/"
        self.file_name = "{}_{}.nxs.h5"
        self.calibration_folder = "/SNS/{}/shared/calibration"

        self.a, self.b, self.c = defaults.get("UnitCellLengths")
        self.alpha, self.beta, self.gamma = defaults.get("UnitCellAngles")

        self.crystal_system = defaults.get("CrystalSystem")
        self.lattice_system = defaults.get("LatticeSystem")

        if self.crystal_system != "Trigonal":
            self.lattice_system = None

        self.max_peaks = defaults.get("MaxPeaks")
        self.strong_threshold = defaults.get("PeakThreshold")

        inst_config = beamlines[self.instrument]

        self.gon_axis = 6 * [None]
        gon = inst_config.get("Goniometer")
        gon_axis_names = inst_config.get("GoniometerAxisNames")
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

        self.wavelength_band = defaults.get(
            "Wavelength", inst_config["Wavelength"]
        )

    def _join(self, items):
        if isinstance(items, list):
            return ",".join(
                [
                    "{}-{}".format(*r) if isinstance(r, list) else str(r)
                    for r in items
                ]
            )
        else:
            return str(items)

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )
        ExtractMonitors(
            InputWorkspace=self.instrument, DetectorWorkspace=self.instrument
        )

    def calculate_limits(self):
        PreprocessDetectorsToMD(
            InputWorkspace=self.instrument, OutputWorkspace="detectors"
        )

        two_theta = max(mtd["detectors"].column("TwoTheta"))
        lamda = min(self.wavelength_band)
        self.Q_max = 4 * np.pi / lamda * np.sin(0.5 * two_theta)

        CreateSingleValuedWorkspace(OutputWorkspace="sample")

        SetUB(
            Workspace="sample",
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        ol = mtd["sample"].sample().getOrientedLattice()
        astar, bstar, cstar = ol.astar(), ol.bstar(), ol.cstar()

        self.Q_min = 2 * np.pi * min([astar, bstar, cstar])
        self.d_max = 2 * np.pi / self.Q_min
        self.d_min = 2 * np.pi / self.Q_max

        self.h_max = np.floor(1 / self.d_min / astar)
        self.k_max = np.floor(1 / self.d_min / bstar)
        self.l_max = np.floor(1 / self.d_min / cstar)

    def _runs_string_to_list(self, runs_str):
        """
        Convert runs string to list.

        Parameters
        ----------
        runs_str : str
            Condensed notation for run numbers.

        Returns
        -------
        runs : list
            Integer run numbers.

        """

        if type(runs_str) is not str:
            runs_str = str(runs_str)

        runs = []
        ranges = runs_str.split(",")

        for part in ranges:
            if ":" in part:
                range_part, *skip_part = part.split(";")
                start, end = map(int, range_part.split(":"))
                skip = int(skip_part[0]) if skip_part else 1

                if start > end or skip <= 0:
                    return None

                runs.extend(range(start, end + 1, skip))
            else:
                runs.append(int(part))

        return runs

    def load_convert_runs(self, ipts, run_nos, tol=0.1, n_proc=10):
        if not isinstance(run_nos, list):
            run_nos = self._runs_string_to_list(run_nos)

        cell_type = (
            self.crystal_system
            if self.lattice_system is None
            else self.lattice_system
        )
        config = {
            "instrument": self.instrument,
            "file_folder": self.file_folder,
            "file_name": self.file_name,
            "output_folder": self.output_folder,
            "wavelength_band": self.wavelength_band,
            "gon_axis": self.gon_axis,
            "Q_max": self.Q_max,
            "Q_min": self.Q_min,
            "max_peaks": self.max_peaks,
            "strong_threshold": self.strong_threshold,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "cell_type": cell_type,
            "crystal_system": self.crystal_system,
            "lattice_system": self.lattice_system,
            "h_max": self.h_max,
            "k_max": self.k_max,
            "l_max": self.l_max,
            "tube_calibration": self.tube_calibration,
            "detector_calibration": self.detector_calibration,
        }

        args_list = [
            (config, ipts, run, i, tol) for i, run in enumerate(run_nos)
        ]

        multiprocessing.set_start_method("spawn", force=True)

        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(_process_run, args_list)

    def finalize_and_save(self, tol=0.1):
        md_files = [
            os.path.join(self.output_folder, f)
            for f in os.listdir(self.output_folder)
            if f.startswith("mdhkl_") and f.endswith(".nxs")
        ]

        for i, md in enumerate(md_files):
            LoadMD(Filename=md, OutputWorkspace="tmp")
            if i == 0:
                CloneWorkspace(InputWorkspace="tmp", OutputWorkspace="merge")
            else:
                PlusMD(
                    LHSWorkspace="tmp",
                    RHSWorkspace="merge",
                    OutputWorkspace="merge",
                )
            os.remove(md)

        peak_files = [
            os.path.join(self.output_folder, f)
            for f in os.listdir(self.output_folder)
            if f.startswith("peaks_") and f.endswith(".nxs")
        ]

        for i, sf in enumerate(peak_files):
            LoadNexus(Filename=sf, OutputWorkspace="tmp")
            if i == 0:
                CloneWorkspace(InputWorkspace="tmp", OutputWorkspace="peaks")
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace="tmp",
                    RHSWorkspace="peaks",
                    OutputWorkspace="peaks",
                )
            os.remove(sf)

        filename = os.path.join(self.output_folder, "peaks.nxs")
        SaveNexus(InputWorkspace="peaks", Filename=filename)

        filename = os.path.join(self.output_folder, "mdhkl.nxs")
        SaveMD(
            InputWorkspace="merge",
            Filename=filename,
            SaveHistory=False,
            SaveLogs=False,
            SaveInstrument=False,
        )

        IndexPeaks(PeaksWorkspace="peaks", Tolerance=tol, RoundHKLs=True)

        filename = os.path.join(self.output_folder, "peaks.mat")
        SaveIsawUB(InputWorkspace="peaks", Filename=filename)

    def run(self):
        self.load_instrument()
        self.calculate_limits()
        self.load_convert_runs(self.ipts, self.nos)
        self.finalize_and_save()


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    params["OutputFolder"] = os.path.dirname(os.path.abspath(config_file))

    peaks = Peaks(params)
    peaks.run()
