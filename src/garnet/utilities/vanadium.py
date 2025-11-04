import os
import sys

import yaml

import numpy as np

import scipy.signal

from mantid.simpleapi import (
    Load,
    LoadNexus,
    SaveNexus,
    LoadEmptyInstrument,
    LoadIsawDetCal,
    LoadParameterFile,
    ApplyCalibration,
    CompressEvents,
    NormaliseByCurrent,
    NormaliseSpectra,
    SortEvents,
    IntegrateFlux,
    Rebin,
    Minus,
    Divide,
    Multiply,
    ConvertUnits,
    CropWorkspace,
    AddSampleLog,
    RemoveLogs,
    CreateGroupingWorkspace,
    GroupDetectors,
    RemoveMaskedSpectra,
    MaskDetectors,
    MaskDetectorsIf,
    MaskBTP,
    ClearMaskFlag,
    InvertMask,
    ExtractMask,
    SetSample,
    SetBeam,
    AbsorptionCorrection,
    MultipleScatteringCorrection,
    CreateSingleValuedWorkspace,
    SmoothNeighbours,
    InterpolatingRebin,
    CopyInstrumentParameters,
    GenerateGoniometerIndependentBackground,
    SaveDetectorsGrouping,
    FilterBadPulses,
    mtd,
)


class Vanadium:
    def __init__(self, config):
        defaults = {
            "Instrument": "TOPAZ",
            "VanadiumIPTS": 31856,
            "VanadiumRuns": None,
            "NoSampleIPTS": 31856,
            "NoSampleRuns": None,
            "OutputFolder": "",
            "DetectorCalibration": None,
            "TubeCalibration": None,
            "InstrumentDefinition": None,
            "SampleShape": "sphere",
            "Diameter": 4,
            "Height": None,
            "BeamDiameter": None,
            "MomentumLimits": [1.8, 18],
            "MaskOptions": None,
            "Grouping": [4, 4],
        }

        defaults.update(config)

        self.instrument = defaults.get("Instrument")

        self.van_ipts = defaults.get("VanadiumIPTS")
        self.van_nos = defaults.get("VanadiumRuns")

        self.bkg_ipts = defaults.get("NoSampleIPTS")
        self.bkg_nos = defaults.get("NoSampleRuns")

        self.output_folder = defaults.get("OutputFolder")

        self.detector_calibration = defaults.get("DetectorCalibration")
        self.tube_calibration = defaults.get("TubeCalibration")
        self.instrument_definition = defaults.get("InstrumentDefinition")

        self.sample_shape = defaults.get("SampleShape")
        self.diameter = defaults.get("Diameter")
        self.height = defaults.get("Height")

        self.beam_diameter = defaults.get("BeamDiameter")

        self.mask_options = defaults.get("MaskOptions") or {}
        self.x_bins, self.y_bins = defaults.get("Grouping")

        self.file_folder = "/SNS/{}/IPTS-{}/nexus/"
        self.file_name = "{}_{}.nxs.h5"
        self.vanadium_folder = "/SNS/{}/shared/Vanadium"

        self.n_bins = 500

        self.k_min, self.k_max = defaults.get("MomentumLimits")
        self.k_step = (self.k_max - self.k_min) / self.n_bins

        self.lamda_min = 2 * np.pi / self.k_max
        self.lamda_max = 2 * np.pi / self.k_min
        self.lamda_step = (self.lamda_max - self.lamda_min) / self.n_bins

        self.mult_scatt = False

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )
        CreateGroupingWorkspace(
            InputWorkspace=self.instrument,
            GroupDetectorsBy="bank",
            OutputWorkspace="group",
        )
        vanadium_folder = self.vanadium_folder.format(self.instrument)
        output_folder = os.path.join(vanadium_folder, self.output_folder)

        self.grouping = os.path.join(output_folder, "grouping.xml")

        SaveDetectorsGrouping(InputWorkspace="group", OutputFile=self.grouping)

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

    def apply_masks(self):
        if self.mask_options.get("Banks") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Bank=self._join(self.mask_options["Banks"]),
            )

        if self.mask_options.get("Pixels") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Pixel=self._join(self.mask_options["Pixels"]),
            )

        if self.mask_options.get("Tubes") is not None:
            MaskBTP(
                Workspace=self.instrument,
                Tube=self._join(self.mask_options["Tubes"]),
            )

        if self.mask_options.get("BankTube") is not None:
            for bank, tube in self.mask_options["BankTube"]:
                MaskBTP(
                    Workspace=self.instrument,
                    Bank=self._join(bank),
                    Tube=self._join(tube),
                )

        if self.mask_options.get("BankTubePixel") is not None:
            for bank, tube, pixel in self.mask_options["BankTubePixel"]:
                MaskBTP(
                    Workspace=self.instrument,
                    Bank=self._join(bank),
                    Tube=self._join(tube),
                    Pixel=self._join(pixel),
                )

        ExtractMask(
            InputWorkspace=self.instrument,
            UngroupDetectors=True,
            OutputWorkspace="mask",
        )

        ClearMaskFlag(Workspace=self.instrument)

        InvertMask(InputWorkspace="mask", OutputWorkspace="active")

        SmoothNeighbours(
            InputWorkspace="active",
            OutputWorkspace="pixels",
            SumPixelsX=self.x_bins,
            SumPixelsY=self.y_bins,
        )

        MaskDetectors(Workspace="pixels", MaskedWorkspace="mask")

    def apply_calibration(self):
        if self.tube_calibration is not None:
            LoadNexus(
                Filename=self.tube_calibration,
                OutputWorkspace="tube_table",
            )
            ApplyCalibration(
                Workspace=self.instrument, CalibrationTable="tube_table"
            )

        if self.detector_calibration is not None:
            ext = os.path.splitext(self.detector_calibration)[1]
            if ext == ".xml":
                LoadParameterFile(
                    Workspace=self.instrument,
                    Filename=self.detector_calibration,
                )
            else:
                LoadIsawDetCal(
                    InputWorkspace=self.instrument,
                    Filename=self.detector_calibration,
                )

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

    def load_runs(self, workspace, ipts, run_nos):
        if not isinstance(run_nos, list):
            run_nos = self._runs_string_to_list(run_nos)

        files_to_load = "+".join(
            [
                os.path.join(
                    self.file_folder.format(self.instrument, ipts),
                    self.file_name.format(self.instrument, run_no),
                )
                for run_no in run_nos
            ]
        )

        Load(
            Filename=files_to_load,
            NumberOfBins=1,
            LoadType="Multiprocess (experimental)",
            AllowList="gd_prtn_chrg",
            OutputWorkspace=workspace,
        )

        FilterBadPulses(InputWorkspace=workspace, OutputWorkspace=workspace)

        NormaliseByCurrent(InputWorkspace=workspace, OutputWorkspace=workspace)

        CopyInstrumentParameters(
            InputWorkspace=self.instrument,
            OutputWorkspace=workspace,
        )

        ConvertUnits(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            Target="Momentum",
        )

        CropWorkspace(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            XMin=self.k_min,
            XMax=self.k_max,
        )

        Rebin(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            Params=[self.k_min, (self.k_max - self.k_min) / 50, self.k_max],
            PreserveEvents=True,
        )

        if mtd[workspace].isGroup():
            vals = []

            for ws in mtd[workspace].getNames():
                pc = mtd[ws].run().getProperty("gd_prtn_chrg")

                val = pc.valueAsStr
                uni = pc.units

                RemoveLogs(Workspace=workspace)

                logs = ["gd_prtn_chrg", "NormalizationFactor"]
                for log in logs:
                    AddSampleLog(
                        Workspace=workspace,
                        LogName=log,
                        LogText="1.0",
                        LogUnit=uni,
                        LogType="Number",
                        NumberType="Double",
                    )

                vals.append(float(val))

            GenerateGoniometerIndependentBackground(
                InputWorkspaces=workspace,
                OutputWorkspace=workspace,
                GroupingFile=self.grouping,
                PercentMin=0,
                PercentMax=95,
            )

            logs = ["gd_prtn_chrg", "NormalizationFactor"]
            for log in logs:
                AddSampleLog(
                    Workspace=workspace,
                    LogName=log,
                    LogText=str(np.mean(vals)),
                    LogUnit=uni,
                    LogType="Number",
                    NumberType="Double",
                )

        SmoothNeighbours(
            InputWorkspace=workspace,
            OutputWorkspace=workspace,
            SumPixelsX=self.x_bins,
            SumPixelsY=self.y_bins,
        )

        MaskDetectors(Workspace=workspace, MaskedWorkspace="mask")

        CompressEvents(
            InputWorkspace=workspace, Tolerance=1e-3, OutputWorkspace=workspace
        )

        pc = mtd[workspace].run().getProperty("gd_prtn_chrg")

        val = pc.valueAsStr
        uni = pc.units

        RemoveLogs(Workspace=workspace)

        logs = ["gd_prtn_chrg", "NormalizationFactor"]
        for log in logs:
            AddSampleLog(
                Workspace=workspace,
                LogName=log,
                LogText=val,
                LogUnit=uni,
                LogType="Number",
                NumberType="Double",
            )

    def subtract_background(self):
        Minus(
            LHSWorkspace="vanadium",
            RHSWorkspace="background",
            OutputWorkspace="vanadium",
        )

    def _vanadium_niobium_lattice_constant(self, x):
        """
        Smith, J. F.; Carlson, O. N. The Nb−V (Niobium-Vanadium) System.
        Bulletin of Alloy Phase Diagrams 1983, 4 (1), 46–49.
        https://doi.org/10.1007/BF02880319.
        """
        y = 2 * x - 1
        a = 3.19199921 + 0.13954993 * y - 0.02242883 * y**2
        return a

    def set_sample_geometry(self, x=0):
        if self.sample_shape == "sphere":
            shape = {
                "Shape": "Sphere",
                "Radius": self.diameter * 0.05,
                "Center": [0.0, 0.0, 0.0],
            }
        else:
            shape = {
                "Shape": "Cylinder",
                "Height": self.height * 0.1,
                "Radius": self.diameter * 0.05,
                "Axis": [0.0, 1.0, 0.0],
                "Center": [0.0, 0.0, 0.0],
            }

        a = self._vanadium_niobium_lattice_constant(x)

        material = {
            "ChemicalFormula": "V{} Nb{}".format(1 - x, x),
            "ZParameter": 2.0,
            "UnitCellVolume": float(a**3),
        }

        SetSample(InputWorkspace="vanadium", Geometry=shape, Material=material)

        if self.beam_diameter is not None:
            beam = {"Shape": "Circle", "Radius": self.beam_diameter * 0.05}
            SetBeam(InputWorkspace="vanadium", Geometry=beam)

        mat = mtd["vanadium"].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        self.sigma_a = sigma_a
        self.sigma_s = sigma_s

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective  # A^-3
        N = mat.totalAtoms

        self.n = n

        V = np.abs(
            mtd["vanadium"].sample().getShape().volume() * 100**3
        )  # cm^3

        rho = (n / N) / 0.6022 * M
        m = rho * V
        r = np.cbrt(0.75 / np.pi * V)

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        mu = mat.numberDensityEffective * (
            mat.totalScatterXSection() + mat.absorbXSection(1.8)
        )

        print("V\n")
        print("absoption cross section: {:.4f} barn\n".format(sigma_a))
        print("scattering cross section: {:.4f} barn\n".format(sigma_s))

        print("linear absorption coefficient: {:.4f} 1/cm\n".format(mu_a))
        print("linear scattering coefficient: {:.4f} 1/cm\n".format(mu_s))
        print("absorption parameter: {:.4f} \n".format(mu * r))

        print("total atoms: {:.4f}\n".format(N))
        print("molar mass: {:.4f} g/mol\n".format(M))
        print("number density: {:.4f} 1/A^3\n".format(n))

        print("mass density: {:.4f} g/cm^3\n".format(rho))
        print("volume: {:.4f} cm^3\n".format(V))
        print("mass: {:.4f} g\n".format(m))
        print("equivalent radius: {:.4} cm".format(r))

        self.r = r * 10

    def apply_absorption_correction(self):
        ConvertUnits(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Target="Wavelength",
        )

        Rebin(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Params=[self.lamda_min, self.lamda_step, self.lamda_max],
            PreserveEvents=True,
        )

        AbsorptionCorrection(
            InputWorkspace="vanadium",
            NumberOfWavelengthPoints=10,
            ElementSize=self.r / 10,
            OutputWorkspace="correction",
        )

        if self.mult_scatt:
            MultipleScatteringCorrection(
                InputWorkspace="vanadium",
                NumberOfWavelengthPoints=20,
                Method="SampleOnly",
                ElementSize=self.r / 20,
                OutputWorkspace="factor",
            )

        CreateSingleValuedWorkspace(OutputWorkspace="unity", DataValue=1)

        Divide(
            LHSWorkspace="unity",
            RHSWorkspace="correction",
            OutputWorkspace="scale",
        )

        if mtd.doesExist("factor_ms_sampleOnly"):
            Minus(
                LHSWorkspace="scale",
                RHSWorkspace="factor_ms_sampleOnly",
                OutputWorkspace="scale",
            )

        Multiply(
            LHSWorkspace="vanadium",
            RHSWorkspace="scale",
            OutputWorkspace="vanadium",
        )

        GroupDetectors(
            InputWorkspace="vanadium",
            CopyGroupingFromWorkspace="group",
            Behaviour="Sum",
            PreserveEvents=False,
            OutputWorkspace="spectra",
        )

        Rebin(
            InputWorkspace="spectra",
            OutputWorkspace="spectra",
            Params=[self.lamda_min, self.lamda_step, self.lamda_max],
            PreserveEvents=False,
        )

        Rebin(
            InputWorkspace="spectra",
            OutputWorkspace="norm",
            Params=[self.lamda_min, self.lamda_max, self.lamda_max],
            PreserveEvents=False,
        )

        Divide(
            LHSWorkspace="spectra",
            RHSWorkspace="norm",
            OutputWorkspace="spectra",
        )

        X = mtd["spectra"].getXDimension()
        lamda_min = X.getMinimum() + X.getBinWidth()
        lamda_max = X.getMaximum() - X.getBinWidth()
        lamda_step = self.lamda_step / 200

        InterpolatingRebin(
            InputWorkspace="spectra",
            OutputWorkspace="spectra",
            Params=[lamda_min, lamda_step, lamda_max],
        )

        ConvertUnits(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Target="Momentum",
        )

        Rebin(
            InputWorkspace="vanadium",
            OutputWorkspace="vanadium",
            Params=[self.k_min, self.k_max, self.k_max],
            PreserveEvents=True,
        )

    def _smooth_data(self, data, units="wavelength"):
        y = mtd[data].extractY().copy()
        x = mtd[data].extractX()

        x = 0.5 * (x[:, 1:] + x[:, :-1])

        for i, (x_val, y_val) in enumerate(zip(x, y)):
            weights = (
                x_val**2 if units == "momentum" else np.ones_like(x_val)
            )

            y_hat = scipy.signal.savgol_filter(
                y_val * weights,
                window_length=25,
                polyorder=1,
            )

            mtd[data].setY(i, y_hat / weights)

    def process_data(self):
        Rebin(
            InputWorkspace="vanadium",
            OutputWorkspace="solid_angle",
            Params=[self.k_min, self.k_max, self.k_max],
            PreserveEvents=False,
        )

        GroupDetectors(
            InputWorkspace="vanadium",
            CopyGroupingFromWorkspace="group",
            Behaviour="Sum",
            PreserveEvents=True,
            OutputWorkspace="incident",
        )

        MaskDetectorsIf(
            InputWorkspace="incident",
            Operator="LessEqual",
            OutputWorkspace="incident",
        )

        RemoveMaskedSpectra(
            InputWorkspace="incident",
            MaskedWorkspace="incident",
            OutputWorkspace="incident",
        )

        SortEvents(InputWorkspace="incident", SortBy="X Value")

        Rebin(
            InputWorkspace="incident",
            OutputWorkspace="incident",
            Params=[self.k_min, self.k_step, self.k_max],
            PreserveEvents=False,
        )

        # self._smooth_data("incident", units="momentum")

        X = mtd["incident"].getXDimension()
        k_min = X.getMinimum() + X.getBinWidth()
        k_max = X.getMaximum() - X.getBinWidth()
        k_step = self.k_step / 200

        InterpolatingRebin(
            InputWorkspace="incident",
            OutputWorkspace="incident",
            Params=[k_min, k_step, k_max],
        )

        IntegrateFlux(
            InputWorkspace="incident",
            NPoints=self.n_bins * 200,
            OutputWorkspace="flux",
        )

        NormaliseSpectra(InputWorkspace="flux", OutputWorkspace="flux")

        MaskDetectorsIf(
            InputWorkspace="solid_angle",
            Operator="LessEqual",
            OutputWorkspace="solid_angle",
        )

    def finalize_and_save(self):
        vanadium_folder = self.vanadium_folder.format(self.instrument)
        output_folder = os.path.join(vanadium_folder, self.output_folder)
        workspaces = [
            "background",
            "incident",
            "flux",
            "spectra",
            "solid_angle",
            "correction",
            "scale",
        ]
        for workspace in workspaces:
            filename = os.path.join(output_folder, workspace + ".nxs")
            SaveNexus(InputWorkspace=workspace, Filename=filename)

    def run(self):
        self.load_instrument()
        self.apply_masks()
        self.apply_calibration()
        self.load_runs("vanadium", self.van_ipts, self.van_nos)
        self.load_runs("background", self.bkg_ipts, self.bkg_nos)
        self.subtract_background()
        self.set_sample_geometry()
        self.apply_absorption_correction()
        self.process_data()
        self.finalize_and_save()


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    norm = Vanadium(params)
    norm.run()
