import os
import sys

import yaml

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from scipy.spatial.transform import Rotation
import scipy.optimize
import scipy.linalg

from mantid.simpleapi import (
    LoadNexus,
    SaveNexus,
    LoadEmptyInstrument,
    ExtractMonitors,
    LoadParameterFile,
    SaveParameterFile,
    SetInstrumentParameter,
    ClearInstrumentParameters,
    LoadIsawPeaks,
    FilterPeaks,
    IndexPeaks,
    CombinePeaksWorkspaces,
    TransformHKL,
    CalculateUMatrix,
    ApplyInstrumentToPeaks,
    SCDCalibratePanels,
    MoveInstrumentComponent,
    RotateInstrumentComponent,
    CreateEmptyTableWorkspace,
    SaveAscii,
    CloneWorkspace,
    DeleteWorkspace,
    PreprocessDetectorsToMD,
    mtd,
)

from mantid.geometry import UnitCell, PointGroupFactory


class Calibration:
    def __init__(self, config):
        defaults = {
            "Instrument": "TOPAZ",
            "InstrumentDefinition": None,
            "PeaksTable": None,
            "OutputFolder": "",
            "UnitCellLengths": [5.431, 5.431, 5.431],
            "UnitCellAngles": [90, 90, 90],
            "CrystalSystem": "Cubic",
            "LatticeSystem": "Cubic",
            "RefineGoniometer": False,
        }

        defaults.update(config)

        self.instrument = defaults.get("Instrument")
        self.instrument_definition = defaults.get("InstrumentDefinition")

        self.peaks = defaults.get("PeaksTable")

        self.output_folder = defaults.get("OutputFolder")

        self.calibration_folder = "/SNS/{}/shared/calibration"

        self.a, self.b, self.c = defaults.get("UnitCellLengths")
        self.alpha, self.beta, self.gamma = defaults.get("UnitCellAngles")

        self.crystal_system = defaults.get("CrystalSystem")
        self.lattice_system = defaults.get("LatticeSystem")

        self.refine_off = defaults.get("RefineGoniometer")

        self.iterations = 3

    def load_peaks(self):
        ext = os.path.splitext(self.peaks)[1]
        if ext == ".nxs":
            LoadNexus(
                OutputWorkspace="peaks",
                Filename=self.peaks,
            )
        else:
            LoadIsawPeaks(
                OutputWorkspace="peaks",
                Filename=self.peaks,
            )

        FilterPeaks(
            InputWorkspace="peaks",
            OutputWorkspace="peaks",
            FilterVariable="Signal/Noise",
            FilterValue=5,
            Operator=">",
        )

        uc = UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )

        self.goniometer_dict = {}

        for peak in mtd["peaks"]:
            peak.setIntensity(0)
            peak.setSigmaIntensity(uc.d(*peak.getIntHKL()))
            peak.setBinCount(0)
            run = peak.getRunNumber()
            R = peak.getGoniometerMatrix().copy()
            self.goniometer_dict[run] = R

        self.d_dict = {}

        banks = mtd["peaks"].column("BankName")

        for i, peak in enumerate(mtd["peaks"]):
            d = peak.getDSpacing()
            d0 = uc.d(*peak.getIntHKL())

            key = banks[i].strip("bank")
            items = self.d_dict.get(key)
            if items is None:
                items = [], []
            items[0].append(d0)
            items[1].append(d)

            self.d_dict[key] = items

        CreateEmptyTableWorkspace(OutputWorkspace="goniometer")

        mtd["goniometer"].addColumn("float", "Refined X Angle")
        mtd["goniometer"].addColumn("float", "Refined Y Angle")
        mtd["goniometer"].addColumn("float", "Refined Z Angle")
        mtd["goniometer"].addColumn("float", "Offset Chi")

    def reindex_peaks(self):
        runs = np.unique(mtd["peaks"].column("RunNumber")).tolist()

        transforms = self.cell_symmetry_matrices()

        for i, run in enumerate(runs):
            FilterPeaks(
                InputWorkspace="peaks",
                OutputWorkspace="tmp",
                FilterVariable="RunNumber",
                FilterValue=run,
                Operator="=",
            )

            CalculateUMatrix(
                PeaksWorkspace="tmp",
                a=self.a,
                b=self.b,
                c=self.c,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
            if i == 0:
                U = mtd["tmp"].sample().getOrientedLattice().getU().copy()

                CloneWorkspace(
                    InputWorkspace="tmp", OutputWorkspace="peaks_ws"
                )
            else:
                UB = mtd["tmp"].sample().getOrientedLattice().getUB().copy()
                B = mtd["tmp"].sample().getOrientedLattice().getB().copy()

                cost = np.inf
                for order, M in transforms.items():
                    Up = UB @ np.linalg.inv(M) @ np.linalg.inv(B)
                    cos_theta = (np.trace(Up.T @ U) - 1) / 2
                    theta = np.arccos(np.clip(cos_theta, -1, 1))
                    if theta < cost:
                        cost = theta
                        T = M.copy()

                hkl_trans = ",".join(9 * ["{}"]).format(*T.flatten())

                TransformHKL(
                    PeaksWorkspace="tmp",
                    HKLTransform=hkl_trans,
                    FindError=False,
                )

                CombinePeaksWorkspaces(
                    LHSWorkspace="peaks_ws",
                    RHSWorkspace="tmp",
                    OutputWorkspace="peaks_ws",
                )

            DeleteWorkspace(Workspace="tmp")

        CloneWorkspace(InputWorkspace="peaks_ws", OutputWorkspace="peaks")

        DeleteWorkspace(Workspace="peaks_ws")

    def cell_symmetry_matrices(self):
        if self.crystal_system == "Cubic":
            symbol = "m-3m"
        elif self.crystal_system == "Hexagonal":
            symbol = "6/mmm"
        elif self.crystal_system == "Tetragonal":
            symbol = "4/mmm"
        elif self.crystal_system == "Trigonal":
            if self.lattice_system == "Rhombohedral":
                symbol = "-3m r"
            elif self.lattice_system == "Hexagonal":
                symbol = "-3m"
        elif self.crystal_system == "Orthorhombic":
            symbol = "mmm"
        elif self.crystal_system == "Monoclinic":
            symbol = "2/m"
        elif self.crystal_system == "Triclinic":
            symbol = "-1"

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transforms = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
                transforms[name] = T

        return transforms

    def initialize_peaks(self):
        runs = np.unique(mtd["peaks"].column("RunNumber")).tolist()

        for i, run in enumerate(runs):
            FilterPeaks(
                InputWorkspace="peaks",
                OutputWorkspace="tmp",
                FilterVariable="RunNumber",
                FilterValue=run,
                Operator="=",
            )

            for peak in mtd["tmp"]:
                peak.setGoniometerMatrix(np.eye(3))

            CalculateUMatrix(
                PeaksWorkspace="tmp",
                a=self.a,
                b=self.b,
                c=self.c,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            RU = mtd["tmp"].sample().getOrientedLattice().getU().copy()

            for peak in mtd["tmp"]:
                peak.setGoniometerMatrix(RU)

            mtd["tmp"].sample().getOrientedLattice().setU(np.eye(3))

            if i == 0:
                CloneWorkspace(
                    InputWorkspace="tmp", OutputWorkspace="peaks_ws"
                )
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace="tmp",
                    RHSWorkspace="peaks_ws",
                    OutputWorkspace="peaks_ws",
                )

            DeleteWorkspace(Workspace="tmp")

        CloneWorkspace(InputWorkspace="peaks_ws", OutputWorkspace="peaks")

        DeleteWorkspace(Workspace="peaks_ws")

        mtd["peaks"].sample().getOrientedLattice().setU(np.eye(3))

        IndexPeaks(PeaksWorkspace="peaks")

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )

        ExtractMonitors(
            InputWorkspace=self.instrument,
            MonitorWorkspace="monitors",
            DetectorWorkspace=self.instrument,
        )

    def _get_output_folder(self):
        calibration_folder = self.calibration_folder.format(self.instrument)
        output_folder = os.path.join(calibration_folder, self.output_folder)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _get_ouput(self, ext=".xml"):
        return os.path.join(self._get_output_folder(), "calibration" + ext)

    def calibrate_instrument(self, iteration):
        SCDCalibratePanels(
            PeakWorkspace="peaks",
            Tolerance=0.2,
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            OutputWorkspace="calibration_table",
            DetCalFilename=self._get_ouput(".DetCal"),
            CSVFilename=self._get_ouput(".csv"),
            XmlFilename=self._get_ouput(".xml"),
            CalibrateT0=False,
            SearchRadiusT0=10,
            CalibrateL1=True,
            SearchRadiusL1=0.5,
            CalibrateBanks=True,
            SearchRadiusTransBank=0.5,
            SearchRadiusRotXBank=15 if self.instrument != "CORELLI" else 0,
            SearchRadiusRotYBank=15,
            SearchRadiusRotZBank=15 if self.instrument != "CORELLI" else 0,
            VerboseOutput=True,
            SearchRadiusSamplePos=0.01,
            TuneSamplePosition=True,
            CalibrateSize=self.instrument != "CORELLI",
            SearchRadiusSize=0.15,
            FixAspectRatio=False,
        )

        LoadParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput(".xml"),
        )

        inst = mtd[self.instrument].getInstrument()
        sample_pos = inst.getComponentByName("sample-position").getPos()

        components = np.unique(mtd["peaks"].column("BankName")).tolist()
        components += ["sample-position"]

        for component in components:
            MoveInstrumentComponent(
                Workspace=self.instrument,
                ComponentName=component,
                X=-sample_pos[0],
                Y=-sample_pos[1],
                Z=-sample_pos[2],
                RelativePosition=True,
            )

        MoveInstrumentComponent(
            Workspace=self.instrument,
            ComponentName="moderator",
            X=0,
            Y=0,
            Z=-sample_pos[2],
            RelativePosition=True,
        )

        ApplyInstrumentToPeaks(
            InputWorkspace="peaks",
            InstrumentWorkspace=self.instrument,
            OutputWorkspace="peaks",
        )

        CloneWorkspace(InputWorkspace="peaks", OutputWorkspace="peaks_ws")

        for peak in mtd["peaks_ws"]:
            run = peak.getRunNumber()
            R = self.goniometer_dict[run]
            peak.setGoniometerMatrix(R)

        IndexPeaks(PeaksWorkspace="peaks_ws")

        SaveNexus(
            InputWorkspace="peaks_ws",
            Filename=self._get_ouput("_{}.nxs".format(iteration)),
        )

        DeleteWorkspace(Workspace="peaks_ws")

        SCDCalibratePanels(
            PeakWorkspace="peaks",
            RecalculateUB=False,
            Tolerance=0.2,
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            OutputWorkspace="calibration_table",
            DetCalFilename=self._get_ouput(".DetCal"),
            CSVFilename=self._get_ouput(".csv"),
            XmlFilename=self._get_ouput(".xml"),
            CalibrateT0=False,
            SearchRadiusT0=0,
            CalibrateL1=False,
            SearchRadiusL1=0.0,
            CalibrateBanks=False,
            SearchRadiusTransBank=0.0,
            SearchRadiusRotXBank=0,
            SearchRadiusRotYBank=0,
            SearchRadiusRotZBank=0,
            VerboseOutput=True,
            SearchRadiusSamplePos=0.0,
            TuneSamplePosition=False,
            CalibrateSize=False,
            SearchRadiusSize=0.0,
            FixAspectRatio=True,
        )

    def gravity_angle(self):
        PreprocessDetectorsToMD(
            InputWorkspace=self.instrument, OutputWorkspace="detectors"
        )

        l2 = np.array(mtd["detectors"].column(1))
        tt = np.array(mtd["detectors"].column(2))
        az = np.array(mtd["detectors"].column(3))

        x = l2 * np.sin(tt) * np.cos(az)
        y = l2 * np.sin(tt) * np.sin(az)

        mask = np.isfinite(x) & np.isfinite(y)

        A = np.sum(x[mask] ** 2)
        B = np.sum(y[mask] ** 2)
        C = np.sum(x[mask] * y[mask])

        delta = np.rad2deg(0.5 * np.arctan2(2 * C, B - A))

        if delta > 45:
            delta -= 90
        elif delta < -45:
            delta += 90

        return -delta

    def calibrate_goniometer(self, iteration):
        G_inst, G_gon, chi_off = self.refine_goniometer()

        SaveAscii(
            InputWorkspace="goniometer",
            Filename=self._get_ouput(".txt"),
        )

        LoadParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput(".xml"),
        )

        inst = mtd[self.instrument].getInstrument()

        components = np.unique(mtd["peaks"].column("BankName")).tolist()

        for component in components:
            pos = list(inst.getComponentByName(component).getPos())
            rot = inst.getComponentByName(component).getRotation()
            quat = [rot.real(), rot.imagI(), rot.imagJ(), rot.imagK()]
            R = Rotation.from_quat(quat, scalar_first=True).as_matrix()

            x, y, z = G_inst.T @ pos
            Rp = G_inst.T @ R

            w = Rotation.from_matrix(Rp).as_rotvec(degrees=True)
            theta = np.linalg.norm(w)
            wx, wy, wz = (w / theta) if not np.isclose(theta, 0) else (0, 0, 1)

            MoveInstrumentComponent(
                Workspace=self.instrument,
                ComponentName=component,
                X=x,
                Y=y,
                Z=z,
                RelativePosition=False,
            )

            RotateInstrumentComponent(
                Workspace=self.instrument,
                ComponentName=component,
                X=wx,
                Y=wy,
                Z=wz,
                Angle=theta,
                RelativeRotation=False,
            )

        ApplyInstrumentToPeaks(
            InputWorkspace="peaks",
            InstrumentWorkspace=self.instrument,
            OutputWorkspace="peaks",
        )

        CloneWorkspace(InputWorkspace="peaks", OutputWorkspace="peaks_ws")

        for peak in mtd["peaks_ws"]:
            run = peak.getRunNumber()
            R = self.goniometer_dict[run]
            omega, chi, phi = self.calculate_goniometer_angles(R)
            Rp = self.calculate_goniometer_matrix(omega, chi + chi_off, phi)
            peak.setGoniometerMatrix(G_gon @ Rp)

        CalculateUMatrix(
            PeaksWorkspace="peaks_ws",
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        IndexPeaks(PeaksWorkspace="peaks_ws")

        SaveNexus(
            InputWorkspace="peaks_ws",
            Filename=self._get_ouput("_{}.nxs".format(iteration)),
        )

        DeleteWorkspace(Workspace="peaks_ws")

        SCDCalibratePanels(
            PeakWorkspace="peaks",
            RecalculateUB=False,
            Tolerance=0.2,
            a=self.a,
            b=self.b,
            c=self.c,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            OutputWorkspace="calibration_table",
            DetCalFilename=self._get_ouput(".DetCal"),
            CSVFilename=self._get_ouput(".csv"),
            XmlFilename=self._get_ouput(".xml"),
            CalibrateT0=False,
            SearchRadiusT0=0,
            CalibrateL1=False,
            SearchRadiusL1=0.0,
            CalibrateBanks=False,
            SearchRadiusTransBank=0.0,
            SearchRadiusRotXBank=0,
            SearchRadiusRotYBank=0,
            SearchRadiusRotZBank=0,
            VerboseOutput=True,
            SearchRadiusSamplePos=0.0,
            TuneSamplePosition=False,
            CalibrateSize=False,
            SearchRadiusSize=0.0,
            FixAspectRatio=True,
        )

        ClearInstrumentParameters(Workspace=self.instrument)

        SetInstrumentParameter(
            Workspace=self.instrument,
            ParameterName="goniometer-tilt",
            ParameterType="String",
            Value=",".join(G_gon.astype(str).flatten().tolist()),
        )

        SetInstrumentParameter(
            Workspace=self.instrument,
            ParameterName="chi-offset",
            ParameterType="Number",
            Value=str(chi_off),
        )

        SaveParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput("_goniometer.xml"),
        )

        LoadParameterFile(
            Workspace=self.instrument,
            Filename=self._get_ouput(".xml"),
        )

    def generate_diagnostic(self, iteration):
        uc = UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )

        d_dict = {}

        banks = mtd["peaks"].column("BankName")

        d_min = np.inf
        d_max = 0

        for i, peak in enumerate(mtd["peaks"]):
            d = peak.getDSpacing()
            d0 = uc.d(*peak.getIntHKL())

            key = banks[i].strip("bank")
            items = d_dict.get(key)
            if items is None:
                items = [], []
            items[0].append(d0)
            items[1].append(d)

            d_dict[key] = items

            if d > d_max:
                d_max = d
            if d < d_min:
                d_min = d

        d_dict = {key: d_dict[key] for key in sorted(d_dict)}

        with PdfPages(self._get_ouput("_{}.pdf".format(iteration))) as pdf:
            for key in d_dict.keys():
                fig, ax = plt.subplots(1, 1, layout="constrained")

                x, y = self.d_dict[key]

                x = np.array(x)
                y = np.array(y)

                ax.plot(
                    x, (y / x - 1) * 100, ".", color="C0", label="Uncalibrated"
                )

                if iteration > 0:
                    x, y = d_dict[key]

                    x = np.array(x)
                    y = np.array(y)

                    ax.plot(
                        x,
                        (y / x - 1) * 100,
                        ".",
                        color="C1",
                        label="Calibrated",
                    )

                ax.axhline(0, linestyle="-", color="k", linewidth=1)
                ax.legend(shadow=True)

                ax.set_title(key)
                ax.minorticks_on()
                ax.set_xlim(d_min, d_max)
                ax.set_ylim(-2, 2)
                ax.set_xlabel(r"$d_0$ [$\AA$]")
                ax.set_ylabel(r"$(d/d_0-1)\times 100$ [%]")

                pdf.savefig(fig)
                plt.close()

                self.d_dict = d_dict

        PreprocessDetectorsToMD(
            InputWorkspace=self.instrument, OutputWorkspace="detectors"
        )

        tt = np.array(mtd["detectors"].column(2))
        az = np.array(mtd["detectors"].column(3))

        kf_x = np.sin(tt) * np.cos(az)
        kf_y = np.sin(tt) * np.sin(az)
        kf_z = np.cos(tt)

        gamma = np.rad2deg(np.arctan2(kf_x, kf_z))
        nu = np.rad2deg(np.arcsin(kf_y))

        fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=100)
        ax.scatter(gamma, nu, c="lightgray", rasterized=True)
        ax.set_aspect(1)
        ax.minorticks_on()
        ax.set_xlabel(r"$\gamma$ [$^\circ$]")
        ax.set_ylabel(r"$\nu$ [$^\circ$]")
        fig.savefig(self._get_ouput("_instrument_{}.pdf".format(iteration)))

    def refine_goniometer(self):
        self.peak_dict = {}

        runs = np.unique(mtd["peaks"].column("RunNumber")).tolist()

        for i, run in enumerate(runs):
            FilterPeaks(
                InputWorkspace="peaks",
                FilterVariable="RunNumber",
                FilterValue=run,
                Operator="=",
                OutputWorkspace="tmp",
            )

            if i == 0:
                CalculateUMatrix(
                    PeaksWorkspace="tmp",
                    a=self.a,
                    b=self.b,
                    c=self.c,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                self.U = mtd["tmp"].sample().getOrientedLattice().getU().copy()
                self.B = mtd["tmp"].sample().getOrientedLattice().getB().copy()

            Q = np.array(mtd["tmp"].column("QLab"))
            hkl = np.array(mtd["tmp"].column("IntHKL"))

            mask = hkl.any(axis=1)

            R = self.goniometer_dict[run]
            omega, chi, phi = self.calculate_goniometer_angles(R)

            self.peak_dict[run] = (omega, chi, phi), Q[mask], hkl[mask]

            DeleteWorkspace(Workspace="tmp")

        return self.optimize_goniometer()

    def calculate_goniometer_angles(self, R):
        return Rotation.from_matrix(R).as_euler("YZY", degrees=True).tolist()

    def calculate_goniometer_matrix(self, omega, chi, phi):
        return Rotation.from_euler(
            "YZY", [omega, chi, phi], degrees=True
        ).as_matrix()

    def calculate_orientation_matrix(self, u0, u1, u2):
        return Rotation.from_rotvec([u0, u1, u2], degrees=True).as_matrix()

    def calculate_tilt_matrix(self, alpha, beta, gamma):
        Gz, Gyx = self.calculate_tilt_matrices(alpha, beta, gamma)
        return Gz @ Gyx

    def calculate_tilt_matrices(self, alpha, beta, gamma):
        Gz = Rotation.from_euler("Z", gamma, degrees=True).as_matrix()
        Gyx = Rotation.from_euler(
            "YX", [beta, alpha], degrees=True
        ).as_matrix()

        return Gz, Gyx

    def calculate_orientation_vector(self, U):
        return Rotation.from_matrix(U).as_rotvec(degrees=True)

    def residual(self, x, peak_dict, func, gamma):
        phi, theta, omega, *params = func(x)

        B = self.B
        U = self.calculate_orientation_matrix(phi, theta, omega)

        UB = np.dot(U, B)

        alpha, beta, chi_off = params

        G = self.calculate_tilt_matrix(alpha, beta, gamma)

        diff = []

        for i, run in enumerate(peak_dict.keys()):
            (omega, chi, phi), Q, hkl = peak_dict[run]
            R = self.calculate_goniometer_matrix(omega, chi + chi_off, phi)
            T = G @ R @ UB * 2 * np.pi
            diff += (np.einsum("ij,lj->li", T, hkl) - Q).flatten().tolist()

        return diff

    def fix_offsets(self, x):
        return *x, 0, 0

    def refine_offsets(self, x):
        return x

    def optimize_goniometer(self):
        phi, theta, omega = self.calculate_orientation_vector(self.U)

        fun = self.refine_offsets if self.refine_off else self.fix_offsets

        gamma = self.gravity_angle()

        x0 = (phi, theta, omega) + (0,) * (2 + self.refine_off)
        args = (self.peak_dict, fun, gamma)

        sol = scipy.optimize.least_squares(self.residual, x0=x0, args=args)

        phi, theta, omega, *params = fun(sol.x)

        alpha, beta, chi_off = params

        mtd["goniometer"].addRow([alpha, beta, gamma, chi_off])

        return *self.calculate_tilt_matrices(alpha, beta, gamma), chi_off

    def run(self):
        self.load_instrument()
        self.load_peaks()
        self.reindex_peaks()
        for iteration in range(self.iterations):
            self.initialize_peaks()
            self.generate_diagnostic(iteration)
            self.calibrate_instrument(iteration)
            self.calibrate_goniometer(iteration)
        self.generate_diagnostic(self.iterations)


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    norm = Calibration(params)
    norm.run()
