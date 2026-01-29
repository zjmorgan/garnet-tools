from mantid.simpleapi import (
    FindPeaksMD,
    PredictPeaks,
    PredictSatellitePeaks,
    CentroidPeaksMD,
    IntegratePeaksMD,
    BinMD,
    PeakIntensityVsRadius,
    FilterPeaks,
    SortPeaksWorkspace,
    DeleteWorkspace,
    ExtractSingleSpectrum,
    CombinePeaksWorkspaces,
    CreatePeaksWorkspace,
    ConvertPeaksWorkspace,
    CopySample,
    CloneWorkspace,
    SaveNexus,
    LoadNexus,
    AddPeakHKL,
    HasUB,
    SetUB,
    mtd,
)

from mantid.kernel import V3D
from mantid.dataobjects import PeakShapeEllipsoid, NoShape
from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
)

from mantid import config

config["Q.convention"] = "Crystallography"

import numpy as np

centering_reflection = {
    "P": "Primitive",
    "I": "Body centred",
    "F": "All-face centred",
    "R": "Rhombohedrally centred, obverse",  # rhomb/hex axes
    "R(obv)": "Rhombohedrally centred, obverse",  # hex axes
    "R(rev)": "Rhombohedrally centred, reverse",  # hex axes
    "A": "A-face centred",
    "B": "B-face centred",
    "C": "C-face centred",
}

diagonstic_keys = [
    "run",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "vol",
    "bkg",
    "bkg_err",
    "intens",
    "sig",
    "voxels",
    "pk_data",
    "pk_norm",
    "bkg_data",
    "bkg_norm",
    "Qx",
    "Qy",
    "Qz",
    "cntrt",
]


class PeaksModel:
    def __init__(self):
        self.edge_pixels = 0
        self.monitor_count = 1

    def reset_satellites(self, peaks):
        mod_mnp = []
        mod_hkl = []

        for peak in mtd[peaks]:
            hkl = peak.getHKL()
            int_hkl = peak.getIntHKL()
            int_mnp = peak.getIntMNP()
            if int_mnp.norm2() > 0:
                mod_mnp.append(np.array(int_mnp))
                mod_hkl.append(np.array(hkl - int_hkl))

        ol = mtd[peaks].sample().getOrientedLattice()

        if len(mod_mnp) > 0:
            mod_vec = np.linalg.pinv(mod_mnp) @ np.array(mod_hkl)

            self.mod_vec_1 = V3D(*mod_vec[0])
            self.mod_vec_2 = V3D(*mod_vec[1])
            self.mod_vec_3 = V3D(*mod_vec[2])
            ol.setModVec1(self.mod_vec_1)
            ol.setModVec2(self.mod_vec_2)
            ol.setModVec3(self.mod_vec_3)

            ol.setModUB(ol.getUB() @ ol.getModHKL())

            max_order = ol.getMaxOrder()

            self.max_order = max_order if max_order > 0 else 1
            self.modUB = ol.getModUB().copy()
            self.modHKL = ol.getModHKL().copy()

            ol.setMaxOrder(self.max_order)

        else:
            self.max_order = 0
            self.modUB = np.zeros((3, 3))
            self.modHKL = np.zeros((3, 3))
            self.mod_vec_1 = V3D(0, 0, 0)
            self.mod_vec_2 = V3D(0, 0, 0)
            self.mod_vec_3 = V3D(0, 0, 0)

            ol.setMaxOrder(self.max_order)

            ol.setModVec1(self.mod_vec_1)
            ol.setModVec2(self.mod_vec_2)
            ol.setModVec3(self.mod_vec_3)

            ol.setModUB(self.modUB)

    def find_peaks(self, md, peaks, max_d, density=50, max_peaks=1000):
        """
        Harvest strong peak locations from Q-sample into a peaks table.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        max_d : float
            Maxium d-spacing enforcing lower limit of peak spacing.
        density : int, optional
            Threshold density. The default is 1000.
        max_peaks : int, optional
            Maximum number of peaks to find. The default is 50.

        """

        FindPeaksMD(
            InputWorkspace=md,
            PeakDistanceThreshold=2 * np.pi / max_d * 0.5,
            MaxPeaks=max_peaks,
            PeakFindingStrategy="VolumeNormalization",
            DensityThresholdFactor=density,
            EdgePixels=self.edge_pixels,
            OutputWorkspace=peaks,
        )

    def remove_aluminum_contamination(self, peaks, d_min, d_max, delta=0.1):
        aluminum = CrystalStructure(
            "4.05 4.05 4.05", "F m -3 m", "Al 0 0 0 1.0 0.005"
        )

        generator = ReflectionGenerator(aluminum)

        hkls = generator.getHKLsUsingFilter(
            d_min, d_max, ReflectionConditionFilter.StructureFactor
        )

        ds = list(generator.getDValues(hkls))

        for peak in mtd[peaks]:
            d_spacing = peak.getDSpacing()
            Q_mod = 2 * np.pi / d_spacing
            for d in ds:
                Q = 2 * np.pi / d
                if Q - delta < Q_mod < Q + delta:
                    peak.setRunNumber(-1)

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="RunNumber",
            FilterValue="-1",
            Operator="!=",
            Criterion="!=",
            BankName="None",
        )

    def centroid_peaks(self, md, peaks, peak_radius):
        """
        Re-center peak locations using centroid within given radius

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius.

        """

        CentroidPeaksMD(
            InputWorkspace=md,
            PeakRadius=peak_radius,
            PeaksWorkspace=peaks,
            OutputWorkspace=peaks,
        )

    def integrate_peaks(
        self,
        md,
        peaks,
        peak_radius,
        radius_scale=0,
        background_inner_fact=np.cbrt(2),
        background_outer_fact=np.cbrt(3),
        method="ellipsoid",
        centroid=True,
    ):
        """
        Integrate peaks using spherical or ellipsoidal regions.
        Ellipsoid integration adapts itself to the peak distribution.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius.
        radius_scale : float, optional
            Radius scale factor with |Q|. The default is 0.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        method : str, optional
            Integration method. The default is 'sphere'.
        centroid : str, optional
            Shift peak position to centroid. The default is True.

        """

        background_inner_radius = peak_radius * background_inner_fact
        background_outer_radius = peak_radius * background_outer_fact

        adaptive = True if radius_scale else False

        IntegratePeaksMD(
            InputWorkspace=md,
            PeaksWorkspace=peaks,
            PeakRadius=peak_radius,
            BackgroundInnerRadius=background_inner_radius,
            BackgroundOuterRadius=background_outer_radius,
            UseOnePercentBackgroundCorrection=True,
            Ellipsoid=True if method == "ellipsoid" else False,
            FixQAxis=True,
            FixMajorAxisLength=False,
            UseCentroid=centroid,
            MaxIterations=5,
            ReplaceIntensity=True,
            IntegrateIfOnEdge=True,
            AdaptiveQBackground=adaptive,
            AdaptiveQMultiplier=radius_scale,
            MaskEdgeTubes=True,
            OutputWorkspace=peaks,
        )

        if method == "sphere" and centroid:
            self.centroid_peaks(md, peaks, peak_radius)

        for peak in mtd[peaks]:
            Q0, Q1, Q2 = peak.getQSampleFrame()

            shape = peak.getPeakShape()

            shape_dict = eval(shape.toJSON())

            if "translation0" in shape_dict.keys():
                Q0 += shape_dict["translation0"]
                Q1 += shape_dict["translation1"]
                Q2 += shape_dict["translation2"]

                R = peak.getGoniometerMatrix()

                Q = np.array([Q0, Q1, Q2])
                Qx, Qy, Qz = R @ Q

                if -4 * np.pi * Qz / np.linalg.norm(Q) ** 2 > 0:
                    peak.setQSampleFrame(V3D(Q0, Q1, Q2))

    def intensity_vs_radius(
        self,
        md,
        peaks,
        peak_radius,
        background_inner_fact=1,
        background_outer_fact=1.5,
        steps=51,
        fix=False,
    ):
        """
        Integrate peak intensity with radius varying from zero to cut off.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius cut off.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        steps : int, optional
            Number of integration steps. The default is 101.
        fix : bool, optional
            Fix the background shell size

        Returns
        -------

        radius : list
            Peak radius.
        sig_noise : list
            Peak signal/noise ratio at lowest threshold.
        intens : list
            Peak intensity.

        """

        background_inner_rad = (
            background_inner_fact * peak_radius if fix else 0
        )
        background_outer_rad = (
            background_outer_fact * peak_radius if fix else 0
        )

        background_inner_fact = 0 if fix else background_inner_fact
        background_outer_fact = 0 if fix else background_outer_fact

        PeakIntensityVsRadius(
            InputWorkspace=md,
            PeaksWorkspace=peaks,
            RadiusStart=0.0,
            RadiusEnd=peak_radius,
            NumSteps=steps,
            BackgroundInnerFactor=background_inner_fact,
            BackgroundOuterFactor=background_outer_fact,
            BackgroundInnerRadius=background_inner_rad,
            BackgroundOuterRadius=background_outer_rad,
            OutputWorkspace=peaks + "_intens_vs_rad",
            OutputWorkspace2=peaks + "_sig/noise_vs_rad",
        )

        n = mtd[peaks + "_sig/noise_vs_rad"].getNumberHistograms()

        ExtractSingleSpectrum(
            InputWorkspace=peaks + "_sig/noise_vs_rad",
            OutputWorkspace=peaks + "_sig/noise_vs_rad/strong",
            WorkspaceIndex=n - 1,
        )

        peak_radius = (
            mtd[peaks + "_sig/noise_vs_rad/strong"].extractX().ravel()
        )
        sig_noise = mtd[peaks + "_sig/noise_vs_rad/strong"].extractY().ravel()

        # ol = mtd['peaks'].sample().getOrientedLattice()
        # hkls = mtd[peaks+'_intens_vs_rad'].getAxis(1).extractValues()
        # hkls = [np.array(hkl.split(' ')).astype(float) for hkl in hkls]
        lamda = np.array([peak.getWavelength() for peak in mtd["peaks"]])

        y = mtd[peaks + "_intens_vs_rad"].extractY()
        x = mtd[peaks + "_intens_vs_rad"].extractX()
        e = mtd[peaks + "_intens_vs_rad"].extractE()

        return peak_radius, sig_noise, x, y, e, lamda

    def intensity_profile(
        self,
        md,
        peaks,
        peak_radius,
        background_inner_fact=1,
        background_outer_fact=1.5,
    ):
        """
        Integrate peak intensity as profile.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integrat region radius cut off.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.

        Returns
        -------

        x : list
            Peak profile.
        y : list
            Peak intensity.
        lamda : list
            Peak wavelength.

        """

        background_inner_radius = peak_radius * background_inner_fact
        background_outer_radius = peak_radius * background_outer_fact

        length = 4 * peak_radius

        IntegratePeaksMD(
            InputWorkspace=md,
            PeakRadius=peak_radius,
            BackgroundInnerRadius=background_inner_radius,
            BackgroundOuterRadius=background_outer_radius,
            UseOnePercentBackgroundCorrection=True,
            Cylinder=True,
            CylinderLength=length,
            PercentBackground=15,
            ProfileFunction="NoFit",
            PeaksWorkspace=peaks,
            OutputWorkspace=peaks + "_profile",
        )

        lamda = np.array(mtd["peaks"].column(5))

        x = ((mtd["ProfilesData"].extractX() + 1) / 100 - 0.5) * length
        y = mtd["ProfilesData"].extractY()

        return x, y, lamda

    def intensity_projection(
        self,
        md,
        peaks,
        peak_radius,
        background_inner_fact=1,
        background_outer_fact=1.5,
    ):
        """
        Integrate peak intensity as profile.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integrat region radius cut off.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.

        Returns
        -------

        x : list
            Peak projection.
        y : list
            Peak intensity.
        theta : list
            Peak angle.

        """

        background_inner_radius = peak_radius * background_inner_fact
        background_outer_radius = peak_radius * background_outer_fact

        IntegratePeaksMD(
            InputWorkspace=md,
            PeakRadius=peak_radius,
            BackgroundInnerRadius=background_inner_radius,
            BackgroundOuterRadius=background_outer_radius,
            UseOnePercentBackgroundCorrection=True,
            Ellipsoid=True,
            FixQAxis=True,
            FixMajorAxisLength=False,
            MaxIterations=3,
            PeaksWorkspace=peaks,
            OutputWorkspace=peaks + "_projection",
        )

        two_theta = []
        radius = []
        intensity = []

        for peak in mtd[peaks + "_projection"]:
            tt = peak.getScattering()
            I = peak.getIntensity()
            Q = peak.getQSampleFrame()

            shape_dict = eval(peak.getPeakShape().toJSON())

            v0 = [float(val) for val in shape_dict["direction0"].split(" ")]
            v1 = [float(val) for val in shape_dict["direction1"].split(" ")]
            v2 = [float(val) for val in shape_dict["direction2"].split(" ")]

            r0 = shape_dict["radius0"]
            r1 = shape_dict["radius1"]
            r2 = shape_dict["radius2"]

            W = np.column_stack([v0, v1, v2])
            V = np.diag([r0**2, r1**2, r2**2])

            A = (W @ V) @ W.T

            n = Q / np.linalg.norm(Q)

            P = np.eye(3) - np.outer(n, n)

            A_proj = P @ A @ P

            i = np.abs(n).argmax()
            cov = np.delete(np.delete(A_proj, i, axis=0), i, axis=1)

            two_theta.append(np.rad2deg(tt))
            radius.append(np.sqrt(np.linalg.eigvals(cov).max()))
            intensity.append(I)

        theta = np.array(two_theta) / 2
        y = np.array(intensity)
        x = np.array(radius)

        return x, y, theta

    def get_number_peaks(self):
        return self.peaks.getNumberPeaks()

    def extract_peaks_roi(self, md, peaks, r_cut, n_bins=21):
        signals = []
        weights = []
        d2s = []
        lamdas = []

        for peak in mtd[peaks]:
            Q = peak.getQSampleFrame()
            lamda = peak.getWavelength()

            extents = [
                Q[0] - r_cut,
                Q[0] + r_cut,
                Q[1] - r_cut,
                Q[1] + r_cut,
                Q[2] - r_cut,
                Q[2] + r_cut,
            ]

            BinMD(
                InputWorkspace=md,
                AxisAligned=False,
                BasisVector0="Q_sample_x,Angstrom^-1,1.0,0.0,0.0",
                BasisVector1="Q_sample_y,Angstrom^-1,0.0,1.0,0.0",
                BasisVector2="Q_sample_z,Angstrom^-1,0.0,0.0,1.0",
                OutputExtents=extents,
                OutputBins=[n_bins, n_bins, n_bins],
                OutputWorkspace="_md_bin",
            )

            signal = mtd["_md_bin"].getSignalArray().copy()
            weight = 1 / mtd["_md_bin"].getErrorSquaredArray()

            dims = [
                mtd["_md_bin"].getDimension(i)
                for i in range(mtd["_md_bin"].getNumDims())
            ]

            xs = [
                np.linspace(
                    dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
                )
                for dim in dims
            ]

            xs = [0.5 * (x[1:] + x[:-1]) - Q[i] for i, x in enumerate(xs)]

            x, y, z = np.meshgrid(*xs, indexing="ij")

            mask = (signal > 0) & np.isfinite(weight)

            if mask.sum() > 5:
                d2s.append(x[mask] ** 2 + y[mask] ** 2 + z[mask] ** 2)
                lamdas.append(lamda)

                signals.append(signal[mask])
                weights.append(weight[mask])

        return signals, weights, d2s, lamdas

    def get_max_d_spacing(self, ws):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Parameters
        ----------
        ws : str
            Workspace with UB defined on oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=ws):
            if hasattr(mtd[ws], "sample"):
                ol = mtd[ws].sample().getOrientedLattice()
            else:
                for i in range(mtd[ws].getNumExperimentInfo()):
                    sample = mtd[ws].getExperimentInfo(i).sample()
                    if sample.hasOrientedLattice():
                        ol = sample.getOrientedLattice()
                        SetUB(Workspace=ws, UB=ol.getUB())
                ol = mtd[ws].getExperimentInfo(i).sample().getOrientedLattice()

            return 1 / min([ol.astar(), ol.bstar(), ol.cstar()])

    def get_UB(self, ws):
        """
        Obtain UB from the oriented lattice.

        Parameters
        ----------
        ws : str
            Workspace with UB defined on oriented lattice.

        Returns
        -------
        UB : 2d-array
            UB matrix.

        """

        if HasUB(Workspace=ws):
            if hasattr(mtd[ws], "sample"):
                ol = mtd[ws].sample().getOrientedLattice()
            else:
                ol = mtd[ws].getExperimentInfo(0).sample().getOrientedLattice()

            return ol.getUB().copy()

    def predict_peaks(self, ws, peaks, centering, d_min, lamda_min, lamda_max):
        """
        Predict peak Q-sample locations with UB and lattice centering.

        +--------+----------------------------------------+
        | Symbol | Reflection condition                   |
        +========+========================================+
        | P      | None                                   |
        +--------+----------------------------------------+
        | I      | :math:`h+k+l=2n`                       |
        +--------+----------------------------------------+
        | F      | :math:`h,k,l` unmixed                  |
        +--------+----------------------------------------+
        | R      | :math:`-h+k+l=3n` or :math:`-h+k+l=3n` |
        +--------+----------------------------------------+
        | R(obv) | :math:`-h+k+l=3n`                      |
        +--------+----------------------------------------+
        | R(rev) | :math:`h-k+l=3n`                       |
        +--------+----------------------------------------+
        | A      | :math:`k+l=2n`                         |
        +--------+----------------------------------------+
        | B      | :math:`l+h=2n`                         |
        +--------+----------------------------------------+
        | C      | :math:`h+k=2n`                         |
        +--------+----------------------------------------+

        Note
        ----
        R-centering refers to obverse/reverse conditions, a common twin law.

        Parameters
        ----------
        ws : str
            Name of workspace to predict peaks with UB.
        peaks : str
            Name of peaks table.
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.

        """

        d_max = self.get_max_d_spacing(ws)

        refl_cond = centering_reflection[centering]

        if centering == "R":  # obverse/reverse
            obv_rev = ["R(obv)", "R(rev)"]
            refl_conds = [centering_reflection[rc] for rc in obv_rev]
        else:
            refl_conds = [centering_reflection[centering]]

        for i, refl_cond in enumerate(refl_conds):
            peaks_cond = peaks + "_{}".format(refl_cond)

            PredictPeaks(
                InputWorkspace=ws,
                WavelengthMin=lamda_min,
                WavelengthMax=lamda_max,
                MinDSpacing=d_min,
                MaxDSpacing=d_max * 1.2,
                ReflectionCondition=refl_cond,
                RoundHKL=True,
                EdgePixels=self.edge_pixels,
                OutputWorkspace=peaks_cond,
            )

            if i == 0:
                CloneWorkspace(
                    InputWorkspace=peaks_cond, OutputWorkspace=peaks
                )
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace=peaks,
                    RHSWorkspace=peaks_cond,
                    OutputWorkspace=peaks,
                )

        self.remove_duplicate_peaks(peaks)

    def predict_modulated_peaks(
        self,
        peaks,
        centering,
        d_min,
        lamda_min,
        lamda_max,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Predict the modulated peak positions based on main peaks.

        Parameters
        ----------
        ws : str
            Name of workspace to predict peaks with UB.
        peaks : str
            Name of main peaks table.
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        d_max = self.get_max_d_spacing(peaks)

        sat_peaks = peaks + "_sat"

        PredictSatellitePeaks(
            Peaks=peaks,
            SatellitePeaks=sat_peaks,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            IncludeIntegerHKL=False,
            IncludeAllPeaksInRange=True,
            WavelengthMin=lamda_min,
            WavelengthMax=lamda_max,
            MinDSpacing=d_min,
            MaxDSpacing=d_max * 10,
        )

        for no in range(mtd[sat_peaks].getNumberPeaks()):
            peak = mtd[sat_peaks].getPeak(no)
            h, k, l = [int(val) for val in peak.getIntHKL()]

            forbidden = False
            if centering == "I":
                if (h + k + l) % 2 != 0:
                    forbidden = True
            elif centering == "F":
                if not ((h % 2 == 0) == (k % 2 == 0) == (l % 2 == 0)):
                    forbidden = True
            elif centering == "A":
                if h % 2 != 0:
                    forbidden = True
            elif centering == "B":
                if k % 2 != 0:
                    forbidden = True
            elif centering == "C":
                if l % 2 != 0:
                    forbidden = True
            elif centering == "R":
                if (-h + k + l) % 3 != 0 and (h - k + l) % 3 != 0:
                    forbidden = True
            elif centering == "R(obv)":
                if (-h + k + l) % 3 != 0:
                    forbidden = True
            elif centering == "R(rev)":
                if (h - k + l) % 3 != 0:
                    forbidden = True

            if forbidden:
                peak.setRunNumber(-1)

        CombinePeaksWorkspaces(
            LHSWorkspace=peaks, RHSWorkspace=sat_peaks, OutputWorkspace=peaks
        )

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="RunNumber",
            FilterValue=-1,
            Operator="!=",
        )

        ol = mtd[peaks].sample().getOrientedLattice()

        ol.setMaxOrder(max_order)
        ol.setModVec1(V3D(*mod_vec_1))
        ol.setModVec2(V3D(*mod_vec_2))
        ol.setModVec3(V3D(*mod_vec_3))

        DeleteWorkspace(Workspace=sat_peaks)

    def predict_satellite_peaks(
        self,
        peaks_ws,
        data_ws,
        centering,
        lamda_min,
        lamda_max,
        d_min,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Locate satellite peaks from goniometer angles.

        Parameters
        ----------
        peaks_ws : str
            Reference peaks table.
        data_ws : str
            Q-sample data with goniometer(s).
        centering : str
            Lattice centering that provides the reflection condition.
        lamda_min : float
            Minimum wavelength.
        lamda_max : float
            Maximum wavelength.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        Rs = self.get_all_goniometer_matrices(data_ws)

        for R in Rs:
            self.set_goniometer(peaks_ws, R)

            self.predict_modulated_peaks(
                peaks_ws,
                centering,
                d_min,
                lamda_min,
                lamda_max,
                mod_vec_1,
                mod_vec_2,
                mod_vec_3,
                max_order,
                cross_terms,
            )

            self.remove_duplicate_peaks(peaks_ws)

    def sort_peaks_by_hkl(self, peaks):
        """
        Sort peaks table by descending hkl values.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace=peaks,
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace=peaks,
            )

    def sort_peaks_by_d(self, peaks):
        """
        Sort peaks table by descending d-spacing.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=peaks,
        )

    def sort_peaks_by_bank(self, peaks):
        """
        Sort peaks table by ascending bank number

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="BankName",
            SortAscending=True,
            OutputWorkspace=peaks,
        )

        for i, peak in enumerate(mtd[peaks]):
            peak.setPeakNumber(i)

    def remove_duplicate_peaks(self, peaks):
        """
        Omit duplicate peaks from different based on indexing.
        Table will be sorted.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        self.sort_peaks_by_hkl(peaks)

        for no in range(mtd[peaks].getNumberPeaks() - 1, 0, -1):
            if (
                mtd[peaks].getPeak(no).getHKL()
                - mtd[peaks].getPeak(no - 1).getHKL()
            ).norm2() == 0:
                mtd[peaks].getPeak(no).setRunNumber(-1)

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="RunNumber",
            FilterValue=-1,
            Operator="!=",
        )

    def get_all_goniometer_matrices(self, ws):
        """
        Extract all goniometer matrices.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.

        Returns
        -------
        Rs: list
            Goniometer matrices.

        """

        Rs = []

        for ei in range(mtd[ws].getNumExperimentInfo()):
            run = mtd[ws].getExperimentInfo(ei).run()

            n_gon = run.getNumGoniometers()

            Rs += [run.getGoniometer(i).getR() for i in range(n_gon)]

        return np.array(Rs)

    def get_bank_names(self, peaks):
        """
        Obtain the bank names.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        Returns
        -------
        banks : str
            Unique bank names.

        """

        return np.unique(mtd[peaks].column("BankName")).tolist()

    def renumber_runs_by_index(self, ws, peaks):
        """
        Re-label the runs by index based on goniometer setting.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.
        peaks : str
            Name of peaks table.

        """

        Rs = self.get_all_goniometer_matrices(ws)

        for no in range(mtd[peaks].getNumberPeaks()):
            peak = mtd[peaks].getPeak(no)

            R = peak.getGoniometerMatrix()

            ind = np.isclose(Rs, R).all(axis=(1, 2))
            i = -1 if not np.any(ind) else ind.tolist().index(True)

            peak.setRunNumber(i + 1)

    def load_peaks(self, filename, peaks):
        """
        Load peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        peaks : str
            Name of peaks table.

        """

        LoadNexus(Filename=filename, OutputWorkspace=peaks)

    def save_peaks(self, filename, peaks):
        """
        Save peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        peaks : str
            Name of peaks table.

        """

        SaveNexus(Filename=filename, InputWorkspace=peaks)

    def convert_peaks(self, peaks):
        """
        Remove instrument from peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        ConvertPeaksWorkspace(PeakWorkspace=peaks, OutputWorkspace=peaks)

        for i in range(mtd[peaks].getNumberPeaks()):
            mtd[peaks].getPeak(i).setPeakShape(NoShape())

    def combine_peaks(self, peaks, merge):
        """
        Merge two peaks workspaces into one.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.
        merge : str
            Name of peaks table to be accumulated.

        """

        if not mtd.doesExist(merge):
            CloneWorkspace(InputWorkspace=peaks, OutputWorkspace=merge)

        else:
            CombinePeaksWorkspaces(
                LHSWorkspace=merge, RHSWorkspace=peaks, OutputWorkspace=merge
            )

            merge_run = mtd[merge].run()
            peaks_run = mtd[peaks].run()

            for key in diagonstic_keys:
                log = "peaks_{}".format(key)
                if peaks_run.hasProperty(log) and merge_run.hasProperty(log):
                    peaks_log = peaks_run.getLogData(log).value
                    peaks_list = np.array(peaks_log).tolist()
                    merge_log = merge_run.getLogData(log).value
                    merge_list = np.array(merge_log).tolist()
                    merge_run[log] = merge_list + peaks_list

    def delete_peaks(self, peaks):
        """
        Remove peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.

        """

        if mtd.doesExist(peaks):
            DeleteWorkspace(Workspace=peaks)

    def remove_weak_peaks(self, peaks, sig_noise=3):
        """
        Filter out weak peaks based on signal-to-noise ratio.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        sig_noise : float, optional
            Minimum signal-to-noise ratio. The default is 3.

        """

        if sig_noise is None:
            intens_sig = np.array(mtd[peaks].column("Intens/SigInt"))
            med = np.nanmedian(intens_sig)
            mad = np.nanmedian(np.abs(intens_sig - med))
            sig_noise = np.max([med - 1.4826 * mad, 3])

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="Signal/Noise",
            FilterValue=sig_noise,
            Operator=">",
            Criterion="!=",
            BankName="None",
        )

    def update_scale_factor(self, peaks, value):
        """
        Update counting statistic refrence value for normalization

        Parameters
        ----------
        value: float
            Monitor count or proton charge

        """

        for i in range(mtd[peaks].getNumberPeaks()):
            mtd[peaks].getPeak(i).setMonitorCount(value)
            mtd[peaks].getPeak(i).setBinCount(value)

    def remove_peaks_by_d_tolerance(self, peaks, tol=0.05):
        """
        Filter out peaks based on d-spacing tolerance.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        tol : float, optional
            d-spacing tolerance. The default is 0.05.

        """

        if HasUB(Workspace=peaks):
            ol = mtd[peaks].sample().getOrientedLattice()

            for no in range(mtd[peaks].getNumberPeaks()):
                peak = mtd[peaks].getPeak(no)
                d = peak.getDspacing()
                d0 = ol.d(*peak.peak.getHKL())
                if np.abs(d / d0 - 1) < tol:
                    peak.setRunNumber(-1)

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=peaks,
                FilterVariable="RunNumber",
                FilterValue=-1,
                Operator="!=",
            )

    def remove_unindexed_peaks(self, peaks):
        """
        Filter out unindexes peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="h^2+k^2+l^2",
            FilterValue=0,
            Operator=">",
            Criterion="!=",
            BankName="None",
        )

    def create_peaks(self, ws, peaks, lean=False):
        """
        Create a new peaks table.

        ws : str
            Name of workspace.
        peaks : str
            Name of peaks table.

        """

        CreatePeaksWorkspace(
            InstrumentWorkspace=ws if not lean else None,
            NumberOfPeaks=0,
            OutputWorkspace=peaks,
            OutputType="Peak" if not lean else "LeanElasticPeak",
        )

        CopySample(
            InputWorkspace=ws,
            OutputWorkspace=peaks,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        ol = mtd[ws].sample().getOrientedLattice()
        mod_vec_1 = V3D(*ol.getModVec(0))
        mod_vec_2 = V3D(*ol.getModVec(1))
        mod_vec_3 = V3D(*ol.getModVec(2))
        max_order = ol.getMaxOrder()

        ol = mtd[peaks].sample().getOrientedLattice()
        ol.setMaxOrder(max_order)
        ol.setModVec1(mod_vec_1)
        ol.setModVec2(mod_vec_2)
        ol.setModVec3(mod_vec_3)

    def add_peak(self, peaks, hkl):
        """
        Add a peak to an existing table.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        hkl : list
            Miller index.

        """

        AddPeakHKL(Workspace=peaks, HKL=hkl)

    def set_goniometer(self, peaks, R):
        """
        Update the goniometer on the run.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        R : 2d-array
            Goniometer matrix.

        """

        mtd[peaks].run().getGoniometer().setR(R)

    def get_peaks_name(self, peaks):
        """
        Name of peaks.

        Returns
        -------
        name : str
            Readable name of peaks.

        """

        peak = mtd[peaks].getPeak(0)

        run = peak.getRunNumber()

        name = "peaks_run#{}"
        return name.format(run)


class PeakModel:
    def __init__(self, peaks):
        self.peaks = peaks

    def get_number_peaks(self):
        """
        Total number of peaks in the table.

        Returns
        -------
        n : int
            Number of peaks

        """

        return mtd[self.peaks].getNumberPeaks()

    def get_UB(self):
        """
        UB matrix.

        Returns
        -------
        UB : 2d-array, 3x3
            UB-matrix

        """

        return mtd[self.peaks].sample().getOrientedLattice().getUB()

    def set_peak_intensity(self, no, intens, sig):
        """
        Update the peak intensity.

        Parameters
        ----------
        no : int
            Peak index number.
        intens : float
            Intensity.
        sig : float
            Uncertainty.

        """

        mtd[self.peaks].getPeak(no).setIntensity(intens)
        mtd[self.peaks].getPeak(no).setSigmaIntensity(sig)

    def get_wavelength(self, no):
        """
        Wavelength of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        lamda : float
            Wavelength in angstroms.

        """

        peak = mtd[self.peaks].getPeak(no)

        return peak.getWavelength()

    def set_wavelength(self, no, lamda):
        """
        Update the wavelength of the peak.

        Parameters
        ----------
        no : int
            Peak index number.
        lamda : float
            Wavelength in angstroms.

        """

        peak = mtd[self.peaks].getPeak(no)

        peak.setWavelength(lamda)

    def get_sample_Q(self, no):
        """
        Scattering vector in Q sample coordinates.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        Q : list
            Scattering Q-vector.

        """

        peak = mtd[self.peaks].getPeak(no)

        return peak.getQSampleFrame()

    def get_angles(self, no):
        """
        Scattering and azimuthal angle of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        two_theta : float
            Scattering (polar) angle in degrees.
        az_phi : float
            Azimuthal angle in degrees.

        """

        peak = mtd[self.peaks].getPeak(no)

        two_theta = np.rad2deg(peak.getScattering())
        az_phi = np.rad2deg(peak.getAzimuthal())

        return two_theta, az_phi

    def get_goniometer_matrix(self, no):
        """
        Goniometer matrix of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        R : 2d-array
            Rotation matrix.

        """

        peak = mtd[self.peaks].getPeak(no)

        return peak.getGoniometerMatrix()

    def get_projection_matrix(self, no):
        two_theta, az_phi = self.get_angles(no)

        two_theta = np.deg2rad(two_theta)
        az_phi = np.deg2rad(az_phi)

        kf_hat = np.array(
            [
                np.sin(two_theta) * np.cos(az_phi),
                np.sin(two_theta) * np.sin(az_phi),
                np.cos(two_theta),
            ]
        )

        ki_hat = np.array([0, 0, 1])

        n = kf_hat - ki_hat
        n /= np.linalg.norm(n)

        R = self.get_goniometer_matrix(no)

        if (R @ n) @ n < 0:
            n *= -1

        u = np.cross(ki_hat, kf_hat)
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        return (R.T @ n).tolist(), (R.T @ u).tolist(), (R.T @ v).tolist()

    def get_projection_peak_origin(self, no):
        projections = self.get_projection_matrix(no)

        UB = self.get_UB()

        W = np.column_stack(projections)
        Q = 2 * np.pi * W.T @ UB @ self.get_hkl(no)

        Wp = np.linalg.inv(W.T @ (2 * np.pi * UB)).T

        return Wp.tolist(), Q.tolist()

    def get_goniometer_angles(self, no):
        """
        Goniometer Euler angles of the peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        angles : list
            Euler angles (YZY convention) in degrees.

        """

        peak = mtd[self.peaks].getPeak(no)
        gon = mtd[self.peaks].run().getGoniometer()

        R = peak.getGoniometerMatrix()
        gon.setR(R)

        return list(gon.getEulerAngles())

    def add_diagonstic_info(self, no, values):
        """
        Log diagnostic info.

        Parameters
        ----------
        no : int
            Peak index number.
        values : list
           Diagnostics.

        """

        peak = mtd[self.peaks].getPeak(no)

        run = int(peak.getRunNumber())
        h, k, l = [int(val) for val in peak.getIntHKL()]
        m, n, p = [int(val) for val in peak.getIntMNP()]

        run_info = mtd[self.peaks].run()
        run_info_keys = run_info.keys()

        vals = [run, h, k, l, m, n, p] + values

        assert len(vals) == len(diagonstic_keys)

        for key, val in zip(diagonstic_keys, vals):
            log = "peaks_{}".format(key)
            if log not in run_info_keys:
                items = [val]
            else:
                items = np.array(run_info.getLogData(log).value).tolist()
                items.append(val)
                items = np.array(items).tolist()
            run_info[log] = items

    def get_peak_name(self, no, merge=False):
        """
        Name of peak.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        name : str
            Readable name of peak.

        """

        peak = mtd[self.peaks].getPeak(no)

        run = peak.getRunNumber()
        hkl = [int(val) for val in peak.getIntHKL()]
        mnp = [int(val) for val in peak.getIntMNP()]
        lamda = peak.getWavelength()

        if HasUB(Workspace=self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()
            mod_1 = ol.getModVec(0)
            mod_2 = ol.getModVec(1)
            mod_3 = ol.getModVec(2)
            h, k, l, m, n, p = *hkl, *mnp
            dh, dk, dl = (
                m * np.array(mod_1) + n * np.array(mod_2) + p * np.array(mod_3)
            )
            d = ol.d(V3D(h + dh, k + dk, l + dl))
        else:
            d = peak.getDSpacing()

        if not merge:
            name = (
                "peak_d={:.4f}_({:d},{:d},{:d})"
                + "_({:d},{:d},{:d})/lambda={:.4f}_run#{:d}"
            )
            return name.format(d, *hkl, *mnp, lamda, run)
        else:
            name = "peak_d={:.4f}_({:d},{:d},{:d})_({:d},{:d},{:d})"
            return name.format(d, *hkl, *mnp)

    def get_hkl(self, no):
        """
        Miller indices.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        hkl : list
            Components of Miller indices.

        """

        peak = mtd[self.peaks].getPeak(no)
        return list(peak.getHKL())

    def get_hklmnp(self, no):
        """
        Miller indices.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        hklmnp : tuple
            Components of Miller indices.

        """

        peak = mtd[self.peaks].getPeak(no)
        hkl = np.array(peak.getIntHKL()).astype(int).tolist()
        mnp = np.array(peak.getIntMNP()).astype(int).tolist()
        return tuple(hkl + mnp)

    def set_hklmnp(self, no, hklmnp):
        """
        Update Miller indices.

        Parameters
        ----------
        no : int
            Peak index number.
        hklmnp : list
            Components of Miller indices.

        """

        h, k, l, m, n, p = hklmnp

        peak = mtd[self.peaks].getPeak(no)
        peak.setIntHKL(V3D(h, k, l))
        peak.setIntMNP(V3D(m, n, p))

    def get_peak_shape(self, no, r_cut=np.inf):
        """
        Obtain the peak shape parameters.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        c0, c1, c2 : float
            Peak center.
        r0, r1, r2 : float
            Principal radii.
        v0, v1, v2 : list
            Principal axis directions.

        """

        Q0, Q1, Q2 = mtd[self.peaks].getPeak(no).getQSampleFrame()

        shape = mtd[self.peaks].getPeak(no).getPeakShape()

        c0, c1, c2 = Q0, Q1, Q2

        if shape.shapeName() == "ellipsoid":
            shape_dict = eval(shape.toJSON())

            c0 = Q0 + shape_dict["translation0"]
            c1 = Q1 + shape_dict["translation1"]
            c2 = Q2 + shape_dict["translation2"]

            v0 = [float(val) for val in shape_dict["direction0"].split(" ")]
            v1 = [float(val) for val in shape_dict["direction1"].split(" ")]
            v2 = [float(val) for val in shape_dict["direction2"].split(" ")]

            r0 = shape_dict["radius0"]
            r1 = shape_dict["radius1"]
            r2 = shape_dict["radius2"]

            r0 = r0 if r0 < r_cut else r_cut
            r1 = r1 if r1 < r_cut else r_cut
            r2 = r2 if r2 < r_cut else r_cut

        else:
            r0 = r1 = r2 = r_cut
            v0, v1, v2 = np.eye(3).tolist()

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def set_peak_center(self, no, c0, c1, c2):
        """
        Update the shape of the peak.

        Parameters
        ----------
        no : int
            Peak index number.
        c0, c1, c2 : float
            Peak center.
        """

        R = mtd[self.peaks].getPeak(no).getGoniometerMatrix()

        Q = np.array([c0, c1, c2])
        Qx, Qy, Qz = R @ Q

        if -4 * np.pi * Qz / np.linalg.norm(Q) ** 2 > 0:
            try:
                mtd[self.peaks].getPeak(no).setQSampleFrame(V3D(c0, c1, c2))
            except Exception as e:
                print("Exception re-centering: {}".format(e))
                return False
            return True
        else:
            return False

    def set_peak_shape(self, no, c0, c1, c2, r0, r1, r2, v0, v1, v2):
        """
        Update the shape of the peak.

        Parameters
        ----------
        no : int
            Peak index number.
        c0, c1, c2 : float
            Peak center.
        r0, r1, r2 : float
            Principal radii.
        v0, v1, v2 : list
            Principal axis directions.

        """

        radii = [r0, r1, r2]

        valid = self.set_peak_center(no, c0, c1, c2)

        if not valid:
            print(self.get_peak_name(no))

        shape = PeakShapeEllipsoid(
            [V3D(*v0), V3D(*v1), V3D(*v2)], radii, radii, radii
        )

        mtd[self.peaks].getPeak(no).setPeakShape(shape)

    def get_detector_id(self, no):
        """
        Obtain the peak detector id number.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        det_id : int
            Detector number.

        """

        return mtd[self.peaks].getPeak(no).getDetectorID()

    def get_d_spacing(self, no):
        """
        Obtain the peak d-spacing.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        d : float
            Interplanar spacing.

        """

        return mtd[self.peaks].getPeak(no).getDSpacing()

    def set_scale_factor(self, no, scale):
        """
        update the peak normalization scale factor.

        Parameters
        ----------
        no : int
            Peak index number.
        scale : float
            Peak scale factor

        """

        mtd[self.peaks].getPeak(no).setBinCount(scale)
        mtd[self.peaks].getPeak(no).setMonitorCount(scale)

    def get_bank_name(self, no):
        """
        Obtain the bank name.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        bank : str
            Bank name.

        """

        return mtd[self.peaks].column("BankName")[no]
