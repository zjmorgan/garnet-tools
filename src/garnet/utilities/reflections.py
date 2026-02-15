import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append(os.path.abspath(os.path.join(directory, "../..")))

from mantid.simpleapi import (
    LoadNexus,
    FilterPeaks,
    SortPeaksWorkspace,
    CombinePeaksWorkspaces,
    StatisticsOfPeaksWorkspace,
    SaveHKL,
    SaveReflections,
    SaveIsawUB,
    LoadIsawUB,
    LoadIsawSpectrum,
    CloneWorkspace,
    CopySample,
    SetGoniometer,
    SetSample,
    LoadSampleShape,
    AddAbsorptionWeightedPathLengths,
    RemoveMaskedSpectra,
    GroupDetectors,
    CreateGroupingWorkspace,
    SolidAngle,
    Divide,
    mtd,
)

import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

np.random.seed(13)

import scipy.optimize
import scipy.interpolate
import scipy.stats

from sklearn.cluster import AgglomerativeClustering

from mantid.kernel import V3D

from mantid import config

config["Q.convention"] = "Crystallography"


from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import argparse

from garnet.reduction.ub import Optimization

point_group_dict = {
    "-1": "-1 (Triclinic)",
    "1": "1 (Triclinic)",
    "2": "2 (Monoclinic, unique axis b)",
    "m": "m (Monoclinic, unique axis b)",
    "2/m": "2/m (Monoclinic, unique axis b)",
    "112": "112 (Monoclinic, unique axis c)",
    "11m": "11m (Monoclinic, unique axis c)",
    "112/m": "112/m (Monoclinic, unique axis c)",
    "222": "222 (Orthorhombic)",
    "2mm": "2mm (Orthorhombic)",
    "m2m": "m2m (Orthorhombic)",
    "mm2": "mm2 (Orthorhombic)",
    "mmm": "mmm (Orthorhombic)",
    "3": "3 (Trigonal - Hexagonal)",
    "32": "32 (Trigonal - Hexagonal)",
    "312": "312 (Trigonal - Hexagonal)",
    "321": "321 (Trigonal - Hexagonal)",
    "31m": "31m (Trigonal - Hexagonal)",
    "3m": "3m (Trigonal - Hexagonal)",
    "3m1": "3m1 (Trigonal - Hexagonal)",
    "-3": "-3 (Trigonal - Hexagonal)",
    "-31m": "-31m (Trigonal - Hexagonal)",
    "-3m": "-3m (Trigonal - Hexagonal)",
    "-3m1": "-3m1 (Trigonal - Hexagonal)",
    "3 r": "3 r (Trigonal - Rhombohedral)",
    "32 r": "32 r (Trigonal - Rhombohedral)",
    "3m r": "3m r (Trigonal - Rhombohedral)",
    "-3 r": "-3 r (Trigonal - Rhombohedral)",
    "-3m r": "-3m r (Trigonal - Rhombohedral)",
    "4": "4 (Tetragonal)",
    "4/m": "4/m (Tetragonal)",
    "4mm": "4mm (Tetragonal)",
    "422": "422 (Tetragonal)",
    "-4": "-4 (Tetragonal)",
    "-42m": "-42m (Tetragonal)",
    "-4m2": "-4m2 (Tetragonal)",
    "4/mmm": "4/mmm (Tetragonal)",
    "6": "6 (Hexagonal)",
    "6/m": "6/m (Hexagonal)",
    "6mm": "6mm (Hexagonal)",
    "622": "622 (Hexagonal)",
    "-6": "-6 (Hexagonal)",
    "-62m": "-62m (Hexagonal)",
    "-6m2": "-6m2 (Hexagonal)",
    "6/mmm": "6/mmm (Hexagonal)",
    "23": "23 (Cubic)",
    "m-3": "m-3 (Cubic)",
    "432": "432 (Cubic)",
    "-43m": "-43m (Cubic)",
    "m-3m": "m-3m (Cubic)",
}


class AbsorptionCorrection:
    def __init__(
        self,
        peaks,
        chemical_formula,
        z_parameter,
        u_vector=[0, 0, 1],
        v_vector=[1, 0, 0],
        params=None,
        filename=None,
    ):
        assert "PeaksWorkspace" in str(type(mtd[peaks]))

        self.peaks = peaks

        volume = mtd[self.peaks].sample().getOrientedLattice().volume()

        assert volume > 0

        self.volume = volume

        assert self.verify_chemical_formula(chemical_formula)

        self.chemical_formula = chemical_formula

        assert z_parameter > 0
        self.z_parameter = z_parameter

        assert len(u_vector) == 3
        assert len(v_vector) == 3

        assert not np.isclose(np.linalg.norm(np.cross(u_vector, v_vector)), 0)

        self.u_vector = u_vector
        self.v_vector = v_vector

        if params is not None:
            assert len(params) == 3

        self.params = params

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        self.set_shape()
        self.set_material()
        self.set_orientation()
        self.calculate_correction()
        self.write_absortion_parameters()

    def verify_chemical_formula(self, formula):
        pattern = (
            r"(?:\((?:[A-Z][a-z]?\d+)\)|[A-Z][a-z]?)(?:\d+(?:\.\d+)?|\.\d+)?"
        )

        parts = re.split(r"[-\s]+", formula.strip())

        return all(re.fullmatch(pattern, part) for part in parts)

    def save_ellipsoid_stl(self, params, filename="/tmp/ellipsoid.stl"):
        l_thickness, l_width, l_height = params
        sph = pv.Icosphere(radius=0.5)

        ell = sph.scale([l_width, l_height, l_thickness], inplace=False)
        ell.save(filename)

    def set_shape(self):
        self.UB = mtd[self.peaks].sample().getOrientedLattice().getUB().copy()

        u = np.dot(self.UB, self.u_vector)
        v = np.dot(self.UB, self.v_vector)

        u /= np.linalg.norm(u)

        w = np.cross(u, v)
        w /= np.linalg.norm(w)

        v = np.cross(w, u)

        T = np.column_stack([v, w, u])

        gon = mtd[self.peaks].run().getGoniometer()

        gon.setR(T)
        self.gamma, self.beta, self.alpha = gon.getEulerAngles("ZYX")

        if self.params is not None:
            self.shapestl = os.path.splitext(self.filename)[0] + ".stl"
            self.save_ellipsoid_stl(self.params, self.shapestl)

    def set_material(self):
        self.mat_dict = {
            "ChemicalFormula": self.chemical_formula,
            "ZParameter": float(self.z_parameter),
            "UnitCellVolume": self.volume,
        }

    def set_orientation(self):
        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="RunNumber",
            SortAscending=False,
        )

        Rs = [peak.getGoniometerMatrix() for peak in mtd[self.peaks]]
        matrix_dict = {}

        runs = []
        for peak in mtd[self.peaks]:
            R = peak.getGoniometerMatrix()

            matrix_tuple = tuple(R.flatten())

            if matrix_tuple in matrix_dict:
                run = matrix_dict[matrix_tuple]
            else:
                ind = np.isclose(Rs, R).all(axis=(1, 2))
                i = -1 if not np.any(ind) else ind.tolist().index(True)
                run = i + 1
                matrix_dict[matrix_tuple] = run

            runs.append(run)
            peak.setBinCount(peak.getRunNumber())
            peak.setRunNumber(run)

        self.runs = np.unique(runs).astype(int).tolist()
        self.Rs = Rs

    def calculate_correction(self):
        peaks = self.peaks + "_corr"

        filename = os.path.splitext(self.filename)[0] + "_abs.pdf"

        self.apply_correction()

        with PdfPages(filename) as pdf:
            for i, (R, run) in enumerate(zip(self.Rs, self.runs)):
                FilterPeaks(
                    InputWorkspace=peaks,
                    FilterVariable="RunNumber",
                    FilterValue=run,
                    Operator="=",
                    OutputWorkspace="_tmp",
                )

                R = mtd["_tmp"].getPeak(0).getGoniometerMatrix()

                gon = mtd["_tmp"].run().getGoniometer()

                gon.setR(R)
                omega, chi, phi = gon.getEulerAngles("YZY")

                LoadSampleShape(
                    InputWorkspace="_tmp",
                    Filename=self.shapestl,
                    Scale="mm",
                    XDegrees=self.alpha,
                    YDegrees=self.beta,
                    ZDegrees=self.gamma,
                    OutputWorkspace="_tmp",
                )

                SetSample(
                    InputWorkspace="_tmp",
                    Material=self.mat_dict,
                )

                SetGoniometer(
                    Workspace="_tmp",
                    Axis0="{},0,1,0,1".format(omega),
                    Axis1="{},0,0,1,1".format(chi),
                    Axis2="{},0,1,0,1".format(phi),
                )

                hkl = np.eye(3)
                s = np.matmul(self.UB, hkl)

                reciprocal_lattice = np.matmul(R, s)

                shape = mtd["_tmp"].sample().getShape()
                mesh = shape.getMesh() * 1000

                mesh_polygon = Poly3DCollection(
                    mesh,
                    edgecolors="k",
                    facecolors="w",
                    alpha=0.5,
                    linewidths=0.1,
                )

                fig, ax = plt.subplots(
                    subplot_kw={"projection": "mantid3d", "proj_type": "persp"}
                )
                ax.add_collection3d(mesh_polygon)

                ax.set_title("run #{}".format(1 + i))
                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                ax.set_zlabel("z [mm]")

                ax.set_mesh_axes_equal(mesh)
                ax.set_box_aspect((1, 1, 1))

                colors = ["r", "g", "b"]
                origin = (
                    ax.get_xlim3d()[1],
                    ax.get_ylim3d()[1],
                    ax.get_zlim3d()[1],
                )

                lims = ax.get_xlim3d()
                factor = (lims[1] - lims[0]) / 4
                origin = (lims[1] - lims[0]) / 4

                for j in range(3):
                    vector = reciprocal_lattice[:, j]
                    vector = vector / np.linalg.norm(vector)
                    ax.quiver(
                        origin,
                        origin,
                        origin,
                        vector[0],
                        vector[1],
                        vector[2],
                        length=factor,
                        color=colors[j],
                        linestyle="-",
                    )

                    ax.view_init(vertical_axis="y", elev=27, azim=50)

                pdf.savefig(fig, dpi=100, bbox_inches=None)
                plt.close(fig)
                plt.close("all")

    def write_absortion_parameters(self):
        mat = mtd["_tmp"].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective  # A^-3
        N = mat.totalAtoms

        V = np.abs(np.prod(self.params) * 0.1**3)  # cm^3

        rho = (n / N) / 0.6022 * M
        m = rho * V * 1000  # mg
        r = np.cbrt(0.75 * np.pi * V)  # cm

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        mu = mat.numberDensityEffective * (
            mat.totalScatterXSection() + mat.absorbXSection(1.8)
        )

        lines = [
            "{}\n".format(self.chemical_formula),
            "absoption cross section: {:.4f} barn\n".format(sigma_a),
            "scattering cross section: {:.4f} barn\n".format(sigma_s),
            "linear absorption coefficient: {:.4f} 1/cm\n".format(mu_a),
            "linear scattering coefficient: {:.4f} 1/cm\n".format(mu_s),
            "absorption parameter: {:.4f} \n".format(mu * r),
            "total atoms: {:.4f}\n".format(N),
            "molar mass: {:.4f} g/mol\n".format(M),
            "number density: {:.4f} 1/A^3\n".format(n),
            "mass density: {:.4f} g/cm^3\n".format(rho),
            "volume: {:.4f} cm^3\n".format(V),
            "mass: {:.4f} mg\n".format(m),
            "equivalent-sphere: {:.4f} cm\n".format(r),
        ]

        for line in lines:
            print(line)

        if self.filename is not None:
            filename = os.path.splitext(self.filename)[0] + "_abs.txt"

            with open(filename, "w") as f:
                for line in lines:
                    f.write(line)

    def apply_correction(self):
        peaks = self.peaks + "_corr"

        for i, (R, run) in enumerate(zip(self.Rs, self.runs)):
            FilterPeaks(
                InputWorkspace=self.peaks,
                FilterVariable="RunNumber",
                FilterValue=run,
                Operator="=",
                OutputWorkspace="_tmp",
            )

            LoadSampleShape(
                InputWorkspace="_tmp",
                Filename=self.shapestl,
                Scale="mm",
                XDegrees=self.alpha,
                YDegrees=self.beta,
                ZDegrees=self.gamma,
                OutputWorkspace="_tmp",
            )

            SetSample(
                InputWorkspace="_tmp",
                Material=self.mat_dict,
            )

            R = mtd["_tmp"].getPeak(0).getGoniometerMatrix()

            gon = mtd["_tmp"].run().getGoniometer()

            gon.setR(R)
            omega, chi, phi = gon.getEulerAngles("YZY")

            SetGoniometer(
                Workspace="_tmp",
                Axis0="{},0,1,0,1".format(omega),
                Axis1="{},0,0,1,1".format(chi),
                Axis2="{},0,1,0,1".format(phi),
            )

            AddAbsorptionWeightedPathLengths(
                InputWorkspace="_tmp",
                ApplyCorrection=False,
                UseSinglePath=False,
            )

            if i == 0:
                CloneWorkspace(InputWorkspace="_tmp", OutputWorkspace=peaks)
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace=peaks,
                    RHSWorkspace="_tmp",
                    OutputWorkspace=peaks,
                )

        CloneWorkspace(InputWorkspace=peaks, OutputWorkspace=self.peaks)

        mat = mtd["_tmp"].sample().getMaterial()

        for peak in mtd[self.peaks]:
            lamda = peak.getWavelength()
            Tbar = peak.getAbsorptionWeightedPathLength()

            mu = mat.numberDensityEffective * (
                mat.totalScatterXSection() + mat.absorbXSection(lamda)
            )

            corr = np.exp(mu * Tbar)

            print(
                "mu = {:4.2f} corr = {:4.2f} Tbar = {:4.2f}".format(
                    mu, corr, Tbar
                )
            )

            peak.setBinCount(corr)
            peak.setIntensity(peak.getIntensity() * corr)
            peak.setSigmaIntensity(peak.getSigmaIntensity() * corr)


class Peaks:
    def __init__(self, peaks, filename, scale=None, point_group=None):
        self.peaks = peaks

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        if scale is not None:
            assert scale > 0

        self.scale = scale

        if point_group is not None:
            assert point_group in point_group_dict.keys()
            point_groups = [point_group]
        else:
            point_groups = list(point_group_dict.keys())

        self.point_groups = point_groups

        self.max_order = 0
        self.modUB = np.zeros((3, 3))
        self.modHKL = np.zeros((3, 3))

    def refine_UB(self, peaks):
        opt = Optimization(peaks)

        ol = mtd[peaks].sample().getOrientedLattice()

        a, b, c = ol.a(), ol.b(), ol.c()
        alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

        if np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90):
            cell = "Cubic"
        elif np.allclose([a, b], c) and np.allclose([alpha, beta], gamma):
            cell = "Rhombohedral"
        elif np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90):
            cell = "Tetragonal"
        elif (
            np.isclose(a, b)
            and np.allclose([alpha, beta], 90)
            and np.isclose(gamma, 120)
        ):
            cell = "Hexagonal"
        elif np.allclose([alpha, beta, gamma], 90):
            cell = "Orthorhombic"
        elif np.allclose([alpha, gamma], 90):
            cell = "Monoclinic"
        else:
            cell = "Triclinic"

        opt.optimize_lattice(cell)

    def rescale_intensities(self):
        maximal = 10000
        scale = 1 if self.scale is None else self.scale
        if mtd[self.peaks].getNumberPeaks() > 1 and self.scale is None:
            I = np.array(mtd[self.peaks].column("Intens"))
            I0 = np.nanpercentile(I, 99)
            scale = maximal / I0
            self.scale = scale
            self.maximal = maximal

        indices = np.arange(mtd[self.peaks].getNumberPeaks())
        for i, peak in zip(indices.tolist(), mtd[self.peaks]):
            peak.setIntensity(scale * peak.getIntensity())
            peak.setSigmaIntensity(scale * peak.getSigmaIntensity())
            peak.setPeakNumber(peak.getRunNumber())
            peak.setBinCount(peak.getRunNumber())
            # peak.setRunNumber(1)

        filename = os.path.splitext(self.filename)[0] + "_scale.txt"
        with open(filename, "w") as f:
            f.write("{:.4e}".format(scale))

    def remove_edge_peaks(self):
        edge_pixels = {
            "TOPAZ": ([24, 232], [24, 232]),
            "MANDI": ([24, 232], [24, 232]),
            "CORELLI": ([1, 15], [24, 232]),
        }

        inst = mtd[self.peaks].getInstrument()

        cols, rows = edge_pixels[inst.getName()]

        for peak in mtd[self.peaks]:
            col = peak.getCol()
            row = peak.getRow()
            if not (cols[0] < col < cols[1]) or not (rows[0] < row < rows[1]):
                peak.setSigmaIntensity(peak.getIntensity())

    # def remove_off_centered(self):
    #     aluminum = CrystalStructure(
    #         "4.05 4.05 4.05", "F m -3 m", "Al 0 0 0 1.0 0.005"
    #     )

    #     copper = CrystalStructure(
    #         "3.61 3.61 3.61", "F m -3 m", "Cu 0 0 0 1.0 0.005"
    #     )

    #     generator_al = ReflectionGenerator(aluminum)
    #     generator_cu = ReflectionGenerator(copper)

    #     d = np.array(mtd[self.peaks].column("DSpacing"))

    #     d_min = np.nanmin(d)
    #     d_max = np.nanmax(d)

    #     hkls_al = generator_al.getUniqueHKLsUsingFilter(
    #         d_min, d_max, ReflectionConditionFilter.StructureFactor
    #     )

    #     hkls_cu = generator_cu.getUniqueHKLsUsingFilter(
    #         d_min, d_max, ReflectionConditionFilter.StructureFactor
    #     )

    #     d_al = np.unique(generator_al.getDValues(hkls_al))
    #     d_cu = np.unique(generator_cu.getDValues(hkls_cu))

    #     ol = mtd[self.peaks].sample().getOrientedLattice()

    #     Q_vol = []
    #     powder_err = []
    #     peak_err = []
    #     Q0_mod = []
    #     Q_rad = []

    #     for peak in mtd[self.peaks]:
    #         h, k, l = peak.getHKL()
    #         d0 = ol.d(h, k, l)
    #         powder_err.append(peak.getDSpacing() / d0 - 1)
    #         Q0_mod.append(2 * np.pi / d0)

    #         shape = peak.getPeakShape()
    #         if shape.shapeName() == "ellipsoid":
    #             ellipsoid = eval(shape.toJSON())

    #             v0 = [float(val) for val in ellipsoid["direction0"].split(" ")]
    #             v1 = [float(val) for val in ellipsoid["direction1"].split(" ")]
    #             v2 = [float(val) for val in ellipsoid["direction2"].split(" ")]

    #             r0 = ellipsoid["radius0"]
    #             r1 = ellipsoid["radius1"]
    #             r2 = ellipsoid["radius2"]

    #         else:
    #             r0 = r1 = r2 = 1e-6
    #             v0, v1, v2 = np.eye(3).tolist()

    #         r = np.array([r0, r1, r2])

    #         U = np.column_stack([v0, v1, v2])
    #         V = np.diag(r**2)
    #         S = np.dot(np.dot(U, V), U.T)

    #         vol = 4 / 3 * np.pi * r0 * r1 * r2

    #         Q_vol.append(vol)

    #         R = peak.getGoniometerMatrix()

    #         two_theta = peak.getScattering()
    #         az_phi = peak.getAzimuthal()

    #         kf_hat = np.array(
    #             [
    #                 np.sin(two_theta) * np.cos(az_phi),
    #                 np.sin(two_theta) * np.sin(az_phi),
    #                 np.cos(two_theta),
    #             ]
    #         )

    #         ki_hat = np.array([0, 0, 1])

    #         n = kf_hat - ki_hat
    #         n /= np.linalg.norm(n)

    #         u = np.cross(ki_hat, kf_hat)
    #         u /= np.linalg.norm(u)

    #         v = np.cross(n, u)
    #         v /= np.linalg.norm(v)

    #         n, u, v = R.T @ n, R.T @ u, R.T @ v

    #         Q0 = 2 * np.pi * ol.getUB() @ np.array([h, k, l])
    #         Q = peak.getQSampleFrame()

    #         W = np.column_stack([n, u, v])

    #         peak_err.append(W @ (Q - Q0))

    #         r0 = np.sqrt(n.T @ (S @ n))
    #         r1 = np.sqrt(u.T @ (S @ u))
    #         r2 = np.sqrt(v.T @ (S @ v))

    #         Q_rad.append([r0, r1, r2])

    #     Q0_mod = np.array(Q0_mod)
    #     powder_err = np.array(powder_err)
    #     peak_err = np.array(peak_err) / Q0_mod[:, np.newaxis]
    #     Q_vol = np.array(Q_vol)
    #     Q_rad = np.array(Q_rad)

    #     powder_med = np.nanmedian(powder_err)
    #     peak_med = np.nanmedian(peak_err, axis=0)

    #     powder_mad = np.nanmedian(np.abs(powder_err - powder_med))
    #     peak_mad = np.nanmedian(np.abs(peak_err - peak_med), axis=0)

    #     powder_min = powder_med - 1.5 * powder_mad
    #     powder_max = powder_med + 1.5 * powder_mad

    #     peak_min = peak_med - 1.5 * peak_mad
    #     peak_max = peak_med + 1.5 * peak_mad

    #     vol_med = np.nanmedian(Q_vol)
    #     vol_mad = np.nanmedian(np.abs(Q_vol - vol_med))

    #     vol_cut = vol_med + 1.5 * vol_mad

    #     radius_med = np.nanmedian(Q_rad, axis=0)

    #     radius_mad = np.nanmedian(np.abs(Q_rad - radius_med), axis=0)

    #     radius_max = radius_med + 1.5 * radius_mad

    #     filename = os.path.splitext(self.filename)[0]

    #     fig, ax = plt.subplots(4, 2, layout="constrained", sharex=True)
    #     ax = ax.T.ravel()
    #     ax[3].set_xlabel("$|Q|$ [$\AA^{-1}$]")
    #     ax[7].set_xlabel("$|Q|$ [$\AA^{-1}$]")
    #     ax[0].set_ylabel("$d/d_0-1$")
    #     ax[1].set_ylabel("$\Delta{Q_1}/|Q|$")
    #     ax[2].set_ylabel("$\Delta{Q_2}/|Q|$")
    #     ax[3].set_ylabel("$\Delta{Q_3}/|Q|$")
    #     ax[4].set_ylabel("$V$ [$\AA^{-3}$]")
    #     ax[5].set_ylabel("$r_1$  [$\AA^{-1}$]")
    #     ax[6].set_ylabel("$r_2$  [$\AA^{-1}$]")
    #     ax[7].set_ylabel("$r_3$  [$\AA^{-1}$]")
    #     ax[0].plot(Q0_mod, powder_err, ".", color="C0", rasterized=True)
    #     ax[1].plot(Q0_mod, peak_err[:, 0], ".", color="C1", rasterized=True)
    #     ax[2].plot(Q0_mod, peak_err[:, 1], ".", color="C2", rasterized=True)
    #     ax[3].plot(Q0_mod, peak_err[:, 2], ".", color="C3", rasterized=True)
    #     ax[4].plot(Q0_mod, Q_vol, ".", color="C4", rasterized=True)
    #     ax[5].plot(Q0_mod, Q_rad[:, 0], ".", color="C5", rasterized=True)
    #     ax[6].plot(Q0_mod, Q_rad[:, 1], ".", color="C6", rasterized=True)
    #     ax[7].plot(Q0_mod, Q_rad[:, 2], ".", color="C7", rasterized=True)
    #     ax[0].axhline(powder_min, color="k", linestyle="--", linewidth=1)
    #     ax[0].axhline(powder_max, color="k", linestyle="--", linewidth=1)
    #     ax[1].axhline(peak_min[0], color="k", linestyle="--", linewidth=1)
    #     ax[1].axhline(peak_max[0], color="k", linestyle="--", linewidth=1)
    #     ax[2].axhline(peak_min[1], color="k", linestyle="--", linewidth=1)
    #     ax[2].axhline(peak_max[1], color="k", linestyle="--", linewidth=1)
    #     ax[3].axhline(peak_min[2], color="k", linestyle="--", linewidth=1)
    #     ax[3].axhline(peak_max[2], color="k", linestyle="--", linewidth=1)
    #     ax[4].axhline(vol_cut, color="k", linestyle="--", linewidth=1)
    #     ax[5].axhline(radius_max[0], color="k", linestyle="--", linewidth=1)
    #     ax[6].axhline(radius_max[1], color="k", linestyle="--", linewidth=1)
    #     ax[7].axhline(radius_max[2], color="k", linestyle="--", linewidth=1)

    #     for i in range(8):
    #         ax[0].minorticks_on()
    #         for d in d_al:
    #             ax[i].axvline(
    #                 2 * np.pi / d, color="k", linestyle=":", linewidth=1
    #             )

    #         for d in d_cu:
    #             ax[i].axvline(
    #                 2 * np.pi / d, color="k", linestyle=":", linewidth=1
    #             )

    #     fig.savefig(filename + "_cont.pdf")

    #     for i, peak in enumerate(mtd[self.peaks]):
    #         powder = powder_err[i] > powder_max or powder_err[i] < powder_min
    #         contamination = (peak_err[i] > peak_max) | (peak_err[i] < peak_min)
    #         background = Q_vol[i] > vol_cut
    #         if contamination.any() or powder.any() or background:
    #             peak.setSigmaIntensity(float("-inf"))

    def median_absolute_devation(self, arr):
        med = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr - med), axis=0)

        return med, mad

    def remove_off_centered(self):
        ol = mtd[self.peaks].sample().getOrientedLattice()

        Q_vol = []
        powder_err = []
        peak_err = []
        Q0_mod = []
        Q_rad = []

        wl, tt = [], []

        for peak in mtd[self.peaks]:
            h, k, l = peak.getHKL()
            d0 = ol.d(h, k, l)
            powder_err.append(peak.getDSpacing() / d0 - 1)
            Q0_mod.append(2 * np.pi / d0)

            shape = peak.getPeakShape()
            if shape.shapeName() == "ellipsoid":
                ellipsoid = eval(shape.toJSON())

                v0 = [float(val) for val in ellipsoid["direction0"].split(" ")]
                v1 = [float(val) for val in ellipsoid["direction1"].split(" ")]
                v2 = [float(val) for val in ellipsoid["direction2"].split(" ")]

                r0 = ellipsoid["radius0"]
                r1 = ellipsoid["radius1"]
                r2 = ellipsoid["radius2"]

            else:
                r0 = r1 = r2 = 1e-6
                v0, v1, v2 = np.eye(3).tolist()

            r = np.array([r0, r1, r2])

            U = np.column_stack([v0, v1, v2])
            V = np.diag(r**2)
            S = np.dot(np.dot(U, V), U.T)

            vol = 4 / 3 * np.pi * r0 * r1 * r2

            Q_vol.append(vol)

            R = peak.getGoniometerMatrix()

            lamda = peak.getWavelength()
            two_theta = peak.getScattering()
            az_phi = peak.getAzimuthal()

            wl.append(lamda)
            tt.append(two_theta)

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

            u = np.cross(ki_hat, kf_hat)
            u /= np.linalg.norm(u)

            v = np.cross(n, u)
            v /= np.linalg.norm(v)

            Q0 = 2 * np.pi * R @ ol.getUB() @ np.array([h, k, l])
            Q = peak.getQLabFrame()

            W = np.column_stack([n, u, v])

            peak_err.append(W @ (Q - Q0))

            r0 = np.sqrt(n.T @ (S @ n))
            r1 = np.sqrt(u.T @ (S @ u))
            r2 = np.sqrt(v.T @ (S @ v))

            Q_rad.append([r0, r1, r2])

        Q0_mod = np.array(Q0_mod)
        powder_err = np.array(powder_err)
        peak_err = np.array(peak_err)  # / Q0_mod[:, np.newaxis] * 100
        Q_vol = np.array(Q_vol)
        Q_rad = np.array(Q_rad)

        mod_peak_err = np.linalg.norm(peak_err, axis=1)

        med_peak_err = np.median(peak_err, axis=0)
        mad_peak_err = np.median(np.abs(peak_err - med_peak_err), axis=0)

        mask = [
            np.abs(peak_err[:, i] - med_peak_err[i]) / mad_peak_err[i] < 4.5
            for i in range(3)
        ]

        wl = np.array(wl)
        tt = np.array(tt)

        filename = os.path.splitext(self.filename)[0]

        with PdfPages(filename + "_cont.pdf") as pdf:
            fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)

            for i in range(3):
                ax[i].scatter(
                    Q0_mod, peak_err[:, i], c=wl, s=0.05, rasterized=True
                )

                ax[i].axhline(
                    med_peak_err[i] + 4.5 * mad_peak_err[i],
                    color="k",
                    lw=1,
                    zorder=-1,
                )

                ax[i].axhline(
                    med_peak_err[i] - 4.5 * mad_peak_err[i],
                    color="k",
                    lw=1,
                    zorder=-1,
                )

            ax[3].scatter(
                Q0_mod,
                mod_peak_err,
                c=wl,
                s=0.05,
                rasterized=True,
            )

            ax[0].minorticks_on()
            ax[3].set_xlabel(r"$Q$ [$\AA^{-1}$]")
            ax[0].set_ylabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")
            ax[1].set_ylabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")
            ax[2].set_ylabel(r"$\Delta{Q}_3$ [$\AA^{-1}$]")
            ax[3].set_ylabel(r"$|\Delta{Q}|$ [$\AA^{-1}$]")

            pdf.savefig(fig, dpi=300, bbox_inches=None)
            plt.close(fig)
            plt.close("all")

            fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)

            for i in range(3):
                ax[i].scatter(
                    Q0_mod[mask[i]],
                    Q_rad[:, i][mask[i]],
                    c="C{}".format(i),
                    s=0.05,
                    rasterized=True,
                )

            ax[3].scatter(
                Q0_mod,
                np.cbrt(np.prod(Q_rad, axis=1)),
                c="C3",
                s=0.05,
                rasterized=True,
            )

            ax[0].minorticks_on()
            ax[3].set_xlabel(r"$Q$ [$\AA^{-1}$]")
            ax[0].set_ylabel(r"$r_1$ [$\AA^{-1}$]")
            ax[1].set_ylabel(r"$r_2$ [$\AA^{-1}$]")
            ax[2].set_ylabel(r"$r_3$ [$\AA^{-1}$]")
            ax[3].set_ylabel(r"$r$ [$\AA^{-1}$]")

            pdf.savefig(fig, dpi=300, bbox_inches=None)
            plt.close(fig)
            plt.close("all")

            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

            sort = np.argsort(Q0_mod)

            for i in range(3):
                ax[i].scatter(
                    Q_rad[sort, i][mask[i][sort]],
                    peak_err[sort, i][mask[i][sort]],
                    c=Q0_mod[sort][mask[i][sort]],
                    s=0.05,
                    rasterized=True,
                )
                ax[i].set_aspect(1)
                ax[i].minorticks_on()
                ax[i].set_xlabel(r"$r_{}$".format(i + 1) + " [$\AA^{-1}$]")

            ax[0].set_ylabel(r"$\Delta{Q}_i$ [$\AA^{-1}$]")

            for i in range(3):
                for j, peak in enumerate(mtd[self.peaks]):
                    if not mask[i][j]:
                        peak.setSigmaIntensity(float("-inf"))

            pdf.savefig(fig, dpi=300, bbox_inches=None)
            plt.close(fig)
            plt.close("all")

    def remove_non_integrated(self):
        for peak in mtd[self.peaks]:
            shape = eval(peak.getPeakShape().toJSON())

            if shape["shape"] == "none":
                peak.setSigmaIntensity(peak.getIntensity())

            elif (
                shape["radius0"] == 0
                or shape["radius1"] == 0
                or shape["radius2"] == 0
            ):
                peak.setSigmaIntensity(float("-inf"))

    def remove_non_indexed(self, tol=0.1):
        UB = mtd[self.peaks].sample().getOrientedLattice().getUB()
        for peak in mtd[self.peaks]:
            hkl = np.array(peak.getHKL())
            Q = np.array(peak.getQSampleFrame())

            Q0 = 2 * np.pi * UB @ hkl

            diff = np.abs(Q / Q0 - 1).max()
            if diff > tol:
                peak.setSigmaIntensity(float("-inf"))

    def load_spectrum(self, filename, instrument):
        LoadIsawSpectrum(
            SpectraFile=filename,
            OutputWorkspace="spectrum",
            InstrumentName=instrument,
        )

    def get_cell(self):
        ol = mtd[self.peaks].sample().getOrientedLattice()
        return ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

    def load_peaks(self):
        LoadNexus(Filename=self.filename, OutputWorkspace=self.peaks)

        merge = self.filename.replace(".nxs", "_diagnostics/merge.nxs")
        if os.path.exists(merge):
            LoadNexus(Filename=merge, OutputWorkspace=self.peaks + "_merge")

        ub_file = self.filename.replace(".nxs", ".mat")

        if os.path.exists(ub_file):
            LoadIsawUB(Filename=ub_file, InputWorkspace=self.peaks)

        self.filename = re.sub(
            r"(_\([-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+\))+",
            "",
            self.filename,
        )

        # self.remove_edge_peaks()
        self.remove_non_integrated()
        self.remove_non_indexed()

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=1,
            Operator=">",
        )

        self.remove_off_centered()

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=1,
            Operator=">",
        )

        run_info = mtd[self.peaks].run()
        run_keys = run_info.keys()

        keys = ["h", "k", "l", "m", "n", "p", "run"]
        vals = ["intens", "sig", "vol"]

        info_dict = {}
        norm_dict = {}
        rate_dict = {}

        items = keys + vals

        log_info = np.all(
            ["peaks_{}".format(item) in run_keys for item in items]
        )
        if log_info:
            h = run_info.getLogData("peaks_h").value
            k = run_info.getLogData("peaks_k").value
            l = run_info.getLogData("peaks_l").value
            m = run_info.getLogData("peaks_m").value
            n = run_info.getLogData("peaks_n").value
            p = run_info.getLogData("peaks_p").value
            run = run_info.getLogData("peaks_run").value

            cntrt = run_info.getLogData("peaks_cntrt").value

            N = run_info.getLogData("peaks_voxels").value
            vol = run_info.getLogData("peaks_vol").value
            pk_data = run_info.getLogData("peaks_pk_data").value
            pk_norm = run_info.getLogData("peaks_pk_norm").value
            bkg_data = run_info.getLogData("peaks_bkg_data").value
            bkg_norm = run_info.getLogData("peaks_bkg_norm").value

            intens = run_info.getLogData("peaks_intens").value
            sig = run_info.getLogData("peaks_sig").value

            for i in range(len(run)):
                key = (run[i], h[i], k[i], l[i], m[i], n[i], p[i])
                vals = (
                    N[i],
                    vol[i],
                    pk_data[i],
                    pk_norm[i],
                    bkg_data[i],
                    bkg_norm[i],
                )
                info_dict[key] = vals
                norm_dict[key] = (intens[i], sig[i])

                rate_dict[run[i]] = cntrt[i]

        filename = os.path.splitext(self.filename)[0]

        x, y = [], []

        for key in rate_dict.keys():
            x.append(key)
            y.append(rate_dict[key])

        rate_ave = np.mean(y)

        fig, ax = plt.subplots(1, 1, sharex=True, layout="constrained")
        ax.set_xlabel("")
        ax.plot(x, y, ".", rasterized=True)
        ax.minorticks_on()
        ax.set_ylabel("Count rate")
        fig.savefig(filename + "_rate.pdf")

        # for peak in mtd[self.peaks]:
        #     run = int(peak.getRunNumber())
        #     rate = rate_dict.get(run)
        #     if rate is None:
        #         scale = 0
        #     else:
        #         scale = rate_ave / rate
        #     peak.setIntensity(scale * peak.getIntensity())
        #     peak.setSigmaIntensity(scale * peak.getSigmaIntensity())

        self.info_dict = info_dict
        self.norm_dict = norm_dict

        x, y = [], []

        for peak in mtd[self.peaks]:
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            run = int(peak.getRunNumber())
            key = (run, h, k, l, m, n, p)
            items = self.info_dict.get(key)

            if items is not None:
                N, vol, pk_data, pk_norm, bkg_data, bkg_norm = items

                lamda = peak.getWavelength()
                two_theta = peak.getScattering()

                norm = np.log10(bkg_norm / pk_norm)

                Q = 4 * np.pi / lamda * np.sin(0.5 * two_theta)

                x.append(Q)
                y.append(norm)

        filename = os.path.splitext(self.filename)[0]

        lamda = np.array(mtd[self.peaks].column("Wavelength"))

        kde = scipy.stats.gaussian_kde(lamda)

        x = np.linspace(lamda.min(), lamda.max(), 1000)

        pdf = kde(x)

        cdf = scipy.integrate.cumulative_trapezoid(pdf, x, initial=0)
        cdf /= cdf[-1]

        lower_bound = x[np.searchsorted(cdf, 0.0001)]
        upper_bound = x[np.searchsorted(cdf, 0.9999)]

        filename = os.path.splitext(self.filename)[0]

        fig, ax = plt.subplots(layout="constrained")
        ax.hist(lamda, bins=100, density=True, color="C0")
        ax.set_xlabel("$\lambda$ [$\AA$]")
        ax.minorticks_on()
        ax.plot(x, pdf, color="C1")
        ax.axvline(lower_bound, color="k", linestyle="--", linewidth=1)
        ax.axvline(upper_bound, color="k", linestyle="--", linewidth=1)
        fig.savefig(filename + ".pdf")

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Wavelength",
            FilterValue=lower_bound,
            Operator=">",
        )

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Wavelength",
            FilterValue=upper_bound,
            Operator="<",
        )

        self.reset_satellite()

        if mtd.doesExist(self.peaks + "_merge"):
            ol = mtd[self.peaks + "_merge"].sample().getOrientedLattice()
            ol.setMaxOrder(self.max_order)
            ol.setModVec1(self.mod_vec_1)
            ol.setModVec2(self.mod_vec_2)
            ol.setModVec3(self.mod_vec_3)
            ol.setModUB(self.modUB)

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=-1,
            Operator=">",
        )

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
        )

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="Intens",
            SortAscending=False,
        )

        # self.renormalize_intensities()

        # self.rescale_intensities()

    def renormalize_intensities(self, flux, solid_angle):
        self.flux_file = flux
        self.solid_angle_file = solid_angle

        LoadNexus(Filename=self.flux_file, OutputWorkspace="flux")
        LoadNexus(Filename=self.solid_angle_file, OutputWorkspace="sa")

        detIDs = mtd["peaks"].column("DetID")

        inds = list(mtd["sa"].getIndicesFromDetectorIDs(detIDs))

        y = mtd["sa"].extractY().ravel()

        for i, peak in enumerate(mtd[self.peaks]):
            ind = inds[i]
            if y[ind] == 0:
                peak.setSigmaIntensity(peak.getIntensity())

        # FilterPeaks(
        #     InputWorkspace=self.peaks,
        #     OutputWorkspace=self.peaks,
        #     FilterVariable="Signal/Noise",
        #     FilterValue=3,
        #     Operator=">",
        # )

        SolidAngle(InputWorkspace="sa", OutputWorkspace="solid_angle")

        Divide(
            LHSWorkspace="sa",
            RHSWorkspace="solid_angle",
            OutputWorkspace="efficiency",
        )

        CreateGroupingWorkspace(
            InputWorkspace="sa",
            GroupDetectorsBy="bank",
            OutputWorkspace="group",
        )

        GroupDetectors(
            InputWorkspace="efficiency",
            OutputWorkspace="scale",
            Behaviour="Average",
            CopyGroupingFromWorkspace="group",
        )

        RemoveMaskedSpectra(InputWorkspace="scale", OutputWorkspace="scale")

        s = mtd["scale"].extractY().ravel()

        y = mtd["flux"].extractY()
        x = mtd["flux"].extractX()

        k = 0.5 * (x[:, 1:] + x[:, :-1])
        y = np.diff(y) * y.shape[1]

        x = 2 * np.pi / k
        z = 2 * np.pi * y / x**2
        y = z * ((x[:, 0] - x[:, -1]) / (k[:, -1] - k[:, 0]))[:, None]

        x = x[:, ::-1]
        y = y[:, ::-1]

        j = np.searchsorted(x[0, :], 1.0)

        scale = y[:, j]

        filename = os.path.splitext(self.filename)[0]

        fig, ax = plt.subplots(1, 1, layout="constrained")
        for i in range(x.shape[0]):
            ax.plot(x[i, :], y[i, :] / scale[i])
        fig.savefig(filename + "_spec.pdf")

        detIDs = mtd["peaks"].column("DetID")

        rows = list(mtd["scale"].getIndicesFromDetectorIDs(detIDs))

        for i, peak in enumerate(mtd[self.peaks]):
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            run = int(peak.getRunNumber())
            key = (run, h, k, l, m, n, p)
            raw_intens, raw_sig = self.norm_dict[key]

            lamda = peak.getWavelength()
            two_theta = peak.getScattering()
            proton_charge = peak.getBinCount()

            L = 0.5 * lamda**4 / np.sin(0.5 * two_theta) ** 2

            corr = np.inf
            row = rows[i]

            col = np.searchsorted(x[row], lamda, side="right") - 1
            corr = y[row, col] * s[row] * proton_charge / scale[row]

            peak.setIntensity(raw_intens / L / corr)
            peak.setSigmaIntensity(raw_sig / L / corr)

    def merge_intensities(self, name=None):
        if name is not None:
            peaks = name
            app = "_{}".format(name).replace(" ", "_")
        else:
            peaks = self.peaks
            app = ""

        filename = os.path.splitext(self.filename)[0] + app + "_merge"

        for peak in mtd[peaks + "_merge"]:
            peak.setIntensity(self.scale * peak.getIntensity())
            peak.setSigmaIntensity(self.scale * peak.getSigmaIntensity())

        for col in ["h", "k", "l", "DSpacing"]:
            SortPeaksWorkspace(
                InputWorkspace=peaks + "_merge",
                OutputWorkspace=peaks + "_merge",
                ColumnNameToSortBy=col,
                SortAscending=False,
            )

        SaveReflections(
            InputWorkspace=peaks + "_merge",
            Filename=filename + "_jana.int",
            Format="Jana",
        )

        SaveReflections(
            InputWorkspace=peaks + "_merge",
            Filename=filename + "_fullprof.int",
            Format="Fullprof",
        )

    def reset_satellite(self, peaks=None):
        mod_mnp = []
        mod_hkl = []
        if peaks is None:
            peaks = self.peaks
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

    def save_peaks(self, name=None, fit_dict=None):
        if name is not None:
            peaks = name
            app = "_{}".format(name).replace(" ", "_")
        else:
            peaks = self.peaks
            app = ""

        self.rescale_intensities()

        filename = os.path.splitext(self.filename)[0] + app

        if mtd.doesExist(self.peaks + "_merge"):
            self.merge_intensities(self.peaks)

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="PeakNumber",
            SortAscending=True,
            OutputWorkspace=peaks,
        )

        self.calculate_statistics(peaks, filename + "_symm.txt")

        self.renumber_peaks(peaks)

        SaveHKL(
            InputWorkspace=peaks,
            Filename=filename + ".hkl",
            DirectionCosines=True,
            ApplyAnvredCorrections=False,
            SortBy="RunNumber",
        )

        SaveReflections(
            InputWorkspace=peaks,
            Filename=filename + ".int",
            Format="Jana",
        )

        self.refine_UB(peaks)

        SaveIsawUB(InputWorkspace=peaks, Filename=filename + ".mat")

        self.resort_hkl(peaks, filename + ".hkl")

        if self.max_order > 0:
            nuclear = peaks + "_nuc"
            satellite = peaks + "_sat"

            for peak in mtd[peaks]:
                peak.setRunNumber(1)
                peak.setPeakNumber(1)

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=nuclear,
                FilterVariable="m^2+n^2+p^2",
                FilterValue=0,
                Operator="=",
            )

            nuc_ol = mtd[nuclear].sample().getOrientedLattice()
            nuc_ol.setMaxOrder(0)
            nuc_ol.setModVec1(V3D(0, 0, 0))
            nuc_ol.setModVec2(V3D(0, 0, 0))
            nuc_ol.setModVec3(V3D(0, 0, 0))
            nuc_ol.setModUB(np.zeros((3, 3)))

            SaveHKL(
                InputWorkspace=nuclear,
                Filename=filename + "_nuc.hkl",
                DirectionCosines=True,
                ApplyAnvredCorrections=False,
                SortBy="RunNumber",
            )

            SaveIsawUB(InputWorkspace=nuclear, Filename=filename + "_nuc.mat")

            self.resort_hkl(nuclear, filename + "_nuc.hkl")

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=satellite,
                FilterVariable="m^2+n^2+p^2",
                FilterValue=0,
                Operator=">",
            )

            sat_ol = mtd[satellite].sample().getOrientedLattice()
            sat_ol.setMaxOrder(self.max_order)
            sat_ol.setModVec1(V3D(*self.modHKL[:, 0]))
            sat_ol.setModVec2(V3D(*self.modHKL[:, 1]))
            sat_ol.setModVec3(V3D(*self.modHKL[:, 2]))
            sat_ol.setModUB(self.modUB)

            SaveHKL(
                InputWorkspace=satellite,
                Filename=filename + "_sat.hkl",
                DirectionCosines=True,
                ApplyAnvredCorrections=False,
                SortBy="RunNumber",
            )

            SaveIsawUB(
                InputWorkspace=satellite, Filename=filename + "_sat.mat"
            )

            self.resort_hkl(satellite, filename + "_sat.hkl")

            if np.linalg.norm(self.modHKL[:, 1]) > 0:
                for i in range(3):
                    if np.linalg.norm(self.modHKL[:, i]) > 0:
                        CloneWorkspace(
                            InputWorkspace=peaks, OutputWorkspace="tmp"
                        )

                        for peak in mtd["tmp"]:
                            int_mnp = peak.getIntMNP()
                            if int_mnp[i] == 0:
                                peak.setIntensity(0)
                                peak.setSigmaIntensity(0)
                            else:
                                peak.setIntMNP(V3D(int_mnp[i], 0, 0))

                        ol = mtd["tmp"].sample().getOrientedLattice()
                        ol.setMaxOrder(self.max_order)
                        ol.setModVec1(V3D(*self.modHKL[:, i]))
                        ol.setModVec2(V3D(0, 0, 0))
                        ol.setModVec3(V3D(0, 0, 0))
                        ol.setModUB(ol.getUB() @ ol.getModHKL())

                        k = i + 1

                        SaveHKL(
                            InputWorkspace="tmp",
                            Filename=filename + "_sat_k={}.hkl".format(k),
                            DirectionCosines=True,
                            ApplyAnvredCorrections=False,
                            SortBy="RunNumber",
                        )

                        SaveIsawUB(
                            InputWorkspace="tmp",
                            Filename=filename + "_sat_k={}.mat".format(k),
                        )

                        self.resort_hkl(
                            "tmp", filename + "_sat_k={}.hkl".format(k)
                        )

    def _angle_distance_matrix(self, Rs):
        G = np.tensordot(Rs, Rs, axes=([1, 2], [1, 2]))
        cos_th = (G - 1.0) / 2.0
        np.clip(cos_th, -1.0, 1.0, out=cos_th)
        D = np.arccos(cos_th)
        np.fill_diagonal(D, 0.0)
        return D

    def renumber_peaks(self, peaks, k=60, decimals=6, linkage="average"):
        Rs_all = []
        for p in mtd[peaks]:
            Rs_all.append(p.getGoniometerMatrix())
        Rs_all = np.asarray(Rs_all)
        N = len(Rs_all)

        flat = np.round(Rs_all.reshape(N, 9), decimals=decimals)
        uniq_rows, inv = np.unique(flat, axis=0, return_inverse=True)
        Rs_unique = np.array(
            [Rs_all[np.where(inv == i)[0][0]] for i in range(len(uniq_rows))]
        )
        M = len(Rs_unique)

        D = self._angle_distance_matrix(Rs_unique)

        n_clusters = min(k, M) if M > 0 else 0
        if n_clusters <= 1:
            labels_unique = (
                np.zeros(M, dtype=int) if M else np.array([], dtype=int)
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage=linkage
            )
            labels_unique = model.fit_predict(D)

        old_ids = np.unique(labels_unique)
        new_id_map = {old: i + 1 for i, old in enumerate(sorted(old_ids))}
        new_labels_unique = np.array(
            [new_id_map[x] for x in labels_unique], dtype=int
        )

        labels_all = new_labels_unique[inv]

        for ind, (peak, lbl) in enumerate(zip(mtd[peaks], labels_all)):
            peak.setRunNumber(int(lbl))
            peak.setPeakNumber(ind)

    def calculate_statistics(self, name, filename):
        FilterPeaks(
            InputWorkspace=name,
            OutputWorkspace=name + "_stats",
            FilterVariable="Signal/Noise",
            FilterValue=3,
            Operator=">",
        )

        point_groups, R_merge = [], []
        for point_group in self.point_groups:
            StatisticsOfPeaksWorkspace(
                InputWorkspace=name + "_stats",
                PointGroup=point_group_dict[point_group],
                OutputWorkspace="stats",
                EquivalentIntensities="Median",
                SigmaCritical=3,
                WeightedZScore=True,
            )

            R_merge.append(mtd["StatisticsTable"].toDict()["Rmerge"][0])
            point_groups.append(point_group)

        i = np.argmin(R_merge)
        point_group = point_groups[i]

        self.point_groups = [point_group]

        StatisticsOfPeaksWorkspace(
            InputWorkspace=name + "_stats",
            PointGroup=point_group_dict[point_group],
            OutputWorkspace="stats",
            EquivalentIntensities="Median",
            SigmaCritical=3,
            WeightedZScore=True,
        )

        ws = mtd["StatisticsTable"]

        column_names = ws.getColumnNames()
        col_widths = [max(len(str(name)), 8) for name in column_names]

        cols = [
            " ".join(
                name.ljust(col_widths[i])
                for i, name in enumerate(column_names)
            )
        ]

        for i in range(ws.rowCount()):
            row_values = []
            for j, val in enumerate(ws.row(i).values()):
                if isinstance(val, float):
                    val = "{:.2f}".format(val)
                row_values.append(str(val).ljust(col_widths[j]))

            cols.append(" ".join(row_values))

        table = "\n".join(cols)

        with open(filename, "w") as f:
            f.write("{}\n".format(point_group))
            f.write(table)

    def resort_hkl(self, peaks, filename):
        ol = mtd[peaks].sample().getOrientedLattice()

        UB = ol.getUB()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        max_order = ol.getMaxOrder()

        hkl_widths = [4, 4, 4]
        info_widths = [8, 8, 4, 8, 8, 9, 9, 9, 9, 9, 9, 6, 7, 7, 4, 9, 8, 7, 7]

        if max_order > 0:
            hkl_widths += hkl_widths

        col_widths = hkl_widths + info_widths

        h, k, l, m, n, p = [], [], [], [], [], []

        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                start = 0
                columns = []
                for width in col_widths:
                    columns.append(line[start : start + width].strip())
                    start += width
                h.append(columns[0])
                k.append(columns[1])
                l.append(columns[2])
                m.append(columns[3] if max_order > 0 else 0)
                n.append(columns[4] if max_order > 0 else 0)
                p.append(columns[5] if max_order > 0 else 0)

        h = np.array(h).astype(int)
        k = np.array(k).astype(int)
        l = np.array(l).astype(int)

        m = np.array(m).astype(int)
        n = np.array(n).astype(int)
        p = np.array(p).astype(int)

        mod_HKL = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        hkl = np.stack([h, k, l]) + np.einsum("ij,jm->im", mod_HKL, [m, n, p])

        s = np.linalg.norm(np.einsum("ij,jm->im", UB, hkl), axis=0)

        hkls = np.round(np.column_stack([*hkl, s]) * 1000, 1).astype(int)
        sort = np.lexsort(hkls.T).tolist()
        with open(filename, "w") as f:
            for i in sort[1:]:
                f.write(lines[i])
            f.write(lines[sort[0]])


def main():
    parser = argparse.ArgumentParser(description="Corrections for integration")

    parser.add_argument(
        "filename",
        type=str,
        help="Peaks Workspace",
    )

    parser.add_argument(
        "-f",
        "--formula",
        type=str,
        default="Yb3-Al5-O12",
        help="Chemical formula",
    )

    parser.add_argument(
        "-z",
        "--zparameter",
        type=float,
        default="8",
        help="Number of formula units",
    )

    parser.add_argument(
        "-g",
        "--pointgroup",
        type=str,
        default=None,
        help="Point group symmetry",
    )

    parser.add_argument(
        "-u",
        "--uvector",
        nargs="+",
        type=float,
        default=[0, 0, 1],
        help="Miller indices along beam",
    )

    parser.add_argument(
        "-v",
        "--vvector",
        nargs="+",
        type=float,
        default=[1, 0, 0],
        help="Miller indices in plane",
    )

    parser.add_argument(
        "-p",
        "--parameters",
        nargs="+",
        type=float,
        default=[0.1, 0.1, 0.1],
        help="Sample Parameters",
    )

    parser.add_argument(
        "-c", "--scale", type=float, default=None, help="Scale factor"
    )

    args = parser.parse_args()

    peaks = Peaks("peaks", args.filename, args.scale, args.pointgroup)
    peaks.load_peaks()

    if (np.array(args.parameters) > 0).all():
        AbsorptionCorrection(
            "peaks",
            args.formula,
            args.zparameter,
            u_vector=args.uvector,
            v_vector=args.vvector,
            params=args.parameters,
            filename=args.filename,
        )

    peaks.save_peaks()


if __name__ == "__main__":
    main()
