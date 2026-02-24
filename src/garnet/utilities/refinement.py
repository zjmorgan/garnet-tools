import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append(os.path.abspath(os.path.join(directory, "../..")))

import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.linalg
import scipy.interpolate
from scipy.spatial.transform import Rotation

from lmfit import Minimizer, Parameters

from mantid.simpleapi import (
    LoadNexus,
    SaveNexus,
    CloneWorkspace,
    CreateSampleWorkspace,
    LoadSampleShape,
    mtd,
)
from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
)
from mantid.kernel import Atom, MaterialBuilder, V3D

from garnet.plots.crystal import CrystalStructurePlot


class NuclearStructureRefinement:
    def __init__(self, cell, space_group, sites, filename, parameters=None):
        self.sites = []
        for site in sites:
            atm, x, y, z, occ = site
            x, y, z = self._wrap([x, y, z])
            self.sites.append([atm, x, y, z, occ, 0])

        self.cell = cell
        self.space_group = space_group

        self.initialize_crystal_structure(self.sites)
        self.initialize_material()
        self.generate_symmetry_transforms()
        self.initialize_unit_cell_atoms()

        self.load_hkls(filename)

        self.models = [
            "type i gaussian",
            "type i lorentzian",
            "type ii",
            "shelx",
            "none",
        ]

        self.thickness = 0.001
        self.width = 0.001
        self.height = 0.001

        if parameters is not None:
            self.update_sample(parameters)

        self.generate_parameters()
        self.spherical_extinction()
        self.spherical_absorption()

        self.filename = filename

    def _eval(self, op, x, y, z):
        return np.array(eval(op, {"x": x, "y": y, "z": z}))

    def _wrap(self, val):
        val = np.array(val)
        mask = val < 0
        val[mask] += 1
        mask = val >= 1
        val[mask] -= 1
        val[np.isclose(val, 0)] = 0
        return val

    def _collapse(self, val):
        val = np.array(val)
        mask = val <= -0.5
        val[mask] += 1
        mask = val > 0.5
        val[mask] -= 1
        val[np.isclose(val, 0)] = 0
        return val

    def _voigt6_from_sym(self, U):
        """
        Pack symmetric 3x3 into [11,22,33,12,13,23].
        """
        return np.array(
            [U[0, 0], U[1, 1], U[2, 2], U[0, 1], U[0, 2], U[1, 2]], dtype=float
        )

    def _sym_from_voigt6(self, v):
        """
        Unpack [11,22,33,12,13,23] into symmetric 3x3.
        """
        U = np.zeros((3, 3), dtype=float)
        U[0, 0], U[1, 1], U[2, 2] = v[0], v[1], v[2]
        U[0, 1] = U[1, 0] = v[3]
        U[0, 2] = U[2, 0] = v[4]
        U[1, 2] = U[2, 1] = v[5]
        return U

    def _voigt6_transform_matrix(self, R):
        """
        Build 6x6 matrix T such that vec(U') = T vec(U)
        for U' = R U R^T, using Voigt ordering [11,22,33,12,13,23]
        (no factor-of-2 convention).
        """
        basis = np.eye(6)
        T = np.zeros((6, 6), dtype=float)
        for a in range(6):
            Ua = self._sym_from_voigt6(basis[a])
            Uap = R @ Ua @ R.T
            T[:, a] = self._voigt6_from_sym(Uap)
        return T

    def initialize_unit_cell_atoms(
        self, tol=1e-6, rcond=None, enforce_identity=True
    ):
        S = np.asarray(self.S, dtype=float)

        # Pre-split rotations/translations
        Rall = S[:, :, :3]  # (n,3,3)
        tall = S[:, :, 3]  # (n,3)

        null_list, x0_list = [], []
        transforms_list, transforms_disp_list = [], []
        atms, js, bs = [], [], []
        null_disp_list = []

        # Optional: detect identity operator index (best-effort)
        if enforce_identity:
            I3 = np.eye(3)
            # identity op should have R≈I and t≈0 (mod 1)
            twrap = self._wrap(tall)
            is_id = np.all(
                np.isclose(Rall, I3, atol=tol, rtol=0), axis=(1, 2)
            ) & np.all(
                np.isclose(twrap, 0.0, atol=tol, rtol=0)
                | np.isclose(twrap, 1.0, atol=tol, rtol=0),
                axis=1,
            )
            id_idx = int(np.argmax(is_id)) if np.any(is_id) else 0
        else:
            id_idx = 0

        for j, site in enumerate(self.sites):
            atm, x, y, z, occ, U = site
            xv = np.asarray(self._wrap([x, y, z]), dtype=float)

            # Orbit (all equivalent positions) and coset representatives
            equiv = self._wrap((Rall @ xv) + tall)  # (n,3)

            # Identify stabilizer: ops that map x onto itself modulo lattice translations.
            # Robust compare via wrapped difference.
            diff = self._wrap(equiv - xv)
            mask = np.all(np.isclose(diff, 0.0, atol=tol, rtol=0), axis=1)

            if enforce_identity and not np.any(mask):
                mask[id_idx] = True

            # Unique equivalent positions (orbit) and store corresponding operators as coset reps
            equiv_q = np.round(equiv / tol) * tol
            uniq_pos, uniq_idx = np.unique(equiv_q, axis=0, return_index=True)
            ops_orbit = S[uniq_idx]  # (m,3,4)
            transforms_list.append(ops_orbit)

            # Expand bookkeeping for each orbit member
            m = len(uniq_idx)
            atms += [atm] * m
            js += [j] * m

            # scattering length (per atom type; repeated per orbit member)
            atom = Atom(atm)
            atm_dict = atom.neutron()
            br = atm_dict["coh_scatt_length_real"]
            bi = atm_dict["coh_scatt_length_img"]
            b = br + bi * 1j
            bs += [b] * m

            # ---- Position refinables: solve (R - I) x = n - t for stabilizer ops ----
            ops_stab = S[mask]
            Rstab = ops_stab[:, :, :3]
            tstab = ops_stab[:, :, 3]

            # Build A and b for all stabilizer ops:
            # (R - I)x = n - t, where n is chosen integer vector so that R*x + t - x - n ≈ 0
            A_blocks = []
            b_blocks = []
            for R, t in zip(Rstab, tstab):
                delta = (
                    (R @ xv) + t - xv
                )  # should be ~ integer vector for stabilizer
                nvec = np.round(delta)  # choose nearest integer shift
                A_blocks.append(R - np.eye(3))
                b_blocks.append(nvec - t)

            A = np.concatenate(A_blocks, axis=0)  # (3k,3)
            bvec = np.concatenate(b_blocks, axis=0)  # (3k,)

            # Particular solution (least-squares) and null space
            x_part, *_ = np.linalg.lstsq(A, bvec, rcond=rcond)
            x_part = np.asarray(x_part, dtype=float)
            x0_list.append(self._wrap(x_part))

            N = scipy.linalg.null_space(A)  # (3, dof)
            null_list.append(N)

            # ---- ADP refinables in Voigt6: U' = R U R^T; translation irrelevant ----
            # Build T for orbit reps (use their rotations)
            T_orbit = np.zeros((m, 6, 6), dtype=float)
            for ii, R in enumerate(ops_orbit[:, :, :3]):
                T_orbit[ii] = self._voigt6_transform_matrix(R)
            transforms_disp_list.append(T_orbit)

            # Stabilizer constraints for U: (T - I) u = 0
            T_stab = np.zeros((Rstab.shape[0], 6, 6), dtype=float)
            for ii, R in enumerate(Rstab):
                T_stab[ii] = self._voigt6_transform_matrix(R)
            AU = np.concatenate(
                [T_stab[ii] - np.eye(6) for ii in range(T_stab.shape[0])],
                axis=0,
            )
            NU = scipy.linalg.null_space(AU)  # (6, dofU)
            null_disp_list.append(NU)

        self.null = null_list
        self.x0 = x0_list
        self.transforms = transforms_list
        self.atms = np.asarray(atms)
        self.js = np.asarray(js, dtype=int)
        self.bs = np.asarray(bs, dtype=complex)
        self.null_disp = null_disp_list
        self.transforms_disp = transforms_disp_list

    def generate_symmetry_transforms(self):
        space_group = self.crystal_structure.getSpaceGroup()

        symops = space_group.getSymmetryOperationStrings()

        S = np.zeros((len(symops), 3, 4))

        S[..., 0] = [
            self._eval(op, 1, 0, 0) - self._eval(op, 0, 0, 0) for op in symops
        ]
        S[..., 1] = [
            self._eval(op, 0, 1, 0) - self._eval(op, 0, 0, 0) for op in symops
        ]
        S[..., 2] = [
            self._eval(op, 0, 0, 1) - self._eval(op, 0, 0, 0) for op in symops
        ]
        S[..., 3] = [self._eval(op, 0, 0, 0) for op in symops]

        self.S = S

    def generate_parameters(self):
        params = Parameters()

        visits = {}

        for i, (site, N, x0) in enumerate(zip(self.sites, self.null, self.x0)):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            key = tuple(np.round([x, y, z], 6))
            if visits.get(key) is None:
                visits[key] = var

            N_inv = np.linalg.pinv(N)

            u = N_inv @ (np.array([x, y, z]) - x0)

            for j, val in enumerate(u):
                params.add(
                    var.format("uval{}".format(j)),
                    value=val,
                    min=-10,
                    max=10,
                )

            params.add(var.format("x"), value=x, min=0, max=1)
            params.add(var.format("y"), value=y, min=0, max=1)
            params.add(var.format("z"), value=z, min=0, max=1)

            n = len(u)
            if n > 0:
                x_const = "+".join(
                    ["{}".format(x0[0])]
                    + [
                        "{}*{}".format(var.format("uval{}".format(j)), N[0, j])
                        for j in range(n)
                        if not np.isclose(N[0, j], 0)
                    ]
                )
                y_const = "+".join(
                    ["{}".format(x0[1])]
                    + [
                        "{}*{}".format(var.format("uval{}".format(j)), N[1, j])
                        for j in range(n)
                        if not np.isclose(N[1, j], 0)
                    ]
                )
                z_const = "+".join(
                    ["{}".format(x0[2])]
                    + [
                        "{}*{}".format(var.format("uval{}".format(j)), N[2, j])
                        for j in range(n)
                        if not np.isclose(N[2, j], 0)
                    ]
                )
                if np.any(N[0, :]):
                    params[var.format("x")].set(expr=x_const)
                else:
                    params[var.format("x")].set(vary=False)
                if np.any(N[1, :]):
                    params[var.format("y")].set(expr=y_const)
                else:
                    params[var.format("y")].set(vary=False)
                if np.any(N[2, :]):
                    params[var.format("z")].set(expr=z_const)
                else:
                    params[var.format("z")].set(vary=False)
            else:
                params[var.format("x")].set(vary=False)
                params[var.format("y")].set(vary=False)
                params[var.format("z")].set(vary=False)

            params.add(
                var.format("occ"),
                value=occ,
                min=0,
                max=1,
                vary=not np.isclose(occ, 1),
            )

        for i, (site, N) in enumerate(zip(self.sites, self.null_disp)):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            params.add(var.format("beta11"), value=0, min=-np.inf, max=np.inf)
            params.add(var.format("beta22"), value=0, min=-np.inf, max=np.inf)
            params.add(var.format("beta33"), value=0, min=-np.inf, max=np.inf)
            params.add(var.format("beta23"), value=0, min=-np.inf, max=np.inf)
            params.add(var.format("beta13"), value=0, min=-np.inf, max=np.inf)
            params.add(var.format("beta12"), value=0, min=-np.inf, max=np.inf)

            n = N.shape[1]

            for j in range(n):
                params.add(
                    var.format("bval{}".format(j)),
                    value=0,
                    min=-np.inf,
                    max=np.inf,
                )

            if n > 0:
                beta11_const = "+".join(
                    [
                        "{}*{}".format(var.format("bval{}".format(j)), N[0, j])
                        for j in range(n)
                        if not np.isclose(N[0, j], 0)
                    ]
                )
                beta22_const = "+".join(
                    [
                        "{}*{}".format(var.format("bval{}".format(j)), N[1, j])
                        for j in range(n)
                        if not np.isclose(N[1, j], 0)
                    ]
                )
                beta33_const = "+".join(
                    [
                        "{}*{}".format(var.format("bval{}".format(j)), N[2, j])
                        for j in range(n)
                        if not np.isclose(N[2, j], 0)
                    ]
                )
                beta23_const = "+".join(
                    [
                        "{}*{}".format(var.format("bval{}".format(j)), N[3, j])
                        for j in range(n)
                        if not np.isclose(N[3, j], 0)
                    ]
                )
                beta13_const = "+".join(
                    [
                        "{}*{}".format(var.format("bval{}".format(j)), N[4, j])
                        for j in range(n)
                        if not np.isclose(N[4, j], 0)
                    ]
                )
                beta12_const = "+".join(
                    [
                        "{}*{}".format(var.format("bval{}".format(j)), N[5, j])
                        for j in range(n)
                        if not np.isclose(N[5, j], 0)
                    ]
                )
                if np.any(N[0, :]):
                    params[var.format("beta11")].set(expr=beta11_const)
                else:
                    params[var.format("beta11")].set(vary=False)
                if np.any(N[1, :]):
                    params[var.format("beta22")].set(expr=beta22_const)
                else:
                    params[var.format("beta22")].set(vary=False)
                if np.any(N[2, :]):
                    params[var.format("beta33")].set(expr=beta33_const)
                else:
                    params[var.format("beta33")].set(vary=False)
                if np.any(N[3, :]):
                    params[var.format("beta23")].set(expr=beta23_const)
                else:
                    params[var.format("beta23")].set(vary=False)
                if np.any(N[4, :]):
                    params[var.format("beta13")].set(expr=beta13_const)
                else:
                    params[var.format("beta13")].set(vary=False)
                if np.any(N[5, :]):
                    params[var.format("beta12")].set(expr=beta12_const)
                else:
                    params[var.format("beta12")].set(vary=False)

        for i, site in enumerate(self.sites):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            key = tuple(np.round([x, y, z], 6))
            visit = visits.get(key)
            if visit != var:
                params.add(var.format("x"), expr=visit.format("x"))
                params.add(var.format("y"), expr=visit.format("y"))
                params.add(var.format("z"), expr=visit.format("z"))
                params.add(var.format("occ"), expr="1-" + visit.format("occ"))
                params.add(var.format("beta11"), expr=visit.format("beta11"))
                params.add(var.format("beta22"), expr=visit.format("beta22"))
                params.add(var.format("beta33"), expr=visit.format("beta33"))
                params.add(var.format("beta23"), expr=visit.format("beta23"))
                params.add(var.format("beta13"), expr=visit.format("beta13"))
                params.add(var.format("beta12"), expr=visit.format("beta12"))

        self.add_absorption_extinction_parameters(params)
        self.add_beam_parameters(params)
        self.add_detector_run_scale_parameters(params)

        F2s = self.calculate_structure_factors(params)
        scale = self.calculate_scale_factor(F2s)

        params.add("scale", value=scale, min=0, max=np.inf)

        self.params = params

    def update_sample(self, parameters):
        if type(parameters) is list:
            self.thickness = parameters[0] / 10
            self.width = parameters[1] / 10
            self.height = parameters[2] / 10
        else:
            self.thickness = parameters / 10
            self.width = parameters / 10
            self.height = parameters / 10

    def add_absorption_extinction_parameters(self, params):
        params.add("param", value=0.001, min=0, max=np.inf)

        angles = ["alpha", "beta", "gamma"]

        for i in range(3):
            params.add(
                "coeff_{}".format(angles[i]),
                value=0.0,
                min=-180,
                max=180,
            )

        vol = np.pi / 6 * self.thickness * self.width * self.height

        width_thickness_ratio = self.width / self.thickness
        height_thickness_ratio = self.height / self.thickness

        params.add(
            "coeff_volume",
            value=vol,
            min=np.pi / 6 * 0.001**3,
            max=np.pi / 6 * 2**3,
        )
        params.add(
            "coeff_width_thickness",
            value=width_thickness_ratio,
            min=0.2,
            max=5,
        )
        params.add(
            "coeff_height_thickness",
            value=height_thickness_ratio,
            min=0.2,
            max=5,
        )

        self.params = params

    def add_beam_parameters(self, params):
        params.add("off_b", value=0.0, min=-0.01, max=0.01)
        params.add("off_c", value=0.0, min=-np.inf, max=np.inf)
        params.add("off_rx", value=0.0, min=-np.inf, max=np.inf)
        params.add("off_rz", value=0.0, min=-np.inf, max=np.inf)

    def add_detector_run_scale_parameters(self, params):
        n = len(self.banks)
        for i in range(n - 1):
            params.add(
                "det_{}".format(i),
                value=1.0,
                min=0,
                max=np.inf,
                vary=False,
            )

        params.add(
            "det_{}".format(n - 1),
            expr=str(n)
            + "-"
            + "-".join(["det_{}".format(i) for i in range(n - 1)]),
        )

        n = len(self.runs)
        for i in range(n - 1):
            params.add(
                "run_{}".format(i),
                value=1.0,
                min=0,
                max=np.inf,
                vary=False,
            )

        params.add(
            "run_{}".format(n - 1),
            expr=str(n)
            + "-"
            + "-".join(["run_{}".format(i) for i in range(n - 1)]),
        )

        for i in range(2):
            params.add(
                "norm_{}".format(i),
                value=0.0,
                vary=True,
            )

    def extract_parameters(self, params, Uiso=0):
        sites = []

        for i, site in enumerate(self.sites):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            x = params[var.format("x")].value
            y = params[var.format("y")].value
            z = params[var.format("z")].value
            occ = params[var.format("occ")].value

            sites.append([name, x, y, z, occ, Uiso])

        param = params["param"].value
        scale = params["scale"].value

        names = [
            "alpha",
            "beta",
            "gamma",
            "volume",
            "width_thickness",
            "height_thickness",
        ]

        coeffs = np.array(
            [params["coeff_{}".format(names[i])].value for i in range(6)]
        )

        dets = np.array(
            [params["det_{}".format(i)].value for i in range(len(self.banks))]
        )

        runs = np.array(
            [params["run_{}".format(i)].value for i in range(len(self.runs))]
        )

        norm = np.array([params["norm_{}".format(i)].value for i in range(2)])

        off = np.array(
            [
                params["off_b"].value,
                params["off_c"].value,
                params["off_rx"].value,
                params["off_rz"].value,
            ]
        )

        return {
            "sites": sites,
            "param": param,
            "scale": scale,
            "coeffs": coeffs,
            "dets": dets,
            "runs": runs,
            "norm": norm,
            "off": off,
        }

    def load_hkls(self, filename):
        if not mtd.doesExist("peaks"):
            LoadNexus(Filename=filename, OutputWorkspace="peaks")

        self.extract_info()

    def extract_info(self):
        self.UB = mtd["peaks"].sample().getOrientedLattice().getUB()

        lamdas, two_thetas, intensity, sigma, G = [], [], [], [], []
        h, k, l, ri_hat, sf_hat, bank, run = [], [], [], [], [], [], []

        banks = mtd["peaks"].column("BankName")

        d_min = np.inf
        for i, peak in enumerate(mtd["peaks"]):
            hkl = peak.getIntHKL()
            mnp = peak.getIntMNP()
            lamda = peak.getWavelength()
            two_theta = peak.getScattering()
            d = peak.getDSpacing()
            if d < d_min:
                d_min = d

            ri = peak.getSourceDirectionSampleFrame()
            sf = peak.getDetectorDirectionSampleFrame()

            I = peak.getIntensity()
            sig = peak.getSigmaIntensity()
            run_no = peak.getRunNumber()

            R = peak.getGoniometerMatrix()

            if mnp.norm2() == 0 and I > 3 * sig and np.isfinite(I):
                lamdas.append(lamda)
                two_thetas.append(two_theta)

                h.append(hkl[0])
                k.append(hkl[1])
                l.append(hkl[2])

                intensity.append(I)
                sigma.append(sig)

                ri_hat.append(ri)
                sf_hat.append(sf)

                bank.append(banks[i])
                run.append(run_no)

                G.append(R)

        scale_factor = 10000 / np.nanpercentile(intensity, 99)

        unique = ReflectionGenerator(self.crystal_structure)
        unique = unique.getHKLsUsingFilter(
            d_min, float("inf"), ReflectionConditionFilter.StructureFactor
        )

        hkls = np.column_stack([h, k, l]).round().astype(int)

        mask = np.array(
            [V3D(*[float(v) for v in val]) in unique for val in hkls]
        )

        self.hkls = hkls[mask]

        self.equiv, self.inverse, self.counts = np.unique(
            self.hkls, return_inverse=True, return_counts=True, axis=0
        )

        self.banks, self.detectors = np.unique(
            np.array(bank)[mask], return_inverse=True, axis=0
        )

        self.runs, self.angles = np.unique(
            np.array(run)[mask], return_inverse=True, axis=0
        )

        self.I_obs = np.array(intensity)[mask] * scale_factor
        self.sig = np.array(sigma)[mask] * scale_factor

        self.lamda = np.array(lamdas)[mask]
        self.two_theta = np.array(two_thetas)[mask]
        self.Tbar = np.zeros_like(lamdas)[mask]

        self.ri_hat = np.array(ri_hat)[mask]
        self.sf_hat = np.array(sf_hat)[mask]
        self.G = np.array(G)[mask]

        pg = self.crystal_structure.getSpaceGroup().getPointGroup()
        self.families = np.unique(
            [pg.getEquivalents(hkl.tolist())[-1] for hkl in self.equiv], axis=0
        )

    def initialize_crystal_structure(self, sites):
        cell_params = " ".join(6 * ["{}"]).format(*self.cell)
        atom_sites = ";".join(
            [" ".join(6 * ["{}"]).format(*site) for site in sites]
        )
        crystal_structure = CrystalStructure(
            cell_params, self.space_group, atom_sites
        )

        self.V = crystal_structure.getUnitCell().volume()

        self.crystal_structure = crystal_structure

    def initialize_material(self):
        material = MaterialBuilder()

        self.chemical_formula, self.Z = self.chemical_formula_z_parameter()

        material.setFormula(self.chemical_formula)
        material.setZParameter(self.Z)
        material.setUnitCellVolume(self.V)

        self.material = material.build()

    def calculate_structure_factors(self, params):
        Fs = self.calculate_structure_amplitudes(params)

        return (Fs * Fs.conj())[self.inverse].real

    def calculate_structure_amplitudes(self, params):
        """
        Compute complex structure amplitudes Fs (per unique hkl) for given `params`.

        This is identical to the inner part of `calculate_structure_factors`
        but returns the complex amplitudes (not squared magnitudes).
        """
        Fs = 0 + 0j

        for i, (site, transform, transform_disp, b) in enumerate(
            zip(self.sites, self.transforms, self.transforms_disp, self.bs)
        ):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            x = params[var.format("x")].value
            y = params[var.format("y")].value
            z = params[var.format("z")].value

            xyz = np.einsum("ijk,k->ij", transform, [x, y, z, 1])

            pf = np.exp(2j * np.pi * np.einsum("ij,kj->ik", xyz, self.equiv))

            beta11 = params[var.format("beta11")].value
            beta22 = params[var.format("beta22")].value
            beta33 = params[var.format("beta33")].value
            beta23 = params[var.format("beta23")].value
            beta13 = params[var.format("beta13")].value
            beta12 = params[var.format("beta12")].value

            beta = np.einsum(
                "ijk,k->ij",
                transform_disp,
                [beta11, beta22, beta33, beta23, beta13, beta12],
            )

            occ = params[var.format("occ")].value

            h2 = [
                self.equiv[:, 0] ** 2,
                self.equiv[:, 1] ** 2,
                self.equiv[:, 2] ** 2,
                2 * self.equiv[:, 1] * self.equiv[:, 2],
                2 * self.equiv[:, 0] * self.equiv[:, 2],
                2 * self.equiv[:, 0] * self.equiv[:, 1],
            ]

            T = np.exp(-np.einsum("ij,jk->ik", beta, h2))

            Fs += np.sum(b * occ * pf * T, axis=0)

        return Fs

    def chemical_formula_z_parameter(self):
        sg = self.crystal_structure.getSpaceGroup()

        scatterers = self.crystal_structure.getScatterers()

        atom_dict = {}

        for scatterer in scatterers:
            atom, x, y, z, occ, _ = scatterer.split(" ")
            x, y, z, occ = float(x), float(y), float(z), float(occ)
            n = len(sg.getEquivalentPositions([x, y, z]))
            if atom_dict.get(atom) is None:
                atom_dict[atom] = [n], [occ]
            else:
                ns, occs = atom_dict[atom]
                ns.append(n)
                occs.append(occ)
                atom_dict[atom] = ns, occs

        chemical_formula = []

        n_atm = []
        n_wgt = []

        for key in atom_dict.keys():
            ns, occs = atom_dict[key]
            n_atm.append(np.sum(ns))
            n_wgt.append(np.sum(np.multiply(ns, occs)))
            if key.isalpha():
                chemical_formula.append(key + "{}")
            else:
                chemical_formula.append("(" + key + ")" + "{}")

        Z = np.gcd.reduce(n_atm)
        n = np.divide(n_wgt, Z)

        chemical_formula = "-".join(chemical_formula).format(*n)

        return chemical_formula, float(Z)

    def spherical_absorption(self):
        fname = os.path.join(directory, "absorption_sphere.csv")

        data = np.loadtxt(
            fname, skiprows=1, delimiter=",", usecols=np.arange(1, 92)
        )
        muR = np.loadtxt(fname, skiprows=1, delimiter=",", usecols=(0,))
        theta = np.loadtxt(
            fname, delimiter=",", max_rows=1, usecols=np.arange(1, 92)
        )

        two_theta = 2 * np.deg2rad(theta)

        f = scipy.interpolate.RectBivariateSpline(
            muR, two_theta, 1 / data, kx=1, ky=1
        )

        self.f = f

    def spherical_absorption_correction(self, muR, two_theta):
        return self.f(muR, two_theta, grid=False)

    def spherical_extinction(self):
        f1 = {}
        f2 = {}

        for model in self.models:
            if "gaussian" in model:
                filename = "secondary_extinction_gaussian_sphere.csv"
            elif "lorentzian" in model:
                filename = "secondary_extinction_lorentzian_sphere.csv"
            else:
                filename = "primary_extinction_sphere.csv"

            fname = os.path.join(directory, filename)

            data = np.loadtxt(
                fname, skiprows=1, delimiter=",", usecols=np.arange(91)
            )
            theta = np.loadtxt(fname, delimiter=",", max_rows=1)

            two_theta = 2 * np.deg2rad(theta)

            f1[model] = scipy.interpolate.interp1d(
                two_theta,
                data[0],
                kind="linear",
                fill_value="extrapolate",
            )
            f2[model] = scipy.interpolate.interp1d(
                two_theta,
                data[1],
                kind="linear",
                fill_value="extrapolate",
            )

        self.f1 = f1
        self.f2 = f2

    def extinction_xs(self, g, F2, two_theta, lamda, Tbar, V):
        k = 1e2  # g -> dimensionless
        xs = k / V**2 * lamda**3 * g / np.sin(two_theta) * Tbar * F2

        return xs

    def extinction_xp(self, r2, F2, lamda, V):
        k = 1e2  #  r2^2 -> um^2
        xp = k / V**2 * lamda**2 * r2 * F2

        return xp

    def extinction_xx(self, x, F2, two_theta, lamda, V):
        k = 1e2  # x -> cm
        xx = k / V**2 * lamda**3 * x / np.sin(two_theta) * F2

        return xx

    def extinction_correction(self, param, F2):
        """
        Calculate structure factor-dependent extinction correction.

        +------------------------+------------------+-----------------------+
        | Model                  | Nature           | Parameter             |
        +========================+==================+=======================+
        | ``Type I, Gaussian``   | Secondary        | Mosaic [unitless]     |
        +------------------------+------------------+-----------------------+
        | ``Type I, Lorentzian`` | Secondary        | Mosaic [unitless]     |
        +------------------------+------------------+-----------------------+
        | ``Type II``            | Primary          | Block size² [um²]     |
        +------------------------+------------------+-----------------------+
        | ``SHELX``              | Phenomenological | Crystal size [cm]     |
        +------------------------+------------------+-----------------------+
        [1] P. J. Becker and P. Coppens, Acta Cryst A 30, 2 (1974).
        [2] P. J. Becker and P. Coppens, Acta Cryst A 30, 2 (1974).
        [3] A. C. Larson, Crystallographic Computing (1970).


        Parameters
        ----------
        param : float
            Exinction parameter.
        F2 : array
            Structure factors (fm^2).

        Returns
        -------
        y : array
            Extinction (unitless).

        """

        two_theta = self.two_theta
        lamda = self.lamda
        Tbar = self.Tbar

        V = self.V
        if self.model.lower() == "type ii":
            xp = self.extinction_xp(param, F2, lamda, V)
            yp = 1 / (1 + self.c1 * xp**self.c2)
            return yp
        elif self.model.lower().startswith("type i"):
            xs = self.extinction_xs(param, F2, two_theta, lamda, Tbar, V)
            ys = 1 / (1 + self.c1 * xs**self.c2)
            return ys
        elif self.model.lower() == "shelx":
            xx = self.extinction_xx(param, F2, two_theta, lamda, V)
            yx = 1 / np.sqrt(1 + xx)
            return yx
        else:
            return np.ones_like(F2)

    def detector_bank_scale_factors(self, params):
        return params[self.detectors]

    def run_angle_scale_factors(self, params):
        return params[self.angles]

    def cost(self, params, F2s, I_obs, sig):
        s = params["scale"].value

        I_calc = s * F2s

        return (I_calc - I_obs) / sig

    def calculate_scale_factor(self, F2s):
        num = np.nansum(F2s * self.I_obs / self.sig**2)
        den = np.nansum(F2s**2 / self.sig**2)
        s = num / den

        params = Parameters()

        params.add("scale", value=s, min=0, max=np.inf)

        out = Minimizer(
            self.cost,
            params,
            fcn_args=(F2s, self.I_obs, self.sig),
            nan_policy="omit",
        )

        result = out.minimize(method="least_squares", loss="soft_l1")

        scale = result.params["scale"].value

        return scale

    def objective_structure(self, params):
        """
        Objective for structure + scale refinement.

        This stage varies positional, occupancy and displacement
        parameters together with the overall scale, while using the
        current absorption/extinction parameters as fixed values.
        """

        p = self.extract_parameters(params)

        scale = p["scale"]

        F2s = self.calculate_structure_factors(params)

        I_calc = F2s * self.y

        return (scale * I_calc - self.I_obs) / self.sig

    def objective_correction(self, params):
        """
        Objective for absorption/extinction refinement.

        Structure and scale are held fixed; F2s are precomputed for the
        current structural model.
        """

        p = self.extract_parameters(params)

        param = p["param"]
        scale = p["scale"]
        coeffs = p["coeffs"]
        dets = p["dets"]
        runs = p["runs"]
        norm = p["norm"]
        off = p["off"]

        y_abs = self.absorption_correction(coeffs)
        y_off = self.wobble_correction(off)
        y_ext = self.extinction_correction(param, self.F2s)
        y_norm = self.normalization_correction(norm)
        y_dets = self.detector_bank_scale_factors(dets)
        y_runs = self.run_angle_scale_factors(runs)

        y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

        I_calc = self.F2s * y

        wr = (scale * I_calc - self.I_obs) / self.sig

        return wr

    def _minimize_stage(
        self,
        objective,
        vary_flags,
        stage_name,
        iter_num,
        n_iter,
        report,
        reduce_fcn=None,
        fixed=[],
    ):
        """
        Helper to minimize a stage by setting parameter vary flags.

        Parameters
        ----------
        objective : callable
            Objective function (e.g., objective_structure, objective_correction).
        vary_flags : dict
            Maps parameter name patterns to bool; e.g. {"coeff_": True, "param": False}.
        stage_name : str
            Stage label for reporting.
        iter_num : int
            Current iteration number (1-indexed).
        n_iter : int
            Total iterations.
        report : bool
            Whether to print report.
        reduce_fcn : str, optional
            Reduction function for Minimizer (e.g. "negentropy").
        fixed : list, optional
            List of parameter names that should never be varied.

        Returns
        -------
        result : lmfit Result
            Minimization result.
        """
        params = self.params.copy()

        for name, par in params.items():
            if name in fixed or par.expr is not None:
                par.vary = False
            elif name == "scale":
                par.vary = True
            else:
                par.vary = False  # Default off
                for pattern, do_vary in vary_flags.items():
                    if pattern in name:
                        par.vary = do_vary
                        break

        kw = {"nan_policy": "omit"}
        if reduce_fcn:
            kw["reduce_fcn"] = reduce_fcn
        out = Minimizer(objective, params, **kw)
        result = out.minimize(method="leastsq", max_nfev=100)
        self.params = result.params

        if report:
            print(f"Iteration {iter_num}/{n_iter} - {stage_name}")
            self.report(result, False, False)

        return result

    def refine(
        self,
        report=True,
        cutoff=10,
        n_iter=3,
        abs_corr=False,
        off_corr=False,
        det_corr=False,
        run_corr=False,
        norm_corr=False,
        ext_model="shelx",
    ):
        """
        Iterative refinement protocol: 7 stages per iteration.

        Per iteration:
          1. Structure + scale (fixed corrections).
          2. Absorption only.
          3. Extinction only.
          4. Wobble only.
          5. Detector scales (optional).
          6. Run/orientation scales (optional).
          7. Normalization scales (optional).
        """
        self.initialize_correction(ext_model)

        struct_params = ["_uval", "_bval", "_occ"]

        fixed = []
        for name, par in self.params.items():
            if par.vary == False or par.expr is not None:
                if any(p in name for p in struct_params):
                    fixed.append(name)

        for i in range(n_iter):
            # Update corrections based on current parameters
            self.F2s = self.calculate_structure_factors(self.params)
            p = self.extract_parameters(self.params)
            coeffs = p["coeffs"]
            param = p["param"]
            norm = p["norm"]
            off = p["off"]
            dets = p["dets"]
            runs = p["runs"]

            y_abs = self.absorption_correction(coeffs)
            y_off = self.wobble_correction(off)
            y_ext = self.extinction_correction(param, self.F2s)
            y_norm = self.normalization_correction(norm)
            y_dets = self.detector_bank_scale_factors(dets)
            y_runs = self.run_angle_scale_factors(runs)

            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

            # --- Stage 1: structure + scale (fixed corrections)
            self._minimize_stage(
                self.objective_structure,
                {p: True for p in struct_params},
                "Stage 1 (structure + scale)",
                i + 1,
                n_iter,
                report,
                fixed=fixed,
            )

            self.F2s = self.calculate_structure_factors(self.params)
            self.calculate_statistics(cutoff)

            # --- Stage 2: absorption only
            self._minimize_stage(
                self.objective_correction,
                {"coeff_": abs_corr},
                "Stage 2 (absorption)",
                i + 1,
                n_iter,
                report and abs_corr,
                fixed=fixed,
            )

            p = self.extract_parameters(self.params)
            coeffs = p["coeffs"]
            y_abs = self.absorption_correction(coeffs)
            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

            # --- Stage 3: extinction only
            self._minimize_stage(
                self.objective_correction,
                {"param": True},
                "Stage 3 (extinction)",
                i + 1,
                n_iter,
                report,
                fixed=fixed,
            )

            p = self.extract_parameters(self.params)
            param = p["param"]
            y_ext = self.extinction_correction(param, self.F2s)
            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

            # --- Stage 4: wobble only
            self._minimize_stage(
                self.objective_correction,
                {"off_": off_corr},
                "Stage 4 (wobble)",
                i + 1,
                n_iter,
                report and off_corr,
                fixed=fixed,
            )

            p = self.extract_parameters(self.params)
            off = p["off"]
            y_off = self.wobble_correction(off)
            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

            # --- Stage 5: detector scales only
            self._minimize_stage(
                self.objective_correction,
                {"det_": det_corr},
                "Stage 5 (detector scales)",
                i + 1,
                n_iter,
                report and det_corr,
                fixed=fixed,
            )

            p = self.extract_parameters(self.params)
            dets = p["dets"]
            y_dets = self.detector_bank_scale_factors(dets)
            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

            # --- Stage 6: run/orientation scales only
            self._minimize_stage(
                self.objective_correction,
                {"run_": run_corr},
                "Stage 6 (run scales)",
                i + 1,
                n_iter,
                report and run_corr,
                fixed=fixed,
            )

            p = self.extract_parameters(self.params)
            runs = p["runs"]
            y_runs = self.run_angle_scale_factors(runs)
            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs

            # --- Stage 7: normalization only
            self._minimize_stage(
                self.objective_correction,
                {"norm_": norm_corr},
                "Stage 7 (normalization)",
                i + 1,
                n_iter,
                report and norm_corr,
                fixed=fixed,
            )

            if norm_corr:
                p = self.extract_parameters(self.params)
                norm = p["norm"]
                y_norm = self.normalization_correction(norm)

            self.y = y_abs * y_off * y_ext * y_norm * y_dets * y_runs
            self.calculate_statistics(cutoff)

            p = self.extract_parameters(self.params)
            sites = p["sites"]
            dets = p["dets"]

            self.initialize_crystal_structure(sites)
            self.initialize_material()

            self.plot_result()
            self.plot_sample_shape()
            self.save_detector_scales(dets)

        self.plot_hkl_families()
        self.save_corrected_peaks()

    def report(self, result, det_corr, run_corr):
        params = result.params
        print("χ² = {:.2f}".format(result.chisqr))
        print("χ²/dof = {:.2f}\n".format(result.redchi))

        p = self.extract_parameters(result.params)

        param = p["param"]
        scale = p["scale"]
        coeffs = p["coeffs"]
        dets = p["dets"]
        runs = p["runs"]
        norm = p["norm"]
        off = p["off"]

        ellip = self.ellipsoid_parameters(coeffs)
        alpha, beta, gamma, thickness, width, height = ellip

        print("scale = {:1.4e}".format(scale))
        print("ext = {:6.4f}".format(param))
        print("abs : {:6.5f} {:6.5f} {:6.5f}".format(thickness, width, height))
        print("    : {:6.1f} {:6.1f} {:6.1f}".format(alpha, beta, gamma))
        print("")

        for i, site in enumerate(self.sites):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            x = params[var.format("x")].value
            y = params[var.format("y")].value
            z = params[var.format("z")].value

            beta11 = params[var.format("beta11")].value
            beta22 = params[var.format("beta22")].value
            beta33 = params[var.format("beta33")].value
            beta23 = params[var.format("beta23")].value
            beta13 = params[var.format("beta13")].value
            beta12 = params[var.format("beta12")].value

            occ = params[var.format("occ")].value

            print(
                "{:3} : {:6.4f} {:6.4f} {:6.4f} {:6.4f}".format(
                    name, x, y, z, occ
                )
            )
            print(
                "    : {:6.4f} {:6.4f} {:6.4f}".format(beta11, beta22, beta33)
            )
            print(
                "    : {:6.4f} {:6.4f} {:6.4f}".format(beta23, beta13, beta12)
            )

        print("")
        print("norm : {:6.4f} {:6.4f}".format(*norm))
        print("")

        b, c, rx, ry = off
        print("beam : {:6.4f} {:6.4f} {:6.4f} {:6.4f}".format(b, c, rx, ry))
        print("")

        if run_corr:
            for i in range(len(self.runs)):
                print("#{:} | {:6.4f}".format(self.runs[i], runs[i]))
            print("")

        if det_corr:
            for i in range(len(self.banks)):
                print("{:} | {:6.4f}".format(self.banks[i], dets[i]))
            print("")

    def calculate_statistics(self, cutoff):
        p = self.extract_parameters(self.params)

        self.sites = p["sites"]
        self.param = p["param"]
        self.scale = p["scale"]
        self.coeffs = p["coeffs"]
        dets = p["dets"]
        runs = p["runs"]
        norm = p["norm"]
        off = p["off"]

        F2s = self.calculate_structure_factors(self.params)

        y_abs = self.absorption_correction(self.coeffs)
        y_off = self.wobble_correction(off)
        y_ext = self.extinction_correction(self.param, F2s)
        y_norm = self.normalization_correction(norm)
        y = y_abs * y_off * y_ext * y_norm

        c = self.detector_bank_scale_factors(dets)
        k = self.run_angle_scale_factors(runs)

        I_calc = F2s * y * c * k

        scale = self.calculate_scale_factor(I_calc)

        self.I_calc = scale * I_calc

        F_obs = np.sqrt(self.I_obs / scale)
        F_calc = np.sqrt(self.I_calc / scale)
        F_sig = 0.5 * self.sig / np.sqrt(self.I_obs * scale)

        self.F_obs_all = F_obs
        self.F_calc_all = F_calc
        self.F_sig_all = F_sig

        mask = np.abs(F_calc - F_obs) < cutoff * F_sig

        inverse = self.inverse[mask]

        F_sig = F_sig[mask]
        F_obs = F_obs[mask]
        F_calc = F_calc[mask]

        w = np.bincount(inverse, weights=1 / F_sig**2)

        F_obs = np.bincount(inverse, weights=F_obs / F_sig**2) / w
        F_calc = np.bincount(inverse, weights=F_calc / F_sig**2) / w
        F_sig = 1 / np.sqrt(w)

        mask = np.abs(F_calc - F_obs) < cutoff * F_sig

        self.F_obs = F_obs[mask]
        self.F_calc = F_calc[mask]
        self.F_sig = F_sig[mask]

        R = np.sum(np.abs(self.F_calc - self.F_obs)) / np.sum(self.F_obs)

        self.R_ref = R * 100  # %

        print("R = {:.2f}%".format(self.R_ref))
        print("----------")
        print("")

    def save_detector_scales(self, params):
        output = os.path.splitext(self.filename)[0]
        with open(output + "_det.txt", "w") as f:
            for i in range(len(params)):
                bank = self.banks[i].replace("bank", "")
                f.write("{:4} {:.4f}\n".format(bank, params[i]))

    def plot_result(self):
        output = os.path.splitext(self.filename)[0]

        F_lim = [np.min(self.F_obs), np.max(self.F_obs)]

        R = self.R_ref

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.plot(F_lim, F_lim, color="C1")
        ax.errorbar(
            self.F_calc,
            self.F_obs,
            yerr=self.F_sig,
            fmt=".",
            color="C0",
            rasterized=True,
        )
        chemical_formula = self.chemical_formula.replace("-", " ")
        Z = int(self.Z)

        title = r"{} | {} | $R = {:.2f}\%$".format(chemical_formula, Z, R)

        ax.set_aspect(1)
        ax.minorticks_on()
        ax.set_xlim(*F_lim)
        ax.set_ylim(*F_lim)
        ax.set_title(title)
        ax.set_xlabel(r"$|F|_\mathrm{calc}$")
        ax.set_ylabel(r"$|F|_\mathrm{obs}$")

        # ax = axs[1]
        # ax.plot(F_lim, F_lim, color="C1")
        # ax.errorbar(
        #     self.F_calc_all,
        #     self.F_obs_all,
        #     yerr=self.F_sig_all,
        #     fmt=".",
        #     color="C0",
        #     rasterized=True,
        # )
        # ax.set_aspect(1)
        # ax.set_xlabel(r"$|F|_\mathrm{calc}$")
        # ax.set_ylabel(r"$|F|_\mathrm{obs}$")

        fig.savefig(output + "_ref.pdf", bbox_inches="tight")

        cs = CrystalStructurePlot(self.sites, self.cell, self.space_group)
        cs.plot(output + "_cryst_struct.pdf")

    def plot_absorption_correction(self, filename=None):
        if self.filename is None and filename is None:
            return

        if filename is None:
            filename = os.path.splitext(self.filename)[0] + "_abs.pdf"

        two_theta_deg = np.rad2deg(self.two_theta)
        lamda = self.lamda
        A = self.T

        fig, ax = plt.subplots(1, 1)

        norm = Normalize(vmin=np.min(A), vmax=np.max(A))
        cmap = colormaps["viridis"]

        scatter = ax.scatter(
            two_theta_deg,
            lamda,
            c=A,
            s=1,
            cmap=cmap,
            norm=norm,
            alpha=1,
        )

        ax.set_xlabel(r"$2\theta$ [degrees]")
        ax.set_ylabel(r"$\lambda$ [$\AA$]")
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(FormatStrFormatter(r"$%d^\circ$"))

        cb = fig.colorbar(scatter, ax=ax, label="Absorption factor A")
        cb.minorticks_on()

        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    def plot_wobble_correction(self, off, filename=None):
        """
        Plot wobble correction scale factors vs rotation angle.

        Shows observed data points and model curves for different wavelengths.
        """
        if self.filename is None and filename is None:
            return

        if filename is None:
            filename = os.path.splitext(self.filename)[0] + "_off.pdf"

        b, c, rx, rz = off

        R = self.G
        lamda = self.lamda

        omega = np.rad2deg(np.arctan2(R[:, 0, 2], R[:, 0, 0]))
        omega[omega < 0] += 360

        x, _, _ = np.einsum("kij,j->ik", R, [rx, 0, rz])
        x = x + c
        y = np.exp(-0.5 * x**2 / (1.0 + b * lamda) ** 2)

        fig, ax = plt.subplots(1, 1)
        ax.plot(omega, y, ",", color="k")

        omega_grid = np.linspace(0, 360, 361)
        R_grid = [
            Rotation.from_euler("y", val, degrees=True).as_matrix()
            for val in omega_grid
        ]

        lamda_min, lamda_max = np.min(lamda), np.max(lamda)
        norm = Normalize(vmin=lamda_min, vmax=lamda_max)
        cmap = colormaps["turbo"]

        for lam in np.linspace(lamda_min, lamda_max, 5):
            xg, _, _ = np.einsum("kij,j->ik", R_grid, [rx, 0, rz])
            xg = xg + c
            yg = np.exp(-0.5 * xg**2 / (1.0 + b * lam) ** 2)
            ax.plot(omega_grid, yg, color=cmap(norm(lam)))

        ax.set_xlabel("Rotation angle")
        ax.set_ylabel("Scale factor")
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(FormatStrFormatter(r"$%d^\circ$"))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=ax, label=r"$\lambda$ [$\AA$]")
        cb.minorticks_on()

        fig.savefig(filename, bbox_inches="tight")

    def plot_hkl_families(self, filename=None):
        """
        Create multi-page PDF with one page per hkl family.
        """
        if self.filename is None and filename is None:
            return

        if filename is None:
            filename = os.path.splitext(self.filename)[0] + "_families.pdf"

        I_calc = self.F2s * self.y * self.scale
        I_obs = self.I_obs
        I_sig = self.sig
        hkls = self.hkls

        pg = self.crystal_structure.getSpaceGroup().getPointGroup()

        with PdfPages(filename) as pdf:
            for family_hkl in self.families:
                family_indices = []
                for i, hkl in enumerate(hkls):
                    equiv = np.array(pg.getEquivalents(hkl.tolist())[-1])
                    if np.isclose(family_hkl, equiv).all():
                        family_indices.append(i)

                if len(family_indices) == 0:
                    continue

                ic = I_calc[family_indices]
                io = I_obs[family_indices]
                eo = I_sig[family_indices]

                fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))

                lim = [
                    np.min([np.min(ic), np.min(io)]),
                    np.max([np.max(ic), np.max(io)]),
                ]
                lim = [lim[0] * 0.9, lim[1] * 1.1]

                ax.plot(lim, lim, "-")
                ax.errorbar(ic, io, yerr=eo, fmt="o", rasterized=True)

                ax.set_xlim(*lim)
                ax.set_ylim(*lim)
                ax.set_aspect("equal")
                ax.set_xlabel(r"$I_\mathrm{calc}$ [arb. units]")
                ax.set_ylabel(r"$I_\mathrm{obs}$ [arb. units]")

                h, k, l = family_hkl
                ax.set_title(
                    f"Family $({h:.0f}, {k:.0f}, {l:.0f})$ | $N = {len(family_indices)}$"
                )
                ax.minorticks_on()

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def save_corrected_peaks(self):
        """
        Apply absorption corrections to all peaks and save to new file.

        """
        output = os.path.splitext(self.filename)[0] + "_corr.nxs"

        CloneWorkspace(InputWorkspace="peaks", OutputWorkspace="peaks_corr")

        p = self.extract_parameters(self.params)
        scale = p["scale"]
        coeffs = p["coeffs"]
        dets = p["dets"]
        runs = p["runs"]
        norm = p["norm"]
        off = p["off"]

        lamdas = []
        two_thetas = []
        ri_hat = []
        sf_hat = []
        G = []

        for peak in mtd["peaks_corr"]:
            lamda = peak.getWavelength()
            two_theta = peak.getScattering()
            ri = peak.getSourceDirectionSampleFrame()
            sf = peak.getDetectorDirectionSampleFrame()
            R = peak.getGoniometerMatrix()

            lamdas.append(lamda)
            two_thetas.append(two_theta)
            ri_hat.append(ri)
            sf_hat.append(sf)

            G.append(R)

        material = self.material
        n = material.numberDensityEffective
        # sigma_tot = material.totalScatterXSection()
        sigma_abs = [material.absorbXSection(lamda) for lamda in lamdas]
        mu = n * np.array(sigma_abs)

        self.lamda = np.array(lamdas)
        self.two_theta = np.array(two_thetas)

        self.mu = mu
        self.ri_hat = np.array(ri_hat)
        self.sf_hat = np.array(sf_hat)
        self.G = np.array(G)

        y_abs = self.absorption_correction(coeffs)
        y_off = self.wobble_correction(off)
        y_norm = self.normalization_correction(norm)

        Tbar = self.Tbar.copy()

        banks_corr = mtd["peaks_corr"].column("BankName")

        for i, peak in enumerate(mtd["peaks_corr"]):
            I = peak.getIntensity()
            sig = peak.getSigmaIntensity()

            run_no = peak.getRunNumber()
            bank_name = banks_corr[i]

            run_scale, bank_scale = 1, 1

            if bank_name in list(self.banks):
                bank_index = int(np.where(self.banks == bank_name)[0][0])
                bank_scale = float(dets[bank_index])

            if run_no in list(self.runs):
                run_index = int(np.where(self.runs == run_no)[0][0])
                run_scale = float(runs[run_index])

            scale = bank_scale * run_scale

            factor = y_abs[i] * y_off[i] * y_norm[i] * scale

            I_corr = I / factor
            sig_corr = sig / factor

            peak.setIntensity(I_corr)
            peak.setSigmaIntensity(sig_corr)
            peak.setAbsorptionWeightedPathLength(Tbar[i])

        SaveNexus(InputWorkspace="peaks_corr", Filename=output)

        CloneWorkspace(InputWorkspace="peaks_corr", OutputWorkspace="peaks")

    def plot_sample_shape(self):
        """
        Plot the refined ellipsoidal sample shape with beam directions.

        """
        output = os.path.splitext(self.filename)[0]

        params = self.ellipsoid_parameters(self.coeffs)
        alpha, beta, gamma, thickness, width, height = params

        width *= 10
        thickness *= 10
        height *= 10

        sph = pv.Icosphere(radius=0.5)

        ell = sph.scale([width, height, thickness], inplace=False)
        ell.save(output + ".stl")

        CreateSampleWorkspace(OutputWorkspace="sample")

        LoadSampleShape(
            InputWorkspace="sample",
            Filename=output + ".stl",
            Scale="mm",
            XDegrees=alpha,
            YDegrees=beta,
            ZDegrees=gamma,
            OutputWorkspace="sample",
        )

        hkl = np.eye(3)
        reciprocal_lattice = np.matmul(self.UB, hkl)

        shape = mtd["sample"].sample().getShape()
        mesh = shape.getMesh() * 1000

        mesh_polygon = Poly3DCollection(
            mesh,
            edgecolors="k",
            facecolors="w",
            alpha=0.5,
            linewidths=0.1,
        )

        D, R, Q = self.calculate_ellipsoid_surface(self.coeffs)

        UB_inv = np.linalg.inv(self.UB)

        v, w, u = R[:, 0], R[:, 1], R[:, 2]

        v, w, u = UB_inv @ v, UB_inv @ w, UB_inv @ u

        u /= np.max(np.abs(u))
        v /= np.max(np.abs(v))
        w /= np.max(np.abs(w))

        self.uvector = u
        self.vvector = v
        self.parameters = thickness, width, height

        fig, ax = plt.subplots(
            subplot_kw={"projection": "mantid3d", "proj_type": "persp"}
        )
        ax.add_collection3d(mesh_polygon)
        ax.set_proj_type("ortho")

        size = "thickness: {:.2} width: {:.2} height: {:.2} mm".format(
            thickness, width, height
        )
        u_vector = "u = [{:.2f} {:.2f} {:.2f}] ".format(*u)
        v_vector = "v = [{:.2f} {:.2f} {:.2f}] ".format(*v)
        w_vector = "w = [{:.2f} {:.2f} {:.2f}] ".format(*w)

        ax.set_title(size + "\n" + u_vector + " " + v_vector + " " + w_vector)
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

        plt.tight_layout()
        fig.savefig(output + "_sample_shape.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def ellipsoid_parameters(self, coeffs):
        alpha, beta, gamma, vol, ratio_b, ratio_c = coeffs

        a = np.cbrt(6 * vol / (np.pi * ratio_b * ratio_c))

        thickness = a
        width = a * ratio_b
        height = a * ratio_c

        return alpha, beta, gamma, thickness, width, height

    def calculate_ellipsoid_surface(self, coeffs):
        params = self.ellipsoid_parameters(coeffs)
        alpha, beta, gamma, thickness, width, height = params

        R = Rotation.from_euler(
            "ZYX", [gamma, beta, alpha], degrees=True
        ).as_matrix()

        D = np.diag([1 / width**2, 1 / height**2, 1 / thickness**2]) * 4

        return D, R, R @ D @ R.T

    def normalization_correction(self, params):
        muRs, muRa = params
        muR = muRs + muRa * self.lamda / 1.8
        return 1 / self.spherical_absorption_correction(muR, self.two_theta)

    def beam_weights(self, sample_points, G, ds, bx, by, sigma):
        """
        Calculate Gaussian beam weights for sample points.

        Parameters
        ----------
        sample_points : ndarray, shape (N, 3)
            Monte Carlo sample points in sample frame
        G : ndarray, shape (P, 3, 3)
            Goniometer matrices for each peak
        ds : ndarray, shape (P, 3)
            Sample center offset in sample frame for each peak
        bx, by : float
            Beam center position in lab frame
        sigma : float
            Gaussian beam width

        Returns
        -------
        bw : ndarray, shape (N, P)
            Beam weights for each sample point and peak
        """
        xs = sample_points.astype(np.float32)
        ds = np.asarray(ds, dtype=np.float32)
        G = np.asarray(G, dtype=np.float32)

        xsp = xs[None, :, :] + ds[:, None, :]
        xl = np.einsum("pij,pnj->pni", G, xsp)

        dx = xl[:, :, 0] - np.float32(bx)
        dy = xl[:, :, 1] - np.float32(by)

        inv2s2 = np.float32(1.0) / (
            np.float32(2.0) * np.float32(sigma) * np.float32(sigma)
        )
        bw = np.exp(-(dx * dx + dy * dy) * inv2s2)

        return bw.T

    def absorption_correction(
        self,
        coeffs,
    ):
        D, R, Q = self.calculate_ellipsoid_surface(coeffs)

        S = (R @ np.diag(1 / np.sqrt(np.diag(D)))).astype(np.float32)
        y = self.sample_points @ S.T
        yQ = y @ Q
        cquad = np.einsum("ij,ij->i", yQ, y) - 1.0

        A, Tbar = self._absorption_factors(
            self.mu.astype(np.float32),
            Q,
            yQ,
            cquad,
            self.ri_hat,
            self.sf_hat,
        )

        self.T = np.clip(A, 1e-8, None)
        self.Tbar = Tbar
        return self.T

    def prepare_absorption_table(self, N=100, seed=42, beta=3):
        rng = np.random.default_rng(seed)

        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        u = rng.random(N).astype(np.float32)
        r = u ** (1.0 / beta)
        p = v * r[:, None]

        w = (3.0 / beta) * (r ** (3.0 - beta))

        self.sample_points = p.astype(np.float32)
        self.sample_weights = w.astype(np.float32)
        self.N_mc = N

    def wobble_correction(self, off):
        """
        Scale model for off-centering/wobble correction.

        Based on WobbleCorrection model: accounts for beam displacement
        and sample misalignment effects on intensity measurements.

        Parameters
        ----------
        off : array-like
            [b, c, rx, rz] where:
            - b: spread (wavelength-dependent scaling)
            - c: center displacement
            - rx: in-plane displacement
            - rz: beam displacement

        Returns
        -------
        array
            Scale factors for each reflection
        """
        b, c, rx, rz = off

        R = self.G
        lamda = self.lamda

        x, _, _ = np.einsum("kij,j->ik", R, [rx, 0, rz])

        x = x + c

        scales = np.exp(-0.5 * x**2 / (1.0 + b * lamda) ** 2)

        return scales

    def initialize_correction(self, model="type II"):
        material = self.material
        n = material.numberDensityEffective
        # sigma_tot = material.totalScatterXSection()
        sigma_abs = [material.absorbXSection(lamda) for lamda in self.lamda]
        self.mu = n * np.array(sigma_abs)
        self.model = model.lower() if type(model) == str else "none"
        self.c1 = self.f1[self.model](self.two_theta)
        self.c2 = self.f2[self.model](self.two_theta)
        self.prepare_absorption_table()

    def _exit_lengths_for_directions(self, Q, yQ, cquad, dirs):
        a = np.einsum("mi,ij,mj->m", dirs, Q, dirs)

        b = 2 * (yQ @ dirs.T)

        disc = b * b - 4 * (cquad[:, None] * a[None, :])
        disc = np.maximum(disc, 0.0)

        eps = 1e-12
        a_safe = np.where(np.abs(a) < eps, np.sign(a) * eps + eps, a)

        t = (-b + np.sqrt(disc)) / (2 * a_safe[None, :])
        t = np.maximum(t, 0.0)
        return t

    def _absorption_factors(self, mu, Q, yQ, cquad, n_in_rev, n_out_rev):
        P = n_in_rev.shape[0]

        dirs = np.vstack([n_in_rev, n_out_rev])
        t = self._exit_lengths_for_directions(Q, yQ, cquad, dirs)

        t_total = t[:, :P] + t[:, P:]

        w = np.exp(-t_total * mu[None, :])
        ws = self.sample_weights[:, None]

        ws_sum = np.sum(ws, axis=0)
        A = np.sum(ws * w, axis=0) / ws_sum

        denom = np.sum(ws * w, axis=0)
        Tbar = np.sum(ws * w * t_total, axis=0) / np.clip(denom, 1e-30, None)

        return A, Tbar
