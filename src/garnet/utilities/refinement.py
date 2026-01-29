import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append(os.path.abspath(os.path.join(directory, "../..")))

import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.linalg
import scipy.interpolate
from scipy.spatial.transform import Rotation

from lmfit import Minimizer, Parameters

from mantid.simpleapi import (
    LoadNexus,
    SaveNexus,
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
        ]

        self.thickness = 0.001
        self.width_ratio = 1
        self.height_ratio = 1

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

    def initialize_unit_cell_atoms(self, tol=1e-4):
        null, x0, transforms, atms, js, bs = [], [], [], [], [], []

        index_map = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

        null_disp, transforms_disp = [], []

        for j, site in enumerate(self.sites):
            atm, x, y, z, occ, U = site
            x, y, z = self._wrap([x, y, z])
            equiv = self._wrap(np.einsum("ijk,k->ij", self.S, [x, y, z, 1]))
            mask = np.isclose(equiv, [x, y, z]).all(axis=1)
            equiv = np.round(equiv / tol) * tol
            equiv, index = np.unique(equiv, axis=0, return_index=True)
            transforms.append(self.S[index])
            atms += [atm] * len(index)
            js += [j] * len(index)
            P = self.S[mask].mean(axis=0)
            A = np.concatenate(
                [
                    self.S[i, :, :3] - np.eye(3)
                    for i in np.arange(len(mask))[mask]
                ],
                axis=0,
            )
            N = scipy.linalg.null_space(A)
            null.append(N)
            x0.append(P @ [x, y, z, 1] - P[:, 3])
            atom = Atom(atm)
            atm_dict = atom.neutron()
            br = atm_dict["coh_scatt_length_real"]
            bi = atm_dict["coh_scatt_length_img"]
            b = br + bi * 1j
            bs.append(b)

            T = np.zeros((len(self.S[index]), 6, 6))
            for m, R in enumerate(self.S[index, :, :3]):
                for a, (i, j) in enumerate(index_map):
                    for b, (k, l) in enumerate(index_map):
                        Tijkl = R[i, k] * R[j, l]
                        if k != l:
                            Tijkl += R[i, l] * R[j, k]
                        T[m, a, b] = Tijkl
            transforms_disp.append(T)
            T = np.zeros((len(self.S[mask]), 6, 6))
            for m, R in enumerate(self.S[mask, :, :3]):
                for a, (i, j) in enumerate(index_map):
                    for b, (k, l) in enumerate(index_map):
                        Tijkl = R[i, k] * R[j, l]
                        if k != l:
                            Tijkl += R[i, l] * R[j, k]
                        T[m, a, b] = Tijkl
            A = np.concatenate(
                [T[i] - np.eye(6) for i in range(len(T))], axis=0
            )
            N = scipy.linalg.null_space(A)
            null_disp.append(N)

        self.null = null
        self.x0 = x0

        self.transforms = transforms
        self.atms = np.array(atms)
        self.js = np.array(js)
        self.bs = np.array(bs)

        self.null_disp = null_disp
        self.transforms_disp = transforms_disp

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
                    var.format("u{}".format(j)),
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
                        "{}*{}".format(var.format("u{}".format(j)), N[0, j])
                        for j in range(n)
                        if not np.isclose(N[0, j], 0)
                    ]
                )
                y_const = "+".join(
                    ["{}".format(x0[1])]
                    + [
                        "{}*{}".format(var.format("u{}".format(j)), N[1, j])
                        for j in range(n)
                        if not np.isclose(N[1, j], 0)
                    ]
                )
                z_const = "+".join(
                    ["{}".format(x0[2])]
                    + [
                        "{}*{}".format(var.format("u{}".format(j)), N[2, j])
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
                    var.format("b{}".format(j)),
                    value=0,
                    min=-np.inf,
                    max=np.inf,
                )

            if n > 0:
                beta11_const = "+".join(
                    [
                        "{}*{}".format(var.format("b{}".format(j)), N[0, j])
                        for j in range(n)
                        if not np.isclose(N[0, j], 0)
                    ]
                )
                beta22_const = "+".join(
                    [
                        "{}*{}".format(var.format("b{}".format(j)), N[1, j])
                        for j in range(n)
                        if not np.isclose(N[1, j], 0)
                    ]
                )
                beta33_const = "+".join(
                    [
                        "{}*{}".format(var.format("b{}".format(j)), N[2, j])
                        for j in range(n)
                        if not np.isclose(N[2, j], 0)
                    ]
                )
                beta23_const = "+".join(
                    [
                        "{}*{}".format(var.format("b{}".format(j)), N[3, j])
                        for j in range(n)
                        if not np.isclose(N[3, j], 0)
                    ]
                )
                beta13_const = "+".join(
                    [
                        "{}*{}".format(var.format("b{}".format(j)), N[4, j])
                        for j in range(n)
                        if not np.isclose(N[4, j], 0)
                    ]
                )
                beta12_const = "+".join(
                    [
                        "{}*{}".format(var.format("b{}".format(j)), N[5, j])
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
        self.add_detector_run_scale_parameters(params)

        F2s = self.calculate_structure_factors(params)
        scale = self.calculate_scale_factor(F2s)

        params.add("scale", value=scale, min=0, max=np.inf)

        self.params = params

    def update_sample(self, parameters):
        if type(parameters) is list:
            self.thickness = parameters[0] / 10
            self.width_ratio = parameters[1] / parameters[0]
            self.height_ratio = parameters[2] / parameters[0]
        else:
            self.thickness = parameters

    def add_absorption_extinction_parameters(self, params):
        params.add("param", value=1, min=np.finfo(float).eps, max=np.inf)

        angles = ["alpha", "beta", "gamma"]

        for i in range(3):
            params.add(
                "coeff_{}".format(angles[i]),
                value=0.0,
                min=-180,
                max=180,
            )

        params.add(
            "coeff_thickness",
            value=self.thickness,
            min=0.0001,
            max=5.0,
        )
        params.add(
            "coeff_width",
            value=self.width_ratio,
            min=0.2,
            max=5,
        )
        params.add(
            "coeff_height",
            value=self.height_ratio,
            min=0.2,
            max=5,
        )

        self.params = params

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

    def extract_parameters(self, params):
        sites = []

        for i, site in enumerate(self.sites):
            name, x, y, z, occ, Uiso = site
            var = "{}{}_".format(name, i) + "{}"

            x = params[var.format("x")].value
            y = params[var.format("y")].value
            z = params[var.format("z")].value
            occ = params[var.format("occ")].value
            Uiso = 0

            sites.append([name, x, y, z, occ, Uiso])

        param = params["param"].value
        scale = params["scale"].value

        names = ["alpha", "beta", "gamma", "thickness", "width", "height"]

        coeffs = np.array(
            [params["coeff_{}".format(names[i])].value for i in range(6)]
        )

        dets = np.array(
            [params["det_{}".format(i)].value for i in range(len(self.banks))]
        )

        runs = np.array(
            [params["run_{}".format(i)].value for i in range(len(self.runs))]
        )

        return sites, param, scale, coeffs, dets, runs

    def load_hkls(self, filename):
        LoadNexus(Filename=filename, OutputWorkspace="peaks")

        self.extract_info()

    def extract_info(self):
        self.UB = mtd["peaks"].sample().getOrientedLattice().getUB()

        lamdas, two_thetas, intensity, sigma = [], [], [], []
        h, k, l, ri_hat, sf_hat, bank, run = [], [], [], [], [], [], []

        coverage_two_theta, coverage_az_phi = [], []

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

        scale_factor = 10000 / np.nanpercentile(intensity, 99)

        unique = ReflectionGenerator(self.crystal_structure)
        unique = unique.getHKLsUsingFilter(
            d_min, float("inf"), ReflectionConditionFilter.Centering
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
            bank, return_inverse=True, axis=0
        )

        self.runs, self.angles = np.unique(run, return_inverse=True, axis=0)

        self.I_obs = np.array(intensity)[mask] * scale_factor
        self.sig = np.array(sigma)[mask] * scale_factor

        self.lamda = np.array(lamdas)[mask]
        self.two_theta = np.array(two_thetas)[mask]
        self.Tbar = np.zeros_like(lamdas)[mask]

        self.ri_hat = np.array(ri_hat)[mask]
        self.sf_hat = np.array(sf_hat)[mask]

        self.two_theta_coverage = np.asarray(coverage_two_theta, float)
        self.az_phi_coverage = np.asarray(coverage_az_phi, float)

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
        Fs = 0

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

        return (Fs * Fs.conj())[self.inverse].real

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
        elif self.model.lower().startswith("type ii"):
            xs = self.extinction_xs(param, F2, two_theta, lamda, Tbar, V)
            ys = 1 / (1 + self.c1 * xs**self.c2)
            return ys
        else:  # shelx
            xx = self.extinction_xx(param, F2, two_theta, lamda, V)
            yx = 1 / np.sqrt(1 + xx)
            return yx

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

        all_params = self.extract_parameters(params)

        sites, param, scale, coeffs, dets, runs = all_params

        F2s = self.calculate_structure_factors(params)

        c = self.detector_bank_scale_factors(dets)
        k = self.run_angle_scale_factors(runs)

        I_calc = F2s * self.y * c * k

        return (scale * I_calc - self.I_obs) / self.sig

    def objective_correction(self, params):
        """
        Objective for absorption/extinction refinement.

        Structure and scale are held fixed; F2s are precomputed for the
        current structural model.
        """

        all_params = self.extract_parameters(params)

        sites, param, scale, coeffs, dets, runs = all_params

        y_abs = self.absorption_correction(coeffs)
        y_ext = self.extinction_correction(param, self.F2s)
        y = y_abs * y_ext

        c = self.detector_bank_scale_factors(dets)
        k = self.run_angle_scale_factors(runs)

        I_calc = self.F2s * y * c * k

        wr = (scale * I_calc - self.I_obs) / self.sig

        return wr

    def refine(
        self,
        report=True,
        cutoff=15,
        n_iter=3,
        abs_corr=True,
        det_corr=False,
        run_corr=False,
        ext_model="shelx",
    ):
        self.initialize_correction(ext_model)

        fixed = []
        for name, par in self.params.items():
            if par.vary == False:
                fixed.append(name)

        for i in range(n_iter):
            self.F2s = self.calculate_structure_factors(self.params)
            _, param, _, coeffs, _, _ = self.extract_parameters(self.params)
            y_abs = self.absorption_correction(coeffs)
            y_ext = self.extinction_correction(param, self.F2s)
            self.y = y_abs * y_ext

            # --- Stage 1: structure + scale, fixed absorption/extinction
            params = self.params.copy()
            for name, par in params.items():
                if name.startswith("coeff_") or name == "param":
                    par.vary = False
                elif name not in fixed:
                    par.vary = True

            out = Minimizer(
                self.objective_structure,
                params,
                nan_policy="omit",
                reduce_fcn=None,
            )

            result = out.minimize(method="leastsq", max_nfev=100)

            if report:
                print(
                    f"Iteration {i + 1}/{n_iter} - "
                    "Stage 1 (structure + scale)"
                )
                self.report(result, det_corr, run_corr)

            self.params = result.params

            self.F2s = self.calculate_structure_factors(self.params)

            self.params = result.params

            self.calculate_statistics(cutoff)

            # --- Stage 2: absorption only
            params = self.params.copy()
            for name, par in params.items():
                if name.startswith("coeff_"):
                    par.vary = abs_corr
                else:
                    par.vary = False

            out = Minimizer(
                self.objective_correction,
                params,
                nan_policy="omit",
                reduce_fcn=None,
            )

            result = out.minimize(method="leastsq", max_nfev=100)

            if report:
                print(f"Iteration {i + 1}/{n_iter} - " "Stage 2 (absorption)")
                self.report(result, det_corr, run_corr)

            self.params = result.params

            # --- Stage 3: extinction only
            params = self.params.copy()
            for name, par in params.items():
                if name == "param":
                    par.vary = True
                else:
                    par.vary = False

            out = Minimizer(
                self.objective_correction,
                params,
                nan_policy="omit",
                reduce_fcn=None,
            )

            result = out.minimize(method="leastsq", max_nfev=100)

            if report:
                print(f"Iteration {i + 1}/{n_iter} - " "Stage 3 (extinction)")
                self.report(result, det_corr, run_corr)

            self.params = result.params

            # --- Stage 4: calibration only
            params = self.params.copy()
            for name, par in params.items():
                if name.startswith("det_"):
                    par.vary = det_corr
                else:
                    par.vary = False

            out = Minimizer(
                self.objective_correction,
                params,
                nan_policy="omit",
                reduce_fcn=None,
            )

            result = out.minimize(method="leastsq", max_nfev=100)

            if report:
                print(f"Iteration {i + 1}/{n_iter} - " "Stage 4 (calibration)")
                self.report(result, det_corr, run_corr)

            self.params = result.params

            # --- Stage 5: orientation only
            params = self.params.copy()
            for name, par in params.items():
                if name.startswith("run_"):
                    par.vary = run_corr
                else:
                    par.vary = False

            out = Minimizer(
                self.objective_correction,
                params,
                nan_policy="omit",
                reduce_fcn=None,
            )

            result = out.minimize(method="leastsq", max_nfev=100)

            if report:
                print(f"Iteration {i + 1}/{n_iter} - " "Stage 5 (orientation)")
                self.report(result, det_corr, run_corr)

            self.params = result.params

            self.calculate_statistics(cutoff)

            self.plot_result()

            self.plot_sample_shape()

            *_, dets, runs = self.extract_parameters(self.params)

            self.save_detector_scales(dets)

        self.save_corrected_peaks()

    def report(self, result, det_corr, run_corr):
        params = result.params
        print("χ² = {:.2f}".format(result.chisqr))
        print("χ²/dof = {:.2f}\n".format(result.redchi))

        all_params = self.extract_parameters(result.params)

        sites, param, scale, coeffs, dets, runs = all_params

        print("scale = {:1.4e}".format(scale))
        print("ext = {:6.4f}".format(param))
        print("abs : {:6.1f} {:6.1f} {:6.1f}".format(*coeffs[:3]))
        print("    : {:6.4f} {:6.4f} {:6.4f}".format(*coeffs[3:]))
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

        if run_corr:
            for i in range(len(self.runs)):
                print("#{:} | {:6.4f}".format(self.runs[i], runs[i]))
            print("")

        if det_corr:
            for i in range(len(self.banks)):
                print("#{:} | {:6.4f}".format(self.banks[i], dets[i]))
            print("")

    def calculate_statistics(self, cutoff):
        all_params = self.extract_parameters(self.params)

        (
            self.sites,
            self.param,
            self.scale,
            self.coeffs,
            dets,
            runs,
        ) = all_params

        F2s = self.calculate_structure_factors(self.params)

        y_abs = self.absorption_correction(self.coeffs)
        y_ext = self.extinction_correction(self.param, F2s)

        c = self.detector_bank_scale_factors(dets)
        k = self.run_angle_scale_factors(runs)

        y = y_abs * y_ext

        I_calc = F2s * y * c * k

        scale = self.calculate_scale_factor(I_calc)

        self.I_calc = scale * I_calc

        F_obs = np.sqrt(self.I_obs / scale)
        F_calc = np.sqrt(self.I_calc / scale)
        F_sig = 0.5 * self.sig / np.sqrt(self.I_obs * scale)

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
        ax.set_aspect(1)
        ax.minorticks_on()
        ax.set_xlim(*F_lim)
        ax.set_ylim(*F_lim)
        ax.set_title(r"{} | $R = {:.2f}\%$".format(chemical_formula, R))
        ax.set_xlabel(r"$|F|_\mathrm{calc}$")
        ax.set_ylabel(r"$|F|_\mathrm{obs}$")
        fig.savefig(output + "_ref.pdf", bbox_inches="tight")

    def save_corrected_peaks(self):
        """
        Apply absorption corrections to all peaks and save to new file.

        """
        output = os.path.splitext(self.filename)[0] + "_corr.nxs"

        LoadNexus(Filename=self.filename, OutputWorkspace="peaks_corr")

        all_params = self.extract_parameters(self.params)

        _, param, scale, coeffs, dets, runs = all_params

        lamdas = []
        two_thetas = []
        ri_hat = []
        sf_hat = []

        for peak in mtd["peaks_corr"]:
            lamda = peak.getWavelength()
            two_theta = peak.getScattering()
            ri = peak.getSourceDirectionSampleFrame()
            sf = peak.getDetectorDirectionSampleFrame()

            lamdas.append(lamda)
            two_thetas.append(two_theta)
            ri_hat.append(ri)
            sf_hat.append(sf)

        material = self.material
        n = material.numberDensityEffective
        sigma_tot = material.totalScatterXSection()
        sigma_abs = [material.absorbXSection(lamda) for lamda in lamdas]
        mu = n * (sigma_tot + np.array(sigma_abs))

        self.mu = mu
        self.ri_hat = np.array(ri_hat)
        self.sf_hat = np.array(sf_hat)

        y_abs = self.absorption_correction(coeffs)

        Tbar = self.Tbar.copy()

        for idx, peak in enumerate(mtd["peaks_corr"]):
            I = peak.getIntensity()
            sig = peak.getSigmaIntensity()

            I_corr = I / y_abs[idx]
            sig_corr = sig / y_abs[idx]

            peak.setIntensity(I_corr)
            peak.setSigmaIntensity(sig_corr)
            peak.setAbsorptionWeightedPathLength(Tbar[idx])

        SaveNexus(InputWorkspace="peaks_corr", Filename=output)

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
        alpha, beta, gamma, scale, ratio_1, ratio_2 = coeffs

        thickness = scale
        width = scale * ratio_1
        height = scale * ratio_2

        return alpha, beta, gamma, thickness, width, height

    def calculate_ellipsoid_surface(self, coeffs):
        params = self.ellipsoid_parameters(coeffs)
        alpha, beta, gamma, thickness, width, height = params

        R = Rotation.from_euler(
            "ZYX", [gamma, beta, alpha], degrees=True
        ).as_matrix()

        D = np.diag([1 / width**2, 1 / height**2, 1 / thickness**2]) / 4

        return D, R, R @ D @ R.T

    def absorption_correction(self, coeffs):
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

    def prepare_absorption_table(self, N=1000, seed=42, beta=3):
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

    def initialize_correction(self, model="type II"):
        material = self.material
        n = material.numberDensityEffective
        sigma_tot = material.totalScatterXSection()
        sigma_abs = [material.absorbXSection(lamda) for lamda in self.lamda]
        self.mu = n * (sigma_tot + np.array(sigma_abs))
        self.model = model.lower()
        self.c1 = self.f1[self.model](self.two_theta)
        self.c2 = self.f2[self.model](self.two_theta)
        self.prepare_absorption_table()

    def _exit_lengths_for_directions(self, Q, yQ, cquad, dirs):
        a = np.einsum("mi,ij,mj->m", dirs, Q, dirs)

        b = 2 * (yQ @ dirs.T)

        disc = b * b - 4 * (cquad[:, None] * a[None, :])
        disc = np.maximum(disc, 0.0)

        t = (-b + np.sqrt(disc)) / (2 * a[None, :])
        return t

    def _absorption_factors(self, mu, Q, yQ, cquad, n_in_rev, n_out_rev):
        P = n_in_rev.shape[0]

        dirs = np.vstack([n_in_rev, n_out_rev])
        t = self._exit_lengths_for_directions(Q, yQ, cquad, dirs)

        t1 = t[:, :P]
        t2 = t[:, P:]
        t_total = t1 + t2

        w = np.exp(-t_total * mu[None, :])
        A = w.mean(axis=0)

        Tbar = (w * t_total).mean(axis=0) / A

        return A, Tbar


# ---

# if __name__ == "__main__":
# filename = "/SNS/CORELLI/IPTS-31429/shared/Attocube_test/normalization/garnet_withattocube_integration/garnet_withattocube_Cubic_I_d(min)=0.70_r(max)=0.20.nxs"
# filename = "/SNS/CORELLI/IPTS-31429/shared/kkl/garnet_4/garnet_4_integration/garnet_4_Cubic_I_d(min)=0.70_r(max)=0.20.nxs"
# filename = "/SNS/MANDI/IPTS-34720/shared/2025B/garnet_2025_3mm_cal_integration/garnet_2025_3mm_cal_Cubic_I_d(min)=1.00_r(max)=0.10.nxs"

# cell = [11.9386, 11.9386, 11.9386, 90, 90, 90]
# space_group = "I a -3 d"
# sites = [
#     ["Yb", 0.125, 0.0, 0.25, 1, 0.0023],
#     ["Al", 0.0, 0.0, 0.0, 1, 0.0023],
#     ["Al", 0.375, 0.0, 0.25, 1, 0.0023],
#     ["O", -0.03, 0.05, 0.149, 1, 0.0023],
# ]

# filename = "/SNS/CORELLI/IPTS-36263/shared/integration/EuAgAs_4K_integration/EuAgAs_4K_Hexagonal_P_(0.0,0.0,0.5)_d(min)=0.70_r(max)=0.20.nxs"

# cell = [4.516, 4.516, 8.107, 90, 90, 120]
# space_group = "P 63/m m c"
# sites = [
#     ["Eu", 0.0, 0.0, 0.0, 1.0, 0.0023],
#     ["Ag", 0.33333333, 0.6666667, 0.75, 1.0, 0.0023],
#     ["As", 0.33333333, 0.66666667, 0.25, 1.0, 0.0023],
# ]

# filename = "/SNS/TOPAZ/IPTS-31856/shared/2025B_Si_cal/Si_AG_cal_integration/Si_AG_cal_Cubic_F_d(min)=0.50_r(max)=0.20.nxs"
# cell = [5.431, 5.431, 5.431, 90, 90, 90]
# space_group = "F d -3 m"
# sites = [["Si", 0.0, 0.0, 0.0, 1.0, 0.0023]]

# nuclear = NuclearStructureRefinement(cell, space_group, sites, filename)
# nuclear.refine(n_iter=50, abs_corr=True, det_corr=True, run_corr=True)
# nuclear.plot_result()
# nuclear.plot_sample_shape()
# nuclear.save_corrected_peaks()
