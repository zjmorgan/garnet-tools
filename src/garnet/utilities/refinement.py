import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import scipy.linalg
import scipy.interpolate

from lmfit import Minimizer, Parameters, fit_report

from mantid.simpleapi import LoadNexus, mtd
from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
)
from mantid.kernel import Atom, MaterialBuilder, V3D


class NuclearStructureRefinement:
    def __init__(self, cell, space_group, sites, filename, l_max=8):
        self.sites = []
        for site in sites:
            atm, x, y, z, occ, U = site
            x, y, z = self._wrap([x, y, z])
            self.sites.append([atm, x, y, z, occ, U])

        self.cell = cell
        self.space_group = space_group

        self.l_max = l_max

        self.initialize_crystal_structure(self.sites)
        self.initialize_material()
        self.generate_symmetry_transforms()
        self.initialize_unit_cell_atoms()

        self.load_hkls(filename)

        self.models = ["type I gaussian", "type I lorentzian", "type II"]

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

        self.S = np.zeros((len(symops), 3, 4))

        self.S[..., 0] = [
            self._eval(op, 1, 0, 0) - self._eval(op, 0, 0, 0) for op in symops
        ]
        self.S[..., 1] = [
            self._eval(op, 0, 1, 0) - self._eval(op, 0, 0, 0) for op in symops
        ]
        self.S[..., 2] = [
            self._eval(op, 0, 0, 1) - self._eval(op, 0, 0, 0) for op in symops
        ]
        self.S[..., 3] = [self._eval(op, 0, 0, 0) for op in symops]

    def generate_parameters(self):
        params = Parameters()

        visits = {}

        for i, (site, N, x0) in enumerate(zip(self.sites, self.null, self.x0)):
            name, x, y, z, occ, Uiso = site
            variable = "{}{}_".format(name, i) + "{}"

            key = tuple(np.round([x, y, z], 6))
            if visits.get(key) is None:
                visits[key] = variable

            N_inv = np.linalg.pinv(N)

            u = N_inv @ (np.array([x, y, z]) - x0)

            for j, val in enumerate(u):
                params.add(
                    variable.format("u{}".format(j)),
                    value=val,
                    min=-10,
                    max=10,
                )

            params.add(variable.format("x"), value=x, min=0, max=1)
            params.add(variable.format("y"), value=y, min=0, max=1)
            params.add(variable.format("z"), value=z, min=0, max=1)

            n = len(u)
            if n > 0:
                x_const = "+".join(
                    ["{}".format(x0[0])]
                    + [
                        "{}*{}".format(
                            variable.format("u{}".format(j)), N[0, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[0, j], 0)
                    ]
                )
                y_const = "+".join(
                    ["{}".format(x0[1])]
                    + [
                        "{}*{}".format(
                            variable.format("u{}".format(j)), N[1, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[1, j], 0)
                    ]
                )
                z_const = "+".join(
                    ["{}".format(x0[2])]
                    + [
                        "{}*{}".format(
                            variable.format("u{}".format(j)), N[2, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[2, j], 0)
                    ]
                )
                if np.any(N[0, :]):
                    params[variable.format("x")].set(expr=x_const)
                else:
                    params[variable.format("x")].set(vary=False)
                if np.any(N[1, :]):
                    params[variable.format("y")].set(expr=y_const)
                else:
                    params[variable.format("y")].set(vary=False)
                if np.any(N[2, :]):
                    params[variable.format("z")].set(expr=z_const)
                else:
                    params[variable.format("z")].set(vary=False)
            else:
                params[variable.format("x")].set(vary=False)
                params[variable.format("y")].set(vary=False)
                params[variable.format("z")].set(vary=False)

            params.add(
                variable.format("occ"),
                value=occ,
                min=0,
                max=1,
                vary=not np.isclose(occ, 1),
            )

        for i, (site, N) in enumerate(zip(self.sites, self.null_disp)):
            name, x, y, z, occ, Uiso = site
            variable = "{}{}_".format(name, i) + "{}"

            params.add(
                variable.format("beta11"), value=0.0, min=-np.inf, max=np.inf
            )
            params.add(
                variable.format("beta22"), value=0.0, min=-np.inf, max=np.inf
            )
            params.add(
                variable.format("beta33"), value=0.0, min=-np.inf, max=np.inf
            )
            params.add(
                variable.format("beta23"), value=0.0, min=-np.inf, max=np.inf
            )
            params.add(
                variable.format("beta13"), value=0.0, min=-np.inf, max=np.inf
            )
            params.add(
                variable.format("beta12"), value=0.0, min=-np.inf, max=np.inf
            )

            n = N.shape[1]

            for j in range(n):
                params.add(
                    variable.format("b{}".format(j)),
                    value=0,
                    min=-np.inf,
                    max=np.inf,
                )

            if n > 0:
                beta11_const = "+".join(
                    [
                        "{}*{}".format(
                            variable.format("b{}".format(j)), N[0, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[0, j], 0)
                    ]
                )
                beta22_const = "+".join(
                    [
                        "{}*{}".format(
                            variable.format("b{}".format(j)), N[1, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[1, j], 0)
                    ]
                )
                beta33_const = "+".join(
                    [
                        "{}*{}".format(
                            variable.format("b{}".format(j)), N[2, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[2, j], 0)
                    ]
                )
                beta23_const = "+".join(
                    [
                        "{}*{}".format(
                            variable.format("b{}".format(j)), N[3, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[3, j], 0)
                    ]
                )
                beta13_const = "+".join(
                    [
                        "{}*{}".format(
                            variable.format("b{}".format(j)), N[4, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[4, j], 0)
                    ]
                )
                beta12_const = "+".join(
                    [
                        "{}*{}".format(
                            variable.format("b{}".format(j)), N[5, j]
                        )
                        for j in range(n)
                        if not np.isclose(N[5, j], 0)
                    ]
                )
                if np.any(N[0, :]):
                    params[variable.format("beta11")].set(expr=beta11_const)
                else:
                    params[variable.format("beta11")].set(vary=False)
                if np.any(N[1, :]):
                    params[variable.format("beta22")].set(expr=beta22_const)
                else:
                    params[variable.format("beta22")].set(vary=False)
                if np.any(N[2, :]):
                    params[variable.format("beta33")].set(expr=beta33_const)
                else:
                    params[variable.format("beta33")].set(vary=False)
                if np.any(N[3, :]):
                    params[variable.format("beta23")].set(expr=beta23_const)
                else:
                    params[variable.format("beta23")].set(vary=False)
                if np.any(N[4, :]):
                    params[variable.format("beta13")].set(expr=beta13_const)
                else:
                    params[variable.format("beta13")].set(vary=False)
                if np.any(N[5, :]):
                    params[variable.format("beta12")].set(expr=beta12_const)
                else:
                    params[variable.format("beta12")].set(vary=False)

        for i, site in enumerate(self.sites):
            name, x, y, z, occ, Uiso = site
            variable = "{}{}_".format(name, i) + "{}"

            key = tuple(np.round([x, y, z], 6))
            visited = visits.get(key)
            if visited != variable:
                params.add(variable.format("x"), expr=visited.format("x"))
                params.add(variable.format("y"), expr=visited.format("y"))
                params.add(variable.format("z"), expr=visited.format("z"))
                params.add(
                    variable.format("occ"), expr="1-" + visited.format("occ")
                )

                params.add(
                    variable.format("beta11"), expr=visited.format("beta11")
                )
                params.add(
                    variable.format("beta22"), expr=visited.format("beta22")
                )
                params.add(
                    variable.format("beta33"), expr=visited.format("beta33")
                )
                params.add(
                    variable.format("beta23"), expr=visited.format("beta23")
                )
                params.add(
                    variable.format("beta13"), expr=visited.format("beta13")
                )
                params.add(
                    variable.format("beta12"), expr=visited.format("beta12")
                )

        self.add_absorption_extinction_parameters(params)

        F2s = self.calculate_structure_factors(params)
        scale = self.calculate_scale_factor(F2s)

        params.add("scale", value=scale, min=0, max=np.inf)

        self.params = params

    def add_absorption_extinction_parameters(self, params):
        params.add("param", value=0.1, min=np.finfo(float).eps, max=np.inf)

        n_feat = self.fs.shape[1]
        for i in range(n_feat):
            params.add(
                "coeff_{}".format(i),
                value=0.0,
                min=-np.inf,
                max=np.inf,
            )

        self.params = params

    def extract_parameters(self, params):
        sites = []

        for i, site in enumerate(self.sites):
            name, x, y, z, occ, Uiso = site
            variable = "{}{}_".format(name, i) + "{}"

            x = params[variable.format("x")].value
            y = params[variable.format("y")].value
            z = params[variable.format("z")].value
            occ = params[variable.format("occ")].value
            Uiso = 0

            sites.append([name, x, y, z, occ, Uiso])

        param = params["param"].value
        scale = params["scale"].value

        coeffs = np.array(
            [
                params["coeff_{}".format(i)].value
                for i in range(self.fs.shape[1])
            ]
        )

        return sites, param, scale, coeffs

    def real_spherical_harmonic(self, l, m, theta, phi):
        x = np.cos(theta)
        am = abs(m)

        N = np.sqrt(
            (2 * l + 1)
            / (4 * np.pi)
            * scipy.special.factorial(l - am)
            / scipy.special.factorial(l + am)
        )

        P = scipy.special.lpmv(am, l, x)

        if m > 0:
            return np.sqrt(2) * N * P * np.cos(am * phi)
        elif m < 0:
            return np.sqrt(2) * N * P * np.sin(am * phi)
        else:
            return N * P

    def feature_vector(self, theta, phi, l_max):
        f = []
        for l in range(0, l_max + 1, 2):
            for m in range(-l, l + 1):
                f.append(self.real_spherical_harmonic(l, m, theta, phi))
        return np.asarray(f, float)

    def load_hkls(self, filename):
        LoadNexus(Filename=filename, OutputWorkspace="peaks")

        lamdas, two_thetas, intensity, sigma = [], [], [], []
        h, k, l, fs = [], [], [], []

        coverage_two_theta, coverage_az_phi = [], []

        d_min = np.inf
        for peak in mtd["peaks"]:
            hkl = peak.getIntHKL()
            mnp = peak.getIntMNP()
            lamda = peak.getWavelength()
            two_theta = peak.getScattering()
            d = peak.getDSpacing()
            if d < d_min:
                d_min = d

            ri = peak.getSourceDirectionSampleFrame()
            sf = peak.getDetectorDirectionSampleFrame()

            two_theta_ri = np.arccos(np.clip(ri[2], -1.0, 1.0))
            az_phi_ri = np.arctan2(ri[1], ri[0])

            two_theta_sf = np.arccos(np.clip(sf[2], -1.0, 1.0))
            az_phi_sf = np.arctan2(sf[1], sf[0])

            ri_vec = self.feature_vector(two_theta_ri, az_phi_ri, self.l_max)
            sf_vec = self.feature_vector(two_theta_sf, az_phi_sf, self.l_max)
            f = 0.5 * (ri_vec + sf_vec)

            I = peak.getIntensity()
            sig = peak.getSigmaIntensity()

            if mnp.norm2() == 0 and I > 3 * sig and np.isfinite(I):
                lamdas.append(lamda)
                two_thetas.append(two_theta)

                h.append(hkl[0])
                k.append(hkl[1])
                l.append(hkl[2])

                intensity.append(I)
                sigma.append(sig)

                fs.append(f)

                coverage_two_theta.extend([two_theta_ri, two_theta_sf])
                coverage_az_phi.extend([az_phi_ri, az_phi_sf])

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

        self.I_obs = np.array(intensity)[mask] * scale_factor
        self.sig = np.array(sigma)[mask] * scale_factor

        self.lamda = np.array(lamdas)[mask]
        self.two_theta = np.array(two_thetas)[mask]
        self.Tbar = np.zeros_like(lamdas)[mask]

        self.fs = np.array(fs)[mask]

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
            variable = "{}{}_".format(name, i) + "{}"

            x = params[variable.format("x")].value
            y = params[variable.format("y")].value
            z = params[variable.format("z")].value

            xyz = np.einsum("ijk,k->ij", transform, [x, y, z, 1])

            pf = np.exp(2j * np.pi * np.einsum("ij,kj->ik", xyz, self.equiv))

            beta11 = params[variable.format("beta11")].value
            beta22 = params[variable.format("beta22")].value
            beta33 = params[variable.format("beta33")].value
            beta23 = params[variable.format("beta23")].value
            beta13 = params[variable.format("beta13")].value
            beta12 = params[variable.format("beta12")].value

            beta = np.einsum(
                "ijk,k->ij",
                transform_disp,
                [beta11, beta22, beta33, beta23, beta13, beta12],
            )

            h2 = [
                self.equiv[:, 0] ** 2,
                self.equiv[:, 1] ** 2,
                self.equiv[:, 2] ** 2,
                2 * self.equiv[:, 1] * self.equiv[:, 2],
                2 * self.equiv[:, 0] * self.equiv[:, 2],
                2 * self.equiv[:, 0] * self.equiv[:, 1],
            ]

            T = np.clip(np.exp(-np.einsum("ij,jk->ik", beta, h2)), 0, 1)

            occ = params[variable.format("occ")].value

            Fs += np.sum(b * occ * pf * T, axis=0)

        return (Fs * Fs.conj())[self.inverse].real

    def plot_sample_shape(
        self,
        n_theta=40,
        n_phi=40,
        elev=20,
        azim=30,
        coverage_angle_deg=1.0,
    ):
        R = self.R
        coeffs = self.coeffs

        two_theta = np.linspace(0, np.pi, n_theta)
        az_phi = np.linspace(0, 2 * np.pi, n_phi)
        two_theta, az_phi = np.meshgrid(two_theta, az_phi)

        x = R * np.sin(two_theta) * np.cos(az_phi)
        y = R * np.sin(two_theta) * np.sin(az_phi)
        z = R * np.cos(two_theta)

        coeffs = np.asarray(coeffs, float)
        tt_flat = two_theta.ravel()
        az_flat = az_phi.ravel()
        feats = np.array(
            [
                self.feature_vector(tt, az, self.l_max)
                for tt, az in zip(tt_flat, az_flat)
            ]
        )
        A_ani = 1.0 + feats @ coeffs
        A_ani = np.clip(
            A_ani.reshape(two_theta.shape),
            np.finfo(float).eps,
            np.finfo(float).max,
        )

        material = self.material
        n = material.numberDensityEffective
        sigma_tot = material.totalScatterXSection()
        sigma_abs = material.absorbXSection(np.median(self.lamda))
        mu = n * (sigma_tot + sigma_abs)

        muR = mu * R / 10.0
        A_iso = self.spherical_absorption_correction(muR, two_theta)

        T_dir = A_iso / A_ani

        T_plot = T_dir.copy()

        two_theta_cov = getattr(self, "two_theta_coverage", None)
        az_phi_cov = getattr(self, "az_phi_coverage", None)

        if (
            two_theta_cov is not None
            and az_phi_cov is not None
            and two_theta_cov.size
        ):
            ux = x / R
            uy = y / R
            uz = z / R
            grid_dirs = np.stack([ux.ravel(), uy.ravel(), uz.ravel()], axis=1)

            cx = np.sin(two_theta_cov) * np.cos(az_phi_cov)
            cy = np.sin(two_theta_cov) * np.sin(az_phi_cov)
            cz = np.cos(two_theta_cov)
            cov_dirs = np.stack([cx, cy, cz], axis=1)

            cosang = cov_dirs @ grid_dirs.T
            max_cos = np.max(cosang, axis=0)

            min_cos = np.cos(np.deg2rad(coverage_angle_deg))
            covered = (max_cos >= min_cos).reshape(two_theta.shape)

            T_plot[~covered] = np.nan

        fig = plt.figure(layout="constrained")
        gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15)

        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 0])
        ax_yz = fig.add_subplot(gs[1, 1])

        if np.isfinite(T_plot).any():
            vmin = np.nanmin(T_plot)
            vmax = np.nanmax(T_plot)
        else:
            vmin = np.min(T_dir)
            vmax = np.max(T_dir)

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
        facecolors = cmap(norm(T_plot))
        ax3d.plot_surface(
            x,  # horizontal axis: x
            z,  # depth axis: z
            y,  # vertical axis: y
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
        )

        ax_xy.pcolormesh(
            x,
            y,
            T_plot,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        ax_xz.pcolormesh(
            x,
            z,
            T_plot,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        ax_yz.pcolormesh(
            z,
            y,
            T_plot,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(T_plot)
        fig.colorbar(
            mappable,
            ax=[ax3d, ax_xy, ax_xz, ax_yz],
            shrink=0.7,
            label=r"$A_\mathrm{iso}/A_\mathrm{ani}$",
        )

        lim = 1.1 * R
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_xlabel(r"$x$ [mm]")
        ax3d.set_ylabel(r"$z$ [mm]")
        ax3d.set_zlabel(r"$y$ [mm]")
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_aspect("equal")

        ax_xy.set_xlim(-lim, lim)
        ax_xy.set_ylim(-lim, lim)
        ax_xy.set_xlabel(r"$x$ [mm]")
        ax_xy.set_ylabel(r"$y$ [mm]")
        ax_xy.set_aspect("equal")

        ax_xz.set_xlim(-lim, lim)
        ax_xz.set_ylim(-lim, lim)
        ax_xz.set_xlabel(r"$x$ [mm]")
        ax_xz.set_ylabel(r"$z$ [mm]")
        ax_xz.set_aspect("equal")

        ax_yz.set_xlim(-lim, lim)
        ax_yz.set_ylim(-lim, lim)
        ax_yz.set_xlabel(r"$z$ [mm]")
        ax_yz.set_ylabel(r"$y$ [mm]")
        ax_yz.set_aspect("equal")

        ax3d.view_init(elev=elev, azim=azim)

        output = os.path.splitext(self.filename)[0]
        fig.savefig(output + "_sample_shape.pdf", bbox_inches="tight")

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

    def initialize_correction(self, R=0, model="type II"):
        material = self.material
        n = material.numberDensityEffective
        sigma_tot = material.totalScatterXSection()
        sigma_abs = [material.absorbXSection(lamda) for lamda in self.lamda]

        self.mu = n * (sigma_tot + np.array(sigma_abs))
        muR = self.mu * R / 10

        self.A_sph = self.spherical_absorption_correction(muR, self.two_theta)

        self.model = model

        self.c1 = self.f1[self.model](self.two_theta)
        self.c2 = self.f2[self.model](self.two_theta)

    def absorption_correction(self, coeffs):
        A_ani = np.clip(
            1 + self.fs @ coeffs, np.finfo(float).eps, np.finfo(float).max
        )
        self.T = self.A_sph / A_ani
        self.Tbar = -np.log(self.T) / self.mu

        return self.T

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
        a = 1e-5

        xs = a**2 / V**2 * lamda**3 * g / np.sin(two_theta) * Tbar * F2

        return xs

    def extinction_xp(self, r2, F2, lamda, V):
        a = 1e-5

        xp = a**2 / V**2 * lamda**2 * r2 * F2

        return xp

    def extinction_correction(self, param, F2):
        two_theta = self.two_theta
        lamda = self.lamda
        Tbar = self.Tbar

        V = self.V
        if self.model == "type II":
            xp = self.extinction_xp(param, F2, lamda, V)
            yp = 1 / (1 + self.c1 * xp**self.c2)
            return yp
        else:
            xs = self.extinction_xs(param, F2, two_theta, lamda, Tbar, V)
            ys = 1 / (1 + self.c1 * xs**self.c2)
            return ys

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
        """Objective for structure + scale refinement.

        This stage varies positional, occupancy and displacement
        parameters together with the overall scale, while using the
        current absorption/extinction parameters as fixed values.
        """

        sites, param, scale, coeffs = self.extract_parameters(params)

        F2s = self.calculate_structure_factors(params)

        I_calc = F2s * self.y

        return (scale * I_calc - self.I_obs) / self.sig

    def objective_correction(self, params):
        """Objective for absorption/extinction refinement.

        Structure and scale are held fixed; F2s are precomputed for the
        current structural model.
        """

        sites, param, scale, coeffs = self.extract_parameters(params)

        y_abs = self.absorption_correction(coeffs)
        y_ext = self.extinction_correction(param, self.F2s)
        y = y_abs * y_ext

        I_calc = self.F2s * y

        wr = (scale * I_calc - self.I_obs) / self.sig

        return np.concatenate((wr, self.T - 1))

    def refine(self, R=1, report=True, cutoff=10, n_iter=3):
        self.R = R

        self.initialize_correction(R)

        fixed = []
        for name, par in self.params.items():
            if par.vary == False:
                fixed.append(name)

        for i in range(n_iter):
            self.F2s = self.calculate_structure_factors(self.params)
            _, param, _, coeffs = self.extract_parameters(self.params)
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

            result = out.minimize(method="leastsq")

            if report:
                print(
                    f"Iteration {i + 1}/{n_iter} - "
                    "Stage 1 (structure + scale)"
                )
                print(fit_report(result))

            self.params = result.params

            self.F2s = self.calculate_structure_factors(self.params)

            # --- Stage 2: absorption/extinction only
            params = self.params.copy()
            for name, par in params.items():
                if name.startswith("coeff_") or name == "param":
                    par.vary = True
                else:
                    par.vary = False

            out = Minimizer(
                self.objective_correction,
                params,
                nan_policy="omit",
                reduce_fcn=None,
            )

            result = out.minimize(method="leastsq")

            if report:
                print(
                    f"Iteration {i + 1}/{n_iter} - "
                    "Stage 2 (absorption/extinction)"
                )
                print(fit_report(result))

            self.params = result.params

        params = self.extract_parameters(self.params)

        self.sites, self.param, self.scale, self.coeffs = params

        F2s = self.calculate_structure_factors(self.params)

        y_abs = self.absorption_correction(self.coeffs)
        y_ext = self.extinction_correction(self.param, F2s)

        y = y_abs * y_ext

        I_calc = F2s * y

        scale = self.calculate_scale_factor(I_calc)

        self.I_calc = scale * I_calc

        F_obs = np.sqrt(self.I_obs / scale)
        F_calc = np.sqrt(self.I_calc / scale)
        F_sig = 0.5 * self.sig / np.sqrt(self.I_obs * scale)

        # F2_obs = self.I_obs / (scale * y)
        # F2_sig = self.sig / (scale * y)
        # F2_calc = F2s.copy()

        mask = np.abs(F_calc - F_obs) < cutoff * F_sig

        w = np.bincount(self.inverse[mask], weights=1 / F_sig[mask] ** 2)

        F_obs = (
            np.bincount(
                self.inverse[mask], weights=F_obs[mask] / F_sig[mask] ** 2
            )
            / w
        )
        F_calc = (
            np.bincount(
                self.inverse[mask], weights=F_calc[mask] / F_sig[mask] ** 2
            )
            / w
        )
        F_sig = 1 / np.sqrt(w)

        # F_obs = np.sqrt(F2_obs)
        # F_calc = np.sqrt(F2_calc)
        # F_sig = 0.5 * F2_sig / np.sqrt(F2_obs)

        mask = np.abs(F_calc - F_obs) < cutoff * F_sig

        self.F_obs = F_obs[mask]
        self.F_calc = F_calc[mask]
        self.F_sig = F_sig[mask]

        self.R_ref = (
            100
            * np.nansum(np.abs(self.F_calc - self.F_obs))
            / np.nansum(self.F_obs)
        )

        print("R = {:.2f}%".format(self.R_ref))

    def plot_result(self):
        output = os.path.splitext(self.filename)[0]

        F_lim = [np.min(self.F_obs), np.max(self.F_obs)]

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
        R = self.R_ref
        ax.set_aspect(1)
        ax.minorticks_on()
        ax.set_xlim(*F_lim)
        ax.set_ylim(*F_lim)
        ax.set_title(r"{} | $R = {:.2f}\%$".format(chemical_formula, R))
        ax.set_xlabel(r"$|F|_\mathrm{calc}$")
        ax.set_ylabel(r"$|F|_\mathrm{obs}$")
        fig.savefig(output + "_ref.pdf", bbox_inches="tight")


# ---

# filename = "/SNS/CORELLI/IPTS-31429/shared/Attocube_test/normalization/garnet_withattocube_integration/garnet_withattocube_Cubic_I_d(min)=0.70_r(max)=0.20.nxs"
filename = "/SNS/CORELLI/IPTS-31429/shared/kkl/garnet_4/garnet_4_integration/garnet_4_Cubic_I_d(min)=0.70_r(max)=0.20.nxs"

cell = [11.9386, 11.9386, 11.9386, 90, 90, 90]
space_group = "I a -3 d"
sites = [
    ["Yb", 0.125, 0.0, 0.25, 1, 0.0023],
    ["Al", 0.0, 0.0, 0.0, 1, 0.0023],
    ["Al", 0.375, 0.0, 0.25, 1, 0.0023],
    ["O", -0.03, 0.05, 0.149, 1, 0.0023],
]


# filename = "/SNS/CORELLI/IPTS-36263/shared/integration/EuAgAs_4K_integration/EuAgAs_4K_Hexagonal_P_(0.0,0.0,0.5)_d(min)=0.70_r(max)=0.20.nxs"

# cell = [4.516, 4.516, 8.107, 90, 90, 120]
# space_group = "P 63/m m c"
# sites = [
#     ["Eu", 0.0, 0.0, 0.0, 1.0, 0.0023],
#     ["Ag", 0.33333333, 0.6666667, 0.75, 1.0, 0.0023],
#     ["As", 0.33333333, 0.66666667, 0.25, 1.0, 0.0023],
# ]

nuclear = NuclearStructureRefinement(cell, space_group, sites, filename)
nuclear.refine(R=0.1, n_iter=1)
nuclear.plot_result()
nuclear.plot_sample_shape()
