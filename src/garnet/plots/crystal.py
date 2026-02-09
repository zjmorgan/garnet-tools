import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from mantid.geometry import CrystalStructure

from garnet.plots.colors import atm_colors


class CrystalStructurePlot:
    def __init__(self, sites, cell, space_group=None):
        self.sites = list(sites)
        self.cell = list(cell)
        self.space_group = space_group

        self.matrices()

    def params(self, a, b, c, alpha, beta, gamma):
        return np.array(
            [
                a**2,
                a * b * np.cos(np.deg2rad(gamma)),
                a * c * np.cos(np.deg2rad(beta)),
                b**2,
                b * c * np.cos(np.deg2rad(alpha)),
                c**2,
            ]
        )

    def matrices(self):
        i, j = np.triu_indices(3)

        g = self.params(*self.cell)
        G = np.zeros((3, 3))
        G[i, j] = g
        G[j, i] = g
        G_ = np.linalg.inv(G)

        self.A = scipy.linalg.cholesky(G, lower=False)
        self.B = scipy.linalg.cholesky(G_, lower=False)

        self.L = np.diag(self.cell[:3])
        self.C = np.dot(self.A, np.linalg.inv(self.L))

        R = np.dot(np.linalg.inv(self.A).T, np.linalg.inv(self.B))

    def wrap(self, val):
        val = np.array(val)
        mask = val < 0
        val[mask] += 1
        mask = val >= 1
        val[mask] -= 1
        val[np.isclose(val, 0)] = 0
        return val

    def get_crystal_structure(self):
        cell_params = " ".join(6 * ["{}"]).format(*self.cell)
        atom_sites = ";".join(
            [" ".join(6 * ["{}"]).format(*site) for site in self.sites]
        )
        crystal_structure = CrystalStructure(
            cell_params, self.space_group, atom_sites
        )

        return crystal_structure

    def plot(self, filename):
        cs = self.get_crystal_structure()
        sg = cs.getSpaceGroup()

        x, y, z, atm = [], [], [], []

        tol = 1e-8
        for j, site in enumerate(self.sites):
            _atm, *vals = site
            _x, _y, _z, _occ, _U = np.array(vals).astype(float).tolist()
            for pos in sg.getEquivalentPositions([_x, _y, _z]):
                xp, yp, zp = self.wrap(np.array(pos))
                offs_x = (0, 1) if np.isclose(xp, 0.0) else (0,)
                offs_y = (0, 1) if np.isclose(yp, 0.0) else (0,)
                offs_z = (0, 1) if np.isclose(zp, 0.0) else (0,)
                for ox in offs_x:
                    for oy in offs_y:
                        for oz in offs_z:
                            atm.append(_atm)
                            x.append(xp + ox)
                            y.append(yp + oy)
                            z.append(zp + oz)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        atm = np.array(atm)

        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection="3d")

        colors = [atm_colors[a] for a in atm]

        rx, ry, rz = np.einsum("ij,jk->ik", self.A, [x, y, z])

        ax3d.scatter(
            rx,
            ry,
            rz,
            s=50,
            color=colors,
        )

        O = np.array([0, 0, 0])

        ar, br, cr = self.A[:, 0], self.A[:, 1], self.A[:, 2]

        corners = np.array(
            [
                O,
                O + ar,
                O + br,
                O + cr,
                O + ar + br,
                O + ar + cr,
                O + br + cr,
                O + ar + br + cr,
            ]
        )

        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]

        for i, j in edges:
            p, q = corners[i], corners[j]
            ax3d.plot(
                [p[0], q[0]],
                [p[1], q[1]],
                [p[2], q[2]],
                color="k",
                lw=1,
                alpha=1,
            )

        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_aspect("equal")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        plt.savefig(filename, bbox_inches="tight", dpi=300)
