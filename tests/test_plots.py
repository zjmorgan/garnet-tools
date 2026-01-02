import os
import numpy as np

import scipy.linalg
import scipy.spatial

from garnet.reduction.integration import PeakEllipsoid
from garnet.plots.peaks import PeakPlot
from garnet.plots.volume import SlicePlot

filepath = os.path.dirname(os.path.abspath(__file__))


def test_slice_plot():
    np.random.seed(13)

    U = scipy.spatial.transform.Rotation.random().as_matrix()

    a, b, c = 5, 5, 7
    alpha, beta, gamma = np.deg2rad([90, 90, 120])

    G = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c**2],
        ]
    )

    B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    UB = U @ B

    W = np.eye(3)

    x0 = np.linspace(-5, 10, 31)
    x1 = np.linspace(-3, 9, 25)
    x2 = np.linspace(-7, 8, 61)

    axes = x0, x1, x2

    X0, X1, X2 = np.meshgrid(*axes, indexing="ij")

    signal = np.ones_like(X0)

    signal[(X0 % 1 == 0) & (X1 % 1 == 0) & (X2 % 1 == 0)] = 1000

    plot = SlicePlot(UB, W)
    plot.calculate_transforms(axes, ["h", "k", "l"], [0, 0, 1])
    plot.make_slice(signal, 0)
    plot.save_plot(os.path.join(filepath, "slice2.png"))

    plot = SlicePlot(UB, W)
    plot.calculate_transforms(axes, ["h", "k", "l"], [0, 1, 0])
    plot.make_slice(signal, 0)
    plot.save_plot(os.path.join(filepath, "slice1.png"))

    plot = SlicePlot(UB, W)
    plot.calculate_transforms(axes, ["h", "k", "l"], [1, 0, 0])
    plot.make_slice(signal, 0)
    plot.save_plot(os.path.join(filepath, "slice0.png"))


# def test_radius_plot():
#     r_cut = 0.25

#     A = 1.2
#     s = 0.1

#     r = np.linspace(0, r_cut, 51)

#     I = A * np.tanh((r / s) ** 3)

#     sphere = PeakSphere(r_cut)

#     radius = sphere.fit(r, I)

#     I_fit, *vals = sphere.best_fit(r)

#     plot = RadiusPlot(r, I, I_fit)

#     plot.add_sphere(radius, *vals)

#     plot.save_plot(os.path.join(filepath, "sphere.png"))


def test_init_peak_plot():
    plot = PeakPlot()

    file = os.path.join(filepath, "ellipsoid_init.png")

    plot.save_plot(file)

    assert os.path.exists(file)


def test_peak_plot():
    np.random.seed(13)

    nx, ny, nz = 21, 24, 31

    Qx_min, Qx_max = 1, 3
    Qy_min, Qy_max = -2.9, 3.1
    Qz_min, Qz_max = -2.2, 1.8

    Q0_x, Q0_y, Q0_z = 2.1, 0.2, -0.2

    sigma_x, sigma_y, sigma_z = 0.15, 0.25, 0.2
    rho_yz, rho_xz, rho_xy = 0.5, -0.1, -0.12

    a = 100
    b = 100.0
    c = 0.002

    sigma_yz = sigma_y * sigma_z
    sigma_xz = sigma_x * sigma_z
    sigma_xy = sigma_x * sigma_y

    cov = np.array(
        [
            [sigma_x**2, rho_xy * sigma_xy, rho_xz * sigma_xz],
            [rho_xy * sigma_xy, sigma_y**2, rho_yz * sigma_yz],
            [rho_xz * sigma_xz, rho_yz * sigma_yz, sigma_z**2],
        ]
    )

    Q0 = np.array([Q0_x, Q0_y, Q0_z])

    signal = np.random.multivariate_normal(Q0, cov, size=1000)

    counts, bins = np.histogramdd(
        signal,
        density=False,
        bins=[nx, ny, nz],
        range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)],
    )

    x_bin_edges, y_bin_edges, z_bin_edges = bins

    Qx = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])
    Qy = 0.5 * (y_bin_edges[1:] + y_bin_edges[:-1])
    Qz = 0.5 * (z_bin_edges[1:] + z_bin_edges[:-1])

    print(
        counts.sum()
        * np.diff(Qx).mean()
        * np.diff(Qy).mean()
        * np.diff(Qz).mean()
    )

    m = 1

    i, j, k = np.array(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ).reshape(3, -1)

    i_min = np.clip(i - m, 0, nx)
    i_max = np.clip(i + m + 1, 0, nx)
    j_min = np.clip(j - m, 0, ny)
    j_max = np.clip(j + m + 1, 0, ny)
    k_min = np.clip(k - m, 0, nz)
    k_max = np.clip(k + m + 1, 0, nz)

    n = nx * ny * nz

    ic = np.random.randint(i_min, i_max, size=n).ravel()
    jc = np.random.randint(j_min, j_max, size=n).ravel()
    kc = np.random.randint(k_min, k_max, size=n).ravel()

    temp = np.copy(counts)
    counts[i, j, k] = counts[ic, jc, kc]
    counts[ic, jc, kc] = temp[i, j, k]

    counts += b + a * (2 * np.random.random(counts.shape) - 1)

    norm = np.full_like(counts, 5)

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing="ij")

    ellipsoid = PeakEllipsoid()

    mask = counts > 0

    ellipsoid.fit(Qx, Qy, Qz, counts, norm, counts, norm, mask, mask, 0.1, 2)

    c, S, *best_fit = ellipsoid.best_fit

    hkl = [1, 2, 3]
    d = 3.14

    wavelength = 3.2887

    angles = 60, 0
    goniometer = [0, 0, 0]

    plot = PeakPlot()

    norm_params = Qx, Qy, Qz, counts, norm, mask, mask, c, S

    I, sigma = ellipsoid.integrate(*norm_params)

    plot.add_ellipsoid_fit(best_fit)

    plot.add_profile_fit(ellipsoid.best_prof)

    plot.add_projection_fit(ellipsoid.best_proj)

    plot.add_ellipsoid(c, S)

    plot.add_peak_info(hkl, d, wavelength, angles, goniometer)

    plot.add_peak_stats(
        ellipsoid.redchi2, ellipsoid.intensity, ellipsoid.sigma
    )

    plot.add_data_norm_fit(*ellipsoid.data_norm_fit)

    plot.update_envelope(*ellipsoid.peak_background_mask)

    plot.add_integral_fit(ellipsoid.integral)

    file = os.path.join(filepath, "ellipsoid.png")

    # if os.path.exists(file):
    #     os.remove(file)

    plot.save_plot(file)

    assert os.path.exists(file)


test_peak_plot()
