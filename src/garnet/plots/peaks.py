import numpy as np

import matplotlib

matplotlib.use("agg")

import matplotlib.style

matplotlib.style.use("fast")

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import scipy.ndimage
import skimage.measure

from garnet.plots.base import BasePlot


class PeakCentroidPlot(BasePlot):
    def __init__(self, c0, c1, c2, Q, r_cut, d_min):
        super(PeakCentroidPlot, self).__init__()

        plt.close("all")

        self.fig, self.ax = plt.subplots(
            1,
            3,
            figsize=(6.4 * 3, 4.8),
            layout="constrained",
            sharex=True,
            sharey=True,
        )

        self.ax[0].plot(c0, Q, ".", color="C0")
        self.ax[1].plot(c1, Q, ".", color="C1")
        self.ax[2].plot(c2, Q, ".", color="C2")

        self.ax[0].minorticks_on()
        self.ax[2].minorticks_on()
        self.ax[1].minorticks_on()

        self.ax[0].set_xlabel("$\Delta |Q|$ [$\AA^{-1}$]")
        self.ax[1].set_xlabel("$\Delta Q_1$ [$\AA^{-1}$]")
        self.ax[2].set_xlabel("$\Delta Q_2$ [$\AA^{-1}$]")

        self.ax[0].set_ylabel("$|Q|$ [$\AA^{-1}$]")

        self.ax[0].set_xlim(-1.2 * r_cut, 1.2 * r_cut)
        self.ax[1].set_xlim(-1.2 * r_cut, 1.2 * r_cut)
        self.ax[2].set_xlim(-1.2 * r_cut, 1.2 * r_cut)

        Q_max = 2 * np.pi / d_min

        self.ax[0].set_ylim(0, 1.2 * Q_max)
        self.ax[1].set_ylim(0, 1.2 * Q_max)
        self.ax[2].set_ylim(0, 1.2 * Q_max)

        self.ax[0].axvline(0, linestyle="--", color="k", linewidth=1)
        self.ax[1].axvline(0, linestyle="--", color="k", linewidth=1)
        self.ax[2].axvline(0, linestyle="--", color="k", linewidth=1)


class PeakProfilePlot(BasePlot):
    def __init__(self, x, y, e, vol, r, r_cut):
        super(PeakProfilePlot, self).__init__()

        plt.close("all")

        self.fig, self.ax = plt.subplots(
            1,
            3,
            figsize=(6.4 * 3, 4.8),
            layout="constrained",
            sharex=True,
            sharey=True,
        )

        self.plot_peaks(*x, *y, *e, r_cut)

        self.plot_peak_bins(*x, *y, *e)

        self.draw_boundary(vol, r)

    def plot_peaks(self, x0, x1, x2, y0, y1, y2, e0, e1, e2, r_cut):
        self.ax[0].errorbar(x0, y0, e0, fmt=".", color="C0")
        self.ax[1].errorbar(x1, y1, e1, fmt=".", color="C1")
        self.ax[2].errorbar(x2, y2, e2, fmt=".", color="C2")

        self.ax[0].minorticks_on()
        self.ax[2].minorticks_on()
        self.ax[1].minorticks_on()

        self.ax[0].set_xlabel("$\Delta |Q|$ [$\AA^{-1}$]")
        self.ax[1].set_xlabel("$\Delta Q_1$ [$\AA^{-1}$]")
        self.ax[2].set_xlabel("$\Delta Q_2$ [$\AA^{-1}$]")

        self.ax[0].set_ylim(-0.5, 2.5)
        self.ax[1].set_ylim(-0.5, 2.5)
        self.ax[2].set_ylim(-0.5, 2.5)

        self.ax[0].set_xlim(-1.2 * r_cut, 1.2 * r_cut)
        self.ax[1].set_xlim(-1.2 * r_cut, 1.2 * r_cut)
        self.ax[2].set_xlim(-1.2 * r_cut, 1.2 * r_cut)

    def plot_peak_bins(self, x0, x1, x2, y0, y1, y2, e0, e1, e2):
        mask0 = y0 > 0
        mask1 = y1 > 0
        mask2 = y2 > 0

        w0 = y0.copy()
        w1 = y1.copy()
        w2 = y2.copy()

        x0_bins = np.histogram_bin_edges(x0[mask0], bins="auto")
        x1_bins = np.histogram_bin_edges(x1[mask1], bins="auto")
        x2_bins = np.histogram_bin_edges(x2[mask2], bins="auto")

        w0_bins, _ = np.histogram(x0[mask0], bins=x0_bins, weights=w0[mask0])
        w1_bins, _ = np.histogram(x1[mask1], bins=x1_bins, weights=w1[mask1])
        w2_bins, _ = np.histogram(x2[mask2], bins=x2_bins, weights=w2[mask2])

        w0_bins /= w0_bins.max()
        w1_bins /= w1_bins.max()
        w2_bins /= w2_bins.max()

        self.ax[0].stairs(w0_bins, x0_bins, color="k", zorder=100)
        self.ax[1].stairs(w1_bins, x1_bins, color="k", zorder=100)
        self.ax[2].stairs(w2_bins, x2_bins, color="k", zorder=100)

    def draw_boundary(self, res, params):
        (r0, r1, r2), (dr0, dr1, dr2) = params

        res_min, res_max = 0, 0
        if len(res) > 1:
            res_min, res_max = np.min(res), np.max(res)

        s0_max = r0 + res_max * dr0
        s1_max = r1 + res_max * dr1
        s2_max = r2 + res_max * dr2

        s0_min = r0 + res_min * dr0
        s1_min = r1 + res_min * dr1
        s2_min = r2 + res_min * dr2

        self.ax[0].axvline(s0_max, linestyle="--", color="k", linewidth=1)
        self.ax[1].axvline(s1_max, linestyle="--", color="k", linewidth=1)
        self.ax[2].axvline(s2_max, linestyle="--", color="k", linewidth=1)

        self.ax[0].axvline(-s0_max, linestyle="--", color="k", linewidth=1)
        self.ax[1].axvline(-s1_max, linestyle="--", color="k", linewidth=1)
        self.ax[2].axvline(-s2_max, linestyle="--", color="k", linewidth=1)

        self.ax[0].axvline(s0_min, linestyle=":", color="k", linewidth=1)
        self.ax[1].axvline(s1_min, linestyle=":", color="k", linewidth=1)
        self.ax[2].axvline(s2_min, linestyle=":", color="k", linewidth=1)

        self.ax[0].axvline(-s0_min, linestyle=":", color="k", linewidth=1)
        self.ax[1].axvline(-s1_min, linestyle=":", color="k", linewidth=1)
        self.ax[2].axvline(-s2_min, linestyle=":", color="k", linewidth=1)


class PeakPlot(BasePlot):
    def __init__(self):
        super(PeakPlot, self).__init__()

        plt.close("all")

        self.fig = plt.figure(figsize=(6.4 * 2, 4.8 * 2), layout="constrained")

        sp = GridSpec(3, 2, figure=self.fig)

        self.gs = []

        gs = GridSpecFromSubplotSpec(
            2,
            3,
            height_ratios=[1, 1],
            width_ratios=[1, 1, 1],
            subplot_spec=sp[0, 1],
        )

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(
            3,
            3,
            height_ratios=[1, 1, 1],
            width_ratios=[1, 1, 1],
            subplot_spec=sp[:2, 0],
        )

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(
            2,
            3,
            height_ratios=[1, 1],
            width_ratios=[1, 1, 1],
            subplot_spec=sp[1, 1],
        )

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(
            1,
            3,
            height_ratios=[1],
            width_ratios=[1, 1, 1],
            subplot_spec=sp[2, 0],
        )

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(
            1, 1, height_ratios=[1], width_ratios=[1], subplot_spec=sp[2, 1]
        )

        self.gs.append(gs)

        self.__init_ellipsoid()
        self.__init_profile()
        self.__init_projection()
        self.__init_norm()
        self.__init_int()

    def __init_ellipsoid(self):
        self.ellip = []
        self.ellip_im = []
        self.ellip_el = []
        self.ellip_sp = []

        x = np.arange(5)
        y = np.arange(6)
        z = y + y.size * x[:, np.newaxis]

        gs = self.gs[0]

        ax = self.fig.add_subplot(gs[0, 0])

        self.ellip.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.ellip_el.append(el)
        self.ellip_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1, 0])

        self.ellip.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm="linear",
        )

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.ellip_el.append(el)
        self.ellip_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[0, 1])

        self.ellip.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.ellip_el.append(el)
        self.ellip_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1, 1])

        self.ellip.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm="linear",
        )

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.ellip_el.append(el)
        self.ellip_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[0, 2])

        self.ellip.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.ellip_el.append(el)
        self.ellip_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1, 2])

        self.ellip.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm="linear",
        )

        self.ellip_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.ellip_el.append(el)
        self.ellip_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        norm = Normalize(0, 29)
        im = ScalarMappable(norm=norm)
        self.cb_el = self.fig.colorbar(im, ax=self.ellip[-2:])
        self.cb_el.ax.minorticks_on()
        # self.cb_el.formatter.set_powerlimits((0, 0))
        # self.cb_el.formatter.set_useMathText(True)

    def __init_int(self):
        self.int_error = []
        self.int_line = []

        gs = self.gs[4]

        ax = self.fig.add_subplot(gs[0, 0])

        ax.minorticks_on()
        ax.set_xlabel(r"$r$")

        x = np.arange(10) - 5
        y = -2 * x**2 + 50
        e = np.sqrt(np.abs(y))

        self.int = ax

        error_cont = ax.errorbar(x, y, e, fmt=".", color="C0", zorder=1)
        plot_line = ax.step(x, y, where="mid", color="C1", zorder=0)

        self.int_error.append(error_cont)
        self.int_line.append(plot_line)

    def __init_profile(self):
        self.prof = []
        self.prof_error = []
        self.prof_step = []
        self.prof_lp = []
        self.prof_rp = []
        self.prof_lb = []
        self.prof_rb = []

        gs = self.gs[1]

        ax = self.fig.add_subplot(gs[0, :])

        ax.minorticks_on()
        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")
        # ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        # ax.yaxis.get_major_formatter().set_useMathText(True)

        x = np.arange(10) - 5
        y = -2 * x**2 + 50
        e = np.sqrt(np.abs(y))

        vl = ax.axvline(x=-1, color="k", linestyle="--")
        vr = ax.axvline(x=+1, color="k", linestyle="--")

        self.prof_lp.append(vl)
        self.prof_rp.append(vr)

        vl = ax.axvline(x=-2, color="k", linestyle="--")
        vr = ax.axvline(x=+2, color="k", linestyle="--")

        self.prof_lb.append(vl)
        self.prof_rb.append(vr)

        error_cont = ax.errorbar(x, y, e, fmt="o", color="C0")
        step_line = ax.step(x, y, where="mid", color="C1")

        self.prof_error.append(error_cont)
        self.prof_step.append(step_line)

        self.prof.append(ax)

        ax = self.fig.add_subplot(gs[1, :])

        ax.minorticks_on()
        ax.set_xlabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")
        # ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        # ax.yaxis.get_major_formatter().set_useMathText(True)

        x = np.arange(10) - 5
        y = -2 * x**2 + 50
        e = np.sqrt(np.abs(y))

        vl = ax.axvline(x=-1, color="k", linestyle="--")
        vr = ax.axvline(x=+1, color="k", linestyle="--")

        self.prof_lp.append(vl)
        self.prof_rp.append(vr)

        vl = ax.axvline(x=-2, color="k", linestyle="--")
        vr = ax.axvline(x=+2, color="k", linestyle="--")

        self.prof_lb.append(vl)
        self.prof_rb.append(vr)

        error_cont = ax.errorbar(x, y, e, fmt="o", color="C0")
        step_line = ax.step(x, y, where="mid", color="C1")

        self.prof_error.append(error_cont)
        self.prof_step.append(step_line)

        self.prof.append(ax)

        ax = self.fig.add_subplot(gs[2, :])

        ax.minorticks_on()
        ax.set_xlabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")
        # ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        # ax.yaxis.get_major_formatter().set_useMathText(True)

        x = np.arange(10) - 5
        y = -2 * x**2 + 50
        e = np.sqrt(np.abs(y))

        vl = ax.axvline(x=-1, color="k", linestyle="--")
        vr = ax.axvline(x=+1, color="k", linestyle="--")

        self.prof_lp.append(vl)
        self.prof_rp.append(vr)

        vl = ax.axvline(x=-2, color="k", linestyle="--")
        vr = ax.axvline(x=+2, color="k", linestyle="--")

        self.prof_lb.append(vl)
        self.prof_rb.append(vr)

        error_cont = ax.errorbar(x, y, e, fmt="o", color="C0")
        step_line = ax.step(x, y, where="mid", color="C1")

        self.prof_error.append(error_cont)
        self.prof_step.append(step_line)

        self.prof.append(ax)

    def __init_projection(self):
        self.proj = []
        self.proj_im = []
        self.proj_el = []
        self.proj_sp = []

        x = np.arange(5)
        y = np.arange(6)
        z = y + y.size * x[:, np.newaxis]

        gs = self.gs[2]

        ax = self.fig.add_subplot(gs[0, 0])

        self.proj.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.proj_el.append(el)
        self.proj_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1, 0])

        self.proj.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm="linear",
        )

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.proj_el.append(el)
        self.proj_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[0, 1])

        self.proj.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.proj_el.append(el)
        self.proj_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1, 1])

        self.proj.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm="linear",
        )

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")
        ax.set_ylabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.proj_el.append(el)
        self.proj_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[0, 2])

        self.proj.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.proj_el.append(el)
        self.proj_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        ax = self.fig.add_subplot(gs[1, 2])

        self.proj.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm="linear",
        )

        self.proj_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        el = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        sp = self._draw_ellipse(ax, 2.5, 3, 1, 1, 0, "r")
        self.proj_el.append(el)
        self.proj_sp.append(sp)

        # line = self._draw_intersecting_line(ax, 2.5, 3)
        # self.ellip_pt.append(line)

        norm = Normalize(0, 29)
        im = ScalarMappable(norm=norm)
        self.cb_proj = self.fig.colorbar(im, ax=self.proj[-2:])
        self.cb_proj.ax.minorticks_on()
        # self.cb_proj.formatter.set_powerlimits((0, 0))
        # self.cb_proj.formatter.set_useMathText(True)

    def __init_norm(self):
        self.norm = []
        self.norm_im = []
        self.norm_pk = []
        self.norm_bkg = []

        x = np.arange(5)
        y = np.arange(6)
        z = y + y.size * x[:, np.newaxis]

        gs = self.gs[3]

        ax = self.fig.add_subplot(gs[0, 0])

        self.norm.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.norm_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        # ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")

        [line] = ax.plot([], [], color="r")
        self.norm_pk.append(line)
        [line] = ax.plot([], [], color="r")
        self.norm_bkg.append(line)

        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")

        ax = self.fig.add_subplot(gs[0, 1])

        self.norm.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.norm_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        # ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r"$\Delta{Q}_2$ [$\AA^{-1}$]")

        [line] = ax.plot([], [], color="r")
        self.norm_pk.append(line)
        [line] = ax.plot([], [], color="r")
        self.norm_bkg.append(line)

        ax.set_xlabel(r"$|Q|$ [$\AA^{-1}$]")

        ax = self.fig.add_subplot(gs[0, 2])

        self.norm.append(ax)

        im = ax.imshow(
            z.T,
            extent=(0, 5, 0, 6),
            origin="lower",
            interpolation="nearest",
            norm="linear",
        )

        self.norm_im.append(im)

        ax.minorticks_on()
        ax.set_aspect(1)
        # ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        [line] = ax.plot([], [], color="r")
        self.norm_pk.append(line)
        [line] = ax.plot([], [], color="r")
        self.norm_bkg.append(line)

        ax.set_xlabel(r"$\Delta{Q}_1$ [$\AA^{-1}$]")

        norm = Normalize(0, 29)
        im = ScalarMappable(norm=norm)

        self.cb_norm = self.fig.colorbar(
            im, ax=self.norm, orientation="horizontal"
        )
        self.cb_norm.ax.minorticks_on()
        # self.cb_norm.formatter.set_powerlimits((0, 0))
        # self.cb_norm.formatter.set_useMathText(True)

    def add_integral_fit(self, xye_fit):
        x, y_fit, y, e = xye_fit

        lines, caps, bars = self.int_error[0]
        lines.set_data(x, y)

        (barsy,) = bars

        yb, yt = y - e, y + e

        n = len(x)

        segments = [np.array([[x[i], yt[i]], [x[i], yb[i]]]) for i in range(n)]

        barsy.set_segments(segments)

        self.int_line[0][0].set_data(x, y_fit)

        self.int.relim()
        self.int.autoscale_view()

    def add_profile_fit(self, xye_fit):
        x, y_fit, y, e = xye_fit[0]

        lines, caps, bars = self.prof_error[0]
        lines.set_data(x, y)

        (barsy,) = bars

        yb, yt = y - e, y + e

        n = len(x)

        segments = [np.array([[x[i], yt[i]], [x[i], yb[i]]]) for i in range(n)]

        barsy.set_segments(segments)

        self.prof_step[0][0].set_data(x, y_fit)

        self.prof[0].relim()
        self.prof[0].autoscale_view()

        # ---

        x, y_fit, y, e = xye_fit[1]

        lines, caps, bars = self.prof_error[1]
        lines.set_data(x, y)

        (barsy,) = bars

        yb, yt = y - e, y + e

        n = len(x)

        segments = [np.array([[x[i], yt[i]], [x[i], yb[i]]]) for i in range(n)]

        barsy.set_segments(segments)

        self.prof_step[1][0].set_data(x, y_fit)

        self.prof[1].relim()
        self.prof[1].autoscale_view()

        # ---

        x, y_fit, y, e = xye_fit[2]

        lines, caps, bars = self.prof_error[2]
        lines.set_data(x, y)

        (barsy,) = bars

        yb, yt = y - e, y + e

        n = len(x)

        segments = [np.array([[x[i], yt[i]], [x[i], yb[i]]]) for i in range(n)]

        barsy.set_segments(segments)

        self.prof_step[2][0].set_data(x, y_fit)

        self.prof[2].relim()
        self.prof[2].autoscale_view()

    def add_projection_fit(self, xye_fit):
        x1, x2, y_fit, y, e = xye_fit[0]

        mask = np.isfinite(e) & (e > 0)
        y[~mask] = np.nan
        y_fit[~mask] = np.nan

        d1 = 0.5 * (x1[1, 0] - x1[0, 0])
        d2 = 0.5 * (x2[0, 1] - x2[0, 0])

        x1_min, x1_max = x1[0, 0] - d1, x1[-1, 0] + d1
        x2_min, x2_max = x2[0, 0] - d2, x2[0, -1] + d2

        vmin, vmax = self._color_limits(y)

        self.proj_im[4].set_data(y.T)
        self.proj_im[4].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.proj_im[4].set_clim(vmin, vmax)

        # ---

        vmin, vmax = self._color_limits(y_fit)

        self.proj_im[5].set_data(y_fit.T)
        self.proj_im[5].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.proj_im[5].set_clim(vmin, vmax)

        # ---

        x0, x2, y_fit, y, e = xye_fit[1]

        mask = np.isfinite(y) & (e > 0)
        y[~mask] = np.nan
        y_fit[~mask] = np.nan

        d0 = 0.5 * (x0[1, 0] - x0[0, 0])
        d2 = 0.5 * (x2[0, 1] - x2[0, 0])

        x0_min, x0_max = x0[0, 0] - d0, x0[-1, 0] + d0
        x2_min, x2_max = x2[0, 0] - d2, x2[0, -1] + d2

        vmin, vmax = self._color_limits(y)

        self.proj_im[2].set_data(y.T)
        self.proj_im[2].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.proj_im[2].set_clim(vmin, vmax)

        # ---

        vmin, vmax = self._color_limits(y_fit)

        self.proj_im[3].set_data(y_fit.T)
        self.proj_im[3].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.proj_im[3].set_clim(vmin, vmax)

        # ---

        x0, x1, y_fit, y, e = xye_fit[2]

        mask = np.isfinite(y) & (e > 0)
        y[~mask] = np.nan
        y_fit[~mask] = np.nan

        d0 = 0.5 * (x0[1, 0] - x0[0, 0])
        d1 = 0.5 * (x1[0, 1] - x1[0, 0])

        x0_min, x0_max = x0[0, 0] - d0, x0[-1, 0] + d0
        x1_min, x1_max = x1[0, 0] - d1, x1[0, -1] + d1

        vmin, vmax = self._color_limits(y)

        self.proj_im[0].set_data(y.T)
        self.proj_im[0].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.proj_im[0].set_clim(vmin, vmax)

        # ---

        vmin, vmax = self._color_limits(y_fit)

        self.proj_im[1].set_data(y_fit.T)
        self.proj_im[1].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.proj_im[1].set_clim(vmin, vmax)

        self.cb_proj.update_normal(self.proj_im[4])
        self.cb_proj.ax.minorticks_on()
        # self.cb_proj.formatter.set_powerlimits((0, 0))
        # self.cb_proj.formatter.set_useMathText(True)

    def add_data_norm_fit(self, xye, params):
        axes, bins, y, e = xye

        x0, x1, x2 = axes

        mask = np.isfinite(e) & (e > 0)

        y[~mask] = np.nan
        e[~mask] = np.nan

        y0 = np.nansum(y, axis=0)  # / np.nanmean(e > 0, axis=0)
        y1 = np.nansum(y, axis=1)  # / np.nanmean(e > 0, axis=1)
        y2 = np.nansum(y, axis=2)  # / np.nanmean(e > 0, axis=2)

        e0 = np.nansum(e**2, axis=0)
        e1 = np.nansum(e**2, axis=1)
        e2 = np.nansum(e**2, axis=2)

        mask_0 = np.isfinite(y0) & (e0 > 0)
        mask_1 = np.isfinite(y1) & (e1 > 0)
        mask_2 = np.isfinite(y2) & (e2 > 0)

        y0[~mask_0] = np.nan
        y1[~mask_1] = np.nan
        y2[~mask_2] = np.nan

        mask = np.isfinite(y)

        if mask.sum() == 0:
            mask = np.ones_like(mask, dtype=bool)

        s0 = 0.5 * (x0[1, 0, 0] - x0[0, 0, 0])
        s1 = 0.5 * (x1[0, 1, 0] - x1[0, 0, 0])
        s2 = 0.5 * (x2[0, 0, 1] - x2[0, 0, 0])

        x0min, x0max = x0[mask].min() - s0 * 2, x0[mask].max() + s0 * 2
        x1min, x1max = x1[mask].min() - s1 * 2, x1[mask].max() + s1 * 2
        x2min, x2max = x2[mask].min() - s2 * 2, x2[mask].max() + s2 * 2

        x0_min, x0_max = x0[0, 0, 0] - s0, x0[-1, 0, 0] + s0
        x1_min, x1_max = x1[0, 0, 0] - s1, x1[0, -1, 0] + s1
        x2_min, x2_max = x2[0, 0, 0] - s2, x2[0, 0, -1] + s2

        vmin, vmax = self._color_limits(y2)

        self.norm_im[0].set_data(y2.T)
        self.norm_im[0].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.norm_im[0].set_clim(vmin, vmax)
        self.norm[0].set_xlim([x0min, x0max])
        self.norm[0].set_ylim([x1min, x1max])

        # ---

        vmin, vmax = self._color_limits(y1)

        self.norm_im[1].set_data(y1.T)
        self.norm_im[1].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.norm_im[1].set_clim(vmin, vmax)
        self.norm[1].set_xlim([x0min, x0max])
        self.norm[1].set_ylim([x2min, x2max])

        # ---

        vmin, vmax = self._color_limits(y0)

        self.norm_im[2].set_data(y0.T)
        self.norm_im[2].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.norm_im[2].set_clim(vmin, vmax)
        self.norm[2].set_xlim([x1min, x1max])
        self.norm[2].set_ylim([x2min, x2max])

        self.cb_norm.update_normal(self.norm_im[2])
        self.cb_norm.ax.minorticks_on()
        # self.cb_norm.formatter.set_powerlimits((0, 0))
        # self.cb_norm.formatter.set_useMathText(True)

        I = r"$I={}$"
        I_sig = "$I/\sigma={:.1f}$"
        B = r"$B={}$"

        self.norm[0].set_title(I.format(self._sci_notation(params[0])))
        self.norm[1].set_title(I_sig.format(params[0] / params[1]))
        self.norm[2].set_title(B.format(self._sci_notation(params[2])))

    def _color_limits(self, y):
        """
        Calculate color limits common for an arrays

        Parameters
        ----------
        y : array-like
            Data array.

        Returns
        -------
        vmin, vmax : float
            Color limits

        """

        vmin, vmax = np.nanmin(y), np.nanmax(y)

        if vmin >= vmax:
            vmin, vmax = 0, 1

        if np.isclose(vmin, vmax) or not np.isfinite([vmin, vmax]).all():
            vmin, vmax = 0, 1

        return vmin, vmax

    def add_ellipsoid_fit(self, xye_fit):
        """
        Three-dimensional ellipsoids.

        Parameters
        ----------
        x0, x1, x2, y, e, fit : 3d-array
            Bins, signal, error, and fit.

        """

        x0, x1, x2, y_fit, y, e = xye_fit

        mask = np.isfinite(e) & (e >= 0)

        y[~mask] = np.nan
        y_fit[~mask] = np.nan

        y0 = np.nansum(y, axis=0)  # / np.nanmean(e > 0, axis=0)
        y1 = np.nansum(y, axis=1)  # / np.nanmean(e > 0, axis=1)
        y2 = np.nansum(y, axis=2)  # / np.nanmean(e > 0, axis=2)

        e0 = np.nansum(e**2, axis=0)
        e1 = np.nansum(e**2, axis=1)
        e2 = np.nansum(e**2, axis=2)

        y0_fit = np.nansum(y_fit, axis=0)  # / np.nanmean(e > 0, axis=0)
        y1_fit = np.nansum(y_fit, axis=1)  # / np.nanmean(e > 0, axis=1)
        y2_fit = np.nansum(y_fit, axis=2)  # / np.nanmean(e > 0, axis=2)

        mask_0 = np.isfinite(y0) & (e0 > 0)
        mask_1 = np.isfinite(y1) & (e1 > 0)
        mask_2 = np.isfinite(y2) & (e2 > 0)

        y0[~mask_0] = np.nan
        y1[~mask_1] = np.nan
        y2[~mask_2] = np.nan

        y0_fit[~mask_0] = np.nan
        y1_fit[~mask_1] = np.nan
        y2_fit[~mask_2] = np.nan

        d0 = 0.5 * (x0[1, 0, 0] - x0[0, 0, 0])
        d1 = 0.5 * (x1[0, 1, 0] - x1[0, 0, 0])
        d2 = 0.5 * (x2[0, 0, 1] - x2[0, 0, 0])

        x0_min, x0_max = x0[0, 0, 0] - d0, x0[-1, 0, 0] + d0
        x1_min, x1_max = x1[0, 0, 0] - d1, x1[0, -1, 0] + d1
        x2_min, x2_max = x2[0, 0, 0] - d2, x2[0, 0, -1] + d2

        vmin, vmax = self._color_limits(y2)

        self.ellip_im[0].set_data(y2.T)
        self.ellip_im[0].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.ellip_im[0].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y2_fit)

        self.ellip_im[1].set_data(y2_fit.T)
        self.ellip_im[1].set_extent((x0_min, x0_max, x1_min, x1_max))
        self.ellip_im[1].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y1)

        self.ellip_im[2].set_data(y1.T)
        self.ellip_im[2].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.ellip_im[2].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y1_fit)

        self.ellip_im[3].set_data(y1_fit.T)
        self.ellip_im[3].set_extent((x0_min, x0_max, x2_min, x2_max))
        self.ellip_im[3].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y0)

        self.ellip_im[4].set_data(y0.T)
        self.ellip_im[4].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.ellip_im[4].set_clim(vmin, vmax)

        vmin, vmax = self._color_limits(y0_fit)

        self.ellip_im[5].set_data(y0_fit.T)
        self.ellip_im[5].set_extent((x1_min, x1_max, x2_min, x2_max))
        self.ellip_im[5].set_clim(vmin, vmax)

        self.cb_el.update_normal(self.ellip_im[4])
        self.cb_el.ax.minorticks_on()
        # self.cb_el.formatter.set_powerlimits((0, 0))
        # self.cb_el.formatter.set_useMathText(True)

    def add_ellipsoid(self, c, S):
        """
        Draw ellipsoid envelopes.

        Parameters
        ----------
        c : 1d-array
            3 component center.
        S : 2d-array
            3x3 covariance matrix.

        """

        r = np.sqrt(np.diag(S))

        rho = [
            S[1, 2] / r[1] / r[2],
            S[0, 2] / r[0] / r[2],
            S[0, 1] / r[0] / r[1],
        ]

        for el, ax in zip(self.ellip_el[0:2], self.ellip[0:2]):
            self._update_ellipse(el, ax, c[0], c[1], r[0], r[1], rho[2])

        for el, ax in zip(self.ellip_el[2:4], self.ellip[2:4]):
            self._update_ellipse(el, ax, c[0], c[2], r[0], r[2], rho[1])

        for el, ax in zip(self.ellip_el[4:6], self.ellip[4:6]):
            self._update_ellipse(el, ax, c[1], c[2], r[1], r[2], rho[0])

        for el, ax in zip(self.proj_el[0:2], self.proj[0:2]):
            self._update_ellipse(el, ax, c[0], c[1], r[0], r[1], rho[2])

        for el, ax in zip(self.proj_el[2:4], self.proj[2:4]):
            self._update_ellipse(el, ax, c[0], c[2], r[0], r[2], rho[1])

        for el, ax in zip(self.proj_el[4:6], self.proj[4:6]):
            self._update_ellipse(el, ax, c[1], c[2], r[1], r[2], rho[0])

        s = np.cbrt(2)

        for el, ax in zip(self.ellip_sp[0:2], self.ellip[0:2]):
            self._update_ellipse(
                el, ax, c[0], c[1], r[0] * s, r[1] * s, rho[2]
            )

        for el, ax in zip(self.ellip_sp[2:4], self.ellip[2:4]):
            self._update_ellipse(
                el, ax, c[0], c[2], r[0] * s, r[2] * s, rho[1]
            )

        for el, ax in zip(self.ellip_sp[4:6], self.ellip[4:6]):
            self._update_ellipse(
                el, ax, c[1], c[2], r[1] * s, r[2] * s, rho[0]
            )

        s = np.sqrt(2)

        for el, ax in zip(self.proj_sp[0:2], self.proj[0:2]):
            self._update_ellipse(
                el, ax, c[0], c[1], r[0] * s, r[1] * s, rho[2]
            )

        for el, ax in zip(self.proj_sp[2:4], self.proj[2:4]):
            self._update_ellipse(
                el, ax, c[0], c[2], r[0] * s, r[2] * s, rho[1]
            )

        for el, ax in zip(self.proj_sp[4:6], self.proj[4:6]):
            self._update_ellipse(
                el, ax, c[1], c[2], r[1] * s, r[2] * s, rho[0]
            )

        self.prof_lp[0].set_xdata([c[0] - r[0]])
        self.prof_rp[0].set_xdata([c[0] + r[0]])

        self.prof_lp[1].set_xdata([c[1] - r[1]])
        self.prof_rp[1].set_xdata([c[1] + r[1]])

        self.prof_lp[2].set_xdata([c[2] - r[2]])
        self.prof_rp[2].set_xdata([c[2] + r[2]])

        self.prof_lb[0].set_xdata([c[0] - 2 * r[0]])
        self.prof_rb[0].set_xdata([c[0] + 2 * r[0]])

        self.prof_lb[1].set_xdata([c[1] - 2 * r[1]])
        self.prof_rb[1].set_xdata([c[1] + 2 * r[1]])

        self.prof_lb[2].set_xdata([c[2] - 2 * r[2]])
        self.prof_rb[2].set_xdata([c[2] + 2 * r[2]])

        self.prof[0].relim()
        self.prof[0].autoscale_view()

        self.prof[1].relim()
        self.prof[1].autoscale_view()

        self.prof[2].relim()
        self.prof[2].autoscale_view()

    def _path(self, mask, x, y, dx, dy):
        if not mask.any():
            return np.array([]), np.array([])

        mask = scipy.ndimage.binary_fill_holes(mask)
        mask = scipy.ndimage.binary_closing(mask, structure=np.ones((3, 3)))

        roi = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)

        contours = skimage.measure.find_contours(roi.astype(float), level=0.5)

        if len(contours) == 0:
            return np.array([]), np.array([])

        c = max(contours, key=len)
        row, col = c[:, 0], c[:, 1]

        hx = x + 0.25 * (row + 0.5) * dx - 0.5 * dx
        hy = y + 0.25 * (col + 0.5) * dy - 0.5 * dy

        if hx.size <= 1:
            return np.array([]), np.array([])

        if hx[0] != hx[-1] or hy[0] != hy[-1]:
            hx = np.r_[hx, hx[0]]
            hy = np.r_[hy, hy[0]]

        return hx, hy

    def update_envelope(self, x0, x1, x2, pk, bkg):
        """
        Draw region-of-interest.
        mask =

        Parameters
        ----------
        pk : 3d-array
            Peak region.
        bkg : 3d-array
            Background shell.

        """

        dx0 = x0[1, 0, 0] - x0[0, 0, 0]
        dx1 = x1[0, 1, 0] - x1[0, 0, 0]
        dx2 = x2[0, 0, 1] - x2[0, 0, 0]

        x0, x1, x2 = x0[0, 0, 0], x1[0, 0, 0], x2[0, 0, 0]

        mask = (np.nansum(pk, axis=0) > 0) | (np.nansum(bkg, axis=0) > 0)

        x, y = self._path(mask, x1, x2, dx1, dx2)

        self.norm_bkg[2].set_data(x, y)

        mask = (np.nansum(pk, axis=1) > 0) | (np.nansum(bkg, axis=1) > 0)

        x, y = self._path(mask, x0, x2, dx0, dx2)

        self.norm_bkg[1].set_data(x, y)

        mask = (np.nansum(pk, axis=2) > 0) | (np.nansum(bkg, axis=2) > 0)

        x, y = self._path(mask, x0, x1, dx0, dx1)

        self.norm_bkg[0].set_data(x, y)

        # ---

        mask = np.nansum(pk, axis=0) > 0

        x, y = self._path(mask, x1, x2, dx1, dx2)

        self.norm_pk[2].set_data(x, y)

        mask = np.nansum(pk, axis=1) > 0

        x, y = self._path(mask, x0, x2, dx0, dx2)

        self.norm_pk[1].set_data(x, y)

        mask = np.nansum(pk, axis=2) > 0

        x, y = self._path(mask, x0, x1, dx0, dx1)

        self.norm_pk[0].set_data(x, y)

    def _update_ellipse(self, ellipse, ax, cx, cy, rx, ry, rho):
        ellipse.set_center((0, 0))

        if not np.isfinite(rho):
            rho = 0

        ellipse.width = 2 * np.sqrt(1 + rho)
        ellipse.height = 2 * np.sqrt(1 - rho)

        if np.isclose(rx, 0):
            rx = 1
        if np.isclose(ry, 0):
            ry = 1

        trans = Affine2D()
        trans.rotate_deg(45).scale(rx, ry).translate(cx, cy)

        ellipse.set_transform(trans + ax.transData)

    def _draw_ellipse(self, ax, cx, cy, rx, ry, rho, color="w"):
        """
        Draw ellipse with center, size, and orientation.

        Parameters
        ----------
        ax : axis
            Plot axis.
        cx, cy : float
            Center.
        rx, ry : float
            Radii.
        rho : float
            Correlation.

        """

        peak = Ellipse(
            (0, 0),
            width=2 * np.sqrt(1 + rho),
            height=2 * np.sqrt(1 - rho),
            linestyle="-",
            edgecolor=color,
            facecolor="none",
            rasterized=False,
            zorder=100,
        )

        self._update_ellipse(peak, ax, cx, cy, rx, ry, rho)

        ax.add_patch(peak)

        return peak

    def _update_intersecting_line(self, line, ax, x0, y0):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if x0 != 0:
            slope = y0 / x0
        else:
            slope = np.inf

        y_at_x_min = slope * (x_min - x0) + y0 if slope != np.inf else y_min
        y_at_x_max = slope * (x_max - x0) + y0 if slope != np.inf else y_max
        x_at_y_min = (y_min - y0) / slope + x0 if slope != 0 else x_min
        x_at_y_max = (y_max - y0) / slope + x0 if slope != 0 else x_max

        points = []
        if y_min <= y_at_x_min <= y_max:
            points.append((x_min, y_at_x_min))
        if y_min <= y_at_x_max <= y_max:
            points.append((x_max, y_at_x_max))
        if x_min <= x_at_y_min <= x_max:
            points.append((x_at_y_min, y_min))
        if x_min <= x_at_y_max <= x_max:
            points.append((x_at_y_max, y_max))

        if len(points) > 2:
            points = sorted(points, key=lambda p: (p[0], p[1]))[:2]
        elif len(points) == 0:
            points = (x_min, y_min), (x_max, y_max)

        (x1, y1), (x2, y2) = points

        line.set_data([x1, x2], [y1, y2])

    def _draw_intersecting_line(self, ax, x0, y0):
        """
        Draw line toward origin.

        Parameters
        ----------
        ax : axis
            Plot axis.
        x0, y0 : float
            Center.

        """

        (line,) = ax.plot([], [], color="k", linestyle="--")

        self._update_intersecting_line(line, ax, x0, y0)

        return line

    def _sci_notation(self, x):
        """
        Represent float in scientific notation using LaTeX.

        Parameters
        ----------
        x : float
            Value to convert.

        Returns
        -------
        s : str
            String representation in LaTeX.

        """

        if np.isfinite(x):
            val = np.floor(np.log10(abs(x)))
            if np.isfinite(val):
                exp = int(val)
                return "{:.2f}\\times 10^{{{}}}".format(x / 10**exp, exp)
            else:
                return "\\infty"
        else:
            return "\\infty"

    def add_peak_info(self, hkl, d, wavelength, angles, gon):
        """
        Add peak information.

        Parameters
        ----------
        hkl : list
            Miller indices.
        d : float,
            Interplanar d-spacing.
        wavelength : float
            Wavelength.
        angles : list
            Scattering and azimuthal angles.
        gon : list
            Goniometer Euler angles.

        """

        self.cb_el.ax.set_ylabel(r"$({:.2f}, {:.2f}, {:.2f})$".format(*hkl))
        self.cb_proj.ax.set_ylabel(r"$d={:.4f}$ [$\AA$]".format(d))

        ellip = self.ellip

        ellip[2].set_title(r"$\lambda={:.4f}$ [$\AA$]".format(wavelength))
        ellip[3].set_title(r"$({:.1f},{:.1f},{:.1f})^\circ$".format(*gon))
        ellip[4].set_title(r"$2\theta={:.2f}^\circ$".format(angles[0]))
        ellip[5].set_title(r"$\phi={:.2f}^\circ$".format(angles[1]))

    def add_peak_stats(self, redchi2, intensity, sigma):
        """
        Add peak statistics.

        Parameters
        ----------
        redchi2 : list
            Reduced chi^2 per degree of freedom.
        intensity : list
            Integrated intensity.
        sigma : list
            Integrated intensity uncertainty.

        """

        label = r"$I={}$ | $I/\sigma={:.1f}$ | $\chi^2_\nu={:.1f}$"

        self.prof[0].set_title(
            label.format(
                self._sci_notation(intensity[0][0]),
                intensity[0][0] / sigma[0][0],
                redchi2[0][0],
            )
        )
        self.prof[1].set_title(
            label.format(
                self._sci_notation(intensity[0][1]),
                intensity[0][1] / sigma[0][1],
                redchi2[0][1],
            )
        )
        self.prof[2].set_title(
            label.format(
                self._sci_notation(intensity[0][2]),
                intensity[0][2] / sigma[0][2],
                redchi2[0][2],
            )
        )

        self.proj[0].set_title(
            r"$I={}$".format(self._sci_notation(intensity[1][0]))
        )
        self.proj[2].set_title(
            r"$I={}$".format(self._sci_notation(intensity[1][1]))
        )
        self.proj[4].set_title(
            r"$I={}$".format(self._sci_notation(intensity[1][2]))
        )

        self.proj[1].set_title(r"$\chi^2_\nu={:.1f}$".format(redchi2[1][0]))
        self.proj[3].set_title(r"$\chi^2_\nu={:.1f}$".format(redchi2[1][1]))
        self.proj[5].set_title(r"$\chi^2_\nu={:.1f}$".format(redchi2[1][2]))

        self.ellip[0].set_title(
            r"$I={}$".format(self._sci_notation(intensity[2]))
        )
        self.ellip[1].set_title(r"$\chi^2_\nu={:.1f}$".format(redchi2[2]))

        I_sig = "$I/\sigma={:.1f}$"

        self.prof[0].set_ylabel(I_sig.format(intensity[1][0] / sigma[1][0]))
        self.prof[1].set_ylabel(I_sig.format(intensity[1][1] / sigma[1][1]))
        self.prof[2].set_ylabel(I_sig.format(intensity[1][2] / sigma[1][2]))

        self.cb_norm.ax.set_xlabel(I_sig.format(intensity[2] / sigma[2]))

        label = r"$I={}$ | $I/\sigma={:.1f}$"

        self.int.set_title(
            label.format(
                self._sci_notation(intensity[-1]), intensity[-1] / sigma[-1]
            )
        )
