import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use("fast")

from matplotlib.backends.backend_pdf import PdfPages


class BasePlot:
    def __init__(self):
        """
        Create a plot.

        """

        plt.close("all")

        self.fig = plt.figure()

    def save_plot(self, filename):
        """
        Save plot.

        Parameters
        ----------
        filename : str
            Path to file.

        """

        self.fig.savefig(filename, bbox_inches="tight")


class Pages:
    def __init__(self, filename):
        self.pdf = PdfPages(filename)

    def add_plot(self, fig):
        self.pdf.savefig(fig)

    def close(self):
        self.pdf.close()
