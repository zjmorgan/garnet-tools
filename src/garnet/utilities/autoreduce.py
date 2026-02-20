import io
import os
import sys
import glob
import base64

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append(os.path.abspath(os.path.join(directory, "../..")))

import numpy as np
import matplotlib.pyplot as plt

from mantid import logger
from mantid.simpleapi import (
    LoadEventNexus,
    SumNeighbours,
    CompressEvents,
    MaskBTP,
    LoadInstrument,
    SaveNexus,
    PreprocessDetectorsToMD,
    CorelliCrossCorrelate,
    mtd,
)


from finddata import publish_plot

from garnet.config.instruments import beamlines

instrument_dict = {
    beamlines[key]["InstrumentName"]: key for key in beamlines.keys()
}


class AutoReduce:
    def __init__(self, filename):
        self.filename = filename

        facility, self.inst, *_ = self.filename.split("/")

        LoadEventNexus(
            Filename=self.filename, OutputWorkspace="data", NumberOfBins=1
        )

        self.run = mtd["data"].getRunNumber()

        self.instrument = instrument_dict[self.inst]

        filepath = os.path.join("/", facility, self.inst, "shared/autoreduce")

        self.idf = glob.glob(os.path.join(filepath, "*_Definition_*.xml"))

        self.files = {}

        self.cc = False

    def elastic(self, time_offset=14000):
        if self.instrument == "CORELLI":
            try:
                CorelliCrossCorrelate(
                    InputWorkspace="data",
                    OutputWorkspace="elastic",
                    TimingOffset=time_offset,
                )
            except RuntimeError as e:
                logger.warning("Cross Correlation failed: {}".format(e))
            else:
                output = self.filename.replace(
                    ".nxs.h5", "_elastic.nxs"
                ).replace("nexus", "shared/autoreduce")
                SaveNexus(InputWorkspace="elastic", Filename=output)
                self.cc = True
                self.compress("elastic")
                self.plot_instrument()

    def compress(self, ws):
        beamline = beamlines[self.instrument]

        c, r = [int(val) for val in beamline["Grouping"].split("x")]
        cols, rows = beamline["BankPixels"]
        mask_cols, mask_rows = beamline["MaskEdges"]

        SumNeighbours(
            InputWorkspace=ws,
            SumX=self.c,
            SumY=self.r,
            OutputWorkspace="lite",
        )
        CompressEvents(InputWorkspace="lite", OutputWorkspace="lite")

        cols //= c
        rows //= r
        mask_cols //= c
        mask_rows //= r

        MaskBTP(
            Workspace="lite",
            Instrument="lite",
            Pixel="0-{},{}-{}".format(mask_rows, rows - mask_rows, rows),
        )
        MaskBTP(
            Workspace="lite",
            Instrument="lite",
            Tube="0-{},{}-{}".format(mask_cols, cols - mask_cols, cols),
        )

        inst = beamline["Name"]
        banks = beamline["MaskBanks"]

        for bank in banks:
            MaskBTP(
                Workspace="lite",
                Instrument=inst,
                Bank=bank,
            )

        LoadInstrument(
            Workspace="lite",
            Filename=self.idf,
            RewriteSpectraMap="True",
        )

        out = "_" + ws if ws != "data" else ""

        output = self.filename.replace(".nxs.h5", out + "_lite.nxs").replace(
            "nexus", "shared/autoreduce"
        )

        SaveNexus(
            InputWorkspace="lite",
            Filename=output,
        )

    def plot_instrument(self):
        PreprocessDetectorsToMD(
            InputWorkspace="lite", OutputWorkspace="detectors"
        )
        tt = np.array(mtd["detectors"].column(2))
        az = np.array(mtd["detectors"].column(3))

        data = mtd["lite"].extractY().ravel()

        kf_x = np.sin(tt) * np.cos(az)
        kf_y = np.sin(tt) * np.sin(az)
        kf_z = np.cos(tt)

        nu = np.rad2deg(np.arcsin(kf_y))
        gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

        figfile = io.BytesIO()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.scatter(gamma, nu, c=data, s=1, norm="asinh", rasterized=True)
        ax.set_aspect(1)
        ax.minorticks_on()
        ax.set_xlabel(r"γ [°]")
        ax.set_ylabel(r"ν [°]")
        ax.xaxis.set_inverted(True)
        fig.savefig(figfile, format="svg", bbox_inches="tight")
        plt.close()

        figfile.seek(0)
        figdata = base64.b64encode(figfile.getvalue()).decode()

        out = "_elastic" if self.cc else ""

        output = self.filename.replace(".nxs.h5", out).replace(
            "nexus", "shared/autoreduce"
        )
        div = '<div><img alt="{}" src="data:image/svg+xml;base64,{}" /></div>'

        self.files[output] = div.format(output, figdata)

    def publish_plots(self):
        request = publish_plot(self.inst, self.run, files=self.files)
        print(request)


if __name__ == "__main__":
    filename = sys.argv[1]
    ar = AutoReduce(filename)
    ar.compress("data")
    ar.plot_instrument()
    ar.elastic()
    ar.publish_plots()
