import sys
import numpy as np

import gemmi

from mantid.simpleapi import mtd


class Macromolecular:
    def __init__(self, peaks="peaks"):
        self.peaks = peaks

        ol = mtd[self.peaks].sample().getOrientedLattice()

        self.a = ol.a()
        self.b = ol.b()
        self.c = ol.c()

        self.alpha = ol.alpha()
        self.beta = ol.beta()
        self.gamma = ol.gamma()

    def write_mtz(self, filename, space_group="P 1"):
        mtz = gemmi.Mtz(with_base=True)
        mtz.set_logging(sys.stdout)

        mtz.spacegroup = gemmi.find_spacegroup_by_name(space_group)

        unit_cell = gemmi.UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )
        mtz.set_cell_for_all(unit_cell)

        mtz.add_column("I", "J")
        mtz.add_column("SIGI", "Q")
        mtz.add_column("WAVEL", "W")
        mtz.add_column("BATCH", "B")

        data = []

        for peak in mtd[self.peaks]:
            h, k, l = peak.getHKL()
            I = peak.getIntensity()
            sig = peak.getSigmaIntensity()
            wl = peak.getWavelength()
            run = peak.getRunNumber()

            data.append([h, k, l, I, sig, wl, run])

        data = np.array(data)

        mtz.set_data(data)
        mtz.write_to_file(filename)
