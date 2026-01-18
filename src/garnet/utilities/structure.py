import sys

import numpy as np

from garnet.utilities.reflections import AbsorptionCorrection, Peaks
from garnet.utilities.refinement import NuclearStructureRefinement

from garnet.reduction.plan import load_YAML, save_YAML
from garnet.reduction.crystallography import space_groups, space_point
from garnet.reduction.integration import Integration


class StructureAnalysis:
    def __init__(self, config):
        defaults = {
            "Filename": None,
            "ExtinctionModel": "Type II",
            "RefineAbsorption": False,
            "ChemicalFormula": "Yb3-Al5-O12",
        }

        defaults.update(config)

        space_group = defaults.get("SpaceGroup")
        point_group = None
        if space_group is not None:
            space_group = space_groups.get(space_group)
            point_group = space_point.get(space_group)

        self.space_group = space_group
        self.point_group = point_group

        self.sites = defaults.get("Sites")

        self.refine_abs = defaults.get("RefineAbsorption")

        if self.refine_abs:
            assert self.space_group is not None
            assert self.sites is not None

        self.load_peaks()
        self.refimenent()
        self.save_peaks()

    def load_peaks(self):
        self.peaks = Peaks("peaks", self.filename, None, self.point_group)
        self.peaks.load_peaks()

    def save_peaks(self):
        self.peaks.save_peaks()

    def apply_correction(self):
        if (np.array(self.parameters) > 0).all():
            AbsorptionCorrection(
                "peaks",
                self.formula,
                self.zparameter,
                u_vector=self.uvector,
                v_vector=self.vvector,
                params=self.parameters,
                filename=self.filename,
            )

    def refimenent(self):
        nuclear = NuclearStructureRefinement(
            self.cell, self.space_group, self.sites, self.filename
        )
        nuclear.refine(n_iter=10)
        nuclear.plot_result()
        nuclear.plot_sample_shape()
        nuclear.save_corrected_peaks()


if __name__ == "__main__":
    filename = sys.argv[1]

    params = load_YAML(filename)

    config = {}
    if params.get("Sample") is not None:
        config.update(params["Sample"])
    if params.get("Material") is not None:
        config.update(params["Material"])
    if params.get("Integration") is not None:
        config.update(params["Integration"])

    inst = Integration(params)
    config["Filename"] = inst.get_file()

    StructureAnalysis(config)
