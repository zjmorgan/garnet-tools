import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import numpy as np

from garnet.utilities.reflections import AbsorptionCorrection, Peaks
from garnet.utilities.refinement import NuclearStructureRefinement
from garnet.utilities.macromolecular import Macromolecular

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.integration import Integration
from garnet.reduction.crystallography import (
    space_groups,
    space_point,
    mantid_to_gemmi,
)


class StructureAnalysis:
    def __init__(self, config, filename):
        defaults = {
            "ExtinctionModel": "SHELX",
            "ChemicalFormula": "Yb3-Al5-O12",
            "ZParameter": 8,
            "UVector": [0, 0, 1],
            "VVector": [1, 0, 0],
            "ThicknessWidthHeight": [0, 0, 0],
        }

        defaults.update(config)

        assert os.path.exists(filename)

        self.filename = filename

        space_group = defaults.get("SpaceGroup")
        point_group = None
        if space_group is not None:
            space_group = space_groups.get(space_group)
            point_group = space_point.get(space_group)

        self.space_group = space_group
        self.point_group = point_group

        self.sites = defaults.get("Sites")
        self.ext_model = defaults.get("ExtinctionModel")

        self.refine_abs = False
        if self.space_group is not None and self.sites is not None:
            self.refine_abs = True

        self.chemical_formula = defaults.get("ChemicalFormula")
        self.z = defaults.get("ZParameter")
        self.uvector = defaults.get("UVector")
        self.vvector = defaults.get("VVector")

        parameters = defaults.get("ThicknessWidthHeight")
        if type(parameters) is not list:
            parameters = [parameters] * 3
        self.parameters = parameters

        self.load_peaks()

        apply_corr = False
        if not self.refine_abs:
            apply_corr = (np.array(self.parameters) > 0).all()

        if apply_corr:
            self.apply_correction()

        if self.sites is not None and self.space_group is not None:
            self.refimenent()

            # if self.refine_abs:
            #     self.apply_correction()

        self.save_peaks()

    def save_mtz(self):
        if self.space_group is not None:
            filename = os.path.splitext(self.filename)[0] + ".mtz"
            sg = mantid_to_gemmi.get(self.space_group)
            if sg is not None:
                mm = Macromolecular("peaks")
                mm.write_mtz(filename, space_group=sg)
            else:
                print("Invalid MTZ space group")
        else:
            print("Provide space group for MTZ")

    def load_peaks(self):
        self.peaks = Peaks("peaks", self.filename, None, self.point_group)
        self.peaks.load_peaks()
        self.cell = self.peaks.get_cell()

    def save_peaks(self):
        self.peaks.save_peaks()

    def apply_correction(self):
        if (np.array(self.parameters) > 0).all():
            AbsorptionCorrection(
                "peaks",
                self.chemical_formula,
                self.z,
                u_vector=self.uvector,
                v_vector=self.vvector,
                params=self.parameters,
                filename=self.filename,
            )

    def refimenent(self, n_iter=20):
        nuclear = NuclearStructureRefinement(
            self.cell,
            self.space_group,
            self.sites,
            self.filename,
            self.parameters,
        )
        nuclear.extract_info()
        nuclear.refine(
            n_iter=n_iter,
            abs_corr=self.refine_abs,
            ext_model=self.ext_model,
        )
        nuclear.plot_result()
        nuclear.plot_sample_shape()

        self.uvector = nuclear.uvector
        self.vvector = nuclear.vvector
        self.parameters = nuclear.parameters
        self.chemical_formula, self.z = nuclear.chemical_formula_z_parameter()


if __name__ == "__main__":
    filename = sys.argv[1]

    rp = ReductionPlan()
    rp.load_plan(filename)
    params = rp.plan

    config = {}
    if params.get("Sample") is not None:
        config.update(params["Sample"])
    if params.get("Material") is not None:
        config.update(params["Material"])
    if params.get("Integration") is not None:
        config.update(params["Integration"])

    inst = Integration(params)

    StructureAnalysis(config, inst.get_file(inst.get_output_file(), ""))
