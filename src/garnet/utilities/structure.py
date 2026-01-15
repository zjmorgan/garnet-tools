import numpy as np

from mantid.simpleapi import CreateSampleWorkspace, LoadCIF, mtd

from garnet.reduction.plan import load_YAML, save_YAML
from garnet.utilities.reflections import AbsorptionCorrection, PrunePeaks
from garnet.utilities.refinement import NuclearStructureRefinement


class CrystalStructure:
    def __init__(self, filename):
        CreateSampleWorkspace(OutputWorkspace="crystal")
        LoadCIF(Workspace="crystal", InputFile=filename)

    def get_space_group(self):
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        return cryst_struct.getSpaceGroup().getHMSymbol().strip()

    def get_lattice_constants(self):
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        uc = cryst_struct.getUnitCell()

        params = uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

        return params

    def get_unit_cell_volume(self):
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        return cryst_struct.getUnitCell().volume()

    def get_scatterers(self):
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        scatterers = cryst_struct.getScatterers()

        scatterers = [atm.split(" ") for atm in list(scatterers)]

        scatterers = [
            [val if val.isalpha() else float(val) for val in scatterer[:-1]]
            for scatterer in scatterers
        ]

        return scatterers

    def get_chemical_formula_z_parameter(self):
        cryst_struct = mtd["crystal"].sample().getCrystalStructure()

        sg = cryst_struct.getSpaceGroup()

        scatterers = self.get_scatterers()

        atom_dict = {}

        for scatterer in scatterers:
            atom, x, y, z, occ = scatterer
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
                chemical_formula.append(key + "{:.3g}")
            else:
                chemical_formula.append("(" + key + ")" + "{:.3g}")

        Z = np.gcd.reduce(n_atm)
        n = np.divide(n_wgt, Z)

        chemical_formula = "-".join(chemical_formula).format(*n)

        return chemical_formula, float(Z)

    def add_material_info(self, filename):
        params = load_YAML(filename)

        chemical_formula, Z = self.get_chemical_formula_z_parameter()

        material = {}
        material["ChemicalFormula"] = chemical_formula
        material["ZParameter"] = Z
        material["SpaceGroup"] = self.get_space_group()
        material["Sites"] = self.get_scatterers()

        params["Material"] = material

        save_YAML(params, filename)


class SampleParameters:
    def __init__(self, shape="sphere", u_vector=[0, 0, 1], v_vector=[1, 0, 0]):
        self.set_sample_shape(shape)

        self.u_vector = u_vector
        self.v_vector = v_vector

    def set_sample_shape(self, shape):
        if shape.lower() == "sphere":
            self.shape = "Sphere"
        elif shape.lower() == "cylinder":
            self.shape = "Cylinder"
        else:
            self.shape = "FlatPlate"

    def add_sample_info(self, filename):
        params = load_YAML(filename)

        sample = {}
        sample["Shape"] = self.shape
        sample["DepthWidthHeight"] = [0.1, 0.1, 0.1]
        sample["IndexAlongDepth"] = self.u_vector
        sample["IndexTangentHeight"] = self.v_vector

        params["Sample"] = sample

        save_YAML(params, filename)


class StructureAnalysis:
    def __init__(self, config):
        defaults = {
            "Filename": None,
            "ExtinctionModel": "Type II",
            "RefineAbsorption": False,
            "SpaceGroup": "I a -3 d",
            "ChemicalFormula": "Yb3-Al5-O12",
        }

        defaults.update(config)

        self.space_group = defaults.get("SpaceGroup")
        self.sites = defaults.get("Sites")

        self.refine_off = defaults.get("RefineAbsorption")

    def refimenent(self):
        nuclear = NuclearStructureRefinement(
            cell, space_group, sites, filename
        )
        nuclear.refine(n_iter=25)
        nuclear.plot_result()
        nuclear.plot_sample_shape()
        nuclear.save_corrected_peaks()
