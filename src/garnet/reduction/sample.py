import os
import re

import numpy as np
import pyvista as pv

from mantid.simpleapi import CreateSampleWorkspace, LoadIsawUB, mtd


class SampleMaterial:
    def __init__(self, plan, filename):
        CreateSampleWorkspace(OutputWorkspace="sample")
        LoadIsawUB(InputWorkspace="sample", Filename=plan["UBFile"])

        sample = plan.get("Sample", {})
        self.set_sample(sample)

        material = plan.get("Material", {})
        self.set_material(material)

        self.set_shape(filename)

        self.sample = True if plan.get("Sample") is not None else False
        self.material = True if plan.get("Material") is not None else False

    def set_sample(self, sample):
        params = sample.get("ThicknessWidthHeight")
        u_vector = sample.get("IndexAlongThickness")
        v_vector = sample.get("IndexTangentHeight")

        if params is None:
            thickness, width, height = 0.001, 0.001, 0.001
            u_vector, v_vector = [0, 0, 1], [1, 0, 0]
        elif u_vector is None and v_vector is None:
            if type(params) is list:
                assert len(params) == 3
                diameter = np.mean(params)
            else:
                diameter = params
            thickness, width, height = diameter, diameter, diameter
            u_vector, v_vector = [0, 0, 1], [1, 0, 0]
        else:
            assert len(u_vector) == 3, "Invalid u-vector"
            assert len(v_vector) == 3, "Invalid v-vector"
            thickness, width, height = params

        assert not np.isclose(
            np.cross(u_vector, v_vector), 0
        ).all(), "Collinear indexing"

        self.u_vector = u_vector
        self.v_vector = v_vector

        self.thickness = thickness
        self.width = width
        self.height = height

    def set_material(self, material):
        chemical_formula = material.get("ChemicalFormula", "Si")
        z_parameter = material.get("ZParameter", 8)

        volume = mtd["sample"].sample().getOrientedLattice().volume()

        assert volume > 0

        self.volume = volume

        assert self.verify_chemical_formula(chemical_formula)

        self.chemical_formula = chemical_formula

        assert z_parameter > 0
        self.z_parameter = z_parameter

    def verify_chemical_formula(self, formula):
        pattern = (
            r"(?:\((?:[A-Z][a-z]?\d+)\)|[A-Z][a-z]?)(?:\d+(?:\.\d+)?|\.\d+)?"
        )

        parts = re.split(r"[-\s]+", formula.strip())

        return all(re.fullmatch(pattern, part) for part in parts)

    def save_ellipsoid_stl(self, filename="/tmp/ellipsoid.stl"):
        sph = pv.Icosphere(radius=0.5)
        params = [self.width, self.height, self.thickness]
        ell = sph.scale(params, inplace=False)
        ell.save(filename)

    def set_shape(self, filename):
        self.UB = mtd["sample"].sample().getOrientedLattice().getUB().copy()

        u = np.dot(self.UB, self.u_vector)
        v = np.dot(self.UB, self.v_vector)

        u /= np.linalg.norm(u)

        w = np.cross(u, v)
        w /= np.linalg.norm(w)

        v = np.cross(w, u)

        T = np.column_stack([v, w, u])

        gon = mtd["sample"].run().getGoniometer()

        gon.setR(T)
        self.gamma, self.beta, self.alpha = gon.getEulerAngles("ZYX")

        self.shapestl = os.path.splitext(filename)[0] + ".stl"
        self.save_ellipsoid_stl(self.shapestl)

    def get_material(self):
        return {
            "ChemicalFormula": self.chemical_formula,
            "ZParameter": float(self.z_parameter),
            "UnitCellVolume": self.volume,
        }

    def get_shape(self):
        return self.shapestl, [self.alpha, self.beta, self.gamma]

    def get_sample_material(self):
        if self.material is not None and self.sample is not None:
            return *self.get_shape(), self.get_material()
        else:
            return None, None, None
