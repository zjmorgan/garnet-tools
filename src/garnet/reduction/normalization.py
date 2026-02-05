import os
import subprocess

from mantid.simpleapi import mtd
from mantid import config

config["Q.convention"] = "Crystallography"

import numpy as np

from garnet.plots.base import Pages
from garnet.plots.volume import SlicePlot
from garnet.reduction.data import DataModel
from garnet.reduction.plan import SubPlan

# from garnet.reduction.sample import SampleMaterial
from garnet.reduction.crystallography import space_point, point_laue
from garnet.config.instruments import beamlines

NORMALIZATION = os.path.abspath(__file__)
directory = os.path.dirname(NORMALIZATION)

filename = os.path.join(directory, "../utilities/sliceview.py")
SLICEVIEW = os.path.abspath(filename)

assert os.path.exists(SLICEVIEW)


class Normalization(SubPlan):
    def __init__(self, plan):
        super(Normalization, self).__init__(plan)

        self.params = plan["Normalization"]
        self.output = plan["OutputName"] + "_normalization"

        self.validate_params()

    def validate_params(self):
        symbols = list(space_point.keys()) + list(point_laue.keys())

        if self.params.get("Symmetry") is not None:
            symmetry = self.params["Symmetry"].replace(" ", "")
            if "x" in symmetry and "y" in symmetry and "z" in symmetry:
                ops = [symm.split(",") for symm in symmetry.split(";")]
                for op in ops:
                    self.check(
                        len(op), "==", 3, "Operators must have length 3"
                    )
            else:
                self.check(
                    symmetry, "in", symbols, f"Unknown Symmetry: {symmetry}"
                )
                if symmetry in space_point:
                    symmetry = space_point[symmetry]
                self.check(
                    symmetry,
                    "in",
                    point_laue.keys(),
                    f"Unknown point group: {symmetry}",
                )
                symmetry = point_laue[symmetry]
            self.params["Symmetry"] = symmetry

        self.check(
            len(self.params["Projections"]),
            "==",
            3,
            "Projections must have length 3",
        )
        det = np.isclose(np.linalg.det(self.params["Projections"]), 0)
        self.check(
            det, "is not", False, "Projections matrix must be invertible"
        )

        self.check(
            len(self.params["Bins"]), "==", 3, "Bins must have length 3"
        )
        for i, val in enumerate(self.params["Bins"]):
            self.check(
                val,
                lambda v: isinstance(v, (int, np.integer)),
                f"Bins[{i}] must be an integer",
            )
            self.check(val, ">", 0, f"Bins[{i}] must be > 0")
        self.check(
            int(np.prod(self.params["Bins"])),
            "<",
            1001**3,
            "Too many bins: memory usage limit exceeded",
        )

        self.check(
            len(self.params["Extents"]), "==", 3, "Extents must have 3 ranges"
        )
        ext_ok = (
            np.diff(np.array(self.params["Extents"], dtype=float), axis=1) >= 0
        ).all()
        self.check(
            bool(ext_ok), "is", True, "Each extent must satisfy min <= max"
        )

    @staticmethod
    def normalize_parallel(plan, runs, proc):
        plan["Runs"] = runs
        plan["ProcName"] = "_p{}".format(proc)

        instance = Normalization(plan)
        instance.proc = proc

        return instance.normalize()

    def view(self, result_file):
        try:
            process = subprocess.Popen(
                ["python", SLICEVIEW, result_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = process.communicate()
            if process.returncode == 0:
                print("First command succeeded:", out.decode().strip())
            else:
                raise subprocess.SubprocessError(err.decode().strip())
        except (FileNotFoundError, subprocess.SubprocessError):
            subprocess.Popen(["mantidpython", SLICEVIEW, result_file])

    def normalize(self):
        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        runs = self.plan["Runs"]

        self.run = 0
        self.runs = len(runs)

        output_file = self.get_output_file()

        # sm = SampleMaterial(self.plan, output_file)

        for run in runs:
            self.run += 1

            print("{}: {:}/{:}".format(self.proc, self.run, len(runs)))

            data.load_data(
                "data",
                self.plan["IPTS"],
                run,
                self.plan.get("Grouping"),
                self.plan.get("TimeCut"),
            )

            data.load_generate_normalization(
                self.plan["VanadiumFile"], self.plan.get("FluxFile")
            )

            data.apply_calibration(
                "data",
                self.plan.get("DetectorCalibration"),
                self.plan.get("TubeCalibration"),
                self.plan.get("GoniometerCalibration"),
            )

            data.preprocess_detectors("data")

            data.apply_mask("data", self.plan.get("MaskFile"))

            data.crop_for_normalization("data")

            data.load_background(
                self.plan.get("BackgroundFile"),
                "data",
                self.plan.get("DetectorCalibration"),
                self.plan.get("TubeCalibration"),
            )

            data.group_pixels("data")

            data.load_clear_UB(self.plan["UBFile"], "data", run)

            # data.absorption_correction("data", *sm.get_sample_material())

            data.convert_to_Q_sample("data", "md", lorentz_corr=False)

            data.normalize_to_hkl(
                "md",
                self.params["Projections"],
                self.params["Extents"],
                self.params["Bins"],
                symmetry=self.params.get("Symmetry"),
            )

        UB_file = output_file.replace(".nxs", ".mat")
        data.save_UB(UB_file, "md")

        data_file = self.get_file(output_file, "data")
        norm_file = self.get_file(output_file, "norm")

        data.save_histograms(data_file, "md_data")
        data.save_histograms(norm_file, "md_norm")

        if mtd.doesExist("md_bkg_data") and mtd.doesExist("md_bkg_norm"):
            data_file = self.get_file(output_file, "bkg_data")
            norm_file = self.get_file(output_file, "bkg_norm")

            data.save_histograms(data_file, "md_bkg_data")
            data.save_histograms(norm_file, "md_bkg_norm")

        mtd.clear()

        return output_file

    def get_file(self, file, ws=""):
        """
        Update filename with identifier name and optional workspace name.

        Parameters
        ----------
        file : str
            Original file name.
        ws : str, optional
            Name of workspace. The default is ''.

        Returns
        -------
        output_file : str
            File with updated name for identifier and workspace name.

        """

        if len(ws) > 0:
            ws = "_" + ws

        fname = self.append_name(file)
        return fname.replace("_p0", "").replace(".nxs", ws + ".nxs")

    def append_name(self, file):
        """
        Update filename with identifier name.

        Parameters
        ----------
        file : str
            Original file name.

        Returns
        -------
        output_file : str
            File with updated name for identifier name.

        """

        append = (
            self.projection_name()
            + self.extents_name()
            + self.binning_name()
            + self.symmetry_name()
            + self.elastic_name()
        )

        name, ext = os.path.splitext(file)

        return name + append + ext

    def elastic_name(self):
        """
        Elastic channel.

        Returns
        -------
        cc : str
           Total or elastic channel.

        """

        elastic = self.plan.get("Elastic")

        return "_cc" if elastic else ""

    def extents_name(self):
        """
        Min/max pairs for each dimensional extents.

        `_[min_0,max_0]_[min_1,max_1]_[min_2,max_2]`

        Returns
        -------
        extents : str
            Underscore separated list.

        """

        extents = self.params.get("Extents")

        return "".join(["_[{},{}]".format(*extent) for extent in extents])

    def binning_name(self):
        """
        Bin size for each dimension.

        `_N0xN1xN2`

        Returns
        -------
        bins : str
            Cross separated integers.

        """

        bins = self.params.get("Bins")

        return "_" + "x".join(np.array(bins).astype(str).tolist())

    def symmetry_name(self):
        """
        Laue group name.
        Spaces are removed and slashes are replaced with underscore.

        Returns
        -------
        symmetry : str
            None or Hermann-Mauguin point group symbol.

        """

        symmetry = self.params.get("Symmetry")

        name = "" if symmetry is None else "_" + symmetry.replace(" ", "")

        return name.replace("/", "_")

    def projection_name(self):
        """
        Axes projections.

        Returns
        -------
        proj : str
            Name of slices.

        """

        W = np.column_stack(self.params["Projections"])

        char_dict = {0: "0", 1: "{1}", -1: "-{1}"}

        chars = ["h", "k", "l"]

        axes = []
        for j in [0, 1, 2]:
            axis = []
            for w in W[:, j]:
                char = chars[np.argmax(W[:, j])]
                axis.append(char_dict.get(w, "{0}{1}").format(w, char))
            axes.append(axis)

        result = []
        for item0, item1 in zip(axes[0], axes[1]):
            if item0 == "0":
                result.append(item1)
            elif item1 == "0":
                result.append(item0)
            elif "-" in item1:
                result.append(item0 + item1)
            else:
                result.append(item0 + "+" + item1)

        proj = "_(" + ",".join(result) + ")" + "_[" + ",".join(axes[2]) + "]"

        return proj

    @staticmethod
    def combine_parallel(plan, files):
        instance = Normalization(plan)

        return instance.combine(files)

    def combine(self, files):
        """
        Merge data and normalization files.

        Parameters
        ----------
        files : list
            Files to be combined.

        """

        output_file = self.get_output_file()
        diag_file = self.get_diagnostic_file("volume")

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        for ind, file in enumerate(files):
            data_file = self.get_file(file, "data")
            norm_file = self.get_file(file, "norm")

            data.load_histograms(data_file, "tmp_data")
            data.load_histograms(norm_file, "tmp_norm")

            data.combine_histograms("tmp_data", "data")
            data.combine_histograms("tmp_norm", "norm")

            bkg_data_file = self.get_file(file, "bkg_data")
            bkg_norm_file = self.get_file(file, "bkg_norm")

            if os.path.exists(bkg_data_file) and os.path.exists(bkg_norm_file):
                data.load_histograms(bkg_data_file, "tmp_bkg_data")
                data.load_histograms(bkg_norm_file, "tmp_bkg_norm")

                data.combine_histograms("tmp_bkg_data", "bkg_data")
                data.combine_histograms("tmp_bkg_norm", "bkg_norm")

                os.remove(bkg_data_file)
                os.remove(bkg_norm_file)

            os.remove(data_file)
            os.remove(norm_file)

        data_file = self.get_file(diag_file, "data")
        norm_file = self.get_file(diag_file, "norm")
        result_file = self.get_file(output_file, "")

        data.divide_histograms("result", "data", "norm")

        UB_file = file.replace(".nxs", ".mat")

        for ws in ["data", "norm", "result"]:
            data.add_UBW(ws, UB_file, self.params["Projections"])

        for ind, file in enumerate(files):
            UB_file = file.replace(".nxs", ".mat")
            if os.path.exists(UB_file):
                os.remove(UB_file)
            STL_file = file.replace(".nxs", ".stl")
            if os.path.exists(STL_file):
                os.remove(STL_file)

        data.save_histograms(data_file, "data", sample_logs=True)
        data.save_histograms(norm_file, "norm", sample_logs=True)
        data.save_histograms(result_file, "result", sample_logs=True)

        signal, error, *_ = data.extract_bin_info("result")
        UB, W, titles, axes = data.extract_axis_info("result")

        for i, vals in enumerate(axes):
            norm = np.zeros(3, dtype=int)
            norm[i] = 1

            plot_name = "slice_{}".format(titles[i].replace(" ", ""))
            pdf = Pages(self.get_plot_file(plot_name.replace(".", ""), ".pdf"))

            plot = SlicePlot(UB, W)
            plot.calculate_transforms(axes, titles, norm)

            for val in vals:
                if np.isclose(np.round(val, 4) % 1, 0):
                    plot.make_slice(signal, val)
                    pdf.add_plot(plot.fig)

                    # name = plot.z_label.replace('.','')+'={:.3f}'.format(val)
                    # plot_name = 'slice_{}'.format(name.replace(' ', ''))
                    # plot.save_plot(self.get_plot_file(plot_name))

            pdf.close()

        view_file = result_file

        if mtd.doesExist("bkg_data") and mtd.doesExist("bkg_norm"):
            data_file = self.get_file(diag_file, "bkg_data")
            norm_file = self.get_file(diag_file, "bkg_norm")

            data.save_histograms(data_file, "bkg_data", sample_logs=True)
            data.save_histograms(norm_file, "bkg_norm", sample_logs=True)

            bkg_output_file = self.get_file(output_file, "bkg")

            data.divide_histograms("bkg_result", "bkg_data", "bkg_norm")
            data.save_histograms(bkg_output_file, "bkg_result")

            data.subtract_histograms("sub", "result", "bkg_result")

            sub_output_file = self.get_file(output_file, "sub_bkg")
            data.save_histograms(sub_output_file, "sub", sample_logs=True)

            view_file = sub_output_file

        self.view(view_file)
