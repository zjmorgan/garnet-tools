import os
import subprocess

from mantid.simpleapi import mtd
from mantid import config

config["Q.convention"] = "Crystallography"

import numpy as np

from garnet.reduction.data import DataModel
from garnet.reduction.peaks import PeaksModel, PeakModel
from garnet.reduction.plan import SubPlan
from garnet.config.instruments import beamlines

PARAMETRIZATION = os.path.abspath(__file__)
directory = os.path.dirname(PARAMETRIZATION)

filename = os.path.join(directory, "../utilities/sliceview.py")
SLICEVIEW = os.path.abspath(filename)

assert os.path.exists(SLICEVIEW)


class Parametrization(SubPlan):
    def __init__(self, plan):
        super(Parametrization, self).__init__(plan)

        self.params = plan["Parametrization"]
        self.output = plan["OutputName"] + "_parametrization"

        self.validate_params()

    def validate_params(self):
        self.check(
            len(self.params["Projections"]),
            "==",
            3,
            "Projections must have length 3",
        )
        det = np.isclose(
            np.linalg.det(np.array(self.params["Projections"], dtype=float)), 0
        )
        self.check(
            det, "is not", False, "Projections matrix must be invertible"
        )

        self.check(
            len(self.params["Bins"]), "==", 3, "Bins must have length 3"
        )
        for i, val in enumerate(self.params["Bins"]):
            self.check(
                isinstance(val, (int, np.integer)),
                "is",
                True,
                f"Bins[{i}] must be an integer",
            )
            self.check(val, ">", 0, f"Bins[{i}] must be > 0")
        self.check(
            int(np.prod(self.params["Bins"])),
            "<",
            101**3,
            "Too many bins: memory usage limit exceeded",
        )

        self.check(
            len(self.params["Extents"]), "==", 3, "Extents must have 3 ranges"
        )
        ext = np.array(self.params["Extents"], dtype=float)
        self.check(
            bool((np.diff(ext, axis=1) >= 0).all()),
            "is",
            True,
            "Each extent must satisfy min <= max",
        )

        if self.params.get("LogName") is None:
            self.params["LogName"] = "SequenceTime"
            self.params["LogExtents"] = [0, 1]
            self.params["LogBins"] = 0

        if self.params.get("MillerIndex") is not None:
            self.check(
                len(self.params["MillerIndex"]),
                "==",
                3,
                "MillerIndex must have length 3",
            )
            self.params["Projections"] = np.eye(3).tolist()

    @staticmethod
    def parametrize_parallel(plan, runs, proc):
        total = plan["Runs"]
        plan["Runs"] = runs
        plan["ProcName"] = "_p{}".format(proc)

        instance = Parametrization(plan)
        instance.proc = proc
        instance.total = total

        return instance.parametrize()

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

    def parametrize(self):
        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        runs = self.plan["Runs"]

        peaks = PeaksModel()

        projections = self.params["Projections"]
        extents = np.array(self.params["Extents"].copy()).tolist()

        if data.laue:
            self.run = 0
            self.runs = len(runs)

            for run in runs:
                self.run += 1

                data.load_data("data", self.plan["IPTS"], run, None)

                data.load_generate_normalization(
                    self.plan["VanadiumFile"], self.plan.get("FluxFile")
                )

                data.apply_calibration(
                    "data",
                    self.plan.get("DetectorCalibration"),
                    self.plan.get("TubeCalibration"),
                    self.plan.get("GoniometerCalibration"),
                )

                data.update_logs_for_time("data")

                log_vals, log_units = data.log_split_info(
                    "data",
                    self.params["LogName"],
                    self.params["LogExtents"],
                    self.params["LogBins"],
                )

                indices, workspaces = data.filter_events(
                    "data", run, self.total
                )

                status = "{}: {:}/{:}".format(self.proc, self.run, len(runs))

                ind = 0

                for index, workspace in zip(indices, workspaces):
                    ind += 1

                    progress = "{:}/{:}".format(ind, len(indices))

                    print(status + " " + progress)

                    data.apply_mask(workspace, self.plan.get("MaskFile"))

                    data.crop_for_normalization(workspace)

                    data.preprocess_detectors(workspace)

                    data.load_background(
                        self.plan.get("BackgroundFile"), workspace
                    )

                    data.group_pixels(workspace)

                    data.load_clear_UB(self.plan["UBFile"], workspace, run)

                    data.convert_to_Q_sample(
                        workspace, "md", lorentz_corr=False
                    )

                    hkl = self.params.get("MillerIndex")
                    if hkl is not None:
                        if not mtd.doesExist("peaks"):
                            peaks.create_peaks("md", "peaks")
                            peaks.add_peak("peaks", hkl)
                            peak = PeakModel("peaks")
                            proj_orig = peak.get_projection_peak_origin(0)
                            projections, origin = proj_orig
                            for i in range(3):
                                for j in range(2):
                                    extents[i][j] += origin[i]

                    data.normalize_to_hkl(
                        "md",
                        projections,
                        extents,
                        self.params["Bins"],
                        "-1",
                    )

                    data.combine_splits(
                        "md",
                        self.params["LogName"],
                        log_vals,
                        log_units,
                        index,
                        self.total,
                    )

        output_file = self.get_output_file()

        if mtd.doesExist("md"):
            UB_file = output_file.replace(".nxs", ".mat")
            data.save_UB(UB_file, "md")

            data_file = self.get_file(output_file, "data")
            norm_file = self.get_file(output_file, "norm")

            data.save_histograms(data_file, "md_data_split", sample_logs=True)
            data.save_histograms(norm_file, "md_norm_split", sample_logs=True)

        if mtd.doesExist("md_bkg_data_split") and mtd.doesExist(
            "md_bkg_norm_split"
        ):
            data_file = self.get_file(output_file, "bkg_data")
            norm_file = self.get_file(output_file, "bkg_norm")

            data.save_histograms(
                data_file, "md_bkg_data_split", sample_logs=True
            )
            data.save_histograms(
                norm_file, "md_bkg_norm_split", sample_logs=True
            )

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
            + self.log_name()
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
        print("_cc" if elastic else "")

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

    def log_name(self):
        """
        Log name.
        Spaces are removed and slashes are replaced with underscore.

        Returns
        -------
        log : str
            None or Hermann-Mauguin point group symbol.

        """

        log_name = self.params.get("LogName")
        log_bins = self.params.get("LogBins")
        log_extents = self.params.get("LogExtents")

        name = (
            ""
            if log_bins == 0
            else "_" + log_name.replace(" ", "").replace("/", "")
        )

        log_vals = [*log_extents, log_bins]

        bin_name = "" if log_bins == 0 else "_[{},{}]x{}".format(*log_vals)

        return name + bin_name

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

        if self.params.get("MillerIndex") is None:
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
        else:
            result = [str(val) for val in self.params["MillerIndex"]]

        proj = "_(" + ",".join(result) + ")"

        if self.params.get("MillerIndex") is None:
            proj += "_[" + ",".join(axes[2]) + "]"

        return proj

    @staticmethod
    def combine_parallel(plan, files):
        instance = Parametrization(plan)

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

            if os.path.exists(data_file) and os.path.exists(norm_file):
                data.load_histograms(data_file, "tmp_data")
                data.load_histograms(norm_file, "tmp_norm")

                data.combine_histograms("tmp_data", "data")
                data.combine_histograms("tmp_norm", "norm")

                os.remove(data_file)
                os.remove(norm_file)

            bkg_data_file = self.get_file(file, "bkg_data")
            bkg_norm_file = self.get_file(file, "bkg_norm")

            if os.path.exists(bkg_data_file) and os.path.exists(bkg_norm_file):
                data.load_histograms(bkg_data_file, "tmp_bkg_data")
                data.load_histograms(bkg_norm_file, "tmp_bkg_norm")

                data.combine_histograms("tmp_bkg_data", "bkg_data")
                data.combine_histograms("tmp_bkg_norm", "bkg_norm")

                os.remove(bkg_data_file)
                os.remove(bkg_norm_file)

        data_file = self.get_file(diag_file, "data")
        norm_file = self.get_file(diag_file, "norm")
        result_file = self.get_file(output_file, "")

        data.divide_histograms("result", "data", "norm")

        UB_file = file.replace(".nxs", ".mat")

        for ws in ["data", "norm", "result"]:
            if os.path.exists(UB_file):
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
