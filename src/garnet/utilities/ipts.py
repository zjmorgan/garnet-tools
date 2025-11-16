import pyoncat

import numpy as np

import sys
import traceback

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QGridLayout,
    QVBoxLayout,
    QComboBox,
    QAbstractItemView,
)
from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from garnet.config.instruments import beamlines


class View(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        login_label = QLabel("ONCat Login")
        user_label = QLabel("Username: ")
        pass_label = QLabel("Password: ")
        self.user_line = QLineEdit()
        self.pass_line = QLineEdit()
        self.pass_line.setEchoMode(QLineEdit.Password)
        self.login_button = QPushButton("Sign In")
        self.refresh_button = QPushButton("Refresh")
        self.message_label = QLabel("Not Signed In")
        self.message_label.setStyleSheet("color: red;")
        self.message_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.layout.addWidget(login_label, 0, 0)
        self.layout.addWidget(user_label, 0, 1)
        self.layout.addWidget(self.user_line, 0, 2, 1, 3)
        self.layout.addWidget(pass_label, 0, 5)
        self.layout.addWidget(self.pass_line, 0, 6, 1, 1)
        self.layout.addWidget(self.login_button, 0, 8)
        self.layout.addWidget(self.message_label, 0, 7)
        self.layout.setColumnMinimumWidth(7, 300)

        instrument_cbox_label = QLabel("Instrument: ")
        self.instrument_cbox = QComboBox(self)
        instruments = ["TOPAZ", "MANDI", "CORELLI", "SNAP", "WAND²", "DEMAND"]
        self.instrument_cbox.addItems(instruments)
        self.layout.addWidget(instrument_cbox_label, 1, 0)
        self.layout.addWidget(self.instrument_cbox, 1, 1, 1, 8)

        ipts_label = QLabel("IPTS: ")
        self.ipts_field = QComboBox(self)
        self.layout.addWidget(ipts_label, 2, 0)
        self.layout.addWidget(self.ipts_field, 2, 1, 1, 7)

        self.name_list = QListWidget()
        self.name_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.layout.addWidget(self.name_list, 3, 0, 3, 4)

        self.runs_label = QLabel("Run Numbers: ")
        self.runs_list = QLineEdit()

        self.exp_cbox = QComboBox(self)
        self.exp_cbox.setEnabled(False)

        self.layout.addWidget(self.runs_label, 7, 0)
        self.layout.addWidget(self.runs_list, 7, 1, 1, 7)
        self.layout.addWidget(self.refresh_button, 7, 8)
        self.layout.addWidget(self.exp_cbox, 2, 8)

        self.plot = FigureCanvas(Figure(figsize=(8, 6)))
        self.layout.addWidget(self.plot, 3, 4, 2, 5)
        self.layout.addWidget(NavigationToolbar(self.plot, self), 5, 4, 1, 5)

    def ipts_entered(self):
        if self.ipts_field.hasAcceptableInput():
            self.update()

    def get_instrument(self):
        return self.instrument_cbox.currentText()

    def get_name(self):
        return self.name_list.selectedItems()

    def get_ipts(self):
        return self.ipts_field.currentText()

    def get_runs(self):
        return self.runs_list.text()

    def get_experiment(self):
        return self.exp_cbox.currentText()

    def connect_ipts(self, update):
        self.ipts_field.activated.connect(update)

    def connect_switch_instrument(self, update):
        self.instrument_cbox.activated.connect(update)

    def connect_select_name(self, update):
        self.name_list.itemSelectionChanged.connect(update)
        self.name_list.itemClicked.connect(update)

    def connect_adjust_runs(self, update):
        self.runs_list.editingFinished.connect(update)

    def connect_login_button(self, update):
        self.login_button.clicked.connect(update)
        self.pass_line.returnPressed.connect(update)

    def connect_select_experiment(self, update):
        self.exp_cbox.activated.connect(update)

    def connect_refresh_button(self, update):
        self.refresh_button.clicked.connect(update)


class Presenter:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_select_name(self.select_name)
        self.view.connect_ipts(self.set_ipts)
        self.view.connect_adjust_runs(self.adjust_runs_list)
        self.view.connect_login_button(self.sign_in)
        self.view.connect_select_experiment(self.set_exp)
        self.view.connect_refresh_button(self.refresh)

        self.switch_instrument()
        self.login = None
        self.data_files = None

    def switch_instrument(self):
        instrument = self.view.get_instrument()
        self.view.ipts_field.clear()
        self.clear()

        if instrument == "DEMAND":
            self.view.exp_cbox.setEnabled(True)
            self.view.runs_label.setText("Scan Numbers:")
        else:
            self.view.exp_cbox.setEnabled(False)
            self.view.runs_label.setText("Run Numbers: ")

        inst_params = self.model.beamline_info(instrument)

        try:
            available_runs = self.model.list_available(self.login, inst_params)
            self.view.ipts_field.addItems(available_runs)
        except AttributeError:
            self.view.message_label.setText("Not Signed In")
            self.view.message_label.setStyleSheet("color: red;")

        self.inst_params = inst_params

    def set_ipts(self):
        ipts = self.view.get_ipts()
        self.clear()
        try:
            self.data_files = self.model.retrieve_data_files(
                self.login, self.inst_params, ipts
            )
            self.model.set_experiments(self.view.exp_cbox, self.data_files)
            self.names = self.model.run_title_dictionary(
                self.data_files, self.inst_params
            )
            if self.view.get_instrument() != "DEMAND":
                self.view.name_list.addItems(list(self.names.keys()))
            else:
                self.set_exp()

        except AttributeError:
            self.view.message_label.setText("Not Signed In")
            self.view.message_label.setStyleSheet("color: red;")
        except pyoncat.InvalidRefreshTokenError:
            self.view.message_label.setText("Login Expired")
            self.view.message_label.setStyleSheet("color: orange;")

    def refresh(self):
        self.set_ipts()

    def set_exp(self):
        self.data_files = self.model.retrieve_data_files(
            self.login, self.inst_params, self.view.get_ipts()
        )
        data_files = self.data_files
        exp = self.view.get_experiment()

        self.clear()
        self.model.set_experiments(self.view.exp_cbox, self.data_files)
        self.view.exp_cbox.setCurrentText(exp)

        mask = np.array([f"exp{exp}" in df["id"] for df in data_files])

        dfs = np.array(data_files)[mask]
        self.data_files = list(dfs)

        self.names = self.model.run_title_dictionary(
            self.data_files, self.inst_params
        )

        self.view.name_list.addItems(list(self.names.keys()))

    def select_name(self):
        names = self.view.get_name()
        runs = []
        run_numbers_list = []
        data_indices = []
        for name in names:
            runs.append(self.names[name.text()])

            rrun_numbers_list, ddata_indices = self.model.run_numbers_indices(
                name.text(), self.data_files, self.names, self.inst_params
            )
            for r in rrun_numbers_list:
                run_numbers_list.append(r)
            for d in ddata_indices:
                data_indices.append(d)

        if len(names) > 0:
            rnl = run_numbers_list[:]
            rnl.sort()
            run_seq = np.split(rnl, np.where(np.diff(rnl) > 1)[0] + 1)
            rs = ",".join(
                [
                    str(s[0]) + ":" + str(s[-1]) if len(s) - 1 else str(s[0])
                    for s in run_seq
                ]
            )

            self.runs = rs
        else:
            self.runs = ",".join(runs)

        self.view.runs_list.setText(self.runs)

        self.adjust_runs_list()

    def adjust_runs_list(self):
        rs = self.view.get_runs()
        try:
            runs_list = self.model.run_numbers_list(rs)
        except:
            # print('Invalid run numbers')
            return

        if self.data_files is None:
            return

        run_numbers_list, data_indices = self.model.run_numbers_indices_1(
            self.data_files, runs_list, self.inst_params
        )

        if len(run_numbers_list) == 0:
            pass
            # self.view.plot.figure.clf()
            # self.view.plot.figure.canvas.draw()
        else:
            gonio_values, gonio_names = self.model.goniometer_values(
                self.data_files, data_indices, self.inst_params
            )
            scale_values = self.model.scale_values(
                self.data_files, data_indices, self.inst_params
            )

            self.plot(
                gonio_values, gonio_names, run_numbers_list, scale_values
            )

    def clear(self):
        self.view.name_list.clear()
        self.view.exp_cbox.clear()
        self.view.runs_list.setText("")
        self.view.plot.figure.clf()
        self.view.plot.figure.canvas.draw()

    def plot(self, gonio_values, gonio_names, run_numbers_list, scale_values):
        self.view.plot.figure.clf()
        self.view.plot.figure.canvas.draw()
        if self.view.get_ipts() == "":
            return

        self.view.plot.figure.subplots_adjust(wspace=0.1, top=0.85, bottom=0.1)
        colors = ["C0", "C1", "C2", "C4"]

        if len(self.model.subplot_limits) == 1:
            ax1 = self.view.plot.figure.subplots()
            if self.view.get_instrument() != "DEMAND":
                for val, lab, c in zip(gonio_values, gonio_names, colors):
                    ax1.plot(run_numbers_list, val, ".", color=c, label=lab)
            else:
                for val, lab, c in zip(gonio_values, gonio_names, colors):
                    v = val[:]
                    for ii in range(len(run_numbers_list)):
                        if ii == 0:
                            ax1.errorbar(
                                run_numbers_list[ii],
                                v[0][ii],
                                yerr=np.array(
                                    [
                                        abs(v[1][ii] - v[0][ii]),
                                        abs(v[2][ii] - v[0][ii]),
                                    ]
                                ),
                                fmt=".",
                                color=c,
                                label=lab,
                                elinewidth=0.5,
                                capsize=2,
                            )
                        else:
                            ax1.errorbar(
                                run_numbers_list[ii],
                                v[0][ii],
                                yerr=np.array(
                                    [
                                        abs(v[1][ii] - v[0][ii]),
                                        abs(v[2][ii] - v[0][ii]),
                                    ]
                                ),
                                fmt=".",
                                color=c,
                                elinewidth=0.5,
                                capsize=2,
                            )

            ax1.set_ylabel("Goniometer Values (degrees)")
            if self.view.get_instrument() != "DEMAND":
                ax1.set_xlabel("Run Number")
            else:
                ax1.set_xlabel("Scan Number")
                ax1.set_title(f"exp{self.view.get_experiment()}")
            ax1.set_xlim(
                self.model.subplot_limits[0][0] - 1,
                self.model.subplot_limits[0][1] + 1,
            )
            ax1.legend(
                fontsize="x-small", loc="upper left", bbox_to_anchor=(0, 1.2)
            )
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.ticklabel_format(style="plain", axis="x", useOffset=False)

            ax2 = ax1.twinx()
            color = "r"
            ax2.set_ylabel(
                f'Scale ({self.inst_params["Scale"].split(".")[-1]})',
                color=color,
            )
            ax2.plot(run_numbers_list, scale_values, ".", color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.set_ylim(
                -0.1 * np.max(scale_values), np.max(scale_values) * 1.1
            )

        else:
            axs = self.view.plot.figure.subplots(
                1,
                len(self.model.subplot_limits),
                sharey=True,
                width_ratios=[
                    l[1] - l[0] + 2 for l in self.model.subplot_limits
                ],
            )

            if self.view.get_instrument() != "DEMAND":
                self.view.plot.figure.supxlabel(
                    "Run Number", fontsize="medium"
                )
            else:
                self.view.plot.figure.supxlabel(
                    "Scan Number", fontsize="medium"
                )
                self.view.plot.figure.suptitle(
                    f"exp{self.view.get_experiment()}", fontsize="medium"
                )
            spacing = len(run_numbers_list) // 6 + 1

            d = 0.5
            kwargs = dict(
                marker=[(-d, -1), (d, 1)],
                markersize=12,
                linestyle="none",
                color="k",
                mec="k",
                mew=1,
                clip_on=False,
            )

            for i, ax in enumerate(axs):
                lim = self.model.subplot_limits[i]
                lim_range = 1  # (lim[1]-lim[0] + 1)*0.2
                ax.set_xlim(lim[0] - lim_range, lim[1] + lim_range)
                ax.set_ylim(
                    np.min(gonio_values) - 10, np.max(gonio_values) + 10
                )
                if lim[0] != lim[1]:
                    xt = np.arange(lim[0], lim[1] + 1)
                    mask = xt % spacing == 0
                    if len(xt[mask]) == 0:
                        ax.set_xticks([lim[0]])
                    else:
                        ax.set_xticks(xt[mask])
                else:
                    ax.set_xticks([lim[0]])
                # ax.set_aspect(0.5)
                # ax.tick_params(axis='x',labelrotation=15)

                if i == 0:
                    ax.set_ylabel("Goniometer Values (degrees)")
                    if self.view.get_instrument() != "DEMAND":
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            ax.plot(
                                run_numbers_list, val, ".", color=c, label=lab
                            )
                    else:
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            v = val[:]
                            for ii in range(len(run_numbers_list)):
                                if ii == 0:
                                    ax.errorbar(
                                        run_numbers_list[ii],
                                        v[0][ii],
                                        yerr=np.array(
                                            [
                                                abs(v[1][ii] - v[0][ii]),
                                                abs(v[2][ii] - v[0][ii]),
                                            ]
                                        ),
                                        fmt=".",
                                        color=c,
                                        label=lab,
                                        elinewidth=0.5,
                                        capsize=2,
                                    )
                                else:
                                    ax.errorbar(
                                        run_numbers_list[ii],
                                        v[0][ii],
                                        yerr=np.array(
                                            [
                                                abs(v[1][ii] - v[0][ii]),
                                                abs(v[2][ii] - v[0][ii]),
                                            ]
                                        ),
                                        fmt=".",
                                        color=c,
                                        elinewidth=0.5,
                                        capsize=2,
                                    )
                    ax.legend(
                        fontsize="x-small",
                        loc="upper left",
                        bbox_to_anchor=(0, 1.2),
                    )
                    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.ticklabel_format(
                        style="plain", axis="x", useOffset=False
                    )
                    ax.spines.right.set_visible(False)
                    ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)
                else:
                    if self.view.get_instrument() != "DEMAND":
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            ax.plot(run_numbers_list, val, ".", color=c)
                    else:
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            v = val[:]
                            for ii in range(len(run_numbers_list)):
                                if ii == 0:
                                    ax.errorbar(
                                        run_numbers_list[ii],
                                        v[0][ii],
                                        yerr=np.array(
                                            [
                                                abs(v[1][ii] - v[0][ii]),
                                                abs(v[2][ii] - v[0][i]),
                                            ]
                                        ),
                                        fmt=".",
                                        color=c,
                                        label=lab,
                                        elinewidth=0.5,
                                        capsize=2,
                                    )
                                else:
                                    ax.errorbar(
                                        run_numbers_list[ii],
                                        v[0][ii],
                                        yerr=np.array(
                                            [
                                                abs(v[1][ii] - v[0][ii]),
                                                abs(v[2][ii] - v[0][ii]),
                                            ]
                                        ),
                                        fmt=".",
                                        color=c,
                                        elinewidth=0.5,
                                        capsize=2,
                                    )

                    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.ticklabel_format(
                        style="plain", axis="x", useOffset=False
                    )
                    ax.spines.left.set_visible(False)
                    ax.tick_params(labelleft=False)
                    # ax.set_yticks([])
                    if i != len(self.model.subplot_limits) - 1:
                        ax.spines.right.set_visible(False)
                        ax.tick_params(labelright=False)
                        ax.tick_params(axis="y", length=0)
                        ax.plot(
                            [0, 0, 1, 1],
                            [0, 1, 0, 1],
                            transform=ax.transAxes,
                            **kwargs,
                        )
                    else:
                        ax.spines.right.set_visible(False)
                        ax.tick_params(axis="y", length=0)
                        # ax.yaxis.tick_right()
                        ax.plot(
                            [0, 0], [0, 1], transform=ax.transAxes, **kwargs
                        )

                ax2 = ax.twinx()
                color = "r"
                ax2.plot(run_numbers_list, scale_values, ".", color=color)
                ax2.tick_params(axis="y", labelcolor=color)
                ax2.set_ylim(
                    -0.1 * np.max(scale_values), np.max(scale_values) * 1.1
                )
                if i < len(self.model.subplot_limits) - 1:
                    ax2.spines.right.set_visible(False)
                    ax2.spines.left.set_visible(False)
                    ax2.tick_params(labelright=False)
                    ax2.tick_params(labelleft=False)
                    ax2.tick_params(axis="y", length=0)
                else:
                    ax2.tick_params(labelright=True)
                    ax2.spines.left.set_visible(False)
                    ax2.tick_params(labelleft=False)
                    ax2.tick_params(axis="y", color=color, labelright=True)
                    # ax2.spines.right.set_color(color)
                    ax2.set_ylabel(
                        f'Scale ({self.inst_params["Scale"].split(".")[-1]})',
                        color=color,
                    )

        self.view.plot.figure.canvas.draw()

    def sign_in(self):
        user = self.view.user_line.text()
        pw = self.view.pass_line.text()

        ONCAT_URL = "https://oncat.ornl.gov"
        CLIENT_ID = "99025bb3-ce06-4f4b-bcf2-36ebf925cd1d"

        oncat = pyoncat.ONCat(
            ONCAT_URL,
            client_id=CLIENT_ID,
            flow=pyoncat.RESOURCE_OWNER_CREDENTIALS_FLOW,
        )

        try:
            oncat.login(user, pw)
        except:
            self.view.message_label.setText("Incorrect Username or Password")
            self.view.message_label.setStyleSheet("color: red;")
            self.view.pass_line.setText("")
            return

        self.login = oncat
        self.view.message_label.setText("Signed In")
        self.view.message_label.setStyleSheet("color: green;")
        # self.view.user_line.setText('')
        # self.view.pass_line.setText('')
        self.switch_instrument()


class Model:
    def __init__(self):
        pass

    def goniometer_entries(self, inst_params):
        goniometer = inst_params["Goniometer"]
        goniometer_engry = inst_params["GoniometerEntry"]

        if inst_params["FancyName"] == "DEMAND":
            goniometer["2theta"] = "2theta"

        projection = []
        for name in goniometer.keys():
            if (
                inst_params["FancyName"] != "DEMAND"
                or inst_params["InstrumentName"] == "WAND"
            ):
                entry = ".".join(
                    [goniometer_engry, name.lower(), "average_value"]
                )
            else:
                min_entry = ".".join(
                    [goniometer_engry, name.lower(), "minimum"]
                )
                entry = ".".join([goniometer_engry, name.lower(), "average"])
                max_entry = ".".join(
                    [goniometer_engry, name.lower(), "maximum"]
                )
                projection.append(min_entry)
                projection.append(max_entry)
            projection.append(entry)

        return projection

    def retrieve_data_files(self, login, inst_params, ipts_number):
        facility = inst_params["Facility"]
        instrument = inst_params["Name"]
        run_number = inst_params["RunNumber"]

        projection = [
            run_number,
            inst_params["Title"],
            inst_params["Scale"],
            "id",
        ]

        projection += self.goniometer_entries(inst_params)

        exts = [inst_params["Extension"]]

        data_files = login.Datafile.list(
            facility=facility,
            instrument=instrument,
            experiment="IPTS-{}".format(ipts_number),
            projection=projection,
            exts=exts,
            tags=["type/raw"],
        )
        return data_files

    def list_available(self, login, inst_params):
        facility = inst_params["Facility"]
        instrument = inst_params["Name"]
        projection = ["id"]

        available_runs = login.Experiment.list(
            facility=facility, instrument=instrument, projection=projection
        )

        available = [""]
        avs = []
        if len(available_runs) != 0:
            for i in available_runs:
                avs.append(int(i["id"].split("-")[-1]))
        avs.sort(reverse=True)
        for i in range(len(avs)):
            available.append(str(avs[i]))

        return available

    def set_experiments(self, cbox, data_files):
        ids = np.array(
            [df["id"].split("/")[-3].strip("expIPTS-") for df in data_files]
        )
        unique = np.unique(ids)
        cbox_entries = []  # ['']
        for i in unique:
            cbox_entries.append(i)
        cbox.addItems(cbox_entries)

    def run_title_dictionary(self, data_files, inst_params):
        title_entry = inst_params["Title"]
        run_number_entry = inst_params["RunNumber"]

        titles = np.array([df[title_entry] for df in data_files])
        run_numbers = np.array(
            [int(df[run_number_entry]) for df in data_files]
        )

        unique_titles = np.unique(titles)

        run_title_dict = {}
        for unique_title in unique_titles:
            runs = run_numbers[titles == unique_title]
            run_seq = np.split(
                runs.astype(str), np.where(np.diff(runs) > 1)[0] + 1
            )
            rs = ",".join(
                [s[0] + ":" + s[-1] if len(s) - 1 else s[0] for s in run_seq]
            )
            run_title_dict[unique_title] = rs

        return run_title_dict

    def run_numbers_list(self, rs):
        run_seq = [np.array(s.split(":")).astype(int) for s in rs.split(",")]
        run_list = [
            np.arange(r[0], r[-1] + 1) if len(r) - 1 else r for r in run_seq
        ]

        return np.array([r for sub_list in run_list for r in sub_list])

    def prepare_runs_for_multiple_plots(self, run_number_list):
        rs = run_number_list.copy()

        rs.sort(0)

        out_list = []
        breaks = [0]
        for i in range(1, len(rs)):
            if rs[i] - rs[i - 1] > 2:
                breaks.append(i)

        if len(breaks) == 1:
            out_list.append([rs[0], rs[-1]])

        else:
            for i in range(len(breaks) - 1):
                out_list.append([rs[breaks[i]], rs[breaks[i + 1] - 1]])
            out_list.append([rs[breaks[-1]], rs[-1]])

        self.subplot_limits = out_list

    def beamline_info(self, bl):
        inst_params = beamlines[bl]

        return inst_params

    def run_numbers_indices(
        self, name, data_files, run_title_dict, inst_params
    ):
        run_number_entry = inst_params["RunNumber"]
        this_run_numbers = self.run_numbers_list(run_title_dict[name])
        run_numbers = np.array(
            [int(df[run_number_entry]) for df in data_files]
        )
        indices = np.arange(len(data_files))

        mask = np.array([i in this_run_numbers for i in run_numbers])

        self.prepare_runs_for_multiple_plots(run_numbers[mask])

        return run_numbers[mask], indices[mask]

    def run_numbers_indices_1(self, data_files, run_number_list, inst_params):
        run_number_entry = inst_params["RunNumber"]
        run_numbers = np.array(
            [int(df[run_number_entry]) for df in data_files]
        )
        indices = np.arange(len(data_files))

        mask = np.array([i in run_number_list for i in run_numbers])

        if sum(mask) != 0:
            self.prepare_runs_for_multiple_plots(run_numbers[mask])

        return run_numbers[mask], indices[mask]

    def goniometer_values(self, data_files, indices, inst_params):
        a = []
        gonio_entry = inst_params["Goniometer"]

        for entry in gonio_entry:
            b = []
            try:
                values = np.array(
                    [
                        float(
                            df[
                                inst_params["GoniometerEntry"]
                                + "."
                                + entry.lower()
                                + ".average_value"
                            ]
                        )
                        for df in data_files
                    ]
                )
            except KeyError:
                values = np.array(
                    [
                        [
                            float(
                                df[
                                    inst_params["GoniometerEntry"]
                                    + "."
                                    + entry.lower()
                                    + ".average"
                                ]
                            ),
                            float(
                                df[
                                    inst_params["GoniometerEntry"]
                                    + "."
                                    + entry.lower()
                                    + ".minimum"
                                ]
                            ),
                            float(
                                df[
                                    inst_params["GoniometerEntry"]
                                    + "."
                                    + entry.lower()
                                    + ".maximum"
                                ]
                            ),
                        ]
                        for df in data_files
                    ]
                )
            except TypeError:
                values = []
                for df in data_files:
                    try:
                        val = df[
                            inst_params["GoniometerEntry"]
                            + "."
                            + entry.lower()
                            + ".average_value"
                        ]
                    except KeyError:
                        val = np.array(
                            [
                                [
                                    float(
                                        df[
                                            inst_params["GoniometerEntry"]
                                            + "."
                                            + entry.lower()
                                            + ".average"
                                        ]
                                    ),
                                    float(
                                        df[
                                            inst_params["GoniometerEntry"]
                                            + "."
                                            + entry.lower()
                                            + ".minimum"
                                        ]
                                    ),
                                    float(
                                        df[
                                            inst_params["GoniometerEntry"]
                                            + "."
                                            + entry.lower()
                                            + ".maximum"
                                        ]
                                    ),
                                ]
                                for df in data_files
                            ]
                        )
                    if val is None:
                        val = np.nan
                    if type(val) is list and val[0] is None:
                        val[0] = np.nan
                        val[1] = np.nan
                        val[2] = np.nan
                    values.append(float(val))
                values = np.array(values)

            b.append([values[i] for i in indices])
            b = np.array(b).T
            a.append(b)

        return a, [i.lower() for i in gonio_entry]

    def scale_values(self, data_files, indices, inst_params):
        scale_entry = inst_params["Scale"]
        if inst_params["FancyName"] == "DEMAND":
            scale_entry += ".average"

        values = np.array([float(df[scale_entry]) for df in data_files])
        if inst_params["Scale"] == "metadata.entry.proton_charge":
            a = [values[i] / 1e12 for i in indices]
        else:
            a = [values[i] for i in indices]
        a = np.array(a).T

        return a


class ExperimentBrowser(QMainWindow):
    __instance = None

    def __new__(cls):
        if ExperimentBrowser.__instance is None:
            ExperimentBrowser.__instance = QMainWindow.__new__(cls)
        return ExperimentBrowser.__instance

    def __init__(self, parent=None):
        super().__init__(parent)

        icon_path = "./icon.png"
        self.setWindowIcon(QIcon(icon_path))
        name = "ipts-experiment-browser"
        self.setWindowTitle(name)
        self.setGeometry(0, 0, 1024, 635)

        main_window = QWidget(self)
        self.setCentralWidget(main_window)
        layout = QVBoxLayout(main_window)

        view = View()
        model = Model()
        self.form = Presenter(view, model)
        layout.addWidget(view)


def handle_exception(exc_type, exc_value, exc_traceback):
    error_message = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Application Error")
    msg_box.setText("An unexpected error occurred. Please see details below:")
    msg_box.setDetailedText(error_message)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.exec_()


if __name__ == "__main__":
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    window = ExperimentBrowser()
    window.show()
    sys.exit(app.exec_())
