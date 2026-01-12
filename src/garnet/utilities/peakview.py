import sys
import os
import traceback
import glob

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QTextEdit,
)
from qtpy.QtGui import QIntValidator, QTextCursor

import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PIL import Image
import numpy as np

try:
    from qdarkstyle.light.palette import LightPalette
    import qdarkstyle

    style = True
except:
    qdarkstyle = None
    style = False


class View(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        yaml_label = QLabel("Config: ")
        self.yaml_line = QLineEdit("")
        self.yaml_line.setReadOnly(True)
        self.yaml_button = QPushButton("Select Config")

        self.layout.addWidget(yaml_label, 0, 0)
        self.layout.addWidget(self.yaml_line, 0, 1, 1, 12)
        self.layout.addWidget(self.yaml_button, 0, 13)

        nxs_label = QLabel("Params: ")
        self.layout.addWidget(nxs_label, 1, 0)

        self.nxs_combo = QComboBox(self)
        self.layout.addWidget(self.nxs_combo, 1, 1, 1, 13)

        hkl_validator = QIntValidator(-100, 100, self)
        mnp_validator = QIntValidator(-1, 1, self)

        peak_label = QLabel("Peak: ")
        hl = QLabel("h: ")
        kl = QLabel("k: ")
        ll = QLabel("l: ")
        ml = QLabel("m: ")
        nl = QLabel("n: ")
        pl = QLabel("p: ")

        self.h_line = QLineEdit("1")
        self.k_line = QLineEdit("0")
        self.l_line = QLineEdit("0")
        self.m_line = QLineEdit("0")
        self.n_line = QLineEdit("0")
        self.p_line = QLineEdit("0")

        self.h_line.setValidator(hkl_validator)
        self.k_line.setValidator(hkl_validator)
        self.l_line.setValidator(hkl_validator)
        self.m_line.setValidator(mnp_validator)
        self.n_line.setValidator(mnp_validator)
        self.p_line.setValidator(mnp_validator)

        self.update_button = QPushButton("Update")

        self.layout.addWidget(peak_label, 2, 0)
        self.layout.addWidget(hl, 2, 1)
        self.layout.addWidget(self.h_line, 2, 2)
        self.layout.addWidget(kl, 2, 3)
        self.layout.addWidget(self.k_line, 2, 4)
        self.layout.addWidget(ll, 2, 5)
        self.layout.addWidget(self.l_line, 2, 6)
        self.layout.addWidget(ml, 2, 7)
        self.layout.addWidget(self.m_line, 2, 8)
        self.layout.addWidget(nl, 2, 9)
        self.layout.addWidget(self.n_line, 2, 10)
        self.layout.addWidget(pl, 2, 11)
        self.layout.addWidget(self.p_line, 2, 12)
        self.layout.addWidget(self.update_button, 2, 13)

        run_label = QLabel("Run: ")
        self.layout.addWidget(run_label, 3, 0)
        self.run_combo = QComboBox(self)
        self.layout.addWidget(self.run_combo, 3, 1, 1, 13)

        self.plot = FigureCanvas(Figure(figsize=(11, 8)))
        self.plot.figure.set_layout_engine("tight")
        self.layout.addWidget(self.plot, 4, 1, 10, 13)
        self.layout.addWidget(NavigationToolbar(self.plot, self), 14, 1, 1, 13)

        self.text_out = ""
        self.output = QTextEdit()
        self.output.setFontFamily("monospace")
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output, 15, 1, 1, 13)

    def connect_select_config(self, update):
        self.yaml_button.clicked.connect(update)

    def connect_nxs_combo(self, update):
        self.nxs_combo.currentTextChanged.connect(update)

    def connect_run_combo(self, update):
        self.run_combo.currentTextChanged.connect(update)

    def connect_peak_updated(self, update):
        self.h_line.returnPressed.connect(update)
        self.k_line.returnPressed.connect(update)
        self.l_line.returnPressed.connect(update)
        self.m_line.returnPressed.connect(update)
        self.n_line.returnPressed.connect(update)
        self.p_line.returnPressed.connect(update)
        self.update_button.clicked.connect(update)

    def connect_output(self, update):
        self.output.textChanged.connect(update)

    def get_nxs(self):
        return self.nxs_combo.currentText()

    def get_yaml(self):
        return self.yaml_line.text()

    def get_hklmnp(self):
        return [
            int(i.text())
            for i in [
                self.h_line,
                self.k_line,
                self.l_line,
                self.m_line,
                self.n_line,
                self.p_line,
            ]
        ]

    def get_run(self):
        return self.run_combo.currentText()


class Presenter:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.connect_select_config(self.select_config)
        self.view.connect_peak_updated(self.update_hkl)
        self.view.connect_output(self.cursor_to_end)
        self.view.connect_nxs_combo(self.update_nexus)
        self.view.connect_run_combo(self.update_run)

        self.initialized = False

    def select_config(self):
        if self.view.yaml_line.text() == "":
            filename, ok = QFileDialog().getOpenFileName(
                self.view,
                "Select file",
                "/SNS/",
                "Garnet config files (*.yaml)",
            )
        else:
            file_folder = "/".join(self.view.yaml_line.text().split("/")[:-1])
            filename, ok = QFileDialog().getOpenFileName(
                self.view,
                "Select file",
                file_folder,
                "Garnet config files (*.yaml)",
            )
        self.clear_all()
        if filename:
            self.view.yaml_line.setText(filename)
            self.view.text_out = filename + "\n"
            self.view.output.setPlainText(self.view.text_out)

            self.init_nxs(filename)
        self.initialized = False

    def init_nxs(self, path):
        self.integrate_path = path.rstrip("yaml").rstrip(".") + "_integration/"
        pattern = os.path.join(self.integrate_path, "**", "*(max)=????.nxs")
        nexus_paths = glob.glob(pattern, recursive=True)
        nexus_paths = [os.path.basename(i) for i in nexus_paths]

        if len(nexus_paths) > 0:
            self.view.nxs_combo.addItems(nexus_paths)
        else:
            self.view.text_out += "\nNo integration files found for " + path
            self.view.output.setPlainText(self.view.text_out)

    def update_hkl(self):
        self.initialized = True
        h, k, l, m, n, p = self.view.get_hklmnp()

        plots_paths = self.model.list_plots(
            self.integrate_path, self.view.get_nxs(), self.view.get_hklmnp()
        )

        if plots_paths != -1:
            plots_sorted = [
                [i.split("#")[-1].strip(".png"), i] for i in plots_paths
            ]
            plots_sorted.sort()
            plots_paths = [i[1] for i in plots_sorted]

        if plots_paths == -1:
            self.view.text_out += "\nNo plots found for (h,k,l) = ({},{},{}) with (m,n,p) = ({},{},{})  \n".format(
                h, k, l, m, n, p
            )
            self.view.run_combo.clear()
            self.clear_plot()
        else:
            self.view.text_out += "\n{} plots found for (h,k,l) = ({},{},{}) with (m,n,p) = ({},{},{})\n  ".format(
                len(plots_paths), h, k, l, m, n, p
            )
            self.view.text_out += (
                "\n  ".join(["/".join(i.split("/")[-2:]) for i in plots_paths])
                + "\n"
            )

            self.view.run_combo.clear()
            for path in plots_paths:
                self.view.run_combo.addItem("/".join(path.split("/")[-2:]))

            # self.model.plot(self.view.plot,self.view.get_run(),self.integrate_path,self.view.get_nxs())

        self.view.output.setPlainText(self.view.text_out)

    def update_run(self):
        if self.view.run_combo.currentText() != "":
            self.model.plot(
                self.view.plot,
                self.view.get_run(),
                self.integrate_path,
                self.view.get_nxs(),
            )

    def update_nexus(self):
        self.view.run_combo.clear()
        self.view.plot.figure.clf()
        self.view.plot.figure.canvas.draw()
        self.view.text_out += "\n" + self.view.get_nxs().split("/")[-1] + "\n"
        self.view.output.setPlainText(self.view.text_out)
        if self.initialized:
            self.update_hkl()

    def cursor_to_end(self):
        cursor = self.view.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.view.output.setTextCursor(cursor)
        self.view.output.ensureCursorVisible()
        self.view.output.setFocus()

    def clear_all(self):
        # self.view.yaml_line.setText('')
        self.view.nxs_combo.clear()
        self.view.run_combo.clear()

        self.clear_plot()

    def clear_plot(self):
        self.view.plot.figure.clf()
        self.view.plot.figure.canvas.draw()


class Model:
    def __init__(self):
        pass

    def list_plots(self, int_path, nxs, hklmnp):
        plots_folder = (
            "/"
            + "/".join(int_path.split("/"))
            + nxs.rstrip("nxs").rstrip(".")
            + "_plots/"
        )
        pattern = os.path.join(
            plots_folder,
            "**",
            "*({},{},{})_({},{},{})".format(*hklmnp),
            "*.png",
        )
        plots_path = glob.glob(pattern, recursive=True)

        if len(plots_path) == 0:
            return -1

        return plots_path

    def plot(self, plot, run, integrate_path, nexus):
        filename = (
            integrate_path + nexus.rstrip("nxs").rstrip(".") + "_plots/" + run
        )

        plot.figure.clf()

        img = np.asarray(Image.open(filename))
        ax = plot.figure.subplots()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        plot.figure.canvas.draw()


class IntegratedPeaksViewer(QMainWindow):
    __instance = None

    def __new__(cls):
        if IntegratedPeaksViewer.__instance is None:
            IntegratedPeaksViewer.__instance = QMainWindow.__new__(cls)
        return IntegratedPeaksViewer.__instance

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("integrated-peaks-viewer")
        self.setGeometry(50, 50, 1250, 1250)

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

    if theme:
        app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))

    window = IntegratedPeaksViewer()
    window.show()
    sys.exit(app.exec_())
