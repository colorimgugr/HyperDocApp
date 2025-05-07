from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt

import sys

# widgets import
from hypercubes.hypercube      import HDF5BrowserWidget
from data_vizualisation.data_vizualisation_tool import Data_Viz_Window
from registration.register_tool          import RegistrationApp

# TODO : initier dans MainWindow les hypercubes et connecter les champs de chaque widget (yeah...big deal)

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp")
        self.resize(1200, 800)
        self.setCentralWidget(QtWidgets.QWidget())

        # add docks
        self._add_dock("File Browser",   HDF5BrowserWidget,   QtCore.Qt.LeftDockWidgetArea)
        self._add_dock("Data Visualization", Data_Viz_Window,  QtCore.Qt.RightDockWidgetArea)
        self._add_dock("Registration",   RegistrationApp,     QtCore.Qt.BottomDockWidgetArea)

        # Tool menu
        view = self.menuBar().addMenu("Tools Selection")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view.addAction(dock.toggleViewAction())

    def _add_dock(self, title, WidgetClass, area):
        widget = WidgetClass(parent=self)
        dock   =  QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)



def excepthook(exc_type, exc_value, exc_traceback):
    """Capture les exceptions et les affiche dans une boîte de dialogue."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Erreur : {error_msg}")  # Optionnel : enregistrer dans un fichier log
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText("Une erreur est survenue :")
    msg_box.setInformativeText(error_msg)
    msg_box.exec_()

def update_font(_app,width=None,_font="Segoe UI",):
    global main

    if not width:
        screen = _app.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()

    else:
        screen_width=width

    if screen_width < 1280:
        font_size = 7
    elif screen_width < 1920:
        font_size = 8
    else:
        font_size = 9

    _app.setFont(QFont(_font, font_size))
    plt.rcParams.update({"font.size": font_size + 3, "font.family": _font})

def check_resolution_change():
    """ Vérifie si la résolution a changé et met à jour la police si besoin """
    global last_width  # On garde la dernière largeur connue
    screen = app.screenAt(main.geometry().center())
    current_width = screen.size().width()

    if current_width != last_width:
        update_font(app,current_width)
        last_width = current_width


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    update_font(app)
    app.setStyle('Fusion')

    main = MainApp()
    main.show()

    # Timer for screen resolution check
    last_width = app.primaryScreen().size().width()
    timer = QTimer()
    timer.timeout.connect(check_resolution_change)
    timer.start(500)  # Vérifie toutes les 500 ms
    sys.exit(app.exec_())

