from PyQt5 import QtWidgets, QtCore
import sys

# widgets import
from hypercubes.hypercube      import HDF5BrowserWidget
from data_vizualisation.data_vizualisation_tool import Data_Viz_Window
from registration.register_tool          import RegistrationApp

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp")
        self.resize(1200, 800)

        # zone centrale (vide ou un simple placeholder)
        self.setCentralWidget(QtWidgets.QWidget())

        # créer et ajouter tes docks
        self._add_dock("HDF5 Browser",   HDF5BrowserWidget,   QtCore.Qt.LeftDockWidgetArea)
        self._add_dock("Data Visualization", Data_Viz_Window,  QtCore.Qt.RightDockWidgetArea)
        self._add_dock("Registration",   RegistrationApp,     QtCore.Qt.BottomDockWidgetArea)

        # menu Affichage pour toggle
        view = self.menuBar().addMenu("Affichage")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view.addAction(dock.toggleViewAction())

    def _add_dock(self, title, WidgetClass, area):
        widget = WidgetClass(parent=self)     # si besoin de parent
        dock   = QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(area, dock)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = MainApp()
    main.show()
    sys.exit(app.exec_())
