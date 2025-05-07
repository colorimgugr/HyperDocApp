#!/usr/bin/env python3
import sys
from PyQt5 import QtWidgets, QtCore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exemple Dockable")
        self.resize(800, 600)

        # Widget central
        textEdit = QtWidgets.QTextEdit()
        textEdit.setText("Contenu central")
        self.setCentralWidget(textEdit)

        # Dock 1: Liste d'items
        dock1 = QtWidgets.QDockWidget("Dock 1", self)
        dock1.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        listWidget = QtWidgets.QListWidget()
        listWidget.addItems(["Item A", "Item B", "Item C"])
        dock1.setWidget(listWidget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock1)

        # Dock 2: Arborescence
        dock2 = QtWidgets.QDockWidget("Dock 2", self)
        features = (
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
        dock2.setFeatures(features)
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(["Colonne 1", "Colonne 2"])
        for i in range(3):
            QtWidgets.QTreeWidgetItem(tree, [f"Noeud {i}", f"Valeur {i}"])
        dock2.setWidget(tree)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock2)

        # Dock 3: Label centré
        dock3 = QtWidgets.QDockWidget("Dock 3", self)
        label = QtWidgets.QLabel("Un label simple dans un dock")
        label.setAlignment(QtCore.Qt.AlignCenter)
        dock3.setWidget(label)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock3)

        # 1. Créer le menu Affichage
        view_menu = self.menuBar().addMenu("&Affichage")

        # 2. Ajouter pour chaque dock son toggleViewAction()
        #    Cela rend l'action “checkable” et synchronise fermeture/réouverture.
        view_menu.addAction(dock1.toggleViewAction())
        view_menu.addAction(dock2.toggleViewAction())
        view_menu.addAction(dock3.toggleViewAction())

        # Facultatif : séparer les groupes
        view_menu.addSeparator()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
