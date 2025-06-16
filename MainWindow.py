from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer,QSize, Qt
from PyQt5.QtGui import QFont,QIcon, QPalette, QColor
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QStyleFactory, QToolBar,QAction,QComboBox,
                             QLabel,QToolButton,QMenu)

import sys
import os
import traceback

# widgets import
from hypercubes.hypercube  import *
from data_vizualisation.data_vizualisation_tool import Data_Viz_Window
from registration.register_tool        import RegistrationApp
from interface.HypercubeManager import HypercubeManager
from data_vizualisation.metadata_tool import MetadataTool

# TODO : initier dans MainWindow les hypercubes et connecter les champs de chaque widget (yeah...big deal)
# TODO : generate metadata position,height, width ,parentCube,name of registered cube or minicube
# TODO : generate a list of basic Metadatas keys with types
# todo : gestion of signal/slot for in tool load hypercubes.

from PyQt5.QtWidgets import QToolBar, QDockWidget
from PyQt5.QtCore    import QSize, Qt
from PyQt5.QtGui     import QIcon

def apply_fusion_border_highlight(app,
                                  border_color: str = "#888888",
                                  title_bg:      str = "#E0E0E0",
                                  separator_hover: str = "#AAAAAA",
                                  window_bg:     str = "#F5F5F5",   # ← ton nouveau fond
                                  base_bg:       str = "#EFEFEF"):  # ← pour QTextEdit, etc.
    # 1) Fusion
    app.setStyle(QStyleFactory.create("Fusion"))

    # 1b) palette customisée
    pal = app.palette()
    pal.setColor(QPalette.Window,        QColor(window_bg))
    pal.setColor(QPalette.Base,          QColor(base_bg))
    app.setPalette(pal)

    # 2) ton QSS existant pour les bordures
    app.setStyleSheet(f"""
    QMainWindow, QWidget#centralwidget {{
        background-color: {window_bg};
    }}
    QDockWidget {{
        border: 1px solid {border_color};
    }}
    QDockWidget::title {{
        background: {title_bg};
        padding: 3px;
        border-bottom: 1px solid {border_color};
        color: black;
        text-align: left;
    }}
    QMainWindow::separator {{
        background-color: {border_color};
        width: 2px; height: 2px; margin: 1px;
    }}
    QMainWindow::separator:hover {{
        background-color: {separator_hover};
    }}
    QSplitter::handle {{
        background-color: {border_color};
    }}
    QSplitter::handle:hover {{
        background-color: {separator_hover};
    }}
    /* assure que les widgets enfants héritent bien de la couleur de fond */
    QDockWidget > QWidget {{
        background-color: {base_bg};
    }}
    """)

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp")
        self.resize(1200, 800)
        self.setCentralWidget(QtWidgets.QWidget())

        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        ICONS_DIR = os.path.join(BASE_DIR, "Hypertool/interface/icons")
        icon_main= "Hyperdoc_logo_transparente_CIMLab.png"
        self.setWindowIcon(QIcon(os.path.join(ICONS_DIR,icon_main)))

        # add docks
        self.file_browser_dock = self._add_file_browser_dock()
        self.data_viz_dock =self._add_dock("Data Visualization", Data_Viz_Window,  QtCore.Qt.RightDockWidgetArea)
        self.reg_dock=self._add_dock("Registration",   RegistrationApp,     QtCore.Qt.BottomDockWidgetArea)
        self.meta_dock=self._add_dock("Metadata",   MetadataTool,     QtCore.Qt.BottomDockWidgetArea)

        # Tool menu
        view = self.menuBar().addMenu("Tools")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view.addAction(dock.toggleViewAction())

        # ─── Toolbar “Quick Tools” ───────────────────────────────────────────
        toolbar = self.addToolBar("Quick Tools")
        toolbar.setIconSize(QSize(48, 48))  # Taille des icônes
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)  #ToolButtonIconOnly ou TextUnderIcon)

        # Action File Browser
        act_file = self.file_browser_dock.toggleViewAction()
        icon_file_browser = "file_browser_icon.png"
        act_file.setIcon(QIcon(os.path.join(ICONS_DIR, icon_file_browser)))  # charge ton icône
        act_file.setToolTip("File Browser")
        toolbar.addAction(act_file)

        # Action Data Viz
        act_data = self.data_viz_dock.toggleViewAction()
        icon_data_viz = "icon_data_viz.svg"
        act_data.setIcon(QIcon(os.path.join(ICONS_DIR, icon_data_viz)))
        act_data.setToolTip("Data Visualization")
        toolbar.addAction(act_data)

        # Action Registration
        act_reg = self.reg_dock.toggleViewAction()
        icon_registration = "registration_icon.png"
        act_reg.setIcon(QIcon(os.path.join(ICONS_DIR, icon_registration)))
        act_reg.setToolTip("Registration")
        toolbar.addAction(act_reg)

        # Action Metadata
        act_met = self.meta_dock.toggleViewAction()
        icon_met = "metadata_icon.png"
        act_met.setIcon(QIcon(os.path.join(ICONS_DIR, icon_met)))
        act_met.setToolTip("Metadata")
        toolbar.addAction(act_met)

        toolbar.addSeparator()

        # Cubes "list"
        self.cubeBtn = QtWidgets.QToolButton(self)
        self.cubeBtn.setText("Cubes list   ")
        self.cubeBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # self.cubeBtn.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        toolbar.addWidget(self.cubeBtn)

        # Création du menu hiérarchique
        self.cubeMenu = QtWidgets.QMenu(self)
        self.cubeBtn.setMenu(self.cubeMenu)

        # Hypercube Manager
        self.hypercube_manager = HypercubeManager()
        reg_widget = self.reg_dock.widget()
        reg_widget.alignedCubeReady.connect(self.hypercube_manager.addCube)  # get signal from register tool

        # Action Add File in list of cubes
        act_add = QAction("Add Cubes", self)
        act_add.triggered.connect(self._on_add_cube)
        toolbar.addAction(act_add)

        # Mise à jour du menu à chaque modification
        self.hypercube_manager.cubesChanged.connect(self._update_cube_menu)
        self._update_cube_menu(self.hypercube_manager.paths)

        act_open_previous = QAction("<", self)
        act_open_previous.setToolTip("Open previous cube in current folder")
        toolbar.addAction(act_open_previous)

        act_open_next = QAction(">", self)
        act_open_next.setToolTip("Open next cube in current folder")
        toolbar.addAction(act_open_next)

        # tools to hide at opening
        self.file_browser_dock.hide()
        self.reg_dock.hide()

    def _on_file_browser_accepted(self, updated_ci: CubeInfoTemp):
        '''
        get OK pressed button of browser with
        '''
        self.hypercube_manager.cubesChanged.emit(self.hypercube_manager.paths)

    def _open_file_browser_for_index(self, index: int):
        """
        Injecte le CubeInfoTemp existant dans le widget,
        pré-remplit les champs, recharge l'arbre et l'affiche.
        """
        ci     = self.hypercube_manager.getCubeInfo(index)
        widget = self.file_browser_dock.widget()

        # 1) Ré-associe l'objet métier et le filepath
        widget.cube_info = ci
        widget.filepath  = ci.filepath

        # 2) Pré-remplissage des QLineEdit
        widget.le_cube.setText       (ci.data_path       or "")
        widget.le_wl.setText         (ci.wl_path         or "")
        widget.le_meta.setText       (ci.metadata_path   or "")
        # comboBox_channel_wl : 0 = First, 1 = Other (à ajuster)
        widget.comboBox_channel_wl.setCurrentIndex(0 if ci.wl_trans else 1)

        # 3) Rebuild de l'arbre HDF5/MAT
        widget._load_file()

        # 4) Affichage du dock
        self.file_browser_dock.show()
        widget.show()

    def _add_dock(self, title, WidgetClass, area):
        widget = WidgetClass(parent=self)
        dock   =  QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _add_file_browser_dock(self) -> QtWidgets.QDockWidget:
        """
        Initialise le File Browser avec un CubeInfoTemp vide,
        connecte son signal accepted, puis l'ajoute en dock.
        """
        # 1) crée un CubeInfoTemp vierge
        ci = CubeInfoTemp(filepath=None)

        # 2) instancie le widget
        widget = HDF5BrowserWidget(
            cube_info=ci,
            filepath=None,
            parent=self,
            closable=True
        )
        # 3) connecte le signal accepted
        widget.accepted.connect(self._on_file_browser_accepted)
        widget.rejected.connect(lambda: None)  # si tu veux réagir à l'annulation

        # 4) création du dock
        dock = QtWidgets.QDockWidget("File Browser", self)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        return dock

    def _on_add_cube(self,paths=None):
        if not isinstance(paths, (list, tuple)) or len(paths) == 0:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select Hypercubes",
                "",
                "All files (*.*)"
            )

        if not paths:
            return

        for path in paths:
            print(path.split('/')[-1])
            ci=CubeInfoTemp(filepath=path)
            self.hypercube_manager.addCube(ci)

        print(self.hypercube_manager.paths)

    def _on_get_cube_info(self, insert_index):
        # 1) Récupère le CubeInfoTemp déjà présent
        ci = self.hypercube_manager._cubes[insert_index]
        filepath = ci.filepath
        if not filepath:
            return

        # 2) Recharge les infos via Hypercube en mode init
        hc = Hypercube(filepath=filepath, cube_info=ci, load_init=True)
        # Optionnel : mettre à jour la forme des données pour affichage
        ci.data_shape = getattr(hc.data, 'shape', None)

        # 3) Nettoie l’objet lourd pour ne pas garder de data en mémoire
        hc.reinit_cube()
        del hc

        # 4) Notifie la mise à jour du manager/UI
        self.hypercube_manager.cubesChanged.emit(self.hypercube_manager.paths)

    def _send_to_metadata(self,index):
        widget = self.meta_dock.widget()
        ci     = self.hypercube_manager.getCubeInfo(index)
        widget.set_cube_info(ci)
        widget.update_combo_meta(init=True)

    # todo : send to vizualisation tool

    def _update_cube_menu(self, paths):
        """Met à jour le menu de cubes avec sous-menus et actions fonctionnelles."""
        self.cubeMenu.clear()
        for idx, path in enumerate(paths):
            # Sous-menu pour chaque cube
            sub = QtWidgets.QMenu(path, self)
            # Envoyer au dock viz
            act_viz = QtWidgets.QAction("Send to Vizualisation tool", self)
            act_viz.triggered.connect(lambda checked, i=idx: self._send_to_dock(i, self.dock1))
            sub.addAction(act_viz)
            # Envoyer au dock reg
            menu_load_reg=QtWidgets.QMenu("Send to Register Tool", sub)
            act_fix = QtWidgets.QAction("Fixed Cube", self)
            act_fix.triggered.connect(
                lambda _, i=idx: self.reg_dock.widget().load_cube(0,self.hypercube_manager.paths[i])
            )
            menu_load_reg.addAction(act_fix)
            # Action Moving
            act_mov = QtWidgets.QAction("Moving Cube", self)
            act_mov.triggered.connect(
                lambda _, i=idx: self.reg_dock.widget().load_cube(1,self.hypercube_manager.paths[i])
            )
            menu_load_reg.addAction(act_mov)
            sub.addMenu(menu_load_reg)
            # Envoyer au dock file browser
            act_browser = QtWidgets.QAction("Send to File Browser", self)
            act_browser.triggered.connect(lambda checked, i=idx: self._open_file_browser_for_index(i))
            sub.addAction(act_browser)

            # Envoyer au dock metadata
            act_meta = QtWidgets.QAction("Send to Metadata", self)
            act_meta.triggered.connect(lambda checked, i=idx: self._send_to_metadata(i))
            sub.addAction(act_meta)

            # Séparateur
            sub.addSeparator()

            # ─── NOUVELLE ACTION “Get Cube Info from File” ─────────────────
            act_get_info = QtWidgets.QAction("Get cube_info from file…", self)
            act_get_info.triggered.connect(lambda _, i=idx: self._on_get_cube_info(i))
            sub.addAction(act_get_info)
            # ───────────────────────────────────────────────────────────────

            # Séparateur
            sub.addSeparator()
            # Supprimer de la liste
            act_rm = QtWidgets.QAction("Remove from list", self)
            act_rm.triggered.connect(lambda checked, i=idx: self.hypercube_manager.removeCube(i))
            sub.addAction(act_rm)
            # Ajouter sous-menu au menu principal
            self.cubeMenu.addMenu(sub)

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
    # apply_fusion_dark_theme(app)
    apply_fusion_border_highlight(app)
    # app.setStyle('Fusion')

    main = MainApp()
    main.show()

    folder=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Samples\minicubes/'
    cube_1='00001-VNIR-mock-up.h5'
    cube_2='00002-VNIR-mock-up.h5'
    paths=[folder+cube_1,folder+cube_2]

    main._on_add_cube(paths)

    # Timer for screen resolution check
    last_width = app.primaryScreen().size().width()
    timer = QTimer()
    timer.timeout.connect(check_resolution_change)
    timer.start(500)  # Vérifie toutes les 500 ms
    sys.exit(app.exec_())