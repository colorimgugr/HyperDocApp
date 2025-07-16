# cd C:\Users\Usuario\Documents\GitHub\Hypertool
# python MainWindow.py
# sys.excepthook = excepthook #set the exception handler
# pyinstaller  --noconfirm --noconsole --exclude-module tensorflow --exclude-module torch --exclude-module matlab --icon="interface/icons/hyperdoc_logo_transparente.ico" --add-data "interface/icons:Hypertool/interface/icons" --add-data "ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv:ground_truth"  --add-data "data_vizualisation/Spatially registered minicubes equivalence.csv:data_vizualisation"  MainWindow.py
# C:\Envs\py37test\Scripts\activate

# GUI Qt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer,QSize, Qt
from PyQt5.QtGui import QFont,QIcon, QPalette, QColor
from PyQt5.QtWidgets import (QStyleFactory, QAction, QPushButton, QSizePolicy,
                             QLabel, QVBoxLayout, QTextEdit,QMessageBox)

## exception gestion
import traceback
import logging

# projects import
from hypercubes.hypercube import *
from data_vizualisation.data_vizualisation_tool import Data_Viz_Window
from registration.register_tool        import RegistrationApp
from hypercubes.hypercube_manager import HypercubeManager
from metadata.metadata_tool import MetadataTool
from ground_truth.ground_truth_tool import GroundTruthWidget

# grafics to control changes
import matplotlib.pyplot as plt

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

class CustomDockTitleBar(QtWidgets.QWidget):
    def __init__(self, dock_widget, style=None):
        super().__init__()
        self.dock = dock_widget
        self.style = style or {}
        self.setObjectName("CustomDockTitleBar")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)

        # Title label
        self.title_label = QtWidgets.QLabel()
        self.title_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        layout.addWidget(self.title_label)

        # Fullscreen button
        self.fullscreen_button = QtWidgets.QPushButton("⛶")
        self.fullscreen_button.setFixedSize(20, 20)
        self.fullscreen_button.setToolTip("Toggle fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        layout.addWidget(self.fullscreen_button)

        # Dock/undock button
        self.dock_button = QtWidgets.QPushButton("🗗")
        self.dock_button.setFixedSize(20, 20)
        self.dock_button.setToolTip("Dock / Undock")
        self.dock_button.clicked.connect(self.toggle_dock)
        layout.addWidget(self.dock_button)

        # Close button
        self.close_button = QtWidgets.QPushButton("✕")
        self.close_button.setFixedSize(20, 20)
        self.close_button.setToolTip("Close")
        self.close_button.clicked.connect(self.dock.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.setAutoFillBackground(True)

        # Disable double-click behavior
        self.installEventFilter(self)
        self.dock.topLevelChanged.connect(self.update_title)
        self.dock.windowTitleChanged.connect(self.update_title)

        self.update_title()
        self.apply_style()

    def toggle_fullscreen(self):
        if not self.dock.isFloating():
            return
        window = self.dock.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def toggle_dock(self):
        if self.dock.isFloating():
            self.dock.setFloating(False)
            self.dock.raise_()
        else:
            self.dock.setFloating(True)
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            center_x = screen.center().x() - self.dock.width() // 2
            center_y = screen.center().y() - self.dock.height() // 2
            self.dock.move(center_x, center_y)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick:
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_title()

    def update_title(self):
        fm = self.fontMetrics()
        available_width = max(40, self.width() - 90)
        elided = fm.elidedText(self.dock.windowTitle(), Qt.ElideRight, available_width)
        self.title_label.setText(elided)
        self.title_label.setToolTip(self.dock.windowTitle())

    def apply_style(self):
        bg     = self.style.get("background", "#E0E0E0")
        border = self.style.get("border", "#888888")
        text   = self.style.get("text", "black")
        pad_top = self.style.get("padding_top", 2)
        pad_left = self.style.get("padding_left", 4)

        self.setStyleSheet(f"""
            CustomDockTitleBar {{
                background-color: {bg};
                border-bottom: 1px solid {border};
            }}
            QLabel {{
                color: {text};
                padding-top: {pad_top}px;
                padding-left: {pad_left}px;
                background-color: transparent;
            }}
            QPushButton {{
                background-color: transparent;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {border};
            }}
        """)

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp")
        self.resize(1200, 800)
        self.setCentralWidget(QtWidgets.QWidget())

        # if getattr(sys, 'frozen', False): # pynstaller case
        #     self.BASE_DIR = sys._MEIPASS
        # else :
        #     self.BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        if getattr(sys, 'frozen', False):
            # Exécution depuis l’exécutable PyInstaller
            self.BASE_DIR = sys._MEIPASS
        else:
            # Exécution en tant que script Python
            CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.BASE_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))

        self.ICONS_DIR = os.path.join(self.BASE_DIR,"Hypertool","interface", "icons")
        # self.ICONS_DIR = os.path.join(self.BASE_DIR, "Hypertool/interface/icons")
        icon_main= "Hyperdoc_logo_transparente_CIMLab.png"
        self.setWindowIcon(QIcon(os.path.join(self.ICONS_DIR,icon_main)))

        # perso style title bar
        self.title_bar_style = {
            "background": "#E0E0E0",  # slightly darker than window background
            "border": "#888888",  # border color
            "text": "black",  # text color
            "padding_top": 2,
            "padding_left": 4
        }

        # make left docks with meta and file browser
        self.file_browser_dock = self._add_file_browser_dock() # left dock with file browser
        self.meta_dock=self._add_dock("Metadata",   MetadataTool,     QtCore.Qt.LeftDockWidgetArea) # add meta to left dock
        # self.tabifyDockWidget(self.file_browser_dock, self.meta_dock)
        # self.meta_dock.raise_() # raise meta and "hide in tab" file browser
        self.meta_dock.setVisible(False) # raise meta and "hide in tab" file browser

        # make "central" dock with visuals tools
        self.data_viz_dock =self._add_dock("Data Visualization", Data_Viz_Window,  QtCore.Qt.RightDockWidgetArea)
        self.reg_dock=self._add_dock("Registration",   RegistrationApp,     QtCore.Qt.RightDockWidgetArea)
        self.gt_dock=self._add_dock("Ground Truth",   GroundTruthWidget,     QtCore.Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self.reg_dock, self.gt_dock)
        self.tabifyDockWidget(self.reg_dock, self.data_viz_dock)
        self.data_viz_dock.raise_()

        # Tool menu
        view = self.menuBar().addMenu("Tools")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view.addAction(dock.toggleViewAction())

        # ─── Toolbar “Quick Tools” ───────────────────────────────────────────
        self.toolbar = self.addToolBar("Quick Tools")
        self.toolbar.setIconSize(QSize(48, 48))  # Taille des icônes
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)  #ToolButtonIconOnly ou TextUnderIcon)

        act_file = self.onToolButtonPress(self.file_browser_dock,icon_name="file_browser_icon.png",tooltip="File Browser")
        act_met = self.onToolButtonPress(self.meta_dock, "metadata_icon.png", "Metadata")
        self.toolbar.addSeparator()
        act_data = self.onToolButtonPress(self.data_viz_dock, "icon_data_viz.svg", "Data Vizualisation")
        act_reg = self.onToolButtonPress(self.reg_dock, "registration_icon.png", "Registration")
        act_gt =self.onToolButtonPress(self.gt_dock, "GT_icon_1.png", "Ground Truth")

        self.toolbar.addSeparator()

        # Cubes "list"
        self.cubeBtn = QtWidgets.QToolButton(self)
        self.cubeBtn.setText("Cubes list   ")
        self.cubeBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # self.cubeBtn.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        self.toolbar.addWidget(self.cubeBtn)

        # Création du menu hiérarchique
        self.cubeMenu = QtWidgets.QMenu(self)
        self.cubeBtn.setMenu(self.cubeMenu)

        # Hypercube Manager
        self.hypercube_manager = HypercubeManager()


        # Action Add File in list of cubes
        act_add = QAction("Open new cube(s)", self)
        act_add.triggered.connect(self._on_add_cube)
        self.toolbar.addAction(act_add)

        # Mise à jour du menu à chaque modification
        self.hypercube_manager.cubesChanged.connect(self._update_cube_menu)
        self._update_cube_menu(self.hypercube_manager.paths)

        # signal from register tool
        reg_widget = self.reg_dock.widget()
        reg_widget.alignedCubeReady.connect(self.hypercube_manager.addCube)  # get signal from register tool

        # Save with menu
        self.saveBtn = QtWidgets.QToolButton(self)
        self.saveBtn.setText("Save Cube")
        # self.saveBtn.setIcon(QIcon(os.path.join(self.ICONS_DIR, "save_icon.png")))
        # todo : add icon for save cube
        self.saveBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.saveMenu = QtWidgets.QMenu(self)
        self.saveBtn.setMenu(self.saveMenu)
        self.toolbar.addWidget(self.saveBtn)

        # update if list modified
        self.hypercube_manager.cubesChanged.connect(self._update_save_menu)
        self._update_save_menu(self.hypercube_manager.paths)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        act_suggestion = QAction("SUGGESTIONS", self)
        act_suggestion.setToolTip("Add a suggestion for the developper")
        act_suggestion.triggered.connect(self.open_suggestion_box)
        self.toolbar.addAction(act_suggestion)

        #### connect tools with hypercube manager for managing changes in cubeInfoTemp
        self.meta_dock.widget().metadataChanged.connect(self.hypercube_manager.updateMetadata)

        self.reg_dock.widget().cube_saved.connect(self.hypercube_manager.add_or_update_cube)
        self.gt_dock.widget().cube_saved.connect(self.hypercube_manager.add_or_update_cube)

        self.hypercube_manager.metadataUpdated.connect(self.meta_dock.widget().on_metadata_updated)
        self.hypercube_manager.metadataUpdated.connect(self.reg_dock.widget().load_cube_info)
        self.hypercube_manager.metadataUpdated.connect(self.gt_dock.widget().load_cube_info)
        # self.hypercube_manager.metadataUpdated.connect(self.data_viz_dock.widget().load_cube_info)

        self.reg_dock.widget().cubeLoaded.connect(lambda fp: self._on_tool_loaded_cube(fp, self.reg_dock.widget()))
        self.meta_dock.widget().cubeLoaded.connect(lambda fp: self._on_tool_loaded_cube(fp, self.meta_dock.widget()))
        # self.data_viz_dock.widget().cubeLoaded.connect(lambda fp: self._on_tool_loaded_cube(fp, self.data_viz_dock.widget()))
        self.gt_dock.widget().cubeLoaded.connect(lambda fp: self._on_tool_loaded_cube(fp, self.gt_dock.widget()))

        # visible docks of rightDock take all space possible

        self.centralWidget().hide()
        # self.showMaximized()

    def open_suggestion_box(self):
        self.suggestion_window = SuggestionWidget()
        self.suggestion_window.show()

    def addOrSyncCube(self, filepath: str) -> CubeInfoTemp:
        """
        Check if the cube is already in the manager.
        If so, return the existing CubeInfoTemp.
        Otherwise, load it from disk, add it to the list, and return it.
        """
        index = self.getIndexFromPath(filepath)
        if index != -1:
            return self._cubes[index]

        # Cube is not in the list → load and add it
        ci = CubeInfoTemp(filepath=filepath)
        hc = Hypercube(filepath=filepath, cube_info=ci, load_init=True)
        ci = hc.cube_info
        self._cubes.append(ci)
        self.cubesChanged.emit(self.paths)
        return ci

    def onToolButtonPress(self, dock, icon_name, tooltip):
        # act = dock.toggleViewAction()
        # act.setIcon(QIcon(os.path.join(self.ICONS_DIR, icon_name)))

        print(os.path.join(self.ICONS_DIR, icon_name))
        act = QAction(QIcon(os.path.join(self.ICONS_DIR, icon_name)), tooltip, self)
        act.setToolTip(tooltip)
        act.setCheckable(False)
        act.triggered.connect(lambda checked, d=dock: (d.show(), d.raise_()))
        # act.toggled.connect(
        #     lambda visible, d=dock: QTimer.singleShot(0, d.raise_) if visible else None
        # )
        self.toolbar.addAction(act)
        return act

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
        # dock.setTitleBarWidget(CustomDockTitleBar(dock))
        dock.setTitleBarWidget(CustomDockTitleBar(dock, style=self.title_bar_style))
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
        dock.setTitleBarWidget(CustomDockTitleBar(dock, style=self.title_bar_style))
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        return dock

    def _update_save_menu(self, paths):
        """Met à jour le menu déroulant du bouton Save Cube avec tous les cubes chargés."""
        self.saveMenu.clear()
        for idx, path in enumerate(paths):
            action = QtWidgets.QAction(os.path.basename(path), self)
            action.triggered.connect(lambda checked, i=idx: self.save_cube(i))
            self.saveMenu.addAction(action)

    def save_cube(self,index=None):
        ci     = self.hypercube_manager.getCubeInfo(index)

        ans=QMessageBox.question(self,'Save modification',f"Sure to save modification on disc for the cube :\n{ci.filepath}", QMessageBox.Yes | QMessageBox.Cancel)
        if ans==QMessageBox.Cancel:
            return

        cube=Hypercube(filepath=ci.filepath,metadata=ci.metadata_temp)
        cube.save(filepath=ci.filepath)
        print(f"cube saves as {ci.filepath}")

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
            ci=CubeInfoTemp(filepath=path)
            self.hypercube_manager.addCube(ci)

        if len(paths)==1:
            qm = QMessageBox()
            ans = qm.question(self, 'Cube loaded',
                              "Do you want to send the loaded cube to the tools ?",
                              qm.Yes | qm.No)
            if ans==qm.Yes:
                try:
                    index = self.hypercube_manager.getIndexFromPath(paths[0])
                    if index != -1:
                        self._send_to_all(index)
                except:
                    QMessageBox.warning(self,'Cube not loaded',
                              "Cube not loaded. Check format.")

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

    def _send_to_gt(self,index):
        widget = self.gt_dock.widget()
        ci = self.hypercube_manager.getCubeInfo(index)
        widget.load_cube(cube_info=ci)

    def _send_to_data_viz(self,index):
        widget = self.data_viz_dock.widget()
        ci = self.hypercube_manager.getCubeInfo(index)
        widget.open_hypercubes_and_GT(filepath=ci.filepath,cube_info=ci)

    def _send_to_registration(self,index,imov):
        widget = self.reg_dock.widget()
        ci = self.hypercube_manager.getCubeInfo(index)
        widget.load_cube(filepath=ci.filepath,cube_info=ci,i_mov=imov)

    def _send_to_all(self,index):
        self._send_to_data_viz(index)
        self._send_to_gt(index)
        self._send_to_metadata(index)
        self.reg_dock.widget().load_cube(1, self.hypercube_manager.paths[index])

    def _on_tool_loaded_cube(self, filepath, widget):
        index = self.hypercube_manager.getIndexFromPath(filepath)
        if index != -1:
            ci = self.hypercube_manager.getCubeInfo(index)
            widget.load_cube_info(ci)
        else:
            ci=Hypercube(filepath,load_init=True).cube_info
            self.hypercube_manager.addCube(ci)

    def _update_cube_menu(self, paths):
        """Met à jour le menu de cubes avec sous-menus et actions fonctionnelles."""
        self.cubeMenu.clear()
        for idx, path in enumerate(paths):
            # Sous-menu pour chaque cube
            sub = QtWidgets.QMenu(path, self)
            # Envoyer dant tous les docs
            act_all = QtWidgets.QAction("Send to All tools", self)
            act_all.triggered.connect(lambda checked, i=idx: self._send_to_all(i))
            sub.addAction(act_all)

            # Séparateur
            sub.addSeparator()

            # Envoyer au dock viz
            act_viz = QtWidgets.QAction("Send to Vizualisation tool", self)
            act_viz.triggered.connect(lambda checked, i=idx: self._send_to_data_viz(i))
            sub.addAction(act_viz)
            # Envoyer au dock reg
            menu_load_reg=QtWidgets.QMenu("Send to Register Tool", sub)
            act_fix = QtWidgets.QAction("Fixed Cube", self)
            act_fix.triggered.connect(
                lambda _, i=idx: self._send_to_registration(i,0)
            )
            menu_load_reg.addAction(act_fix)
            # Action Moving
            act_mov = QtWidgets.QAction("Moving Cube", self)
            act_mov.triggered.connect(
                lambda _, i=idx:  self._send_to_registration(i,1)
            )
            menu_load_reg.addAction(act_mov)
            sub.addMenu(menu_load_reg)

            # send to file browser
            act_browser = QtWidgets.QAction("Send to File Browser", self)
            act_browser.triggered.connect(lambda checked, i=idx: self._open_file_browser_for_index(i))
            sub.addAction(act_browser)

            # Envoyer au dock metadata
            act_meta = QtWidgets.QAction("Send to Metadata", self)
            act_meta.triggered.connect(lambda checked, i=idx: self._send_to_metadata(i))
            sub.addAction(act_meta)

            # Envoyer au dock gt
            act_gt = QtWidgets.QAction("Send to GT", self)
            act_gt.triggered.connect(lambda checked, i=idx: self._send_to_gt(i))
            sub.addAction(act_gt)

            # Séparateur
            sub.addSeparator()

            # Get Cube Info from File”
            # act_get_info = QtWidgets.QAction("Get cube_info from file…", self)
            # act_get_info.triggered.connect(lambda _, i=idx: self._on_get_cube_info(i))
            # sub.addAction(act_get_info)

            # Save Cube
            act_save = QtWidgets.QAction("Save modification to disc", self)
            act_save.triggered.connect(lambda checked, i=idx: self.save_cube(i))
            sub.addAction(act_save)

            # Séparateur
            sub.addSeparator()
            # Supprimer de la liste
            act_rm = QtWidgets.QAction("Remove from list", self)
            act_rm.triggered.connect(lambda checked, i=idx: self.hypercube_manager.removeCube(i))
            sub.addAction(act_rm)
            # Ajouter sous-menu au menu principal
            self.cubeMenu.addMenu(sub)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "You may loose unsaved modifications. \nAre you sure you want to quit the application?",
            QMessageBox.Ok | QMessageBox.Cancel
        )

        if reply == QMessageBox.Ok:
            event.accept()  # Proceed with closing the app
        else:
            event.ignore()  # Cancel the close event

# Configure error logging
# Get absolute path of log folder (support PyInstaller frozen mode)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    exe_dir = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(__file__)
    exe_dir = base_path

log_dir = os.path.join(exe_dir, "log")
os.makedirs(log_dir, exist_ok=True)  # ← crée le dossier s’il n’existe pas

logging.basicConfig(
    filename=os.path.join(log_dir, "error.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

suggestion_logger = logging.getLogger("suggestion_logger")
suggestion_handler = logging.FileHandler(os.path.join(log_dir, "suggestions.log"))
suggestion_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
suggestion_logger.addHandler(suggestion_handler)
suggestion_logger.setLevel(logging.INFO)

class SuggestionWidget(QWidget):
    """ Window to get suggestion of users """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suggestion Box")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Write your suggestion or feedback below:"))
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_suggestion)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def submit_suggestion(self):
        text = self.text_edit.toPlainText().strip()
        if text:
            suggestion_logger.info(text)
            self.close()
            QMessageBox.information(None, "Thank you",
                                    "Suggestion has been logged in 'suggestions.log'.")  # show to user that comment has been taken into account
        else:
            self.submit_button.setText("Please write something!")

class ErrorDialog(QDialog):
    """ Window to open in development phase to describe exception seen"""
    def __init__(self, error_text):
        super().__init__()
        self.setWindowTitle("Unexpected Error")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("An unexpected error occurred:"))
        layout.addWidget(QLabel(error_text))

        layout.addWidget(QLabel("If you wish, describe what you were doing before the error:"))
        self.user_input = QTextEdit()
        layout.addWidget(self.user_input)

        self.btn_send = QPushButton("OK")
        self.btn_send.clicked.connect(self.accept)
        layout.addWidget(self.btn_send)

        self.setLayout(layout)

    def get_user_comment(self):
        return self.user_input.toPlainText()

def excepthook(exc_type, exc_value, exc_traceback):
    """Capture exception and send it to log file, and show warning message"""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    logging.error("Uncaught exception:\n%s", error_msg) # log error

    # Show dialog debug to the user
    dlg = ErrorDialog(str(exc_value))
    dlg.exec_()

    user_comment = dlg.get_user_comment()

    if user_comment.strip():
        logging.error("User comment: %s", user_comment)

    QMessageBox.information(None, "Log Saved", "Error has been logged in 'error.log'.") # show to user that comment has been taken into account

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
    """ Check if resolution change to adapt font size """
    global last_width  # On garde la dernière largeur connue
    center_point = main.geometry().center()
    screen = app.screenAt(center_point)

    if screen is not None:
        current_width = screen.size().width()
        if current_width != last_width:
            update_font(app, current_width)
            last_width = current_width
    else:
        print("[⚠️] Fenêtre en dehors de tout écran détecté. Résolution inchangée.")

if __name__ == "__main__":

    # sys.excepthook = excepthook #set the exception handler

    app = QtWidgets.QApplication(sys.argv)

    update_font(app)
    apply_fusion_border_highlight(app)

    main = MainApp()
    main.show()

    try:
        import matlab.engine

        print(" [ :-) ] matlab.engine loaded with success")
    except Exception as e:
        print(f" [ !!! ] Failed to load matlab.engine: {e}")

    # Timer for screen resolution check
    last_width = app.primaryScreen().size().width()
    timer = QTimer()
    timer.timeout.connect(check_resolution_change)
    timer.start(500)  # Vérifie toutes les 500 ms
    sys.exit(app.exec_())