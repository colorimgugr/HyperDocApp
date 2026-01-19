# cd C:\Users\Usuario\Documents\GitHub\Hypertool
# python MainWindow.py
# sys.excepthook = excepthook #set the exception handler
# pyinstaller  --noconfirm --noconsole --exclude-module tensorflow --exclude-module torch --exclude-module matlab --icon="interface/icons/hyperdoc_logo_transparente.ico" --add-data "interface/icons:Hypertool/interface/icons" --add-data "hypercubes/white_ref_reflectance_data:hypercubes/white_ref_reflectance_data" --add-data "ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv:ground_truth"  --add-data "data_vizualisation/Spatially registered minicubes equivalence.csv:data_vizualisation" --add-data "illumination/Illuminants.csv:illumination" --add-data "unmixing/data:unmixing/data"  MainWindow.py
# C:\Envs\py37test\Scripts\activate

# GUI Qt

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtGui import QFont,QIcon, QPalette, QDesktopServices
from PyQt5.QtWidgets import (QStyleFactory, QAction, QSizePolicy,QPushButton,
                             QTextEdit,QToolTip, QCheckBox, QWidgetAction,)

## Python import
import traceback
import logging

## bloc non important warning
import warnings
warnings.filterwarnings("ignore", message="Parameters with non-lowercase names")

# projects import
from hypercubes.hypercube import *
from data_vizualisation.data_vizualisation_tool import Data_Viz_Window
from registration.register_tool        import RegistrationApp
from interface.hypercube_manager import HypercubeManager
from metadata.metadata_tool import MetadataTool
from ground_truth.ground_truth_tool import GroundTruthWidget
from minicube.minicube_tool import MiniCubeTool
from identification.identification_tool import IdentificationWidget
from illumination.illumination_tool import IlluminationWidget
from unmixing.unmixing_tool import UnmixingTool

# grafics to control changes
import matplotlib.pyplot as plt

def apply_fusion_border_highlight(app,
                                  border_color: str = "#888888",
                                  title_bg:      str = "#E0E0E0",
                                  separator_hover: str = "#AAAAAA",
                                  window_bg:     str = "#F5F5F5",
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

class CubeDestinationDialog(QtWidgets.QDialog):
    """
    Petite boîte de dialogue qui propose d'envoyer le cube
    vers un outil particulier, ou juste l'ajouter à la liste.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Send cube to tool")
        self.choice = None  # stockera un petit mot-clé décrivant le choix

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Cube added to list.\nWhere do you want to send it?")
        layout.addWidget(label)

        # Layout de boutons
        btn_layout = QtWidgets.QGridLayout()
        row = 0

        def add_button(text, choice_key, row, col):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda: self._set_choice_and_accept(choice_key))
            btn_layout.addWidget(btn, row, col)
            return btn

        # Ligne 1
        add_button("Data Visualization", "data_viz", row, 0)
        add_button("Ground Truth", "gt", row, 1)
        row += 1

        # Ligne 2
        add_button("Metadata", "meta", row, 0)
        add_button("File Browser", "browser", row, 1)
        row += 1

        # Ligne 3
        add_button("Minicube tool", "minicube", row, 0)
        add_button("Illumination", "illumination", row, 1)
        row += 1

        # Ligne 4 : Registration
        add_button("Registration (Fixed)", "reg_fix", row, 0)
        add_button("Registration (Moving)", "reg_mov", row, 1)
        row += 1

        # Ligne 5 : Identification
        add_button("Identification VNIR", "ident_vnir", row, 0)
        add_button("Identification SWIR", "ident_swir", row, 1)
        row += 1

        # Ligne 6 : Unmixing
        add_button("Unmixing VNIR", "unmix_vnir", row, 0)
        add_button("Unmixing SWIR", "unmix_swir", row, 1)

        layout.addLayout(btn_layout)

        # Boutons Cancel / Close en bas (optionnel)
        btn_close = QtWidgets.QPushButton("Just add to list")
        btn_close.clicked.connect(self.reject)
        layout.addWidget(btn_close)

    def _set_choice_and_accept(self, choice_key):
        # choice_key peut être None (Just add), ou une string comme "data_viz"
        self.choice = choice_key
        self.accept()

class GlobalToolTipFilter(QtCore.QObject):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.ToolTip and not self.enabled:
            return True  # on bloque l'affichage du tooltip
        return False

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp")
        self.resize(1200, 800)
        self.setCentralWidget(QtWidgets.QWidget())

        if getattr(sys, 'frozen', False):
            # if from .exe de pyinstaler
            self.BASE_DIR = sys._MEIPASS
        else:
            # from Python script
            CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.BASE_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))

        self.active_cube=Hypercube() # solo cube que esta cargado con sus datos en la app gnl.

        self.ICONS_DIR = os.path.join(self.BASE_DIR,"Hypertool","interface", "icons")
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

        # Hypercube Manager
        self.hypercube_manager = HypercubeManager()

        # make left docks with meta and file browser
        self.file_browser_dock = self._add_file_browser_dock() # left dock with file browser
        self.file_browser_dock.setVisible(False) # raise meta and "hide in tab" file browser
        self.meta_dock=self._add_dock("Metadata",   MetadataTool,     QtCore.Qt.LeftDockWidgetArea) # add meta to left dock
        # self.tabifyDockWidget(self.file_browser_dock, self.meta_dock)
        # self.meta_dock.raise_() # raise meta and "hide in tab" file browser
        self.meta_dock.setVisible(False) # raise meta and "hide in tab" file browser

        # make "central" dock with visuals tools
        self.data_viz_dock =self._add_dock("Data Visualization", Data_Viz_Window,  QtCore.Qt.RightDockWidgetArea)
        self.reg_dock=self._add_dock("Registration",   RegistrationApp,     QtCore.Qt.RightDockWidgetArea)
        self.gt_dock=self._add_dock("Ground Truth",   GroundTruthWidget,     QtCore.Qt.RightDockWidgetArea)
        self.minicube_dock=self._add_dock("Minicube Extract",   MiniCubeTool,     QtCore.Qt.RightDockWidgetArea)
        self.identification_dock=self._add_dock("Identification", IdentificationWidget, QtCore.Qt.RightDockWidgetArea)
        self.unmixing_dock=self._add_dock("Unmixing", UnmixingTool, QtCore.Qt.RightDockWidgetArea)
        self.illumination_dock=self._add_dock("Illumination", IlluminationWidget, QtCore.Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self.reg_dock, self.gt_dock)
        self.tabifyDockWidget(self.reg_dock, self.data_viz_dock)
        self.tabifyDockWidget(self.reg_dock, self.minicube_dock)
        self.tabifyDockWidget(self.reg_dock, self.identification_dock)
        self.tabifyDockWidget(self.reg_dock, self.illumination_dock)
        self.tabifyDockWidget(self.reg_dock, self.unmixing_dock)

        self.gt_dock.raise_()

        # Tool menu
        view = self.menuBar().addMenu("Tools")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view.addAction(dock.toggleViewAction())

        # ─── Toolbar “Quick Tools” ───────────────────────────────────────────
        self.toolbar = self.addToolBar("Quick Tools")
        self.toolbar.setIconSize(QSize(48, 48))  # Taille des icônes
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)  #ToolButtonIconOnly ou TextUnderIcon)

        act_file = self.onToolButtonPress(self.file_browser_dock,icon_name="file_browser_icon.png",    tooltip="File Browser — Browse cube files, inspect internal datasets, and send selections to the tools.")
        act_met = self.onToolButtonPress(self.meta_dock, "metadata_icon.png",     "Metadata — View and edit cube metadata with safe synchronization.")
        self.toolbar.addSeparator()
        act_mini=self.onToolButtonPress(self.minicube_dock, "minicube_icon.png",     "Minicube Extract — Crop regions of interest and export compact minicubes while preserving parent metadata.")
        act_data = self.onToolButtonPress(self.data_viz_dock, "icon_data_viz.svg", "Data Visualization — Explore cubes of the dataset interactively")
        act_illumination = self.onToolButtonPress(self.illumination_dock, "illumination_icon.png", "Illumination — Inspect illumination effects")
        act_reg = self.onToolButtonPress(self.reg_dock, "registration_icon.png",     "Registration — Align a moving cube to a fixed cube using feature matching and geometric transforms.")
        act_gt =self.onToolButtonPress(self.gt_dock, "GT_icon_1.png",     "Ground Truth — Create, edit, and export labeled maps (manual selection + supervised/unsupervised segmentation).")
        self.toolbar.addSeparator()
        act_ident=self.onToolButtonPress(self.identification_dock,"Ident_icon.png",    "Identification — Run ink/substrate identification workflows using the dataset.")
        act_unmix=self.onToolButtonPress(self.unmixing_dock,"unmixing_icon.png",    "Unmixing — Estimate per-pixel abundances by fitting mixtures of reference spectra and visualize abundance maps.")

        self.toolbar.addSeparator()

        # Cubes "list"
        self.cubeBtn = QtWidgets.QToolButton(self)
        self.cubeBtn.setText("Cubes list   ")
        self.cubeBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.cubeBtn.setToolTip(
            "Cubes list — Select the loaded cube to manage."
        )
        # self.cubeBtn.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        self.toolbar.addWidget(self.cubeBtn)

        # Création du menu hiérarchique
        self.cubeMenu = QtWidgets.QMenu(self)
        self.cubeBtn.setMenu(self.cubeMenu)

        # Action Add File in list of cubes
        act_add = QAction("Open new cube(s)", self)
        act_add.setToolTip(
            "Open new cube(s) — Load one or multiple hyperspectral cubes and add them to the cubes list."
        )
        act_add.triggered.connect(self._on_add_cube)
        self.toolbar.addAction(act_add)

        # signal from register tool
        reg_widget = self.reg_dock.widget()

        # Save with menu
        self.saveBtn = QtWidgets.QToolButton(self)
        self.saveBtn.setText("Save Cube")
        self.saveBtn.setToolTip(
            "Save cube — Export the selected cube to disk (including updated metadata)."
        )
        # self.saveBtn.setIcon(QIcon(os.path.join(self.ICONS_DIR, "save_icon.png")))
        self.saveBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.saveMenu = QtWidgets.QMenu(self)
        self.saveBtn.setMenu(self.saveMenu)
        self.toolbar.addWidget(self.saveBtn)

        # update if list modified
        self.hypercube_manager.cubes_changed.connect(lambda: self._update_save_menu(self.hypercube_manager.paths))
        self._update_save_menu(self.hypercube_manager.paths)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        act_suggestion = QAction("SUGGESTIONS", self)
        act_suggestion.setToolTip(
            "Suggestions — Send feedback or feature requests to improve the tool."
        )
        act_suggestion.triggered.connect(self.open_suggestion_box)
        self.toolbar.addAction(act_suggestion)

        # Checkbox "Show tooltips" dans la toolbar (à droite, près de SUGGESTIONS)
        self.checkBox_show_tooltips = QCheckBox("Show Tooltips for help")
        self.checkBox_show_tooltips.setToolTip("Enable/disable all tooltips in the application")

        tooltips_action = QWidgetAction(self)
        tooltips_action.setDefaultWidget(self.checkBox_show_tooltips)
        self.toolbar.addAction(tooltips_action)

        # Connexion + état initial
        self.checkBox_show_tooltips.toggled.connect(self._on_show_tooltips_toggled)
        self.checkBox_show_tooltips.setChecked(True)

        #### connect tools with hypercube manager for managing changes in cubeInfoTemp
        self.meta_dock.widget().metadataChanged.connect(self.hypercube_manager.update_metadata)

        self.hypercube_manager.metadata_updated.connect(self.data_viz_dock.widget().on_metadata_updated)
        self.hypercube_manager.metadata_updated.connect(self.meta_dock.widget().on_metadata_updated)
        self.hypercube_manager.metadata_updated.connect(self.reg_dock.widget().load_cube_info)
        self.hypercube_manager.metadata_updated.connect(self.gt_dock.widget().load_cube_info)

        self.hypercube_manager.cubes_changed.connect(self._update_cube_menu)

        self.reg_dock.widget().cubeLoaded.connect(lambda hc: self._on_tool_loaded_cube(hc, self.reg_dock.widget()))
        self.meta_dock.widget().cubeLoaded.connect(lambda hc: self._on_tool_loaded_cube(hc, self.meta_dock.widget()))
        self.gt_dock.widget().cubeLoaded.connect(lambda hc: self._on_tool_loaded_cube(hc, self.gt_dock.widget()))

        self._connect_tool_cube_saved_signals()

        # visible docks of rightDock take all space possible

        self.centralWidget().hide()
        # self.showMaximized()

        self.checkBox_show_tooltips.toggled.connect(self._on_show_tooltips_toggled)
        self.checkBox_show_tooltips.setChecked(True)  # ou valeur depuis QSettings

    def _on_show_tooltips_toggled(self, checked: bool):
        # tooltip_filter est global (défini dans __main__), on y accède via QApplication
        app = QtWidgets.QApplication.instance()
        # on l’a stocké comme attribut pour pouvoir le retrouver
        app._global_tooltip_filter.enabled = checked

    def _connect_tool_cube_saved_signals(self):
        """
        Connecte les signaux cube_saved des outils à la mise à jour du menu.
        """

        def handle_cube_saved(ci):
            self.hypercube_manager.add_or_sync_cube(ci.filepath)
            QtCore.QTimer.singleShot(0, lambda: self._update_cube_menu(self.hypercube_manager.paths))
            QtCore.QTimer.singleShot(0, lambda: self._update_save_menu(self.hypercube_manager.paths))
            print(f"[INFO] Cube ajouté/synchronisé : {ci.filepath}")

        for tool in [
            self.reg_dock.widget(),
            self.gt_dock.widget(),
            self.minicube_dock.widget(),
        ]:
            try:
                tool.cube_saved.connect(handle_cube_saved)
            except AttributeError:
                print(f"[Warning] Le widget {tool.__class__.__name__} n’a pas de signal 'cube_saved'.")

    def open_suggestion_box(self):
        # Public distribution: redirect suggestions to an online form.
        url = "https://docs.google.com/forms/d/e/1FAIpQLSff0dXWaO57mmHGBjkNVxKFjhxLHEYXgzifMafPH8soU93PdA/viewform?usp=publish-editor"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Suggestions / Feedback")
        msg.setText("Submitting a suggestion will open a Google Form in your web browser.")
        msg.setInformativeText(
            "If you continue, your default browser will be opened on the feedback form.\n\nContinue?"
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)

        if msg.exec_() != QMessageBox.Ok:
            return

        opened = QDesktopServices.openUrl(QUrl(url))
        if not opened:
            QMessageBox.warning(
                self,
                "Unable to open browser",
                "The application could not open your browser automatically.\n\n"
                f"Please open this link manually:\n{url}",
            )

    def open_suggestion_box_intern(self):
        self.suggestion_window = SuggestionWidget()
        self.suggestion_window.show()

    def onToolButtonPress(self, dock, icon_name, tooltip):
        # act = dock.toggleViewAction()
        # act.setIcon(QIcon(os.path.join(self.ICONS_DIR, icon_name)))

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
        """
        Called when HDF5BrowserWidget OK is pressed.
        """
        self.hypercube_manager.metadata_updated.emit(updated_ci)

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
        ci = CubeInfoTemp(_filepath=None)

        # 2) instancie le widget
        widget = HDF5BrowserWidget(
            cube_info=ci,
            filepath=None,
            parent=self,
            closable=False
        )

        # 3) connecte le signal accepted
        widget.accepted.connect(self._on_file_browser_accepted)
        self.hypercube_manager.metadata_updated.connect(widget.load_from_cubeinfo)
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
            action.triggered.connect(lambda checked, p=path: self.save_cube(p))
            self.saveMenu.addAction(action)

    def _on_cubes_changed(self, paths: list):
        self.cubeMenu.clear()

        for path in paths:
            action = QtWidgets.QAction(Path(path).name, self)  # n’afficher que le nom de fichier
            # stocke le chemin complet dans les data de l’action
            action.setData(path)
            self.cubeMenu.addAction(action)

    def save_cube(self,filepath):
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        ans=QMessageBox.question(self,'Save modification',f"Sure to save modification on disc for the cube :\n{ci.filepath}", QMessageBox.Yes | QMessageBox.Cancel)
        if ans==QMessageBox.Cancel:
            return

        filters = (
            "Supported files (*.h5 *.mat *.hdr);;"
            "HDF5 files (*.h5);;"
            "MATLAB files (*.mat);;"
            "ENVI header (*.hdr)"
        )

        filepath_new,_ = QFileDialog.getSaveFileName(parent=self,caption="Save cube As…",directory=filepath,filter=filters)
        hc.save(filepath=filepath_new,meta_from_cube_info=True)

        if filepath_new!=filepath:
            self.hypercube_manager.rename_cube(filepath, filepath_new)
            self._update_cube_menu(self.hypercube_manager.paths)
            ci = self.hypercube_manager.add_or_sync_cube(filepath_new)

        print(f"cube saves as {ci.filepath}")

    def _on_add_cube(self,paths=None):
        filters = (
            "Supported files (*.h5 *.mat *.hdr);;"
            "HDF5 files (*.h5);;"
            "MATLAB files (*.mat);;"
            "ENVI header (*.hdr)"
        )

        if not isinstance(paths, (list, tuple)) or len(paths) == 0:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select Hypercubes",
                "",
                filters
            )

        if not paths:
            return

        for path in paths:
            ci = self.hypercube_manager.add_or_sync_cube(path)

        self._update_cube_menu(self.hypercube_manager.paths)

        if len(paths) == 1:
            filepath = paths[0]
            ci = self.hypercube_manager.add_or_sync_cube(filepath)
            hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

            print(f"Cube '{paths[0]}' loaded in cache for use.")

            dlg = CubeDestinationDialog(self)
            result = dlg.exec_()

            # Si Cancel / fermeture → on ne fait rien de plus (cube déjà dans la liste)
            if result != QtWidgets.QDialog.Accepted:
                return

            choice = dlg.choice

            # None ou chaîne vide → Just add to cube list
            if not choice:
                return

            try:
                if choice == "all":
                    self._send_to_all(filepath)
                elif choice == "data_viz":
                    self._send_to_data_viz(filepath)
                elif choice == "gt":
                    self._send_to_gt(filepath)
                elif choice == "meta":
                    self._send_to_metadata(filepath)
                elif choice == "browser":
                    self._send_to_browser(filepath)
                elif choice == "minicube":
                    self._send_to_minicube(filepath)
                elif choice == "illumination":
                    self._send_to_illumination(filepath)
                elif choice == "reg_fix":
                    self._send_to_registration(filepath, 0)
                elif choice == "reg_mov":
                    self._send_to_registration(filepath, 1)
                elif choice == "ident_vnir":
                    self._send_to_identification(filepath, 0)
                elif choice == "ident_swir":
                    self._send_to_identification(filepath, 1)
                elif choice == "unmix_vnir":
                    self._send_to_unmix(filepath, 0)
                elif choice == "unmix_swir":
                    self._send_to_unmix(filepath, 1)
            except Exception:
                QMessageBox.warning(
                    self,
                    'Cube not loaded',
                    "Cube not loaded to selected tool. Check format."
                )

    def calib_cube(self,filepath):
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)
        hc.calibrating_from_image_extract()

    def _send_to_metadata(self,filepath,show_tab=True):
        widget = self.meta_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        if show_tab:
            self.meta_dock.raise_()

        widget.set_cube_info(ci)
        widget.update_combo_meta(init=True)

    def _send_to_gt(self,filepath,show_tab=True):
        widget = self.gt_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        if show_tab:
            self.gt_dock.raise_()

        # print(f'[send GT] ci.filepath : {ci.filepath}')
        # print(f'[send GT] hc.ci.filepath : {ci.filepath}')

        if hc.data is None:
            widget.load_cube(cube_info=ci)
        else:
            widget.load_cube(cube_info=ci,cube=hc)

    def _send_to_data_viz(self,filepath,show_tab=True):
        widget = self.data_viz_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)
        print(f'[send VIZ] ci.filepath : {ci.filepath}')
        print(f'[send VIZ] hc.ci.filepath : {ci.filepath}')

        if show_tab:
            self.data_viz_dock.raise_()

        if hc.data is None:
            widget.open_hypercubes_and_GT(filepath=ci.filepath, cube_info=ci)
        else:
            hc.cube_info = ci
            widget.open_hypercubes_and_GT(filepath=ci.filepath, cube_info=ci, cube=hc)

    def _send_to_registration(self,filepath,imov,show_tab=True):
        widget = self.reg_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        if show_tab:
            self.reg_dock.raise_()

        print(f'[send REG] ci.filepath : {ci.filepath}')
        print(f'[send REG] hc.ci.filepath : {ci.filepath}')
        if hc.data is None:
            widget.load_cube(filepath=ci.filepath,cube_info=ci,i_mov=imov)
        else:
            print('Try registered with cube sended')
            hc.cube_info = ci
            widget.load_cube(filepath=ci.filepath,cube_info=ci,i_mov=imov,cube=hc)

    def _send_to_identification(self,filepath,icube,show_tab=True):
        widget = self.identification_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        range=['VNIR','SWIR'][icube]

        if show_tab:
            self.identification_dock.raise_()

        if hc.data is None:
            widget.load_cube(filepath=ci.filepath,cube_info=ci,range=range)
        else:
            print('Try registered with cube sended')
            hc.cube_info = ci
            widget.load_cube(filepath=ci.filepath,cube_info=ci,range=range,cube=hc)
            print(f'[SEND TO IDENT] path : {filepath} of range {range}')
        pass

    def _send_to_unmix(self,filepath,icube,show_tab=True):
        widget = self.unmixing_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        range=['VNIR','SWIR'][icube]

        if show_tab:
            self.unmixing_dock.raise_()

        if hc.data is None:
            widget.load_cube(filepath=ci.filepath,cube_info=ci,range=range)
        else:
            print('Try registered with cube sended')
            hc.cube_info = ci
            widget.load_cube(filepath=ci.filepath,cube_info=ci,range=range,cube=hc)
            print(f'[SEND TO IDENT] path : {filepath} of range {range}')
        pass

    def _send_to_minicube(self,filepath,show_tab=True):
        widget = self.minicube_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        if show_tab:
            self.minicube_dock.raise_()

        if hc.data is None:
            widget.load_cube(cube_info=ci)
        else:
            widget.load_cube(cube_info=ci, cube=hc)

    def _send_to_browser(self, filepath,show_tab=True):
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)
        widget = self.file_browser_dock.widget()

        if show_tab:
            self.file_browser_dock.raise_()

         # 1) Ré-associe l'objet métier et le filepath
        widget.cube_info = ci
        widget.filepath  = ci.filepath

        # 2) Pré-remplissage des QLineEdit
        widget.le_cube.setText(ci.data_path       or "")
        widget.le_wl.setText(ci.wl_path         or "")
        widget.le_meta.setText(ci.metadata_path   or "")
        widget.le_gtmap.setText (ci.gtmap_path or "")
        # comboBox_channel_wl : 0 = First, 1 = Other (à ajuster)
        widget.comboBox_channel_wl.setCurrentIndex(0 if ci.wl_trans else 1)

        # 3) Rebuild de l'arbre HDF5/MAT
        widget._load_file()

        # 4) Affichage du dock
        # widget.show()

    def _send_to_illumination(self, filepath,show_tab=True):
        print('Send to illumination')
        widget = self.illumination_dock.widget()
        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        if show_tab:
            self.illumination_dock.raise_()

        if hc.data is None:
            widget.on_load_cube(cube_info=ci)
        else:
            widget.on_load_cube(cube_info=ci, cube=hc)

    def _send_to_all(self,filepath):

        ci = self.hypercube_manager.add_or_sync_cube(filepath)
        hc = self.hypercube_manager.get_loaded_cube(filepath, cube_info=ci)

        self._send_to_data_viz(filepath,show_tab=False)
        self._send_to_gt(filepath,show_tab=False)
        self._send_to_metadata(filepath,show_tab=False)
        self._send_to_registration(filepath,1,show_tab=False)
        self._send_to_browser(filepath,show_tab=False)
        self._send_to_minicube(filepath,show_tab=False)
        self._send_to_illumination(filepath,show_tab=False)
        self._send_to_minicube(filepath,show_tab=False)

        kind=(min(hc.wl)>600)
        self._send_to_identification(filepath, kind, show_tab=False)
        self._send_to_unmix(filepath, kind, show_tab=False)

    def _on_tool_loaded_cube(self, hc: Hypercube, tool_widget):
        # Chemin résolu du cube
        path = hc.filepath

        # Détecte si le cube existait déjà
        already_registered = path in self.hypercube_manager.paths

        # 1) Récupère ou crée le CubeInfoTemp « officiel »
        ci = self.hypercube_manager.add_or_sync_cube(path)

        # 2) Fait pointer l'objet Hypercube sur ce ci
        hc.cube_info = ci

        # 3) Enregistre le Hypercube dans le cache LRU du manager
        #    (get_loaded_cube va le charger s'il n'y est pas déjà)
        self.hypercube_manager.get_loaded_cube(path, cube_info=ci)

        # 4) Notification utilisateur / log
        print(f"[MainWindow] Cube loaded from {tool_widget.__class__.__name__}: {path}")
        if already_registered:
            print(f"[MainWindow] Cube already present in list: {path}")

        self._update_cube_menu(self.hypercube_manager.paths)

    def _update_cube_menu(self, paths):
        """Met à jour le menu de cubes avec sous-menus et actions fonctionnelles."""
        self.cubeMenu.clear()
        for idx, path in enumerate(paths):
            sub = QtWidgets.QMenu(path, self)
            # send to tools
            act_all = QtWidgets.QAction("Send to All tools", self)
            act_all.triggered.connect(lambda checked, p=path: self._send_to_all(p))
            sub.addAction(act_all)

            # Séparateur
            sub.addSeparator()

            # Envoyer au dock mini
            act_mini = QtWidgets.QAction("Send to Minicube tool", self)
            act_mini.triggered.connect(lambda checked, p=path: self._send_to_minicube(p))
            sub.addAction(act_mini)

            # Envoyer au dock viz
            act_viz = QtWidgets.QAction("Send to Vizualisation tool", self)
            act_viz.triggered.connect(lambda checked, p=path: self._send_to_data_viz(p))
            sub.addAction(act_viz)
            # Envoyer au dock reg
            menu_load_reg=QtWidgets.QMenu("Send to Register Tool", sub)
            act_fix = QtWidgets.QAction("Fixed Cube", self)
            act_fix.triggered.connect(
                lambda _, p=path: self._send_to_registration(p,0)
            )
            menu_load_reg.addAction(act_fix)
            # Action Moving
            act_mov = QtWidgets.QAction("Moving Cube", self)
            act_mov.triggered.connect(
                lambda _, p=path:  self._send_to_registration(p,1)
            )
            menu_load_reg.addAction(act_mov)
            sub.addMenu(menu_load_reg)

            # send to file browser
            act_browser = QtWidgets.QAction("Send to File Browser", self)
            act_browser.triggered.connect(lambda checked, p=path: self._send_to_browser(p))
            sub.addAction(act_browser)

            # Envoyer au dock metadata
            act_meta = QtWidgets.QAction("Send to Metadata", self)
            act_meta.triggered.connect(lambda checked, p=path: self._send_to_metadata(p))
            sub.addAction(act_meta)

            # Envoyer au dock gt
            act_gt = QtWidgets.QAction("Send to GT", self)
            act_gt.triggered.connect(lambda checked, p=path: self._send_to_gt(p))
            sub.addAction(act_gt)

            # Envoyer au dock ident
            menu_load_ident = QtWidgets.QMenu("Send to Identification Tool", sub)
            act_ident_vnir = QtWidgets.QAction("VNIR Cube", self)
            act_ident_vnir.triggered.connect(
                lambda _, p=path: self._send_to_identification(p, 0)
            )
            menu_load_ident.addAction(act_ident_vnir)

            act_ident_swir = QtWidgets.QAction("SWIR Cube", self)
            act_ident_swir.triggered.connect(
                lambda _, p=path: self._send_to_identification(p, 1)
            )
            menu_load_ident.addAction(act_ident_swir)

            sub.addMenu(menu_load_ident)

            # Envoyer au dock unmix
            menu_load_unmix = QtWidgets.QMenu("Send to Unmix Tool", sub)
            act_unmix_vnir = QtWidgets.QAction("VNIR Cube", self)
            act_unmix_vnir.triggered.connect(
                lambda _, p=path: self._send_to_unmix(p, 0)
            )
            menu_load_unmix.addAction(act_unmix_vnir)

            act_unmix_swir = QtWidgets.QAction("SWIR Cube", self)
            act_unmix_swir.triggered.connect(
                lambda _, p=path: self._send_to_unmix(p, 1)
            )
            menu_load_unmix.addAction(act_unmix_swir)

            sub.addMenu(menu_load_unmix)

            # Envoyer au dock illum
            act_illum = QtWidgets.QAction("Send to Illumination", self)
            act_illum.triggered.connect(lambda checked, p=path: self._send_to_illumination(p))
            sub.addAction(act_illum)

            # White calibration
            sub.addSeparator()
            act_calib = QtWidgets.QAction("Process white calibration", self)
            act_calib.triggered.connect(lambda checked, p=path: self.calib_cube(p))
            sub.addAction(act_calib)

            # Save Cube
            sub.addSeparator()
            act_save = QtWidgets.QAction("Save modification to disc", self)
            act_save.triggered.connect(lambda checked, p=path: self.save_cube(p))
            sub.addAction(act_save)

            sub.addSeparator()
            # Quit from list
            act_rm = QtWidgets.QAction("Remove from list", self)
            act_rm.triggered.connect(lambda checked, p=path: self.remove_cube(p))
            sub.addAction(act_rm)
            # Ajouter sous-menu au menu principal
            self.cubeMenu.addMenu(sub)

    def remove_cube(self,filepath):
        self.hypercube_manager.remove_cube(filepath)
        self._update_cube_menu(self.hypercube_manager.paths)

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
    plt.rcParams.update({"font.size": font_size + 2, "font.family": _font})

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

if __name__ == "__main__":

    sys.excepthook = excepthook #set the exception handler

    app = QtWidgets.QApplication(sys.argv)

    tooltip_filter = GlobalToolTipFilter(enabled=True)
    app.installEventFilter(tooltip_filter)
    app._global_tooltip_filter = tooltip_filter

    update_font(app)
    apply_fusion_border_highlight(app)

    main = MainApp()
    main.show()

    try :
        folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\Samples\minicubes/'
        fname = '00189-VNIR-mock-up.h5'
        filepath = os.path.join(folder, fname)
        main._on_add_cube([filepath,filepath.replace('189','191')])
    except:
        pass

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