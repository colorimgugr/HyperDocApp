import os
import numpy as np
import cv2
from PyQt5.QtGui    import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox,QInputDialog , QSplitter,QGraphicsView,QLabel
from PyQt5.QtCore import Qt, QEvent, QRect, QRectF,QPointF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.path import Path

from scipy.spatial import distance as spdist

from hypercubes.hypercube import Hypercube
from registration.register_tool import ZoomableGraphicsView
# Import the compiled UI
from ground_truth.ground_truth_window import Ui_GroundTruthWidget

# todo : give GT labels names and number for RGB code ? -> save GT in new dataset of file + png
# todo : check if dobbles in samples at end of selection process -> keep last selection.
# todo : band selection on spectra graph
# todo : show number of pixel selected and number of pixel of each class after segmentation
# todo : loading cube to solve

class GroundTruthWidget(QWidget, Ui_GroundTruthWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up UI from compiled .py
        self.setupUi(self)

        self.selecting_pixels = False # mode selection ref activated
        self._pixel_selecting = False  # for manual pixel selection for dragging mode
        self.erase_selection = False # erase mode on or off
        self._pixel_coords = []  # collected  (x,y) during dragging
        self._preview_mask = None # temp mask during dragging pixel selection
        self.class_colors = {}  # color of each class
        self._cmap = None
        n0 = self.nclass_box.value()
        self._cmap = cm.get_cmap('jet', n0)         # same jet as final segmentation

        # Replace placeholders with custom widgets
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)

        self._promote_canvas('spec_canvas', FigureCanvas)

        self.viewer_left.viewport().installEventFilter(self)
        self.viewer_right.viewport().installEventFilter(self)

        # Enable live spectrum tracking
        self.viewer_left.viewport().setCursor(Qt.CrossCursor) # curseur croix
        self.viewer_left.viewport().setMouseTracking(True)

        # Promote spec_canvas placeholder to FigureCanvas
        self.spec_canvas_layout = self.spec_canvas.layout() if hasattr(self.spec_canvas, 'layout') else None
        self._init_spectrum_canvas()
        self.spec_canvas.setVisible(False)
        self.show_selection=True
        self.live_spectra_update=True

        # State variables
        self.cube = None
        self.data = None
        self.wl= None
        self.cls_map = None
        self.samples = {} # to save pixels spectra samples for GT
        self.sample_coords = {c: set() for c in self.samples.keys()} # to remember coord of pixel samples
        self.alpha = self.horizontalSlider_transparency_GT.value() / 100.0
        self.mode = self.comboBox_ClassifMode.currentText()
        self.hyps_rgb_chan_DEFAULT=[0,0,0] #default rgb channels (in int nm)
        self.hyps_rgb_chan=[0,0,0] #current rgb (in int nm)
        self.class_means = {} #for spectra of classe

        # Connect widget signals
        self.load_btn.clicked.connect(self.load_cube)
        self.run_btn.clicked.connect(self.run)
        self.comboBox_ClassifMode.currentIndexChanged.connect(self.set_mode)
        self.pushButton_class_selection.toggled.connect(self.on_toggle_selection)
        self.pushButton_erase_selected_pix.toggled.connect(self.on_toggle_erase)
        self.checkBox_see_selection_overlay.toggled.connect(self.toggle_show_selection)
        self.pushButton_merge.clicked.connect(self.merge_selec_GT)

        # RGB sliders <-> spinboxes
        self.sliders_rgb = [self.horizontalSlider_red_channel, self.horizontalSlider_green_channel,
                            self.horizontalSlider_blue_channel]
        self.spinBox_rgb = [self.spinBox_red_channel, self.spinBox_green_channel, self.spinBox_blue_channel]

        for sl, sp in zip(self.sliders_rgb,  self.spinBox_rgb):
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sl.valueChanged.connect(self.show_image)

        self.radioButton_rgb_user.toggled.connect(self.modif_sliders)
        self.radioButton_rgb_default.toggled.connect(self.modif_sliders)
        self.radioButton_grayscale.toggled.connect(self.modif_sliders)

        # Transparency slider
        self.horizontalSlider_transparency_GT.valueChanged.connect(self.on_alpha_change)

        # Live spectrum checkbox
        self.live_cb.stateChanged.connect(self.toggle_live)

        self.distance_funcs = {
            'sqeuclidean': spdist.sqeuclidean,
            'cosine': spdist.cosine,
            'correlation': spdist.correlation,
            'canberra': spdist.canberra,
        }

        # init stretch of each layout in QSplitters
        self.splitter.setStretchFactor(0, 1) #init stretch of images and spectra
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([600, 600])
        self.splitter.setHandleWidth(2)

        self.splitter_2.setStretchFactor(0,4) #init stretch of image hyp and GT
        self.splitter_2.setStretchFactor(1, 1)

        # style poignée QSplitter
        self.splitter.setHandleWidth(2)
        self.splitter_2.setHandleWidth(4)
        self.splitter.setStyleSheet("""QSplitter::handle {background-color: darkgray;}""")
        self.splitter_2.setStyleSheet("""QSplitter::handle {background-color: darkgray;}""")

    def start_pixel_selection(self):

        self.show_selection=True
        self.pushButton_class_selection.setText("Stop Selection")
        self.pushButton_erase_selected_pix.setChecked(False)

        if len(self.samples)>0 :
            reply = QMessageBox.question(
                self, "Erase selection?",
                "Do you want to erase previous selection?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                H, W = self.selection_mask_map.shape
                # Réinitialise à -1 (aucune classe)
                self.selection_mask_map[:] = -1
                self.samples.clear()

        self.frame_legend.setVisible(True)
        self.selecting_pixels = True
        self.viewer_left.setDragMode(QGraphicsView.NoDrag)
        self.viewer_left.setCursor(Qt.CrossCursor)
        self.viewer_left.viewport().setCursor(Qt.CrossCursor)
        self.show_image()

    def toggle_show_selection(self):

        self.show_selection = self.checkBox_see_selection_overlay.isChecked()
        self.show_image()

    def stop_pixel_selection(self):

        self.selecting_pixels = False
        self.frame_legend.setVisible(False)

        # ready to select
        self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)
        self.viewer_left.setCursor(Qt.ArrowCursor)
        self.viewer_left.viewport().setCursor(Qt.ArrowCursor)

        # remet le bouton à l'état initial
        self.pushButton_class_selection.setText("Start Selection")
        self.pushButton_class_selection.setChecked(False)

        # efface tout preview en cours
        self.selecting_pixels = False

        # enfin, on affiche l'image normale (sans preview ni sélection en cours)
        self.show_image()

    def on_toggle_erase(self, checked):
        self.erase_selection = checked

        if checked:
            self._pixel_selecting=False
            self.stop_pixel_selection()

            self.show_selection = True

            self.pushButton_erase_selected_pix.setText("Stop Erasing")
            self.pushButton_class_selection.setChecked(False)
            self.viewer_left.setDragMode(QGraphicsView.NoDrag)
            self.viewer_left.setCursor(Qt.CrossCursor)

        else:
            self.pushButton_erase_selected_pix.setText("Erase Pixels")
            self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewer_left.unsetCursor()

    def on_toggle_selection(self, checked: bool):

        if checked:
            self.erase_selection=False
            self.start_pixel_selection()
            self.update_legend()

        else:
            # fin du mode sélection
            self.stop_pixel_selection()

    def merge_selec_GT(self):
        """
        Fusionne les annotations manuelles (selection_mask_map)
        dans la carte de segmentation algorithmique (cls_map),
        puis met à jour les prototypes (moyennes et écart-types).
        """

        if self.cls_map is None:
            QMessageBox.warning(
                self, "Warning",
                "No segmentation done : Perform a segmentation and try again if needed."
            )
            return


        mask = (self.selection_mask_map >= 0) ##mask of manual selected

        if not mask.any():
            QMessageBox.information(
                self, "Info",
                "Not selected pixel to mergi with segmented result"
            )
            return

        self.checkBox_enable_segment.setChecked(False) # secure selection by disabled segmentation

        self.cls_map[mask] = self.selection_mask_map[mask] # assign manual selecte class in segmented image

        # update class prototypes
        unique_labels = np.unique(self.cls_map)
        new_means = {}
        new_stds = {}
        for c in unique_labels:
            # collecte les coordonnées dont cls_map == c
            ys, xs = np.where(self.cls_map == c)
            # construit un tableau (N_c x B) de leurs spectres
            spectra = np.stack([self.data[y, x, :] for x, y in zip(xs, ys)], axis=0)
            # moyenne et écart-type
            new_means[c] = np.mean(spectra, axis=0)
            new_stds[c] = np.std(spectra, axis=0)

        self.class_means = new_means
        self.class_stds = new_stds

        n = len(unique_labels)
        self._cmap = cm.get_cmap('jet', n)

        self.show_image()

    def modif_sliders(self):
        max_wl = int(self.wl[-1])
        min_wl = int(self.wl[0])
        wl_step = int(self.wl[1] - self.wl[0])

        default=self.radioButton_rgb_default.isChecked()

        if self.radioButton_grayscale.isChecked():
            self.label_red_channel.setText('')
            self.label_green_channel.setText('')
            self.label_blue_channel.setText('Gray')
        else:
            self.label_red_channel.setText('Red')
            self.label_green_channel.setText('Green')
            self.label_blue_channel.setText('Blue')

        for i, element in enumerate(self.sliders_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[i])
            else:
                element.setValue(self.hyps_rgb_chan[i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i == 2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

        for i, element in enumerate(self.spinBox_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[i])
            else:
                element.setValue(self.hyps_rgb_chan[i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i == 2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

        self.show_image()

    def _init_spectrum_canvas(self):
        placeholder = getattr(self, 'spec_canvas')
        parent = placeholder.parent()
        from PyQt5.QtWidgets import QSplitter

        # Crée le canvas
        self.spec_fig = Figure()
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title('Spectrum')

        # Remplace dans le splitter ou dans le layout
        if isinstance(parent, QSplitter):
            idx = parent.indexOf(placeholder)
            placeholder.deleteLater()
            parent.insertWidget(idx, self.spec_canvas)
        elif parent.layout() is not None:
            layout = parent.layout()
            idx = layout.indexOf(placeholder)
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            layout.insertWidget(idx, self.spec_canvas)
        else:
            placeholder.deleteLater()
            self.verticalLayout.addWidget(self.spec_canvas)

        self.spec_canvas.setVisible(True)

    def _handle_selection(self, coords):
        """Prompt for class and store spectra of the given coordinates."""
        cls, ok = QInputDialog.getInt(
            self, "Class", "Class label number:", 0, 0, self.nclass_box.value() - 1
        )
        if not ok:
            return

        # append spectra
        if cls not in self.class_colors:
            # si _cmap n'existe pas encore, on le crée à la volée
            if self._cmap is None:
                from matplotlib import cm
                n = self.nclass_box.value()
                self._cmap = cm.get_cmap('jet', n)
            rgba = self._cmap(cls)
            r, g, b = (int(255 * rgba[i]) for i in range(3))
            self.class_colors[cls] = (b, g, r)

        for x, y in coords:
            if not (0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]):
                continue

            # A) s’il appartenait déjà à une autre classe, on l’enlève
            old = self.selection_mask_map[y, x]
            if old >= 0 and old != cls:
                # retirer coord de sample_coords[old] et de samples[old]
                if (x, y) in self.sample_coords.get(old, set()):
                    self.sample_coords[old].remove((x, y))
                # reconstruire la liste des spectres pour old
                self.samples[old] = [
                    self.data[yy, xx, :]
                    for (xx, yy) in self.sample_coords.get(old, ())
                ]

            # B) on (ré)assigne le pixel à la classe cls
            self.selection_mask_map[y, x] = cls
            # ajouter dans sample_coords et samples si pas déjà présent
            if (x, y) not in self.sample_coords.setdefault(cls, set()):
                self.sample_coords.setdefault(cls, set()).add((x, y))
                self.samples.setdefault(cls, []).append(self.data[y, x, :])

            # 3) rafraîchir l’affichage
        self.show_image()
        self.update_legend()

    def _handle_erasure(self, coords):

        for x, y in coords:
            cls = self.selection_mask_map[y, x]
            if cls >= 0:
                # enlève du mask
                self.selection_mask_map[y, x] = -1
                # enlève des sets et listes
                if (x, y) in self.sample_coords.get(cls, set()):
                    self.sample_coords[cls].remove((x, y))
                # reconstruit self.samples[cls]
                self.samples[cls] = [
                    self.data[yy, xx, :]
                    for (xx, yy) in self.sample_coords.get(cls, [])
                ]
                # si plus d'exemples pour cette classe, tu peux aussi
                # nettoyer class_colors, class_means, class_stds si tu veux

        self.show_image()

    def eventFilter(self, source, event):
        mode = self.comboBox_pixel_selection_mode.currentText()

        # 1) Clic souris
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and (self.selecting_pixels or self.erase_selection):
            print('Clicked OK')
            pos = self.viewer_left.mapToScene(event.pos())
            x0, y0 = int(pos.x()), int(pos.y())
            if mode == 'pixel':
                # on commence la collecte
                self._pixel_selecting = True
                self._pixel_coords = [(x0, y0)]
                return True
            elif mode == 'rectangle':
                # début du drag
                from PyQt5.QtWidgets import QRubberBand
                self.origin = event.pos()
                self.rubberBand = QRubberBand(QRubberBand.Rectangle,
                                              self.viewer_left.viewport())
                self.rubberBand.setGeometry(self.origin.x(),
                                            self.origin.y(), 1, 1)
                self.rubberBand.show()
                return True
            elif mode == 'ellipse':
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                from PyQt5.QtGui import QPen

                self.origin = event.pos()
                pen = QPen(Qt.red)
                pen.setStyle(Qt.DashLine)
                self.ellipse_item = QGraphicsEllipseItem()
                self.ellipse_item.setPen(pen)
                self.ellipse_item.setBrush(Qt.transparent)
                self.viewer_left.scene().addItem(self.ellipse_item)
                return True

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            self.live_spectra_update=not self.live_spectra_update

        # 2) Mouvement souris → mise à jour de la selection en cours
        if event.type() == QEvent.MouseMove and self._pixel_selecting and mode == 'pixel':
            pos = self.viewer_left.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())

            if (x, y) not in self._pixel_coords:
                self._pixel_coords.append((x, y))
            if self._preview_mask is None:
                H, W = self.data.shape[:2]
                self._preview_mask = np.zeros((H, W), dtype=bool)

            self._preview_mask[y, x] = True
            self.show_image(preview=True)

            return True

        if event.type() == QEvent.MouseMove and hasattr(self, 'rubberBand'):
            self.rubberBand.setGeometry(
                QRect(self.origin, event.pos()).normalized()
            )
            return True

        if event.type() == QEvent.MouseMove and mode == 'ellipse' and hasattr(self, 'ellipse_item'):
            sc_orig = self.viewer_left.mapToScene(self.origin)
            sc_now = self.viewer_left.mapToScene(event.pos())
            x0, y0 = sc_orig.x(), sc_orig.y()
            x1, y1 = sc_now.x(), sc_now.y()
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self.ellipse_item.setRect(rect)
            return True

        # 3) Relâchement souris → calcul de la sélection

        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and mode == 'pixel' and self._pixel_selecting :
            print('realeased OK')
            # get pixels
            coords = self._pixel_coords.copy()
            #  Si au moins 3 points, propose de fermer le cheminif min 3 points, propose contour
            if len(coords) >= 3:
                reply = QMessageBox.question(
                    self, "Close Path?",
                    "You have selected multiple pixels.\n"
                    "Do you want to close the path and include all pixels inside the contour?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    pts = np.array(coords)
                    poly = Path(pts)
                    x0, y0 = pts[:, 0].min().astype(int), pts[:, 1].min().astype(int)
                    x1, y1 = pts[:, 0].max().astype(int), pts[:, 1].max().astype(int)
                    filled = list(coords)
                    for yy in range(y0, y1 + 1):
                        for xx in range(x0, x1 + 1):
                            if poly.contains_point((xx, yy)):
                                filled.append((xx, yy))

                    # to avoid dobbles
                    seen = set()
                    coords = []
                    for p in filled:
                        if p not in seen:
                            seen.add(p)
                            coords.append(p)

            if self.erase_selection:
                self._handle_erasure(coords)
            else :
                self._handle_selection(coords) # close selection

            # ready to new selection
            self._pixel_selecting = False
            self._erase_selecting = False
            self._preview_mask = None
            return True

        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'rubberBand'):
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            # coins en coordonnées image
            tl = self.viewer_left.mapToScene(rect.topLeft())
            br = self.viewer_left.mapToScene(rect.bottomRight())
            x0, y0 = int(tl.x()), int(tl.y())
            x1, y1 = int(br.x()), int(br.y())
            # liste de tous les pixels dans le rectangle
            coords = [
                (xx, yy)
                for yy in range(max(0, min(y0, y1)), min(self.data.shape[0], max(y0, y1) + 1))
                for xx in range(max(0, min(x0, x1)), min(self.data.shape[1], max(x0, x1) + 1))
            ]

            if self.erase_selection:
                self._handle_erasure(coords)
            else:
                self._handle_selection(coords)  # close selection

            del self.rubberBand
            return True

        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'ellipse_item'):
            rect = self.ellipse_item.rect()
            self.viewer_left.scene().removeItem(self.ellipse_item)
            del self.ellipse_item

            cx, cy = rect.center().x(), rect.center().y()
            rx, ry = rect.width() / 2, rect.height() / 2
            x0, x1 = int(rect.left()), int(rect.right())
            y0, y1 = int(rect.top()), int(rect.bottom())

            coords = []
            for yy in range(max(0, y0), min(self.data.shape[0], y1 + 1)):
                for xx in range(max(0, x0), min(self.data.shape[1], x1 + 1)):
                    if ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1:
                        coords.append((xx, yy))

            if self.erase_selection:
                self._handle_erasure(coords)
            else:
                self._handle_selection(coords)  # close selection
            return True

        # 4) Mouvement souris pour le live spectrum
        if source is self.viewer_left.viewport() and event.type() == QEvent.MouseMove and self.live_spectra_update:
            if self.live_cb.isChecked() and self.data is not None:
                pos = self.viewer_left.mapToScene(event.pos())
                self.update_spectra(x=int(pos.x()),y=int(pos.y()))

            return super().eventFilter(source, event)

    def update_spectra(self,x=None,y=None):
        self.spec_ax.clear()
        x_graph = self.wl

        if x is not None and y is not None:
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                spectrum = self.data[y, x, :]
                # Spectre du pixel
                self.spec_ax.plot(x_graph, spectrum, label='Pixel')

        # Spectres GT moyens ± std
        if self.checkBox_seeGTspectra.isChecked() and hasattr(self, 'class_means'):
            for c, mu in self.class_means.items():
                std = self.class_stds[c]
                color = self._cmap(c)
                self.spec_ax.fill_between(
                    x_graph, mu - std, mu + std,
                    color=color, alpha=0.3, linewidth=0
                )
                self.spec_ax.plot(
                    x_graph, mu, '--',
                    color=color, label=f'Classe {c}'
                )
            if self.spec_ax.get_legend_handles_labels()[1]:
                self.spec_ax.legend(loc='upper right', fontsize='small')
            self.spec_ax.set_title(f'Spectra')
            self.spec_canvas.setVisible(True)
            self.spec_canvas.draw()

    def on_alpha_change(self, val):
        self.alpha = val / 100.0
        self.show_image()

    def toggle_live(self, state):
        if not state:
            self.spec_canvas.setVisible(False)
        else:
            self.update_spectra()
            self.live_spectra_update=True

    def load_cube(self,path=None):
        if self.cls_map : # if work done, stop to permit saving before continue.
            reply = QMessageBox.question(
                self, "Erase previous selection ?",
                "Do you want to erase previous cube work ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
               return

        if not path :
            print('Ask path for cube')
            path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not path:
                return
        try:
            self.cube = Hypercube(filepath=path, load_init=True)
            self.data = self.cube.data
            self.wl= self.cube.wl
            if self.wl[-1]<1100 and self.wl[0]>350:
                self.hyps_rgb_chan_DEFAULT = [610, 540, 435]
            elif self.wl[-1]>=1100:
                self.hyps_rgb_chan_DEFAULT = [1605, 1205, 1005]
            else:
                mid=int(len(self.wl)/2)
                self.hyps_rgb_chan_DEFAULT = [self.wl[0], self.wl[mid], self.wl[-1]]

            self.reset_state()
            self.modif_sliders()
            self.show_image()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le cube: {e}")

    def reset_state(self):
        """
        Réinitialise tous les états liés au cube courant pour repartir d'un état vierge.
        """
        # 1. Segmentation algorithmique
        self.cls_map = None
        # 2. Sélection manuelle
        if self.data is not None:
            H, W = self.data.shape[:2]
            self.selection_mask_map = np.full((H, W), -1, dtype=int)
        # 3. Samples et prototypes
        self.samples = {}
        self.sample_coords = {}
        self.class_means = {}
        self.class_stds = {}
        self.class_colors = {}
        self._cmap = None
        # 4. Masques de preview/erase
        self._preview_mask = None
        if hasattr(self, '_erase_mask'):
            self._erase_mask = None
        # 5. UI
        # Masquer le canvas de spectres
        self.spec_canvas.setVisible(False)
        # Réinitialiser la légende
        self.update_legend()
        # Remettre le slider de threshold à 100%
        if hasattr(self, 'horizontalSlider_threshold'):
            self.horizontalSlider_threshold.setValue(100)
        # Remettre toggles
        self.selecting_pixels = False
        self.erase_selection = False
        self.pushButton_class_selection.setChecked(False)
        self.pushButton_erase_selected_pix.setChecked(False)
        # 6. Rafraîchir l'affichage
        self.show_image()

    def set_mode(self):
        self.mode = self.comboBox_ClassifMode.currentText()
        self.spec_canvas.setVisible(False)
        self.show_image()

    def show_image(self, preview=False):
        if self.data is None:
            return

        #hyp image
        H, W, B = self.data.shape
                # Get band indices from spinboxes for RGB
        self.hyps_rgb_chan = [self.spinBox_red_channel.value(),
               self.spinBox_green_channel.value(),
               self.spinBox_blue_channel.value()]

        idx = [np.argmin(np.abs(self.hyps_rgb_chan[j] - self.wl)) for j in range(3)]
        if self.radioButton_grayscale.isChecked():
            idx=[idx[2],idx[2],idx[2]]

        rgb = self.data[:, :, idx]
        rgb = (rgb / np.max(rgb) * 255).astype(np.uint8)

        # overlay of GT

        if self.cls_map is None:
            overlay = rgb.copy()
        else:
            seg8 = (self.cls_map.astype(np.float32) / (self.nclass_box.value() - 1) * 255).astype(np.uint8)
            cmap = cv2.applyColorMap(seg8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(rgb, 1-self.alpha, cmap, self.alpha, 0)

        if self.cls_map is None:
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            pix2 = self._np2pixmap(blank)
        else:
            seg8 = (self.cls_map.astype(np.float32) / (self.nclass_box.value() - 1) * 255).astype(np.uint8)
            pix2 = self._np2pixmap(cv2.applyColorMap(seg8, cv2.COLORMAP_JET))
        self.viewer_right.setImage(pix2)

        # overlay of selection blended to GT overlay

        current = overlay.copy()
        if self.selection_mask_map is not None and self.show_selection:
            mixed = overlay.copy()
            α = 0.7
            for cls, color in self.class_colors.items():
                mask2d = (self.selection_mask_map == cls)
                if not mask2d.any():
                    continue

                layer = np.zeros_like(overlay)
                layer[:] = color

                blended = cv2.addWeighted(overlay, 1 - α, layer, α, 0)

                mask3 = mask2d[:, :, None]
                current = np.where(mask3, blended, current)

        self.current_composite = current

        if preview and self._preview_mask is not None:
            base = self.current_composite
            layer = np.zeros_like(base)
            layer[..., :] = 0,0,255  # BGR = (0,0,0)
            mixed = cv2.addWeighted(base, 1-0.1, layer, 0.1, 0)
            mask3 = self._preview_mask[:, :, None]
            result = np.where(mask3, mixed, base)
            self.viewer_left.setImage(self._np2pixmap(result))
            return

        self.viewer_left.setImage(self._np2pixmap(self.current_composite))

    def update_legend(self):
        """
        Construit un petit carré coloré + numéro de classe
        dans frame_legend pour chaque classe actuelle.
        """
        # 1) Vider l’ancien contenu
        for i in reversed(range(self.frame_legend.layout().count())):
            w = self.frame_legend.layout().itemAt(i).widget()
            self.frame_legend.layout().removeWidget(w)
            w.deleteLater()

        # Si pas en mode sélection, on ne montre rien
        if not self.selecting_pixels:
            return

        # 2) Pour chaque classe, crée un QLabel avec un pixmap carré coloré
        for c in sorted(self.class_colors):
            b, g, r = self.class_colors[c]
            lbl = QLabel(str(c))
            lbl.setFixedSize(30, 20)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"background-color: rgb({r},{g},{b});"
                "color: white;"
                "border-radius: 3px;"
                "font-weight: bold;"
            )
            self.frame_legend.layout().addWidget(lbl)

    def _replace_placeholder(self, name, widget_cls, **kwargs):
        placeholder = getattr(self, name)
        parent = placeholder.parent()
        # Determine if parent is a layout container or a splitter
        if parent.layout() is not None:
            layout = parent.layout()
            idx = layout.indexOf(placeholder)
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            widget = widget_cls(**kwargs) if kwargs else widget_cls()
            setattr(self, name, widget)
            layout.insertWidget(idx, widget)
        elif isinstance(parent, QSplitter):
            # Handle QSplitter parent
            idx = parent.indexOf(placeholder)
            placeholder.deleteLater()
            widget = widget_cls(**kwargs) if kwargs else widget_cls()
            setattr(self, name, widget)
            parent.insertWidget(idx, widget)
        else:
            # Fallback: direct replacement
            placeholder.deleteLater()
            widget = widget_cls(**kwargs) if kwargs else widget_cls()
            setattr(self, name, widget)

    def _promote_canvas(self, name, canvas_cls):
        placeholder = getattr(self, name)
        parent = placeholder.parent()
        from PyQt5.QtWidgets import QSplitter

        # Crée le nouveau canvas
        canvas = canvas_cls()
        # Supprime l’ancien placeholder
        placeholder.deleteLater()

        if isinstance(parent, QSplitter):
            # cas splitter : insère au même emplacement
            idx = parent.indexOf(placeholder)
            parent.insertWidget(idx, canvas)
        else:
            # cas layout classique
            layout = parent.layout() or self.verticalLayout
            layout.addWidget(canvas)

        # Conserve refs pour live spectrum
        self.spec_canvas = canvas
        self.spec_fig = getattr(canvas, 'figure', None) or Figure()
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title('Spectrum')
        canvas.setVisible(False)

    def compute_distance(self, u, v):
        name = self.comboBox_distance.currentText()
        fn   = self.distance_funcs.get(name)
        return fn(u, v)

    def run(self):

        if not self.checkBox_enable_segment.isChecked():
            QMessageBox.warning(self, "Warning", "Enable segmentation with checkbox !")
            return

        if self.data is None:
            QMessageBox.warning(self, "Warning", "Load a cube !")
            return

        flat = self.data.reshape(-1, self.data.shape[2])

        # 1) Unsupervised
        if self.mode == 'Unsupervised':
            from sklearn.cluster import KMeans
            n = self.nclass_box.value()
            kmeans = KMeans(n_clusters=n).fit(flat)
            labels = kmeans.labels_
            # stocke moyennes, écarts et colormap
            self.class_means = {i: kmeans.cluster_centers_[i] for i in range(n)}
            self.class_stds = {i: np.std(flat[labels == i], axis=0) for i in range(n)}
            self._cmap = cm.get_cmap('jet', n)
            # Final : reshape et affichage
            H, W = self.data.shape[:2]
            self.cls_map = labels.reshape(H, W)
            self.show_image()

        elif self.mode == 'Supervised':
            # 1) Récupère les prototypes des classes labellisées
            classes = sorted(self.samples.keys())
            if not classes:
                QMessageBox.warning(self, "Warning", "Select references pixels and try again !")
                return

            means = {c: np.mean(np.vstack(self.samples[c]), axis=0) for c in classes}

            thr_pct = self.slider_class_thr.value()
            thr_frac = thr_pct / 100.0  # 0.0–1.0

            other_label = len(classes)
            labels = np.full(flat.shape[0], other_label, dtype=int)

            for i, pix in enumerate(flat):
                dists = np.array([self.compute_distance(pix, means[c])
                                  for c in classes])
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]

                if thr_pct == 100:
                    labels[i] = classes[min_idx]
                else:
                    # on normalise la distance entre 0 et 1 sur cet exemple
                    max_dist = dists.max() if dists.max() > 0 else 1.0
                    norm_dist = min_dist / max_dist
                    if norm_dist <= thr_frac:
                        labels[i] = classes[min_idx]

            self.class_means = means
            self.class_stds = {c: np.std(np.vstack(self.samples[c]), axis=0) for c in classes}

            # 5) reshape et préparation de l’affichage
            H, W = self.data.shape[:2]
            self.cls_map = labels.reshape(H, W)

            # on prend K = nombre de classes + 1 for “other”
            n_colors = other_label + 1
            self._cmap = cm.get_cmap('jet', n_colors)

            self.show_image()
            self.update_legend()
            return

    def _np2pixmap(self, img):
        from PyQt5.QtGui import QImage, QPixmap
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()


if __name__=='__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = GroundTruthWidget()
    folder=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test/'
    file_name='00001-SWIR-mock-up.h5'
    filepath=folder+file_name
    w.load_cube(filepath)
    w.show()
    sys.exit(app.exec_())

