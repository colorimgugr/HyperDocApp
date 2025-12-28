from PyQt5.QtWidgets import (QProgressBar,QLabel,QDialog, QVBoxLayout,QGraphicsLineItem, QGraphicsItem,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QRubberBand,QGraphicsRectItem,QHBoxLayout,QPushButton
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor
from PyQt5.QtCore import QRect, QPoint, Qt, QSize,pyqtSignal, QPointF,QRectF,QLineF, QObject, pyqtSlot, QRunnable

import traceback

import cv2
import os
import numpy as np

def np_to_qpixmap(img):
    if len(img.shape) == 2:
        try:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        except:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            qimg = QImage(img.tobytes(), img.shape[1], img.shape[0],img.shape[1], QImage.Format_Grayscale8)

    else:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).copy()

class ZoomableGraphicsView(QGraphicsView):
    middleClicked = pyqtSignal(QPointF) # suppres features
    moveFeatureStart = pyqtSignal(QPointF) # movefeatures
    moveFeatureUpdate = pyqtSignal(QPointF)
    moveFeatureEnd = pyqtSignal()
    selectionChanged = pyqtSignal()

    def __init__(self,cursor=Qt.CrossCursor):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setCursor(cursor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item = None  # check if image loaded

        # Rubber band selection
        self.enable_rect_selection = True
        self.origin = QPoint()
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self._selecting = False
        self.last_rect_item = None
        self.rect_coords = None

        # if overlay
        self._selection_overlay = None

        # Moving point
        self.editing_match = None  # (match_index, side: "left"/"right")
        self.setMouseTracking(True)  # enable mouseMoveEvent without press

        # zoom
        self._scale_factor = 1.0
        self.min_scale = 0.2
        self.max_scale = 100.0
        self.zoom_step = 1.25

    def setImage(self, pixmap):
        # self.clear_rectangle()
        self.scene().clear()
        self._selection_overlay = None
        self.last_rect_item = None
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))

    def fitImage(self, scale_factor=0.8):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.scale(scale_factor, scale_factor)

    def wheelEvent(self, event):
        if event.angleDelta().y() == 0:
            return

        if event.angleDelta().y() > 0:
            zoom = self.zoom_step
        else:
            zoom = 1 / self.zoom_step

        new_scale = self._scale_factor * zoom

        # Clamp
        if new_scale < self.min_scale:
            return  # ← stop net : plus aucun effet
        if new_scale > self.max_scale:
            return

        self.scale(zoom, zoom)
        self._scale_factor = new_scale

    def mousePressEvent(self, event):
        self.viewport().setCursor(Qt.CrossCursor)

        if event.button() == Qt.RightButton and self.pixmap_item:
            if not self.enable_rect_selection:
                # Keypoint move (View Matches mode)
                scene_pos = self.mapToScene(event.pos())
                self.moveFeatureStart.emit(scene_pos)
                event.accept()
                return
            else:
                # Rectangle selection
                self.origin = event.pos()
                self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_band.show()
                self._selecting = True
                event.accept()
                return

        if (event.button() == Qt.RightButton and self.pixmap_item and self.enable_rect_selection): # rectangle selection
            self.viewport().setCursor(Qt.CrossCursor)
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self._selecting = True

        elif event.button() == Qt.MiddleButton and self.pixmap_item: # feature supress
            scene_pos = self.mapToScene(event.pos())
            self.middleClicked.emit(scene_pos)

        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier and self.pixmap_item: # feature supress
            scene_pos = self.mapToScene(event.pos())
            self.middleClicked.emit(scene_pos)
            event.accept()
            return

        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.AltModifier and self.pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            self.moveFeatureStart.emit(scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.viewport().setCursor(Qt.CrossCursor)
        if self._selecting:
            max_w = self.viewport().width() - 1
            max_h = self.viewport().height() - 1
            x = min(max(event.pos().x(), 0), max_w)
            y = min(max(event.pos().y(), 0), max_h)
            rect = QRect(self.origin, QPoint(x, y)).normalized()
            self.rubber_band.setGeometry(rect)

        elif self.editing_match is not None:
            scene_pos = self.mapToScene(event.pos())
            self.moveFeatureUpdate.emit(scene_pos)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.viewport().setCursor(Qt.CrossCursor)
        if event.button() == Qt.RightButton and self.pixmap_item and self._selecting:
            self.rubber_band.hide()
            # self.viewport().setCursor(Qt.OpenHandCursor)
            self.viewport().setCursor(Qt.CrossCursor)
            self._selecting = False

            # Convert view coords to scene coords
            start_scene = self.mapToScene(self.origin)
            end_scene = self.mapToScene(event.pos())

            # Convert scene coords to pixmap (image) coords
            p1 = self.pixmap_item.mapFromScene(start_scene)
            p2 = self.pixmap_item.mapFromScene(end_scene)

            x1, y1 = int(p1.x()), int(p1.y())
            x2, y2 = int(p2.x()), int(p2.y())

            # Clamp aux bords de l'image
            w, h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # Erase previous rectangle
            self.clear_selection_overlay()
            self.clear_rectangle()

            # Création du rectangle avec coins clampés
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            width, height = x_max - x_min, y_max - y_min

            if width < 2 or height < 2:
                self.rect_coords = None
                self.last_rect_item = None
                return

            self.rect_coords = [x_min, y_min, width, height]
            pen = QPen(QColor(255, 255, 0, 255))  # jaune très visible (mieux que rouge sur certaines scènes)
            pen.setWidth(3)  # épaisseur
            pen.setCosmetic(True)  # épaisseur constante à l’écran (indépendante du zoom)
            pen.setStyle(Qt.SolidLine)

            self.last_rect_item = self.scene().addRect(x_min, y_min, width, height, pen)
            self.last_rect_item.setZValue(20)

        if self.editing_match is not None:
            self.moveFeatureEnd.emit()
            self.editing_match = None

        self.selectionChanged.emit()
        super().mouseReleaseEvent(event)

    def get_rect_coords(self):
        return self.rect_coords

    def clear_rectangle(self):
        if self.last_rect_item:
            try:
                self.scene().removeItem(self.last_rect_item)
            except:
                None
        self.last_rect_item = None
        self.rect_coords=None

    def add_selection_overlay(self, rect: QRectF,surface=True):
        # Remove previous overlay if any
        if hasattr(self, '_selection_overlay') and self._selection_overlay is not None:
            try:
                self.scene().removeItem(self._selection_overlay)
            except RuntimeError:
                pass  # item déjà supprimé côté C++
            self._selection_overlay = None

        overlay = QGraphicsRectItem(rect)
        if surface:
            overlay.setBrush(QColor(0, 255, 0, 80))  # Green with transparency
        pen = QPen(QColor(255, 255, 0, 255), 2, Qt.DashLine)
        pen.setCosmetic(True)
        overlay.setPen(pen)
        overlay.setZValue(20)
        self.scene().addItem(overlay)
        self._selection_overlay = overlay

    def clear_selection_overlay(self):
        if hasattr(self, '_selection_overlay') and self._selection_overlay is not None:
            self.scene().removeItem(self._selection_overlay)
            self._selection_overlay = None

    def clear_overlay_items(self):
        items = getattr(self, "_overlay_items", [])
        if not items:
            self._overlay_items = []
            return
        sc = self.scene()
        for it in items:
            try:
                sc.removeItem(it)
            except Exception:
                pass
        self._overlay_items = []

    def add_cross_overlay(self, x, y, half_size=8, color=QColor(255, 255, 0), width=2):
        """
        Draw a cross centered at (x,y) in SCENE coordinates.
        Cosmetic pen => constant thickness regardless of zoom.
        """
        sc = self.scene()
        if sc is None:
            return

        if not hasattr(self, "_overlay_items"):
            self._overlay_items = []

        pen = QPen(color)
        pen.setWidth(int(width))
        pen.setCosmetic(True)  # KEY: thickness constant on screen

        # Horizontal line
        h = QGraphicsLineItem(QLineF(x - half_size, y, x + half_size, y))
        h.setPen(pen)
        h.setZValue(10_000)

        # Vertical line
        v = QGraphicsLineItem(QLineF(x, y - half_size, x, y + half_size))
        v.setPen(pen)
        v.setZValue(10_000)

        sc.addItem(h)
        sc.addItem(v)
        self._overlay_items.extend([h, v])


class LoadingDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(
        self,
        message="Loading.",
        filename=None,
        parent=None,
        cancellable: bool = False,
        cancel_text: str = "Cancel",
        block_close: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._cancellable = bool(cancellable)
        self._block_close = bool(block_close)
        self._cancel_requested = False

        layout = QVBoxLayout(self)

        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        if filename:
            self.label_file = QLabel(f"<i>{os.path.basename(filename)}</i>")
            self.label_file.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.label_file)
        else:
            self.label_file = None

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate (animated)
        self.progress.setTextVisible(True)
        self.progress.setFormat("In progress…")  # “dynamic” busy style, not 0%
        layout.addWidget(self.progress)

        # Optional cancel button row
        self.btn_cancel = None
        if self._cancellable:
            row = QHBoxLayout()
            row.addStretch(1)

            self.btn_cancel = QPushButton(cancel_text, self)
            self.btn_cancel.clicked.connect(self._on_cancel_clicked)
            row.addWidget(self.btn_cancel)

            layout.addLayout(row)

        self.setFixedWidth(350)

    def _on_cancel_clicked(self):
        self._cancel_requested = True
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(False)

        # UX feedback
        self.label.setText("Cancel requested…")
        self.progress.setFormat("Stopping…")

        self.cancel_requested.emit()

    def was_cancel_requested(self) -> bool:
        return self._cancel_requested

    def set_message(self, message: str):
        self.label.setText(message)

    def set_busy_text(self, text: str):
        # changes the text displayed inside the progress bar
        self.progress.setFormat(text)

    def set_indeterminate(self, on: bool):
        if on:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            # Optionnel: si tu veux éviter le 0% affiché
            if self.progress.value() == 0:
                self.progress.setValue(100)

    def set_progress(self, value: int):
        # Si on reçoit un progrès déterminé, on bascule automatiquement
        if self.progress.maximum() == 0:
            self.set_indeterminate(False)
        self.progress.setValue(int(value))

    def closeEvent(self, event):
        # Optionally block user from closing via the window [X] while cancellable & not canceled
        if self._cancellable and self._block_close and not self._cancel_requested:
            event.ignore()
            return
        super().closeEvent(event)

# ======================================================================
# Cube loading worker (generic, cancellable)
# ======================================================================

class LoadCubeSignals(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(object)   # Hypercube
    error = pyqtSignal(str)
    canceled = pyqtSignal()

class LoadCubeWorker(QRunnable):
    """
    Generic worker to load a Hypercube (or equivalent object) in background.

    Parameters
    ----------
    load_fn : callable
        Function with no arguments that performs the actual loading
        and RETURNS the loaded cube.
    """

    def __init__(self, load_fn):
        super().__init__()
        self.signals = LoadCubeSignals()
        self._load_fn = load_fn
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            self.signals.started.emit()

            if self._cancel_requested:
                self.signals.canceled.emit()
                return

            cube = self._load_fn()

            if self._cancel_requested:
                self.signals.canceled.emit()
                return

            self.signals.finished.emit(cube)

        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"Cube loading failed:\n{e}\n\n{tb}")
