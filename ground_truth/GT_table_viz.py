import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QPushButton, QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QFileDialog, QMessageBox, QGridLayout, QLabel, QFrame, QSplitter,
    QInputDialog,QColorDialog
)
from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSlot
from PyQt5.QtGui import QBrush, QColor

class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        # Copie du DataFrame et mémorisation du nombre de lignes initiales
        self._df = df.copy()
        self._initial_rows = len(self._df)

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            value = self._df.iat[index.row(), index.column()]
            return str(value)
        # Highlight specific columns
        if role == Qt.BackgroundRole:
            col_name = self._df.columns[index.column()]
            if col_name in ['Materials/binder', 'Label in GT']:
                return QBrush(QColor('#fff2cc'))  # light background
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
            else:
                return str(self._df.index[section])
        return None

    def flags(self, index):
        base_flags = super().flags(index)
        if not index.isValid():
            return base_flags
        # Les lignes existantes (avant ajout) ne sont pas éditables
        if index.row() < self._initial_rows:
            return base_flags | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        # Les nouvelles lignes sont éditables
        return base_flags | Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        # N'autorise l'édition que pour les lignes ajoutées
        if role == Qt.EditRole and index.row() >= self._initial_rows:
            row, col = index.row(), index.column()
            try:
                orig = self._df.iat[row, col]
                converted = type(orig)(value)
            except Exception:
                converted = value
            self._df.iat[row, col] = converted
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

class NewLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add new label")
        layout = QFormLayout(self)
        self.name_edit = QLineEdit(self)
        layout.addRow("Name of new label :", self.name_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def label_name(self):
        return self.name_edit.text().strip()

class LabelWidget(QWidget):
    def __init__(self, csv_path,class_info={0:[None,None,(0,0,255)],1:[None,None,(0,255,0)]}):
        super().__init__()
        self.current_path = csv_path
        try:
            df, cols = load_and_prepare_df(csv_path)
        except Exception:
            # If fail, create empty with no columns first
            df = pd.DataFrame()
            cols = None
        self.default_cols = cols
        self.model = PandasModel(df)
        self.class_info=class_info

        # Vue principale
        self.tableView = QTableView()
        self.tableView.setModel(self.model)

        # Boutons pour gérer labels
        self.btn_keep=QPushButton("Keep to cube info")
        self.btn_keep.clicked.connect(self.keep_gt_cube_info)
        self.btn_load = QPushButton("Load table")
        self.btn_load.clicked.connect(self.load_table)
        self.btn_choose = QPushButton("Assign from table")
        self.btn_choose.clicked.connect(self.on_choose_label)
        self.btn_new = QPushButton("Add a new label")
        self.btn_new.clicked.connect(self.on_new_label)
        self.btn_delete = QPushButton("Delete added label")
        self.btn_delete.clicked.connect(self.on_delete_label)
        self.btn_save = QPushButton("Save modified table")
        self.btn_save.clicked.connect(self.save_table)

        # Layout
        # Création des vboxes et des séparateurs
        cols_titles = ["Classe idx", "GT idx", "GT name", "GT color"]
        self.vboxes = [QVBoxLayout() for _ in cols_titles]
        for vbox, title in zip(self.vboxes, cols_titles):
            header = QLabel(title)
            header.setAlignment(Qt.AlignCenter)
            header.setStyleSheet("font-weight:bold;")
            header.setFixedHeight(20)
            hdr_font = header.font()
            hdr_font.setPointSize(hdr_font.pointSize() + 2)
            header.setFont(hdr_font)
            vbox.addWidget(header)

            # Ligne horizontale sous le titre
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            vbox.addWidget(line)


        label_container=QWidget()
        label_hbox = QHBoxLayout(label_container)
        for i, vbox in enumerate(self.vboxes):
            # colonne dans un widget
            w = QWidget()
            w.setLayout(vbox)
            label_hbox.addWidget(w)
            # trait vertical sauf après la dernière colonne
            if i < len(self.vboxes) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setFrameShadow(QFrame.Sunken)
                sep.setLineWidth(2)
                label_hbox.addWidget(sep)

        # --- 3) Assemblage main : tableView | label_box ---
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(label_container)
        splitter.addWidget(self.tableView)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # taille caractere
        font = self.font()
        font.setPointSize(font.pointSize() + 2)
        self.setFont(font)

        # --- 4) Boutons en bas ---
        btn_hbox = QHBoxLayout()
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_keep)
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_load)
        btn_hbox.addWidget(self.btn_choose)
        btn_hbox.addWidget(self.btn_new)
        btn_hbox.addWidget(self.btn_delete)
        btn_hbox.addWidget(self.btn_save)
        btn_hbox.addStretch()

        # --- 5) Layout final ---
        outer_vbox = QVBoxLayout(self)
        outer_vbox.addWidget(splitter)
        outer_vbox.addLayout(btn_hbox)

        # Enfin, remplir la colonne de labels
        self.fill_label_box()

    def keep_gt_cube_info(self):
        pass

    def fill_label_box(self):
        """
        Remplit chaque VBox à partir de self.class_info = {
            key: [cls_idx, gt_idx, gt_name, (r,g,b)], …
        }
        """
        # On vide d’abord tout (si recyclage possible)
        for vbox in self.vboxes:
            # supprime tous les widgets enfants
            while vbox.count()>2:
                item = vbox.takeAt(2)
                w = item.widget()
                if w:
                    w.deleteLater()

        dic = self.class_info

        # Remplissage des lignes
        for key in dic:
            gt_idx, gt_name, gt_color=dic[key]
            # Col 0 : classe idx
            lbl = QLabel(str(key))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(20)
            self.vboxes[0].addWidget(lbl)
            # Col 1 : GT idx
            lbl = QLabel(str(gt_idx))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(20)
            self.vboxes[1].addWidget(lbl)
            # Col 2 : GT name
            lbl = QLabel(str(gt_name))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(20)
            self.vboxes[2].addWidget(lbl)
            # Col 3 : swatch couleur
            lbl = QLabel()
            lbl.setFixedHeight(20)
            r, g, b = gt_color
            lbl.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

            self.vboxes[3].addWidget(lbl)

    @pyqtSlot()
    def on_choose_label(self):
        index = self.tableView.currentIndex()
        if not index.isValid():
            return

        row = index.row()
        df = self.model._df  # ton DataFrame pandas

        # 1) Lecture des données de la ligne sélectionnée
        new_gt = int(df.at[row, "Label in GT"])
        new_name = df.at[row, "Materials/binder"]
        new_color = (df.at[row,"R"],df.at[row,"G"],df.at[row,"B"])

        # 2) Choix de la classe à mettre à jour
        keys = [str(k) for k in self.class_info.keys()]
        item, ok = QInputDialog.getItem(
            self,
            "Assign to class",
            "Choose class index to assign:",
            keys,
            0,
            False
        )
        if not ok:
            return

        cls_key = int(item)

        # Update
        print(cls_key,new_gt,new_name, new_color)
        self.class_info[cls_key] = [new_gt, new_name, new_color]

        # 4) Rafraîchissement de l’affichage
        self.fill_label_box()

    @pyqtSlot()
    def on_new_label(self):
        dlg = NewLabelDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            name = dlg.label_name()
            if not name:
                return
            color = QColorDialog.getColor(QColor(255,255,255), self)
            if not color.isValid():
                return
            r, g, b, _ = color.getRgb()
            df = self.model._df
            existing = pd.to_numeric(df['Label in GT'], errors='coerce').dropna().astype(int)
            new_gt = next(i for i in range(1, len(existing) + 2) if i not in set(existing))
            new_row = {col: '' for col in df.columns}
            new_row['Label in GT'] = new_gt
            new_row['Materials/binder'] = name
            # si colonnes R,G,B existent, on les utilise
            for comp, colname in zip((r,g,b), ['R','G','B']):
                if colname in new_row:
                    new_row[colname] = comp
            self.model._df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            self.model.layoutChanged.emit()
            last = self.model.rowCount() - 1
            sel = self.model.index(last, 0)
            self.tableView.scrollTo(sel)
            self.tableView.setCurrentIndex(sel)

    @pyqtSlot()
    def on_delete_label(self):
        index = self.tableView.currentIndex()
        if not index.isValid():
            return

        row = index.row()
        # On n’autorise que la suppression des lignes ajoutées,
        # pas celles d’origine
        if row < self.model._initial_rows:
            QMessageBox.information(
                self, "Warning",
                "You can not suppress original lines."
            )
            return

        # Confirmation
        reply = QMessageBox.question(
            self,
            "Confirme delete",
            "Sure you want to delete this label ?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # On retire la ligne du DataFrame
        df = self.model._df
        df = df.drop(index=row).reset_index(drop=True)
        # Mise à jour du modèle
        self.model.beginResetModel()
        self.model._df = df
        # On ne change _initial_rows : les anciennes restent protégées
        self.model.endResetModel()


        # On rafraîchit le panneau de labels
        self.fill_label_box()

    @pyqtSlot()
    def save_table(self):
        print('saving')
        path, _ = QFileDialog.getSaveFileName(self, "Save in CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            # Sauvegarde sans l'index pandas
            self.model._df.to_csv(path, index=False)
            print(f"Table sauvegardée dans : {path}")

    @pyqtSlot()
    def load_table(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un CSV", "", "CSV Files (*.csv);;All Files (*)")
        try:
            df, cols = load_and_prepare_df(path, self.default_cols)
        except Exception:
            res = QMessageBox.question(
                self, "Erreur de chargement",
                "Fichier non chargé. Voulez-vous en créer un nouveau ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if res == QMessageBox.Yes and self.default_cols:
                df = pd.DataFrame(columns=self.default_cols)
                cols = self.default_cols
            else:
                return
        self.model = PandasModel(df)
        self.tableView.setModel(self.model)
        self.current_path = path
        self.default_cols = cols
        print(f"CSV chargé : {path}")

def load_and_prepare_df(path, cols=None):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    if cols:
        # ensure same column order
        df = df.reindex(columns=cols)
    else:
        cols = list(df.columns)
        if 'Label in GT' in cols:
            cols.insert(0, cols.pop(cols.index('Label in GT')))
        df = df[cols]
    return df, cols

if __name__ == "__main__":
    app = QApplication(sys.argv)
    csv_file= 'ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv'
    win = LabelWidget(csv_file)
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())

