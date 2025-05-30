import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QPushButton, QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QFileDialog,
    QMessageBox
)
from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSlot
from PyQt5.QtGui import QBrush, QColor


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


class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
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
            return str(self._df.iat[index.row(), index.column()])
        if role == Qt.BackgroundRole:
            col = self._df.columns[index.column()]
            if col in ['Materials/binder', 'Label in GT']:
                return QBrush(QColor('#fff2cc'))
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
            return str(self._df.index[section])
        return None

    def flags(self, index):
        base = super().flags(index)
        if not index.isValid():
            return base
        if index.row() < self._initial_rows:
            return base | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return base | Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and index.row() >= self._initial_rows:
            try:
                orig = self._df.iat[index.row(), index.column()]
                val = type(orig)(value)
            except Exception:
                val = value
            self._df.iat[index.row(), index.column()] = val
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False


class NewLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nouveau label")
        layout = QFormLayout(self)
        self.name_edit = QLineEdit(self)
        layout.addRow("Nom du label :", self.name_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def label_name(self):
        return self.name_edit.text().strip()


class LabelWidget(QWidget):
    def __init__(self, csv_path):
        super().__init__()
        self.current_path = csv_path
        # Try load default df
        try:
            df, cols = load_and_prepare_df(csv_path)
        except Exception:
            # If fail, create empty with no columns first
            df = pd.DataFrame()
            cols = None
        self.default_cols = cols
        self.model = PandasModel(df)

        self.tableView = QTableView()
        self.tableView.setModel(self.model)

        # Buttons
        self.btn_load = QPushButton("Charger CSV")
        self.btn_load.clicked.connect(self.load_table)
        self.btn_choose = QPushButton("Choisir un label")
        self.btn_choose.clicked.connect(self.on_choose_label)
        self.btn_new = QPushButton("Nouveau label")
        self.btn_new.clicked.connect(self.on_new_label)
        self.btn_save = QPushButton("Enregistrer")
        self.btn_save.clicked.connect(self.save_table)

        # Layout
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.tableView)
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.btn_load)
        hbox.addWidget(self.btn_choose)
        hbox.addWidget(self.btn_new)
        hbox.addWidget(self.btn_save)
        vbox.addLayout(hbox)

    @pyqtSlot()
    def load_table(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
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

    @pyqtSlot()
    def on_choose_label(self):
        idx = self.tableView.currentIndex()
        if idx.isValid():
            print(f"Label choisi : {self.model.data(idx, Qt.DisplayRole)}")

    @pyqtSlot()
    def on_new_label(self):
        dlg = NewLabelDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            name = dlg.label_name()
            if name:
                df = self.model._df
                existing = pd.to_numeric(df['Label in GT'], errors='coerce').dropna().astype(int)
                new_gt = next(i for i in range(1, len(existing) + 2) if i not in set(existing))
                new_row = {col: '' for col in df.columns}
                new_row['Label in GT'] = new_gt
                new_row['Materials/binder'] = name
                self.model._df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                self.model.layoutChanged.emit()
                last = self.model.rowCount() - 1
                sel = self.model.index(last, 0)
                self.tableView.scrollTo(sel)
                self.tableView.setCurrentIndex(sel)
                print(f"Nouveau label ajouté : GT={new_gt}, Material={name}")

    @pyqtSlot()
    def save_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le CSV", self.current_path, "CSV Files (*.csv);;All Files (*)")
        if path:
            self.model._df.to_csv(path, index=False)
            print(f"Table sauvegardée dans : {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    file = "Materials labels and palette assignation - Materials_labels_palette.csv"
    widget = LabelWidget(file)
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())
