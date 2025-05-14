from PyQt5 import QtWidgets, QtCore
from dataclasses import dataclass, field
from typing import Optional, Dict, List,Union
import numpy as np

from hypercubes.hypercube import CubeInfoTemp

class HypercubeManager(QtCore.QObject):
    """Manage a collection of CubeInfoTemp objects and emit updates."""
    cubesChanged = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        # store CubeInfoTemp instances
        self._cubes: List[CubeInfoTemp] = []

    def addCube(self, filepath: str):
        """
        Add a new cube if not already present. Emits updated list of filepaths.
        """
        if not filepath:
            return
        # prevent duplicates
        if any(ci.filepath == filepath for ci in self._cubes):
            return
        cube = CubeInfoTemp(filepath=filepath)
        self._cubes.append(cube)
        self.cubesChanged.emit(self.paths)

    def removeCube(self, index: int):
        """
        Remove a cube by index. Emits updated list of filepaths.
        """
        if 0 <= index < len(self._cubes):
            self._cubes.pop(index)
            self.cubesChanged.emit(self.paths)

    def clearCubes(self):
        """Clear all cubes and emit update."""
        self._cubes.clear()
        self.cubesChanged.emit(self.paths)

    def getCubeInfo(self, index: int) -> CubeInfoTemp:
        """Return the CubeInfoTemp at the given index."""
        return self._cubes[index]

    @property
    def paths(self) -> List[str]:
        """List of cube filepaths."""
        return [ci.filepath for ci in self._cubes]

    @property
    def cubes(self) -> List[CubeInfoTemp]:
        """List of all CubeInfoTemp objects."""
        return list(self._cubes)
