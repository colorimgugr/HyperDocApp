from PyQt5 import QtWidgets, QtCore
from dataclasses import dataclass, field
from typing import Optional, Dict, List,Union
import numpy as np

from hypercubes.hypercube import CubeInfoTemp,Hypercube

class HypercubeManager(QtCore.QObject):
    """Manage a collection of CubeInfoTemp objects and emit updates."""
    cubesChanged = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        # store CubeInfoTemp instances
        self._cubes: List[CubeInfoTemp] = []

    def addCube(self, ci):
        """
        Add a new cube if not already present. Emits updated list of filepaths.
        """

        if not isinstance(ci,CubeInfoTemp):
            if isinstance(ci,Hypercube):
                cube=ci
                ci=ci.cube_info
            else:
                print("Problem for loading cube info")
                return

        filepath=ci.filepath
        if not filepath:
            return

        # prevent duplicates
        if any(ci.filepath == filepath for ci in self._cubes):
            return

        if len(ci.metadata_temp) ==0:
            hc = Hypercube(filepath=filepath, load_init=True)
            ci=hc.cube_info

        self._cubes.append(ci)
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


if __name__ == "__main__":
    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Samples\minicubes/'
    cube_1 = '00001-VNIR-mock-up.h5'
    cube_2 = '00002-VNIR-mock-up.h5'
    paths = [folder + cube_1, folder + cube_2]
    hm=HypercubeManager()
    for path in paths:
        ci=CubeInfoTemp(filepath=path)
        hm.addCube(ci)

