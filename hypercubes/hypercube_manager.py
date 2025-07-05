from PyQt5 import QtWidgets, QtCore
from dataclasses import dataclass, field
from typing import Optional, Dict, List,Union
import numpy as np

from hypercubes.hypercube import CubeInfoTemp,Hypercube

class HypercubeManager(QtCore.QObject):
    """Manage a collection of CubeInfoTemp objects and emit updates."""
    cubesChanged = QtCore.pyqtSignal(list)  # emit a change in cube list
    metadataUpdated = QtCore.pyqtSignal(object)  # to emit a change in a cube_info attribute

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

    def getIndexFromPath(self, filepath: str) -> int:
        """
        Return the index of the cube with the given filepath.
        Returns -1 if not found.
        """
        for i, ci in enumerate(self._cubes):
            if ci.filepath == filepath:
                return i
        return -1

    def compare_metadata_dicts(self,d1, d2):
        if d1.keys() != d2.keys():
            return False

        for key in d1:
            v1 = d1[key]
            v2 = d2[key]
            print(v1,v2)

            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False
            elif isinstance(v1, list) and isinstance(v2, list):
                if v1 != v2:
                    return False
            elif isinstance(v1, dict) and isinstance(v2, dict):
                if v1 != v2:
                    return False
            else:
                if v1 != v2:
                    return False

        return True

    def updateMetadata(self, updated_ci: CubeInfoTemp):
        index = self.getIndexFromPath(updated_ci.filepath)
        if index == -1:
            print(f"[Warning] Cube not found: {updated_ci.filepath}")
            return

        # to check if modif in hypercube ok before emitting
        test = not self.compare_metadata_dicts(self._cubes[index].metadata_temp, updated_ci.metadata_temp)
        if test:
            self._cubes[index] = updated_ci
            updated_ci.modif = True
            self.metadataUpdated.emit(updated_ci)

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

