from PyQt5 import QtWidgets, QtCore
from typing import  List
import numpy as np
from collections import OrderedDict


from hypercubes.hypercube import CubeInfoTemp,Hypercube

class HypercubeManager(QtCore.QObject):
    """Manage a collection of CubeInfoTemp objects and emit updates."""
    cubesChanged = QtCore.pyqtSignal(list)  # emit a change in cube list
    metadataUpdated = QtCore.pyqtSignal(object)  # to emit a change in a cube_info attribute

    def __init__(self):
        super().__init__()
        # store CubeInfoTemp instances
        self._cubes_info_list: List[CubeInfoTemp] = []
        self._cube_cache: OrderedDict[str, Hypercube] = OrderedDict()
        self._max_cache_size = 5
        self._updating_metadata = False # flag

    def addCube(self, ci):
        """
        Add a new cube if not already present. Emits updated list of filepaths.
        """

        if not isinstance(ci,CubeInfoTemp):
            if isinstance(ci,Hypercube):
                ci=ci.cube_info
            else:
                print("Problem for loading cube info")
                return

        filepath=ci.filepath
        if not filepath:
            return

        # prevent duplicates
        if any(ci.filepath == filepath for ci in self._cubes_info_list):
            return

        if len(ci.metadata_temp) ==0:
            print(f"[ !!! ] Warning: metadata_temp is empty for {filepath}. Will remain unloaded until accessed.")

        self._cubes_info_list.append(ci)
        self.cubesChanged.emit(self.paths)

    def ensureLoadedCubeInfo(self, ci: CubeInfoTemp) -> CubeInfoTemp:

        if ci.data_path is None:
            self.get_loaded_cube(ci.filepath, cube_info=ci)
        return ci

    def get_loaded_cube(self, filepath: str,cube_info=None) -> Hypercube:
        """
        Return cached Hypercube if present. Otherwise, load it,
        add to cache (with LRU policy), and return.
        """
        if filepath in self._cube_cache:
            self._cube_cache.move_to_end(filepath)
            return self._cube_cache[filepath]

        # Load cube and update cache
        hc = Hypercube(filepath=filepath, cube_info=cube_info, load_init=True)

        self.add_to_cache(hc)

        return hc

    def add_to_cache(self, hc: Hypercube):
        """
        Add a Hypercube to the cache, respecting the LRU policy.

        If the cache exceeds the maximum size, removes the least recently used cube.
        """
        ci = hc.cube_info
        self._cube_cache[ci.filepath] = hc
        self._cube_cache.move_to_end(ci.filepath)  # Mark as recently used

        if len(self._cube_cache) > self._max_cache_size:
            removed_fp, removed_hc = self._cube_cache.popitem(last=False)
            print(f"[LRU] Cache full. Removed oldest cube: {removed_fp}")

    def removeCube(self, index: int):
        """
        Remove a cube by index. Emits updated list of filepaths.
        """
        if 0 <= index < len(self._cubes_info_list):
            self._cubes_info_list.pop(index)
            self.cubesChanged.emit(self.paths)

    def clearCubes(self):
        """Clear all cubes and emit update."""
        self._cubes_info_list.clear()
        self.cubesChanged.emit(self.paths)

    def getCubeInfo(self, index: int) -> CubeInfoTemp:
        """Return the CubeInfoTemp at the given index."""
        return self._cubes_info_list[index]

    def getIndexFromPath(self, filepath: str) -> int:
        """
        Return the index of the cube with the given filepath.
        Returns -1 if not found.
        """
        for i, ci in enumerate(self._cubes_info_list):
            if ci.filepath == filepath:
                return i
        return -1

    def compare_metadata_dicts(self,d1, d2):
        if d1.keys() != d2.keys():
            return False

        for key in d1:
            v1 = d1[key]
            v2 = d2[key]

            try:
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
            except :
                return False

        return True

    def updateMetadata(self, updated_ci: CubeInfoTemp):
        if self._updating_metadata:
            return

        index = self.getIndexFromPath(updated_ci.filepath)
        if index == -1:
            print(f"[Warning] Cube not found: {updated_ci.filepath}")
            return

        # to check if modif in hypercube ok before emitting
        test = not self.compare_metadata_dicts(self._cubes_info_list[index].metadata_temp, updated_ci.metadata_temp)
        if test:
            self._updating_metadata = True
            self._cubes_info_list[index] = updated_ci
            updated_ci.modif = True
            self.metadataUpdated.emit(updated_ci)
            self._updating_metadata = True

    def add_or_update_cube(self, ci: CubeInfoTemp):
        """
        Add cube if new, or update metadata if it already exists.
        Emits cubesChanged or metadataUpdated accordingly.
        """
        index = self.getIndexFromPath(ci.filepath)

        if index == -1:
            # New cube
            self._cubes_info_list.append(ci)
            self.cubesChanged.emit(self.paths)
        else:
            # Existing cube → update metadata
            self.updateMetadata(ci)

    def addOrSyncCube(self, filepath: str) -> CubeInfoTemp:
        """
        Check if the cube is already in the manager.
        If so, return the existing CubeInfoTemp.
        Otherwise, load it from disk, add it to the list, and return it.
        """
        index = self.getIndexFromPath(filepath)
        if index != -1:
            return self._cubes_info_list[index]

        # Cube is not in the list → load and add it
        ci = CubeInfoTemp(filepath=filepath)
        hc = Hypercube(filepath=filepath, cube_info=ci, load_init=True)
        ci = hc.cube_info
        self._cubes_info_list.append(ci)
        self.cubesChanged.emit(self.paths)
        return ci

    @property
    def paths(self) -> List[str]:
        """List of cube filepaths."""
        return [ci.filepath for ci in self._cubes_info_list]

    @property
    def cubes(self) -> List[CubeInfoTemp]:
        """List of all CubeInfoTemp objects."""
        return list(self._cubes_info_list)


if __name__ == "__main__":
    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Samples\minicubes/'
    cube_1 = '00001-VNIR-mock-up.h5'
    cube_2 = '00002-VNIR-mock-up.h5'
    paths = [folder + cube_1, folder + cube_2]
    hm=HypercubeManager()
    for path in paths:
        ci=CubeInfoTemp(filepath=path)
        hm.addCube(ci)

