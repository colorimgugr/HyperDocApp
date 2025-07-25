from collections import OrderedDict
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

from hypercubes.hypercube import (Hypercube, CubeInfoTemp)

class HypercubeManager(QObject):
    """
    Manager for Hypercube instances and their metadata.
    Maintains an LRU cache of loaded Hypercube objects, and a registry
    of CubeInfoTemp objects, one per unique filepath.
    """
    cubes_changed = pyqtSignal(list)
    metadata_updated = pyqtSignal(object)  # emits CubeInfoTemp

    def __init__(self, max_cache_size: int = 10):
        super().__init__()
        # Mapping from resolved filepath -> CubeInfoTemp
        self._cubes_info = OrderedDict()
        # LRU cache: resolved filepath -> Hypercube
        self._cube_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self._updating_metadata = False

    @property
    def paths(self) -> list:
        """List of registered cube filepaths."""
        return list(self._cubes_info.keys())

    def get_cube_info_by_index(self, index: int) -> CubeInfoTemp:
        """Renvoie le CubeInfoTemp à la position index dans la liste."""
        path = self.paths[index]
        return self._cubes_info[path]

    def add_or_sync_cube(self, filepath: str) -> CubeInfoTemp:
        """
        Add a new cube or return existing CubeInfoTemp for the given filepath.
        If the cube is not yet registered, load it into cache (with init) and emit cubes_changed.
        """
        path = str(Path(filepath).resolve())
        # Return existing instance if present
        if path in self._cubes_info:
            return self._cubes_info[path]

        # Create new CubeInfoTemp and load initial data
        ci = CubeInfoTemp(_filepath=path)
        hc = Hypercube(filepath=path, cube_info=ci, load_init=True)

        # Register and cache
        self._cubes_info[path] = ci
        self._add_to_cache(path, hc)

        # Notify listeners
        self.cubes_changed.emit(self.paths)
        return ci

    def get_loaded_cube(self, filepath: str, cube_info: CubeInfoTemp = None) -> Hypercube:
        """
        Return a loaded Hypercube instance for the given filepath.
        If not in cache, load it (using provided cube_info or register new), and enforce LRU limits.
        """
        path = str(Path(filepath).resolve())
        # If cached, move to end and return
        if path in self._cube_cache:
            hc = self._cube_cache.pop(path)
            self._cube_cache[path] = hc
            return hc

        # Not cached: ensure CubeInfoTemp exists
        ci = cube_info or self.add_or_sync_cube(path)
        hc = Hypercube(filepath=path, cube_info=ci, load_init=True)
        self._add_to_cache(path, hc)
        return hc

    def _add_to_cache(self, key: str, value) -> None:
        """
        Insert into LRU cache and evict oldest if above limit.
        """
        self._cube_cache[key] = value
        while len(self._cube_cache) > self.max_cache_size:
            self._cube_cache.popitem(last=False)

    def update_metadata(self, filepath: str, key: str, value) -> None:
        """
        Update metadata for a registered cube and emit metadata_updated.
        """
        path = str(Path(filepath).resolve())
        ci = self._cubes_info.get(path)
        if not ci or self._updating_metadata:
            return

        self._updating_metadata = True
        ci.metadata_temp[key] = value
        self.metadata_updated.emit(ci)
        self._updating_metadata = False

    def remove_cube(self, filepath: str) -> None:
        """
        Remove a cube and its cache entry if present, and emit cubes_changed.
        """
        path = str(Path(filepath).resolve())
        removed = False
        if path in self._cubes_info:
            del self._cubes_info[path]
            removed = True
        if path in self._cube_cache:
            del self._cube_cache[path]
            removed = True
        if removed:
            self.cubes_changed.emit(self.paths)

    def clear_cubes(self) -> None:
        """
        Clear all registered cubes and cache, and emit cubes_changed.
        """
        self._cubes_info.clear()
        self._cube_cache.clear()
        self.cubes_changed.emit(self.paths)


    def rename_cube(self, old_path: str, new_path: str) -> None:
        """
        Change the registered filepath of a cube from old_path to new_path,
        preserving its CubeInfoTemp and Hypercube instances, and emit cubes_changed.
        """
        old = str(Path(old_path).resolve())
        new = str(Path(new_path).resolve())

        if old not in self._cubes_info:
            return

        # 1) Met à jour _cubes_info
        ci = self._cubes_info.pop(old)
        ci.filepath = new      # met à jour l'attribut filepath via votre setter
        self._cubes_info[new] = ci

        # 2) Met à jour le cache LRU (_cube_cache) si le cube était chargé
        if old in self._cube_cache:
            hc = self._cube_cache.pop(old)
            self._cube_cache[new] = hc

        # 3) Émet le signal pour rafraîchir vos menus/listes
        self.cubes_changed.emit(self.paths)

