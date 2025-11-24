"""
Unmixing backend for Hypertool.

This module exposes small, dependency-light functions you can call from your
Qt GUI threads via workers. It standardizes array shapes and provides:

- Endmember extraction (EEA): ATGP, N-FINDR (via pysptools)
- Classic unmixers: UCLS, NNLS, FCLS (via pysptools)
- Sparse unmixing: SUnSAL (compact ADMM implementation here, no heavy deps)

Array shape conventions
-----------------------
- Cube (H, W, L): L bands, HxW pixels
- Spectral matrix Y: (L, N) with N = H*W, columns are pixel spectra
- Endmembers E: (L, p) where p is the number of endmembers
- Abundances A: (p, N) or (p, H, W) after reshape

Notes
-----
- All functions are pure (no global state); safe for use from worker threads.
- Inputs are converted to float64; outputs are float64.
- pysptools is optional but recommended. Functions that need it will raise a
  clear ImportError if it is missing.

Example (blind pipeline)
------------------------
>>> Y = vectorize_cube(cube)          # (L, N)
>>> E,idx = extract_endmembers_nfindr(Y, p=5)
>>> A = unmix_fcls(E, Y)              # classic ANC+ASC
>>> A_maps = abundances_to_maps(A, H, W)

Example (sparse with a library of endmembers)
--------------------------------------------
>>> E = load_library(...)             # (L, p)
>>> A = unmix_sunsal(E, Y, lam=1e-3, positivity=True, sum_to_one=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.signal import savgol_filter
import numpy as np

_aliases = {'int': int, 'float': float, 'bool': bool}
for _name, _typ in _aliases.items():
    if _name not in np.__dict__:
        setattr(np, _name, _typ)

# ---------- Utilities (vectorize/devectorize) ---------------------------------

def vectorize_cube(cube: np.ndarray) -> np.ndarray:
    """Convert a cube (H, W, L) to spectral matrix Y (L, N).

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube with shape (H, W, L).

    Returns
    -------
    Y : np.ndarray
        Spectral matrix with shape (L, N) where N = H*W.
    """
    if cube.ndim != 3:
        raise ValueError("cube must have shape (H, W, L)")
    H, W, L = cube.shape
    Y = np.ascontiguousarray(cube.reshape(H * W, L).T, dtype=np.float64)
    return Y

def abundances_to_maps(A: np.ndarray, H: int, W: int) -> np.ndarray:
    """Reshape abundances (p, N) to (p, H, W).
    """
    if A.ndim != 2:
        raise ValueError("A must have shape (p, N)")
    p, N = A.shape
    if N != H * W:
        raise ValueError("H*W must equal N in A=(p,N)")
    return A.reshape(p, H, W)

# ---------- Normalization utilities (cube & spectral matrix) -------------------

def normalize_cube(cube: np.ndarray, mode: str = 'L2', eps: float = 1e-12) -> np.ndarray:
    """Normalize a hyperspectral cube per-pixel.

    Parameters
    ----------
    cube : np.ndarray
        Input cube with shape (H, W, L).
    mode : {'none', 'L2', 'L1', 'sum1'}
        - 'none':    no normalization.
        - 'L2':      divide each pixel spectrum by its L2 norm (sqrt of sum of squares).
        - 'L1'/'sum1': divide each pixel spectrum by its L1 norm (sum of absolute values). In
                       standard reflectance data (non-negative), this is equivalent to enforcing
                       that the band-wise values sum to 1 ("sum = 1").
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    cube_n : np.ndarray
        Normalized cube with the same shape and dtype float64.
    """
    if cube.ndim != 3:
        raise ValueError("cube must have shape (H, W, L)")
    H, W, L = cube.shape
    X = np.asarray(cube, dtype=np.float64).reshape(-1, L)
    if mode is None or mode.lower() == 'none':
        return np.ascontiguousarray(X.reshape(H, W, L), dtype=np.float64)
    m = mode.lower()
    if m == 'l2':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
    elif m in ('l1', 'sum1'):
        norms = np.sum(np.abs(X), axis=1, keepdims=True)
    else:
        raise ValueError("mode must be one of {'none','L2','L1','sum1'}")
    norms = np.maximum(norms, eps)
    Xn = X / norms
    return np.ascontiguousarray(Xn.reshape(H, W, L), dtype=np.float64)

def normalize_spectra(Y: np.ndarray, mode: str = 'L2', eps: float = 1e-12) -> np.ndarray:
    """Normalize a spectral matrix Y (L, N) column-wise (per pixel).

    Parameters
    ----------
    Y : np.ndarray
        Spectral matrix (L, N).
    mode : {'none', 'L2', 'L1', 'sum1'}
        Same semantics as in normalize_cube().
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    Yn : np.ndarray (L, N)
        Normalized spectral matrix, float64.
    """
    if Y.ndim != 2:
        raise ValueError("Y must have shape (L, N)")
    L, N = Y.shape
    X = np.asarray(Y, dtype=np.float64)
    if mode is None or mode.lower() == 'none':
        return np.ascontiguousarray(X, dtype=np.float64)
    m = mode.lower()
    if m == 'l2':
        norms = np.linalg.norm(X, axis=0, keepdims=True)
    elif m in ('l1', 'sum1'):
        norms = np.sum(np.abs(X), axis=0, keepdims=True)
    else:
        raise ValueError("mode must be one of {'none','L2','L1','sum1'}")
    norms = np.maximum(norms, eps)
    return np.ascontiguousarray(X / norms, dtype=np.float64)

def _auto_sg_window(n_bands: int,
                    max_win: int = 15,
                    min_win: int = 5) -> int:
    """
    Choisit une taille de fenêtre impaire raisonnable pour Savitzky–Golay.
    """
    # fenêtre max bornée par max_win et par le nb de bandes
    w = min(max_win, n_bands)
    # on force à être impair
    if w % 2 == 0:
        w -= 1
    # on évite les fenêtres trop petites
    if w < min_win:
        w = max(3, n_bands if n_bands % 2 == 1 else n_bands - 1)
    return max(3, w)


def _apply_savgol(data: np.ndarray,
                  wl: Optional[np.ndarray],
                  order: int,
                  axis: int,
                  polyorder: int = 2) -> np.ndarray:
    """
    Applique Savitzky–Golay (dérivée d'ordre 0,1,2) le long de l'axe spectral.
    """
    arr = np.asarray(data, dtype=float)
    n_bands = arr.shape[axis]

    window_length = _auto_sg_window(n_bands, max_win=15, min_win=5)

    if wl is not None and len(wl) >= 2:
        wl = np.asarray(wl, dtype=float)
        delta = float(np.mean(np.diff(wl)))
    else:
        delta = 1.0

    out = savgol_filter(
        arr,
        window_length=window_length,
        polyorder=polyorder,
        deriv=order,       # 0 = lissage, 1 = 1ère dérivée, 2 = 2nde
        delta=delta,
        axis=axis,
        mode="interp",
    )
    return out

def preprocess_spectra(data: np.ndarray,
                       mode: str = "raw",
                       wl: Optional[np.ndarray] = None,
                       axis: int = -1) -> np.ndarray:
    """
    Applique le prétraitement choisi le long de l'axe spectral.
    mode: 'raw' | 'deriv1' | 'deriv2'
    """
    m = (mode or "raw").lower()
    if m in ("raw", "none"):
        return np.asarray(data, dtype=float)

    if m in ("deriv1", "first", "first derivative", "1st"):
        return _apply_savgol(data, wl=wl, order=1, axis=axis)
    if m in ("deriv2", "second", "second derivative", "2nd"):
        return _apply_savgol(data, wl=wl, order=2, axis=axis)

    raise ValueError(f"Unknown preprocess mode: {mode}")

# ---------- Optional dependency: pysptools wrappers ----------------------------

def _require_pysptools():
    try:
        import pysptools  # noqa: F401
    except Exception as e:
        raise ImportError(
            "This function requires 'pysptools'. Install it with:\n"
            "    pip install pysptools\n"
            f"Original import error: {e}"
        )

def extract_endmembers_atgp(data: np.ndarray, p: int) -> np.ndarray:
    """Extract endmembers with ATGP via pysptools.

    Parameters
    ----------
    Y : (L, N) array
    p : number of endmembers

    Returns
    -------
    E : (L, p) endmember matrix
    """
    _require_pysptools()
    from pysptools.eea import ATGP
    # pysptools expects (N, L) pixels x bands
    X = data
    atgp = ATGP()
    E_list = atgp.extract(X, p, normalize=False)
    indices_list=atgp.get_idx()
    indices=np.asarray(indices_list, dtype=np.int32)
    E = np.asarray(E_list, dtype=np.float64).T  # to (L, p)
    return E,indices

def extract_endmembers_nfindr(data: np.ndarray, p: int, maxit: int = 3) -> np.ndarray:
    """Extract endmembers with N-FINDR via pysptools.

    Parameters
    ----------
    Y : (L, N) array
    p : number of endmembers
    maxit : number of refinement iterations (default 3)

    Returns
    -------
    E : (L, p) endmember matrix
    """
    _require_pysptools()
    from pysptools.eea import NFINDR
    X = data
    nf = NFINDR()
    E_list = nf.extract(X, p, maxit=maxit, normalize=False)
    indices_list=nf.get_idx()
    indices=np.asarray(indices_list, dtype=np.int32)
    E = np.asarray(E_list, dtype=np.float64).T
    return E,indices

def unmix_ucls(E: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Unconstrained LS via pysptools (UCLS).

    Returns A with shape (p, N).
    """
    _require_pysptools()
    from pysptools.abundance_maps.amaps import UCLS
    L, p = E.shape
    if Y.shape[0] != L:
        raise ValueError("Y and E must share the same number of bands (rows)")
    X = Y.T.astype(np.float64, copy=False)
    U = E.T.astype(np.float64, copy=False)  # (p, L)

    A = UCLS(X,U).T

    return np.ascontiguousarray(A, dtype=np.float64)

def unmix_nnls(E: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Non-negative LS via pysptools (NNLS). Returns (p, N)."""
    _require_pysptools()
    from pysptools.abundance_maps.amaps import NNLS
    L, p = E.shape
    if Y.shape[0] != L:
        raise ValueError("Y and E must share the same number of bands (rows)")
    X = Y.T.astype(np.float64, copy=False)
    U = E.T.astype(np.float64, copy=False)  # (p, L)

    A = NNLS(X, U).T

    return np.ascontiguousarray(A, dtype=np.float64)

def unmix_fcls(E: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fully Constrained LS via pysptools (ANC+ASC). Returns (p, N)."""
    _require_pysptools()
    from pysptools.abundance_maps.amaps import FCLS
    L, p = E.shape
    if Y.shape[0] != L:
        raise ValueError("Y and E must share the same number of bands (rows)")
    X = Y.T.astype(np.float64, copy=False)
    U = E.T.astype(np.float64, copy=False)  # (p, L)

    A = FCLS(X, U).T

    return np.ascontiguousarray(A, dtype=np.float64)

# ---------- SUnSAL (compact ADMM) ---------------------------------------------
@dataclass
class SUnSALParams:
    lam: float = 1e-3        # L1 weight
    positivity: bool = True  # a >= 0
    sum_to_one: bool = False # sum(a) = 1 (ASC)
    rho: float = 1.0         # ADMM penalty
    max_iter: int = 1000
    tol: float = 1e-4        # relative change tolerance

def _proj_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto the probability simplex {a >= 0, sum(a) = 1}.
    Implementation of the O(N log N) algorithm.
    """
    if v.ndim != 1:
        raise ValueError("_proj_simplex expects a 1D vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, v.size + 1)) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def _sunsal_admm_single(E: np.ndarray, y: np.ndarray, prm: SUnSALParams) -> np.ndarray:
    """Solve for a single pixel: min_a 0.5||Ea - y||^2 + lam||a||_1  s.t. a>=0, optional sum(a)=1.
    Returns a (p,).
    """
    L, p = E.shape
    if y.shape[0] != L:
        raise ValueError("y and E must have consistent number of bands")

    ET = E.T
    # Precompute (E^T E + rho I)^{-1}
    G = ET @ E + prm.rho * np.eye(p)
    Ginv = np.linalg.inv(G)

    a = np.zeros(p, dtype=np.float64)
    z = np.zeros(p, dtype=np.float64)
    u = np.zeros(p, dtype=np.float64)

    Ey = ET @ y

    for _ in range(prm.max_iter):
        a_old = a
        # Quadratic update
        a = Ginv @ (Ey + prm.rho * (z - u))
        # Constraint handling
        if prm.sum_to_one:
            # ASC implies positivity via projection
            a = _proj_simplex(a)
        elif prm.positivity:
            a = np.maximum(a, 0.0)
        # Soft-threshold on z (l1)
        v = a + u
        z = np.sign(v) * np.maximum(np.abs(v) - prm.lam / prm.rho, 0.0)
        if (not prm.sum_to_one) and prm.positivity:
            z = np.maximum(z, 0.0)
        # Dual update
        u = u + a - z
        # Stopping criterion (relative change)
        denom = np.linalg.norm(a_old) + 1e-12
        if np.linalg.norm(a - a_old) / denom < prm.tol:
            break

    return a

def unmix_sunsal(
    E: np.ndarray,
    Y: np.ndarray,
    lam: float = 1e-3,
    positivity: bool = True,
    sum_to_one: bool = False,
    rho: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Sparse unmixing via SUnSAL (ADMM).

    Parameters
    ----------
    E : (L, p)
    Y : (L, N)
    lam : L1 regularization weight
    positivity : enforce a >= 0
    sum_to_one : enforce sum(a) = 1 (ASC). If True, positivity is implied.
    rho : ADMM penalty parameter
    max_iter : maximum iterations per pixel
    tol : relative change tolerance

    Returns
    -------
    A : (p, N) abundances
    """
    E = np.asarray(E, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    L, p = E.shape
    if Y.shape[0] != L:
        raise ValueError("E and Y must have the same number of bands (rows)")

    prm = SUnSALParams(lam=lam, positivity=positivity, sum_to_one=sum_to_one,
                       rho=rho, max_iter=max_iter, tol=tol)

    N = Y.shape[1]
    A = np.empty((p, N), dtype=np.float64)
    for i in range(N):
        A[:, i] = _sunsal_admm_single(E, Y[:, i], prm)
    return A

# ---------- Similarity metrics (cGFC, MSE) ------------------------------------

def metric_cgfc(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-12) -> float:
    """
    1 - GFC(y, y_hat) où GFC = (y·y_hat) / (||y|| ||y_hat||).
    """
    y = np.asarray(y, dtype=float).ravel()
    yh = np.asarray(y_hat, dtype=float).ravel()
    num = float(np.dot(y, yh))
    ny = float(np.linalg.norm(y)) + eps
    nyh = float(np.linalg.norm(yh)) + eps
    gfc = num / (ny * nyh)
    return 1.0 - gfc


def metric_mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    MSE(y, y_hat) = mean((y - y_hat)^2).
    """
    y = np.asarray(y, dtype=float).ravel()
    yh = np.asarray(y_hat, dtype=float).ravel()
    diff = yh - y
    return float(np.mean(diff * diff))

def _metric_cost_grad(E: np.ndarray,
                      y: np.ndarray,
                      a: np.ndarray,
                      metric: str = "cGFC",
                      eps: float = 1e-12) -> tuple[float, np.ndarray]:
    """
    Calcule (cost, grad) pour une métrique donnée.
    - E: (L, p)
    - y: (L,)
    - a: (p,)
    Retourne:
      cost: scalaire
      grad: (p,) gradient dJ/da
    """
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()

    L, p = E.shape
    if y.shape[0] != L:
        raise ValueError("E and y must have consistent number of bands")

    z = E @ a  # (L,)

    m = (metric or "cGFC").lower()

    if m in ("mse", "ls", "l2"):
        # J(a) = mean((Ea - y)^2)
        diff = z - y                 # (L,)
        cost = float(np.mean(diff * diff))
        grad = (2.0 / L) * (E.T @ diff)   # (p,)
        return cost, grad

    if m == "cgfc":
        # J(a) = 1 - (y·z)/(||y|| ||z||)
        ny = float(np.linalg.norm(y)) + eps
        nz = float(np.linalg.norm(z)) + eps
        num = float(np.dot(y, z))        # y·z
        gfc = num / (ny * nz)
        cost = 1.0 - gfc

        # gradient de J par rapport à a
        # dJ/dz = - d(GFC)/dz
        # GFC = (y·z)/(ny * nz)
        # => dGFC/dz = ( y * nz - (y·z)/nz * (z/nz) ) / (ny * nz)
        # (attention aux eps)
        y_vec = y
        z_vec = z
        # terme intermédiaire
        # d(GFC)/dz
        dG_dz = (y_vec * nz - (num / nz) * (z_vec / nz)) / (ny * nz)
        dJ_dz = -dG_dz
        grad = E.T @ dJ_dz   # (p,)
        return cost, grad

    raise ValueError(f"Unknown metric: {metric}")

def _unmix_metric_pg_single(E: np.ndarray,
                            y: np.ndarray,
                            metric: str = "cGFC",
                            anc: bool = True,
                            asc: bool = True,
                            max_iter: int = 200,
                            step: float = 1e-2,
                            tol: float = 1e-5) -> np.ndarray:
    """
    Unmixing pour un pixel unique par descente de gradient projetée sur ANC/ASC.

    min_a  J(a) = metric(Ea, y)
    s.t.   a >= 0 (si anc)
           sum(a) = 1 (si asc)

    Retourne a de shape (p,).
    """
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    L, p = E.shape
    if y.shape[0] != L:
        raise ValueError("E and y must have consistent number of bands")

    # ---- init ----
    if asc:
        # uniforme sur le simplexe
        a = np.full(p, 1.0 / p, dtype=float)
    elif anc:
        # non-negatif mais pas forcément normalisé
        a = np.zeros(p, dtype=float)
        a[0] = 1.0
    else:
        a = np.zeros(p, dtype=float)

    last_cost = None

    for it in range(max_iter):
        cost, grad = _metric_cost_grad(E, y, a, metric=metric)

        # étape de descente
        a_new = a - step * grad

        # projection sur le domaine admissible
        if asc and anc:
            a_new = _proj_simplex(a_new)  # déjà défini pour SUnSAL
        elif anc:
            a_new = np.maximum(a_new, 0.0)

        # critère d'arrêt
        if last_cost is not None:
            rel = abs(cost - last_cost) / (abs(last_cost) + 1e-12)
            if rel < tol:
                a = a_new
                break

        a = a_new
        last_cost = cost

    return a

def unmix_metric(E: np.ndarray,
                 Y: np.ndarray,
                 metric: str = "cGFC",
                 anc: bool = True,
                 asc: bool = True,
                 max_iter: int = 200,
                 step: float = 1e-2,
                 tol: float = 1e-5) -> np.ndarray:
    """
    Unmixing par métrique générale (cGFC, MSE, ...) pour tous les pixels.
    E : (L, p)
    Y : (L, N)
    Retourne A : (p, N)
    """
    E = np.asarray(E, dtype=float)
    Y = np.asarray(Y, dtype=float)
    L, p = E.shape
    if Y.shape[0] != L:
        raise ValueError("E and Y must share the same number of bands (rows)")

    N = Y.shape[1]
    A = np.zeros((p, N), dtype=float)

    for j in range(N):
        y = Y[:, j]
        a_j = _unmix_metric_pg_single(
            E, y,
            metric=metric,
            anc=anc,
            asc=asc,
            max_iter=max_iter,
            step=step,
            tol=tol,
        )
        A[:, j] = a_j

    return A

# ---------- Helpers: sanity checks, preprocessing -----------------------------

# --- Endmember groups utilities (UI ↔ backend) ---

def build_dictionary_from_groups(groups: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Concatenate group matrices into a single dictionary.

    Parameters
    ----------
    groups : dict[str, np.ndarray]
        Mapping {group_name: E_g} with E_g shape (L, p_g).
        All E_g must share the same L (bands).

    Returns
    -------
    E : (L, p_total) ndarray
        Concatenated dictionary [E_g1 | E_g2 | ...].
    labels : (p_total,) ndarray of dtype object
        Per-column group label (same order as E columns).
    index_map : dict[str, np.ndarray]
        Mapping {group_name: indices} giving column indices of E for each group.
    """
    if not groups:
        raise ValueError("'groups' must be a non-empty dict")
    names = list(groups.keys())
    # Validate shapes and collect
    E_list = []
    labels_list = []
    index_map = {}
    offset = 0
    L_ref = None
    for name in names:
        Eg = np.asarray(groups[name], dtype=np.float64)
        if Eg.ndim != 2:
            raise ValueError(f"Group '{name}' must be 2D (L, p_g)")
        L, p_g = Eg.shape
        if L_ref is None:
            L_ref = L
        elif L != L_ref:
            raise ValueError(f"All groups must have same L (bands). Got {L} vs {L_ref} for '{name}'.")
        E_list.append(Eg)
        labels_list.append(np.array([name]*p_g, dtype=object))
        index_map[name] = np.arange(offset, offset + p_g, dtype=int)
        offset += p_g
    E = np.concatenate(E_list, axis=1)
    labels = np.concatenate(labels_list, axis=0)
    return E, labels, index_map


def regroup_abundances_by_labels(A: np.ndarray, labels: np.ndarray) -> dict:
    """Sum abundances per group label.

    Parameters
    ----------
    A : (p, N) abundances
    labels : (p,) array-like of group names (object/str)

    Returns
    -------
    grouped : dict[str, np.ndarray]
        Mapping {group_name: a_sum} with a_sum shape (N,).
    """
    A = np.asarray(A, dtype=np.float64)
    labels = np.asarray(labels)
    if A.ndim != 2:
        raise ValueError("A must have shape (p, N)")
    if labels.shape[0] != A.shape[0]:
        raise ValueError("labels length must equal number of rows in A (p)")
    grouped = {}
    for name in np.unique(labels):
        mask = (labels == name)
        grouped[str(name)] = A[mask, :].sum(axis=0)
    return grouped


def split_abundances_by_labels(A: np.ndarray, labels: np.ndarray) -> dict:
    """Return abundances per group without summation.
    Each entry has shape (p_g, N).
    """
    A = np.asarray(A, dtype=np.float64)
    labels = np.asarray(labels)
    out = {}
    for name in np.unique(labels):
        mask = (labels == name)
        out[str(name)] = A[mask, :]
    return out


def normalize_endmembers(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each endmember spectrum to unit norm (column-wise)."""
    E = np.asarray(E, dtype=np.float64)
    norms = np.linalg.norm(E, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    return E / norms


def clip_abundances(A: np.ndarray, positivity: bool = True, asc: bool = False) -> np.ndarray:
    """Clip abundances to respect positivity/ASC approximately (post-processing)."""
    A = np.asarray(A, dtype=np.float64)
    if asc:
        # project each pixel onto simplex
        for i in range(A.shape[1]):
            A[:, i] = _proj_simplex(A[:, i])
        return A
    if positivity:
        return np.maximum(A, 0.0)
    return A


def abundance_maps_by_group(A: np.ndarray, labels: np.ndarray, H: int, W: int) -> dict:
    """Return (H, W) abundance maps aggregated per group.

    Parameters
    ----------
    A : (p, N) ndarray
        Abundance matrix (columns are pixels).
    labels : (p,) array-like
        Group label for each dictionary atom (column of E / row of A).
    H, W : int
        Image height and width. Must satisfy H*W == N.

    Returns
    -------
    maps : dict[str, np.ndarray]
        Mapping {group_name: map} where map has shape (H, W).
    """
    A = np.asarray(A, dtype=np.float64)
    labels = np.asarray(labels)
    if A.ndim != 2:
        raise ValueError("A must have shape (p, N)")
    p, N = A.shape
    if labels.shape[0] != p:
        raise ValueError("labels length must equal number of rows in A (p)")
    if H * W != N:
        raise ValueError("H*W must equal N in A=(p, N)")

    maps: dict[str, np.ndarray] = {}
    for name in np.unique(labels):
        mask = (labels == name)
        a_sum = A[mask, :].sum(axis=0)
        maps[str(name)] = a_sum.reshape(H, W)
    return maps

# ---------- End of module -----------------------------------------------------
