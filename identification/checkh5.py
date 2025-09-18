# checkh5.py — version IDE-friendly (tout dans /Metadata, pas de CLI)

import os
import numpy as np
import h5py
import textwrap

# Matplotlib optionnel
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------- Helpers d'affichage / décodage ----------

def _decode_bytes(x):
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return x

def _is_small_str(s, maxlen=256):
    try:
        return isinstance(s, str) and len(s) <= maxlen
    except Exception:
        return False

def _pretty_array(arr, max_items=20):
    a = np.asarray(arr)
    if a.dtype.kind in ('O', 'S', 'U'):
        flat = [_decode_bytes(v) for v in a.ravel()]
        if len(flat) <= max_items:
            return flat if a.ndim == 1 else np.array(flat).reshape(a.shape).tolist()
        head = flat[:max_items] + ["..."]
        return head
    return f"ndarray shape={a.shape}, dtype={a.dtype}"

def _print_kv(indent, k, v, max_items=20):
    if isinstance(v, (bytes, np.bytes_)):
        v = _decode_bytes(v)
    if isinstance(v, np.ndarray):
        v = _pretty_array(v, max_items=max_items)
    if isinstance(v, str) and not _is_small_str(v):
        v = textwrap.shorten(v, width=200, placeholder=" …")
    print(f"{indent}- {k}: {v}")

# ---------- Lecture / affichage de /Metadata ----------

def print_metadata_group(g, path="/Metadata", indent="  ", max_items=20):
    """Affiche récursivement le groupe /Metadata (attrs, datasets, sous-groupes)."""
    print(f"[{path}]")
    if len(g.attrs):
        for k, v in g.attrs.items():
            _print_kv(indent, k, v, max_items=max_items)
    for name, obj in g.items():
        if isinstance(obj, h5py.Dataset):
            val = obj[()]
            _print_kv(indent, name, val, max_items=max_items)
        elif isinstance(obj, h5py.Group):
            print_metadata_group(obj, path=f"{path}/{name}", indent=indent, max_items=max_items)
        else:
            print(f"{indent}- {name}: (unknown node)")

def get_metadata_dict(f):
    """Retourne un dict Python plat/imbriqué depuis /Metadata (utile si besoin)."""
    def _read_group(grp):
        out = {}
        for k, v in grp.attrs.items():
            out[k] = _decode_bytes(v)
        for name, obj in grp.items():
            if isinstance(obj, h5py.Dataset):
                val = obj[()]
                out[name] = _decode_bytes(val) if isinstance(val, (bytes, np.bytes_)) else val
            elif isinstance(obj, h5py.Group):
                out[name] = _read_group(obj)
        return out
    return _read_group(f["Metadata"]) if "Metadata" in f else {}

# ---------- Détection de contenu à prévisualiser ----------

def wavelength_to_band_indices(wl_array, targets=(650, 550, 450)):
    wl = np.asarray(wl_array).astype(float).ravel()
    return [int(np.argmin(np.abs(wl - t))) for t in targets]

def _normalize_img01(img, eps=1e-8):
    img = img.astype(np.float32)
    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)
    return np.clip((img - lo) / max(hi - lo, eps), 0, 1)

def _get_palette(meta_dict):
    pal = meta_dict.get("palette")
    if pal is None:
        return None
    pal = np.asarray(pal)
    if pal.ndim == 2 and pal.shape[1] == 3:
        return pal.astype(np.uint8) if pal.dtype != np.uint8 else pal
    return None

def _get_labels(meta_dict):
    raw = meta_dict.get("class_labels")
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.dtype.kind in ("S", "O", "U"):
        return [str(_decode_bytes(x)) for x in arr.ravel().tolist()]
    return [str(x) for x in arr.ravel().tolist()]

def choose_display_from_file(path):
    """
    Ouvre le .h5 et renvoie (mode, payload)
      - mode = "class_map": payload = {"class_map", "palette", "labels"}
      - mode = "cube":      payload = {"cube", "wl"}
      - mode = "none":      payload = {}
    """
    with h5py.File(path, "r") as f:
        # 1) class_map prioritaire
        if "class_map" in f:
            cm = f["class_map"][()]
            meta = get_metadata_dict(f)
            return "class_map", {
                "class_map": cm,
                "palette": _get_palette(meta),
                "labels": _get_labels(meta),
            }
        # 2) cube hyperspectral
        if "DataCube" in f:
            raw = f["DataCube"][()]
            if raw.ndim != 3:
                return "none", {}
            # on suppose (B,H,W) et on transpose → (H,W,B)
            cube = np.transpose(raw, (1, 2, 0)) if raw.shape[0] < raw.shape[-1] else raw
            wl = None
            if "Metadata" in f and "wl" in f["Metadata"]:
                try:
                    wl = f["Metadata"]["wl"][()]
                except Exception:
                    wl = None
            return "cube", {"cube": cube, "wl": wl}
        return "none", {}


# ---------- Fonctions “one-liners” pour l’IDE ----------

def overview(path, max_items=20):
    """Affiche le contenu racine + le dictionnaire /Metadata."""
    if not os.path.isfile(path):
        print(f"Erreur: fichier introuvable: {path}")
        return
    with h5py.File(path, "r") as f:
        print(f"Fichier: {path}\n")
        print("[Contenu racine]")
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                print(f"  - {k}: dataset shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  - {k}: group")
            else:
                print(f"  - {k}: (unknown node)")
        if "Metadata" in f:
            print("")
            print_metadata_group(f["Metadata"], max_items=max_items)
        else:
            print("\nAvertissement: groupe /Metadata non trouvé.")

def preview(path, title=None):
    """Affiche une prévisualisation (class_map avec palette, ou DataCube RGB)."""
    if not HAS_MPL:
        print("Matplotlib indisponible — impossible d'afficher une prévisualisation.")
        return
    mode, payload = choose_display_from_file(path)
    print(f"[Preview] mode = {mode}")
    if mode == "class_map":
        cm = np.asarray(payload["class_map"], dtype=np.int32)
        pal = payload.get("palette")
        labels = payload.get("labels")
        if pal is not None:
            idx = np.clip(cm, 0, pal.shape[0]-1)
            rgb = pal[idx]  # (H,W,3) uint8
            plt.figure()
            plt.imshow(rgb)
            plt.title(title or "class_map")
            plt.axis("off")
        else:
            plt.figure()
            plt.imshow(cm)
            plt.title(title or "class_map (sans palette)")
            plt.axis("off")
        if labels is not None and len(labels) <= 30:
            import matplotlib.patches as mpatches
            handles = []
            for i in np.unique(cm):
                lab = labels[i] if i < len(labels) else f"class_{i}"
                if pal is not None and i < len(pal):
                    color = tuple((pal[i] / 255.0).tolist())
                    patch = mpatches.Patch(color=color, label=f"{i}: {lab}")
                else:
                    patch = mpatches.Patch(label=f"{i}: {lab}")
                handles.append(patch)
            if handles:
                plt.legend(handles=handles, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

    elif mode == "cube":
        cube = payload["cube"]
        wl = payload.get("wl")
        H, W, B = cube.shape
        if B >= 3:
            if wl is not None and np.size(wl) == B:
                iR, iG, iB = wavelength_to_band_indices(wl, (650, 550, 450))
            else:
                iR, iG, iB = 0, 1, 2
            R = _normalize_img01(cube[:, :, iR])
            G = _normalize_img01(cube[:, :, iG])
            Bc = _normalize_img01(cube[:, :, iB])
            rgb = np.stack([R, G, Bc], axis=-1)
        else:
            rgb = np.repeat(_normalize_img01(cube[:, :, 0])[..., None], 3, axis=-1)
        plt.figure()
        plt.imshow(rgb)
        plt.title(title or "DataCube RGB")
        plt.axis("off")
        plt.show()
    else:
        print("Rien à afficher (ni class_map, ni DataCube).")

if __name__ == "__main__":
    path = r"C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\identification\saved/Test_Substrate_LDA_map.h5"
    overview(path)
    preview(path)