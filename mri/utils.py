import hashlib
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

FILENAME_RE = re.compile(r"^(?P<prefix>[RP])_(?P<contrast>[A-Za-z0-9]+)_(?P<field>[0-9.]+T)_(?P<id>\d+)\.nii\.gz$")

INDEX_CACHE_DIR = Path(__file__).resolve().parent.parent / ".index_cache"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.groupdict()


def find_volumes(root: Path, split: str, contrast: str, field: str, prefix: str):
    """root/<split>/<contrast>/<field>/{R,P}_<contrast>_<field>_<id>.nii.gz"""
    d = root / split / contrast / field
    if not d.is_dir():
        raise FileNotFoundError(f"expected directory not found: {d}")
    out = {}
    for p in sorted(d.glob(f"{prefix}_{contrast}_{field}_*.nii.gz")):
        meta = parse_filename(p)
        if meta is None:
            continue
        out[meta["id"]] = p
    return out


from functools import lru_cache


def _read_volume(path: Path) -> np.ndarray:
    """Reoriented to RAS+ canonical, matching the official MRIxFields2026 baseline's
    `mrixfields.data.utils.load_nifti` -- this dataset should already be on a
    consistent grid/orientation, but canonicalizing defensively here keeps training
    slices in the same orientation convention `infer.py` uses at inference time
    (which must canonicalize, since it has to reorient predictions back for
    arbitrary/unknown input orientations).

    Deliberately NOT cached -- see `load_volume` below for the cached entry point.
    This one exists so the one-time dataset-indexing scan (in mri/datasets.py) can
    load volumes from multiple threads without them all serializing on lru_cache's
    single lock, which would defeat the parallelism.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=np.float32)


@lru_cache(maxsize=64)
def load_volume(path: Path) -> np.ndarray:
    """Cached entry point for repeated random-access reads during training."""
    return _read_volume(path)


def _volume_signature(path: Path) -> tuple:
    st = path.stat()
    return (str(path), st.st_size, int(st.st_mtime))


def index_cache_path(volumes, slice_axis: int, min_fg_frac: float) -> Path:
    """A stable cache file path for a given set of volumes + indexing params.
    `volumes` is anything whose values are Paths (dict, or an iterable of Paths).
    Scanning every retrospective volume to find foreground slices can take hours on
    slow/contended shared storage -- this lets that only ever happen once per
    (file set, size, mtime, params) combination."""
    paths = volumes.values() if isinstance(volumes, dict) else volumes
    sig = sorted(_volume_signature(p) for p in paths)
    payload = json.dumps({"sig": sig, "axis": slice_axis, "frac": min_fg_frac}, sort_keys=True)
    key = hashlib.sha1(payload.encode()).hexdigest()
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_CACHE_DIR / f"{key}.json"


def load_index_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return json.load(f)


def save_index_cache(cache_path: Path, data) -> None:
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(cache_path)


def foreground_slice_indices(volume: np.ndarray, axis: int, min_fg_frac: float = 0.02):
    """Indices along `axis` where the fraction of non-background (>0) voxels exceeds a threshold."""
    n = volume.shape[axis]
    keep = []
    for i in range(n):
        sl = np.take(volume, i, axis=axis)
        fg_frac = float((sl > 1e-4).mean())
        if fg_frac >= min_fg_frac:
            keep.append(i)
    return keep


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def random_crop_pair(a: np.ndarray, b: np.ndarray, patch_size: int):
    """Crop the same location from two spatially-aligned 2D arrays. Pads with zeros if smaller than patch_size."""
    h, w = a.shape
    ph = max(patch_size, h)
    pw = max(patch_size, w)
    if (ph, pw) != (h, w):
        a = np.pad(a, ((0, ph - h), (0, pw - w)))
        b = np.pad(b, ((0, ph - h), (0, pw - w)))
        h, w = ph, pw
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return a[top:top + patch_size, left:left + patch_size], b[top:top + patch_size, left:left + patch_size]


def random_crop_single(a: np.ndarray, patch_size: int):
    h, w = a.shape
    ph = max(patch_size, h)
    pw = max(patch_size, w)
    if (ph, pw) != (h, w):
        a = np.pad(a, ((0, ph - h), (0, pw - w)))
        h, w = ph, pw
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return a[top:top + patch_size, left:left + patch_size]


def center_crop_pair(a: np.ndarray, b: np.ndarray, patch_size: int):
    h, w = a.shape
    ph = max(patch_size, h)
    pw = max(patch_size, w)
    if (ph, pw) != (h, w):
        a = np.pad(a, ((0, ph - h), (0, pw - w)))
        b = np.pad(b, ((0, ph - h), (0, pw - w)))
        h, w = ph, pw
    top = (h - patch_size) // 2
    left = (w - patch_size) // 2
    return a[top:top + patch_size, left:left + patch_size], b[top:top + patch_size, left:left + patch_size]


def sync_augment_pair(a: np.ndarray, b: np.ndarray):
    if random.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    if random.random() < 0.5:
        a, b = a[::-1, :], b[::-1, :]
    k = random.randint(0, 3)
    if k:
        a, b = np.rot90(a, k), np.rot90(b, k)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def augment_single(a: np.ndarray):
    if random.random() < 0.5:
        a = a[:, ::-1]
    if random.random() < 0.5:
        a = a[::-1, :]
    k = random.randint(0, 3)
    if k:
        a = np.rot90(a, k)
    return np.ascontiguousarray(a)


def to_chw_tensor(a: np.ndarray) -> torch.Tensor:
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(a).unsqueeze(0)
