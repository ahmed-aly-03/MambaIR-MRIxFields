import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

FILENAME_RE = re.compile(r"^(?P<prefix>[RP])_(?P<contrast>[A-Za-z0-9]+)_(?P<field>[0-9.]+T)_(?P<id>\d+)\.nii\.gz$")


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


@lru_cache(maxsize=64)
def load_volume(path: Path) -> np.ndarray:
    """Loads reoriented to RAS+ canonical, matching the official MRIxFields2026
    baseline's `mrixfields.data.utils.load_nifti` -- this dataset should already be
    on a consistent grid/orientation, but canonicalizing defensively here keeps
    training slices in the same orientation convention `infer.py` uses at inference
    time (which must canonicalize, since it has to reorient predictions back for
    arbitrary/unknown input orientations)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=np.float32)


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
