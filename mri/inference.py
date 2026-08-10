"""Shared full-volume inference helpers, used by both infer.py (submission
packaging) and evaluate.py (metric computation against known ground truth).
Kept in one place so the canonical-orientation round trip -- the part most
likely to have a subtle bug if duplicated -- only exists once.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from . import model as mri_model


def predict_volume(model, volume: np.ndarray, slice_axis: int, device) -> np.ndarray:
    """Slice-by-slice inference along `slice_axis`. Input/output stay in [0, 1]
    (matching how train.py's datasets feed the model -- no [-1,1] rescale)."""
    output = np.zeros_like(volume, dtype=np.float32)
    n = volume.shape[slice_axis]
    with torch.no_grad():
        for i in range(n):
            idx = [slice(None)] * volume.ndim
            idx[slice_axis] = i
            idx = tuple(idx)
            sl = np.clip(volume[idx], 0.0, 1.0).astype(np.float32)
            tensor = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(tensor).clamp(0.0, 1.0)
            output[idx] = pred.squeeze(0).squeeze(0).cpu().numpy()
    return output


def run_one_volume(model, src_path: Path, slice_axis: int, device) -> nib.Nifti1Image:
    """Canonical-orientation round trip + background remasking, matching the
    official baseline's Baseline/scripts/inference.py `predict_volume`/`main`.
    Returns the prediction back in `src_path`'s own original orientation
    (correct for saving as a submission file); callers that need to compare
    against a ground-truth volume loaded via `nib.as_closest_canonical` should
    re-canonicalize the returned image first (see evaluate.py)."""
    original_img = nib.load(str(src_path))
    original_affine = original_img.affine
    original_ornt = nib.io_orientation(original_affine)

    canonical_img = nib.as_closest_canonical(original_img)
    canonical_ornt = nib.io_orientation(canonical_img.affine)
    canonical_data = canonical_img.get_fdata(dtype=np.float32)

    pred = predict_volume(model, canonical_data, slice_axis, device)

    transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
    pred = nib.orientations.apply_orientation(pred, transform)

    # Skull-stripped input -> background should stay exactly 0; the network's
    # global residual can leak small nonzero values there, so remask using the
    # original (un-reoriented) input as a brain mask.
    src_arr = original_img.get_fdata(dtype=np.float32)
    pred = pred * (src_arr > 1e-6).astype(pred.dtype)

    return nib.Nifti1Image(pred.astype(np.float32), original_affine, header=original_img.header)


def build_model(args, device):
    net = mri_model.build_mambairv2(
        in_chans=1,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        window_size=args.window_size,
        inner_rank=args.inner_rank,
        num_tokens=args.num_tokens,
        d_state=args.d_state,
        convffn_kernel_size=args.convffn_kernel_size,
        img_size=args.patch_size,
    )
    net = net.to(device)
    mri_model.load_checkpoint(net, args.checkpoint, map_location=device)
    net.eval()
    return net


def subject_id_of(path: Path) -> str:
    return path.name.replace(".nii.gz", "").split("_")[-1]
