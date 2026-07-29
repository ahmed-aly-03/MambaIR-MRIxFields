#!/usr/bin/env python
"""Run a trained MambaIRv2 checkpoint over Validating_prospective and package a
Task 2 (0.1T -> higher field) T2FLAIR submission, per the official MRIxFields2026
challenge repo's spec (github.com/mrixfields/MRIxFields2026, Submission/README.md):

- Predictions: NIfTI float32, shape (X, Y, 30) -- the axial slab z in [150, 180)
  of the full (364, 436, 364) grid -- intensity in [0, 1].
- Filename: P_{MOD}_{TARGET_FIELD}_{ID:04d}.nii.gz (source ID kept, field tag
  swapped to the target field).
- Directory: <submission-dir>/task2/<MOD>/<SRC>_to_<TGT>/pred/<file>
- Zip root must be task2/ itself: `cd <submission-dir> && zip -r ~/task2.zip task2/`

Scope: T2FLAIR, 0.1T -> 3T only (matching train.py). Validation-phase released
0.1T subject IDs are 0001, 0002, 0003 (also per Submission/README.md).

This does NOT produce seg/ (SynthSeg segmentation) -- that needs the separate
SynthSeg tool from the official repo (Baseline/scripts/segment_predictions.py or
Evaluation/segment.py). Per the official spec, a submission with zero seg files
still scores SSIM/nRMSE/LPIPS normally; only Dice/Volume come back null.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

from mri import model as mri_model

# Single source of truth in the official repo: Baseline/mrixfields/zclip_constants.py
Z_CLIP_RANGE = (150, 180)


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
    official baseline's Baseline/scripts/inference.py `predict_volume`/`main`."""
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


def clip_axial(img: nib.Nifti1Image, z_range=Z_CLIP_RANGE) -> nib.Nifti1Image:
    """Axial slice [z0, z1) via nibabel's slicer, which shifts the affine origin
    to match -- must be done this way (not raw array slicing) so pred and the
    private GT end up on the same physical grid. See build_submission.py /
    Submission/README.md "Axial slice range" in the official repo."""
    z0, z1 = z_range
    return img.slicer[:, :, z0:z1]


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


def make_zip(submission_dir: Path, task: str, zip_path: Path):
    task_dir = submission_dir / task
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(task_dir.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(submission_dir))


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None,
                                help="YAML file of model-architecture defaults, e.g. configs/t2flair_0p1T_to_3T.yaml "
                                     "(the same file used for train.py -- unrelated training-only keys are ignored)")
    config_args, remaining_argv = config_parser.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[config_parser])
    p.add_argument("--checkpoint", required=True, help="a checkpoint saved by train.py, e.g. runs/.../stage2/stage2_best.pth")
    p.add_argument("--data-root", default=None, help="ChallengeData/ root; used to default --input-dir")
    p.add_argument("--input-dir", default=None,
                   help="directory of source-field .nii.gz volumes to run inference on "
                        "(default: <data-root>/Validating_prospective/<contrast>/<source-field>/)")
    p.add_argument("--subject-ids", nargs="+", default=None,
                   help="restrict to these IDs (default: all files found in --input-dir)")
    p.add_argument("--submission-dir", required=True, help="output root; files land in <submission-dir>/task2/...")
    p.add_argument("--contrast", default="T2FLAIR")
    p.add_argument("--source-field", default="0.1T")
    p.add_argument("--target-field", default="3T")
    p.add_argument("--slice-axis", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-full-volumes", default=None,
                   help="optional directory to also dump the full (uncropped) predicted volumes, for visual QC")
    p.add_argument("--zip", action="store_true", help="also write <submission-dir>/../task2.zip with the correct root")

    # model architecture (must match the checkpoint's training config)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=174)
    p.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    p.add_argument("--num-heads", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--inner-rank", type=int, default=64)
    p.add_argument("--num-tokens", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--convffn-kernel-size", type=int, default=5)

    if config_args.config:
        with open(config_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in p._actions}
        cfg = {k: v for k, v in cfg.items() if k in known}  # silently ignore train-only keys (stage1_*, stage2_*, ...)
        p.set_defaults(**cfg)

    args = p.parse_args(remaining_argv)
    if args.input_dir is None:
        if args.data_root is None:
            p.error("need --input-dir, or --data-root to default it")
        args.input_dir = str(Path(args.data_root) / "Validating_prospective" / args.contrast / args.source_field)
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: CUDA not available, falling back to CPU (mamba_ssm's selective scan needs a GPU; this will fail)")

    model = build_model(args, device)
    print(f"loaded {args.checkpoint}")

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(f"P_{args.contrast}_{args.source_field}_*.nii.gz"))
    if args.subject_ids:
        wanted = set(args.subject_ids)
        files = [f for f in files if subject_id_of(f) in wanted]
    if not files:
        raise RuntimeError(f"no P_{args.contrast}_{args.source_field}_*.nii.gz files found in {input_dir}")
    print(f"running inference on {len(files)} volumes from {input_dir}")

    pair = f"{args.source_field}_to_{args.target_field}"
    pred_dir = Path(args.submission_dir) / "task2" / args.contrast / pair / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.save_full_volumes:
        full_dir = Path(args.save_full_volumes)
        full_dir.mkdir(parents=True, exist_ok=True)

    for src_path in tqdm(files, desc="inference"):
        subject_id = subject_id_of(src_path)

        full_img = run_one_volume(model, src_path, args.slice_axis, device)

        if args.save_full_volumes:
            nib.save(full_img, str(full_dir / f"P_{args.contrast}_{args.target_field}_{subject_id}.nii.gz"))

        clipped = clip_axial(full_img)
        out_path = pred_dir / f"P_{args.contrast}_{args.target_field}_{subject_id}.nii.gz"
        nib.save(clipped, str(out_path))

    print(f"wrote {len(files)} predictions to {pred_dir}")
    print("NOTE: no seg/ written -- Dice/Volume will score as null; SSIM/nRMSE/LPIPS score normally.")
    print("      Run the official repo's Baseline/scripts/segment_predictions.py (or Evaluation/segment.py) "
          "on these predictions first if you also want Dice/Volume.")

    if args.zip:
        zip_path = Path(args.submission_dir).parent / "task2.zip"
        make_zip(Path(args.submission_dir), "task2", zip_path)
        print(f"wrote {zip_path} (verify with: unzip -l {zip_path} | head -3 -- first entry should be task2/)")


if __name__ == "__main__":
    main()
