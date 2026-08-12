#!/usr/bin/env python
"""Compute nRMSE / SSIM / LPIPS / PSNR (+ L1) for a trained checkpoint against
real ground truth, on paired Training_prospective subjects -- typically
whichever subject was held out from Stage 2 via --stage2-val-ids, since that's
the only place we actually have both a source-field input and a target-field
ground truth for the same volunteer.

This is NOT infer.py: infer.py runs on Validating_prospective, which has no
released ground truth, so it can only produce predictions, not scores. This
script needs ground truth and therefore only works on Training_prospective.

Voxel-level metrics (nRMSE, SSIM, LPIPS) are implemented to match the official
MRIxFields2026 evaluator's exact formulas (github.com/mrixfields/MRIxFields2026,
Submission/evaluation-2026/evaluate.py), confirmed by reading that file directly:
  - nrmse = ||pred - target||_2 / ||target||_2, over the whole volume, no mask
  - ssim: skimage.metrics.structural_similarity computed per axial slice, with
    data_range = target.max() - target.min() taken over the WHOLE volume (not
    per-slice), averaged over slices (slices where the target is ~constant are
    skipped, matching the official implementation)
  - lpips: AlexNet backbone (`lpips` package), inputs scaled to [-1, 1] and
    replicated to 3 channels, computed per axial slice and averaged (slices
    where the target is ~all-zero are skipped)

PSNR is NOT part of the official metric set -- confirmed absent from the
official evaluator's code. Included anyway since it was explicitly requested;
computed with max_val=1.0 (the data's normalized intensity range). L1 is also
reported for continuity with train.py's periodic stage-2 validation.

Reports TWO scopes per subject, since real challenge scoring happens on a
cropped axial slab, not the whole brain:
  - "official_slab": z in [150, 180) of the canonical (364, 436, 364) grid --
    the same Z_CLIP_RANGE infer.py/the official repo use for submissions. This
    is the number that's actually representative of competition scoring.
  - "full_volume": the entire brain, for a broader sense of restoration
    quality beyond that one 30-slice sliver.

NOT computed: Dice overlap / normalized volume consistency. Both require
running SynthSeg segmentation first (Baseline/scripts/segment_predictions.py
or Evaluation/segment.py in the official repo) -- out of scope here, same
caveat as infer.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity
from tqdm import tqdm

from mri import utils as mri_utils
from mri.inference import build_model, run_one_volume

Z_CLIP_RANGE = (150, 180)


def compute_nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    norm = np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / norm) if norm > 1e-10 else 0.0


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mse = float(np.mean((pred - target) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, slice_axis: int = 2) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    data_range = target.max() - target.min()
    if data_range < 1e-10:
        return 1.0
    vals = []
    for i in range(pred.shape[slice_axis]):
        idx = [slice(None)] * pred.ndim
        idx[slice_axis] = i
        idx = tuple(idx)
        ps, ts = pred[idx], target[idx]
        if ts.max() - ts.min() < 1e-10:
            continue
        vals.append(structural_similarity(ps, ts, data_range=data_range))
    return float(np.mean(vals)) if vals else 1.0


def compute_lpips(pred: np.ndarray, target: np.ndarray, device, slice_axis: int = 2) -> float:
    import lpips as lpips_module
    fn = lpips_module.LPIPS(net="alex").to(device)
    fn.eval()
    pred_n = pred.astype(np.float64) * 2.0 - 1.0
    target_n = target.astype(np.float64) * 2.0 - 1.0

    def _2d(p, t):
        pt = torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        tt = torch.from_numpy(t).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            return float(fn(pt, tt).item())

    vals = []
    for i in range(pred.shape[slice_axis]):
        idx = [slice(None)] * pred.ndim
        idx[slice_axis] = i
        idx = tuple(idx)
        if np.abs(target_n[idx]).max() < 1e-10:
            continue
        vals.append(_2d(pred_n[idx], target_n[idx]))
    return float(np.mean(vals)) if vals else 0.0


def all_metrics(pred: np.ndarray, target: np.ndarray, device, slice_axis: int = 2, label: str = "") -> dict:
    l1 = float(np.mean(np.abs(pred.astype(np.float64) - target.astype(np.float64))))
    psnr = compute_psnr(pred, target)
    nrmse = compute_nrmse(pred, target)
    print(f"    [{label}] l1/psnr/nrmse done, running ssim ({pred.shape[slice_axis]} slices)...", flush=True)
    ssim = compute_ssim(pred, target, slice_axis)
    print(f"    [{label}] ssim done, running lpips (loads AlexNet on first call -- "
          f"needs internet the very first time) ...", flush=True)
    lpips_val = compute_lpips(pred, target, device, slice_axis)
    print(f"    [{label}] lpips done", flush=True)
    return {"l1": l1, "psnr": psnr, "nrmse": nrmse, "ssim": ssim, "lpips": lpips_val}


def evaluate_subject(model, lq_path: Path, gt_path: Path, slice_axis: int, device) -> tuple:
    print(f"  running full-volume inference on {lq_path.name} ...", flush=True)
    pred_img = run_one_volume(model, lq_path, slice_axis, device)
    print("  inference done", flush=True)
    # run_one_volume returns pred in lq_path's own original orientation (correct
    # for saving as a submission file); re-canonicalize so pred, gt, and lq are
    # all compared/saved in the same space, matching the official evaluator's own
    # load_nifti() (which canonicalizes both pred and target files it loads).
    pred_canon = nib.as_closest_canonical(pred_img)
    gt_canon = nib.as_closest_canonical(nib.load(str(gt_path)))
    lq_canon = nib.as_closest_canonical(nib.load(str(lq_path)))

    pred = pred_canon.get_fdata(dtype=np.float32)
    gt = gt_canon.get_fdata(dtype=np.float32)

    if pred.shape != gt.shape:
        raise RuntimeError(
            f"shape mismatch for {lq_path.name} vs {gt_path.name}: pred {pred.shape} vs gt {gt.shape}")

    z0, z1 = Z_CLIP_RANGE
    full = all_metrics(pred, gt, device, slice_axis, label="full_volume")
    slab = all_metrics(pred[:, :, z0:z1], gt[:, :, z0:z1], device, slice_axis, label="official_slab")
    metrics = {"full_volume": full, "official_slab": slab}
    images = {"input": lq_canon, "prediction": pred_canon, "ground_truth": gt_canon}
    return metrics, images


def save_samples(images: dict, out_dir: Path, subject_id: str, slice_axis: int = 2) -> None:
    """Writes input/prediction/ground_truth as NIfTI (for browsing in a viewer) plus
    a single labeled side-by-side PNG of a representative slice (for dropping
    straight into a slide/report)."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        nib.save(img, str(out_dir / f"{name}.nii.gz"))

    z0, z1 = Z_CLIP_RANGE
    mid = (z0 + z1) // 2  # representative slice: middle of the officially-scored slab
    order = ["input", "prediction", "ground_truth"]
    titles = {"input": "Input", "prediction": "Prediction", "ground_truth": "Ground truth"}

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, key in zip(axes, order):
        idx = [slice(None)] * 3
        idx[slice_axis] = mid
        sl = images[key].get_fdata(dtype=np.float32)[tuple(idx)]
        ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=1)
        ax.set_title(titles[key], fontsize=11)
        ax.axis("off")
    fig.suptitle(f"subject {subject_id} -- axial slice {mid}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None,
                                help="YAML file of model-architecture defaults, e.g. configs/t2flair_0p1T_to_3T.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[config_parser])
    p.add_argument("--checkpoint", required=True, help="e.g. runs/t2flair_0p1T_3T/stage2/stage2_best.pth")
    p.add_argument("--data-root", required=True, help="ChallengeData/ root")
    p.add_argument("--split", default="Training_prospective",
                   help="only Training_prospective has ground truth for paired subjects")
    p.add_argument("--subject-ids", nargs="+", required=True,
                   help="e.g. 0009 -- typically whatever --stage2-val-ids the run used")
    p.add_argument("--contrast", default="T2FLAIR")
    p.add_argument("--source-field", default="0.1T")
    p.add_argument("--target-field", default="3T")
    p.add_argument("--slice-axis", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="where to write the metrics JSON")
    p.add_argument("--save-samples", default=None,
                   help="optional directory to also save input/prediction/ground_truth NIfTI volumes "
                        "plus a labeled side-by-side comparison.png, one subfolder per subject")

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

    return p.parse_args(remaining_argv)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: CUDA not available, falling back to CPU")

    model = build_model(args, device)
    print(f"loaded {args.checkpoint}")

    root = Path(args.data_root)
    lq_volumes = mri_utils.find_volumes(root, args.split, args.contrast, args.source_field, prefix="P")
    gt_volumes = mri_utils.find_volumes(root, args.split, args.contrast, args.target_field, prefix="P")

    pair = f"{args.source_field}_to_{args.target_field}"
    per_subject = {}
    for sid in tqdm(args.subject_ids, desc="evaluating"):
        if sid not in lq_volumes or sid not in gt_volumes:
            print(f"WARNING: subject {sid} missing from {args.source_field} or {args.target_field} "
                  f"under {args.split}/{args.contrast} -- skipping")
            continue
        metrics, images = evaluate_subject(model, lq_volumes[sid], gt_volumes[sid], args.slice_axis, device)
        per_subject[sid] = metrics
        slab = metrics["official_slab"]
        print(f"[{sid}] official_slab: nRMSE {slab['nrmse']:.4f}  SSIM {slab['ssim']:.4f}  "
              f"LPIPS {slab['lpips']:.4f}  PSNR {slab['psnr']:.2f}  L1 {slab['l1']:.4f}")

        if args.save_samples:
            sample_dir = Path(args.save_samples) / args.contrast / pair / sid
            save_samples(images, sample_dir, sid, args.slice_axis)
            print(f"  saved samples to {sample_dir}")

    def mean_across_subjects(scope):
        keys = next(iter(per_subject.values()))[scope].keys()
        return {k: float(np.mean([v[scope][k] for v in per_subject.values()
                                   if np.isfinite(v[scope][k])])) for k in keys}

    payload = {
        "checkpoint": args.checkpoint,
        "contrast": args.contrast,
        "source_field": args.source_field,
        "target_field": args.target_field,
        "subject_ids": list(per_subject.keys()),
        "per_subject": per_subject,
        "mean": {
            "full_volume": mean_across_subjects("full_volume") if per_subject else {},
            "official_slab": mean_across_subjects("official_slab") if per_subject else {},
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
