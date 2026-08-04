#!/usr/bin/env python
"""Two-stage finetuning of MambaIRv2 for MRIxFields Task 2 (0.1T -> higher field),
T2FLAIR only.

Stage 1 (unpaired, retrospective): PatchNCE contrastive loss only, no discriminator.
    The model is applied to a source-field (0.1T) patch to produce y = G(x); the model
    is then applied again to y (detached) to re-extract features at the same tap
    points. PatchNCE pulls together features at matching spatial locations between the
    two passes and pushes apart features at other locations of the same image -- this
    needs no ground-truth target for x and therefore works on the unpaired retrospective
    pool. An optional identity term does the same thing starting from a real
    target-field patch, to discourage the model from altering target-domain images.

Stage 2 (paired, prospective): standard supervised L1 + SSIM finetuning on the very
    few paired 0.1T<->target-field volunteers, initialized from the stage-1 checkpoint.

See README.md for setup and example commands.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from mri import model as mri_model
from mri.datasets import PairedSliceDataset, UnpairedSliceDataset
from mri.losses import PatchNCELoss, PatchSampleMLP, SupervisedLoss, ssim
from mri.utils import set_seed


class FeatureExtractor:
    """Grabs the output of a fixed set of named submodules via forward hooks."""

    def __init__(self, model: torch.nn.Module, layer_names):
        self.feats = {}
        modules = dict(model.named_modules())
        self.handles = [modules[name].register_forward_hook(self._hook(name)) for name in layer_names]

    def _hook(self, name):
        def fn(_module, _inp, out):
            self.feats[name] = out
        return fn

    def clear(self):
        self.feats = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def default_nce_layers(model, num_taps: int = 4):
    num_layers = len(model.layers)
    positions = torch.linspace(0, num_layers - 1, steps=max(num_taps - 1, 1)).tolist()
    idx = sorted(set(int(round(p)) for p in positions))
    return ["conv_first"] + [f"layers.{i}" for i in idx]


def build_model_for_stage(args, device):
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
        use_checkpoint=args.use_checkpoint,
    )
    return net.to(device)


def run_stage1(args, model, device, log):
    root = Path(args.data_root)
    dataset = UnpairedSliceDataset(
        root, args.contrast, args.source_field, args.target_field,
        patch_size=args.patch_size, slice_axis=args.slice_axis, min_fg_frac=args.min_fg_frac,
        index_workers=args.index_workers,
    )
    loader = DataLoader(dataset, batch_size=args.stage1_batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    log(f"[stage1] {len(dataset)} source slices from {args.source_field}, "
        f"{len(dataset.tgt_slices)} target slices from {args.target_field}")

    layer_names = default_nce_layers(model, args.nce_num_taps)
    extractor = FeatureExtractor(model, layer_names)
    sampler = PatchSampleMLP(layer_names, feature_dim=args.embed_dim, num_patches=args.nce_num_patches).to(device)
    nce = PatchNCELoss(nce_t=args.nce_t).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(sampler.parameters()), lr=args.stage1_lr, betas=(0.9, 0.999))

    def nce_direction(src_img):
        extractor.clear()
        y = model(src_img)
        feats_src = copy.copy(extractor.feats)
        extractor.clear()
        _ = model(y.detach())
        feats_y = copy.copy(extractor.feats)

        proj_src, ids = sampler(feats_src)
        proj_y, _ = sampler(feats_y, patch_ids=ids)
        losses = [nce(proj_y[name], proj_src[name]) for name in layer_names]
        return sum(losses) / len(losses)

    model.train()
    step = 0
    out_dir = Path(args.out_dir) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while step < args.stage1_iters:
        for batch in loader:
            if step >= args.stage1_iters:
                break
            A = batch["A"].to(device, non_blocking=True)
            B_img = batch["B"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = args.nce_main_weight * nce_direction(A)
            if args.idt_weight > 0:
                loss = loss + args.idt_weight * nce_direction(B_img)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                log(f"[stage1] step {step}/{args.stage1_iters} loss {loss.item():.4f} "
                    f"({(time.time() - t0) / max(step, 1):.2f}s/it)")
            if step % args.ckpt_every == 0 and step > 0:
                mri_model.save_checkpoint(model, str(out_dir / f"stage1_iter{step:06d}.pth"))
            step += 1

    mri_model.save_checkpoint(model, str(out_dir / "stage1_final.pth"))
    extractor.remove()
    log(f"[stage1] done, saved {out_dir / 'stage1_final.pth'}")
    return out_dir / "stage1_final.pth"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    l1_sum, ssim_sum, n = 0.0, 0.0, 0
    for batch in loader:
        lq = batch["lq"].to(device)
        gt = batch["gt"].to(device)
        pred = model(lq).clamp(0, 1)
        l1_sum += torch.nn.functional.l1_loss(pred, gt, reduction="sum").item()
        ssim_sum += ssim(pred, gt).item() * lq.shape[0]
        n += lq.shape[0]
    model.train()
    return l1_sum / (n * lq.shape[-1] * lq.shape[-2]), ssim_sum / n


def run_stage2(args, model, device, log):
    root = Path(args.data_root)
    val_ids = set(args.stage2_val_ids)
    train_ids = [i for i in args.stage2_subject_ids if i not in val_ids]

    train_set = PairedSliceDataset(root, args.contrast, args.source_field, args.target_field,
                                    train_ids, patch_size=args.patch_size, slice_axis=args.slice_axis,
                                    min_fg_frac=args.min_fg_frac, train=True)
    train_loader = DataLoader(train_set, batch_size=args.stage2_batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    log(f"[stage2] train subjects {train_ids}: {len(train_set)} paired slices")

    val_loader = None
    if val_ids:
        val_set = PairedSliceDataset(root, args.contrast, args.source_field, args.target_field,
                                      sorted(val_ids), patch_size=args.patch_size, slice_axis=args.slice_axis,
                                      min_fg_frac=args.min_fg_frac, train=False)
        val_loader = DataLoader(val_set, batch_size=args.stage2_batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        log(f"[stage2] val subjects {sorted(val_ids)}: {len(val_set)} paired slices")

    criterion = SupervisedLoss(l1_weight=args.l1_weight, ssim_weight=args.ssim_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_iters)

    out_dir = Path(args.out_dir) / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ssim = -1.0

    model.train()
    step = 0
    t0 = time.time()
    while step < args.stage2_iters:
        for batch in train_loader:
            if step >= args.stage2_iters:
                break
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lq)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                log(f"[stage2] step {step}/{args.stage2_iters} loss {loss.item():.4f} "
                    f"({(time.time() - t0) / max(step, 1):.2f}s/it)")

            if val_loader is not None and step % args.val_every == 0 and step > 0:
                val_l1, val_ssim = evaluate(model, val_loader, device)
                log(f"[stage2] step {step} val L1 {val_l1:.4f} SSIM {val_ssim:.4f}")
                if val_ssim > best_ssim:
                    best_ssim = val_ssim
                    mri_model.save_checkpoint(model, str(out_dir / "stage2_best.pth"),
                                               extra={"val_ssim": val_ssim, "step": step})

            if step % args.ckpt_every == 0 and step > 0:
                mri_model.save_checkpoint(model, str(out_dir / f"stage2_iter{step:06d}.pth"))
            step += 1

    mri_model.save_checkpoint(model, str(out_dir / "stage2_final.pth"))
    log(f"[stage2] done, saved {out_dir / 'stage2_final.pth'}")


def make_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    return log


def parse_args():
    # --config is handled in a first pass so its values become argparse defaults;
    # any flag also given explicitly on the command line still wins over the config file.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML file of CLI flag defaults, e.g. configs/t2flair_0p1T_to_3T.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[config_parser])
    p.add_argument("--data-root", default=None, help="path to ChallengeData/ (contains Training_retrospective, Training_prospective); required, via CLI or --config")
    p.add_argument("--out-dir", default=None, help="required, via CLI or --config")
    p.add_argument("--contrast", default="T2FLAIR")
    p.add_argument("--source-field", default="0.1T")
    p.add_argument("--target-field", default="3T")
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--slice-axis", type=int, default=2, help="NIfTI axis to slice 2D images from")
    p.add_argument("--min-fg-frac", type=float, default=0.05)
    p.add_argument("--index-workers", type=int, default=8,
                   help="threads used for the one-time foreground-slice indexing scan (cached to .index_cache/ after)")

    # model / pretrained
    p.add_argument("--pretrained", default=None, help="path to an upstream MambaIRv2 .pth (3-channel) to adapt")
    p.add_argument("--init-checkpoint", default=None, help="resume/init from a checkpoint saved by this script (already 1-channel)")
    p.add_argument("--embed-dim", type=int, default=174)
    p.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    p.add_argument("--num-heads", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--inner-rank", type=int, default=64)
    p.add_argument("--num-tokens", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--convffn-kernel-size", type=int, default=5)
    p.add_argument("--use-checkpoint", action="store_true",
                   help="gradient checkpointing (trades compute for a large activation-memory cut, "
                        "especially valuable for stage 1's double forward+backward pass)")

    # stage 1 (unpaired PatchNCE)
    p.add_argument("--stage1-iters", type=int, default=20000)
    p.add_argument("--stage1-batch-size", type=int, default=4)
    p.add_argument("--stage1-lr", type=float, default=2e-4)
    p.add_argument("--nce-t", type=float, default=0.07)
    p.add_argument("--nce-num-patches", type=int, default=256)
    p.add_argument("--nce-num-taps", type=int, default=4, help="how many layers (incl. conv_first) to tap for PatchNCE")
    p.add_argument("--nce-main-weight", type=float, default=1.0)
    p.add_argument("--idt-weight", type=float, default=0.0,
                   help="identity PatchNCE term on real target-field patches; 0 disables it (2x cheaper)")

    # stage 2 (paired supervised)
    p.add_argument("--stage2-iters", type=int, default=5000)
    p.add_argument("--stage2-batch-size", type=int, default=8)
    p.add_argument("--stage2-lr", type=float, default=1e-5)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--stage2-subject-ids", nargs="+", default=None,
                   help="prospective volunteer IDs to use, e.g. 0006 0007 0009 (default: auto-discover)")
    p.add_argument("--stage2-val-ids", nargs="+", default=[],
                   help="subset of --stage2-subject-ids held out for validation (leave-subject-out)")

    if config_args.config:
        with open(config_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"unknown key(s) in --config {config_args.config}: {sorted(unknown)}")
        p.set_defaults(**cfg)

    args = p.parse_args(remaining_argv)
    if not args.data_root:
        p.error("--data-root is required (via CLI or --config)")
    if not args.out_dir:
        p.error("--out-dir is required (via CLI or --config)")
    return args


def discover_subject_ids(root: Path, contrast: str, source_field: str, target_field: str):
    from mri.utils import find_volumes
    lq = find_volumes(root, "Training_prospective", contrast, source_field, prefix="P")
    gt = find_volumes(root, "Training_prospective", contrast, target_field, prefix="P")
    return sorted(set(lq) & set(gt))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    log = make_logger(Path(args.out_dir))
    log(f"args: {vars(args)}")

    model = build_model_for_stage(args, device)

    if args.init_checkpoint:
        mri_model.load_checkpoint(model, args.init_checkpoint, map_location=device)
        log(f"loaded init checkpoint {args.init_checkpoint}")
    elif args.pretrained:
        mri_model.adapt_pretrained_checkpoint(model, args.pretrained)
    else:
        log("WARNING: no --pretrained or --init-checkpoint given, training from random init")

    stage1_ckpt = None
    if args.stage in ("1", "both"):
        stage1_ckpt = run_stage1(args, model, device, log)

    if args.stage in ("2", "both"):
        if args.stage == "2" and stage1_ckpt is None and not args.init_checkpoint:
            log("WARNING: running stage 2 without a stage-1/init checkpoint; "
                "make sure --pretrained or --init-checkpoint was set")
        if not args.stage2_subject_ids:
            args.stage2_subject_ids = discover_subject_ids(
                Path(args.data_root), args.contrast, args.source_field, args.target_field)
            log(f"auto-discovered paired subject IDs: {args.stage2_subject_ids}")
        run_stage2(args, model, device, log)


if __name__ == "__main__":
    main()
