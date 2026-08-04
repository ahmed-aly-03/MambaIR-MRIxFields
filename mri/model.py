from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# The vendored MambaIR repo's arch code does `from basicsr...`, expecting the
# MambaIR/ checkout itself (which contains its own basicsr/ package) to be on
# sys.path -- not any pip-installed `basicsr`.
_MAMBAIR_ROOT = Path(__file__).resolve().parent.parent / "MambaIR"
if str(_MAMBAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAMBAIR_ROOT))

from basicsr.archs.mambairv2_arch import MambaIRv2  # noqa: E402


def build_mambairv2(
    in_chans: int = 1,
    embed_dim: int = 174,
    depths=(6, 6, 6, 6, 6, 6),
    num_heads=(6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    inner_rank: int = 64,
    num_tokens: int = 128,
    d_state: int = 16,
    convffn_kernel_size: int = 5,
    mlp_ratio: float = 2.0,
    img_size: int = 64,
    img_range: float = 1.0,
    resi_connection: str = "1conv",
    use_checkpoint: bool = False,
) -> MambaIRv2:
    """MambaIRv2 configured for field-to-field MRI translation, not spatial SR:
    - in_chans=1 (single-channel MRI, not RGB)
    - upsampler='' and upscale=1 => the SR pixel-shuffle head is skipped entirely;
      the reconstruction head is a single Conv2d(embed_dim, 1, kernel_size=3) with a
      global residual (this is MambaIRv2's built-in "denoising" branch -- see
      `basicsr/archs/mambairv2_arch.py` forward(), the `else` clause), which is
      exactly "remove the SR upsampling head and replace it with a conv3x3 head".
    - use_checkpoint enables gradient checkpointing of each of the 36 individual
      AttentiveLayer sub-blocks (6 ASSB stages x depth 6), trading recompute for a
      large activation-memory cut during backward. NOTE: upstream MambaIRv2 accepts
      this flag at every level (MambaIRv2/ASSB/BasicBlock) but never actually used it
      -- it was a no-op. We patched BasicBlock.forward() in the vendored
      basicsr/archs/mambairv2_arch.py to actually wrap each AttentiveLayer call in
      torch.utils.checkpoint.checkpoint(). Stage 1 runs two full forward+backward
      passes per step at full patch resolution, which is easy to push past 80GB
      without this.
    """
    return MambaIRv2(
        img_size=img_size,
        patch_size=1,
        in_chans=in_chans,
        embed_dim=embed_dim,
        d_state=d_state,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        inner_rank=inner_rank,
        num_tokens=num_tokens,
        convffn_kernel_size=convffn_kernel_size,
        mlp_ratio=mlp_ratio,
        upscale=1,
        img_range=img_range,
        upsampler="",
        resi_connection=resi_connection,
        use_checkpoint=use_checkpoint,
    )


def _unwrap_state_dict(ckpt: dict) -> dict:
    for key in ("params_ema", "params", "state_dict"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def adapt_pretrained_checkpoint(model: MambaIRv2, ckpt_path: str, verbose: bool = True) -> MambaIRv2:
    """Load a pretrained (3-channel, usually SR/pixel-shuffle-head) MambaIRv2 checkpoint
    into a 1-channel, no-upsample-head model:

    - conv_first.weight: pretrained shape is (embed_dim, 3, 3, 3). We average over the
      input-channel dim to get (embed_dim, 1, 3, 3), matching the classic RGB->grayscale
      first-conv adaptation. conv_first.bias is copied as-is (channel-independent).
    - conv_last.* (and any pixel-shuffle head modules: conv_before_upsample, upsample,
      conv_up1/2, conv_hr): never loaded. These are either absent from the target model
      (pixel-shuffle head) or architecturally a fresh single-channel output layer that
      must be trained from scratch, so they're left at their random initialization.
    - Everything else (patch_embed/unembed, ASSB body layers, conv_after_body, norm) is
      copied when shapes match exactly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src_sd = _unwrap_state_dict(ckpt)
    dst_sd = model.state_dict()

    loaded, skipped, mismatched = [], [], []

    for key, dst_tensor in dst_sd.items():
        if key.startswith("conv_last") or key.startswith("conv_before_upsample") or \
           key.startswith("upsample") or key.startswith("conv_up") or key.startswith("conv_hr"):
            skipped.append(key)
            continue

        if key == "conv_first.weight":
            src_w = src_sd.get(key)
            if src_w is None:
                skipped.append(key)
                continue
            if src_w.shape[0] != dst_tensor.shape[0]:
                mismatched.append(key)
                continue
            dst_sd[key] = src_w.mean(dim=1, keepdim=True)
            loaded.append(key)
            continue

        src_tensor = src_sd.get(key)
        if src_tensor is None:
            skipped.append(key)
        elif src_tensor.shape == dst_tensor.shape:
            dst_sd[key] = src_tensor
            loaded.append(key)
        else:
            mismatched.append(key)

    model.load_state_dict(dst_sd, strict=True)

    if verbose:
        print(f"[adapt_pretrained_checkpoint] loaded {len(loaded)} tensors from {ckpt_path}")
        print(f"[adapt_pretrained_checkpoint] kept randomly initialized (head/new): {len(skipped)} -> {skipped}")
        if mismatched:
            print(f"[adapt_pretrained_checkpoint] shape mismatch, kept random init: {mismatched}")

    return model


def save_checkpoint(model: nn.Module, path: str, extra: dict | None = None):
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return ckpt
