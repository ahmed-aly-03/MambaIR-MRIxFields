import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSampleMLP(nn.Module):
    """Per-layer 2-layer projection head + L2 normalization, as in CUT's PatchSampleF.
    MambaIRv2 keeps the channel count equal to embed_dim at every tap point (conv_first
    output and every ASSB block output), so a single feature_dim covers all layers and
    the heads can be built eagerly (needed so their params exist before the optimizer
    is constructed).
    """

    def __init__(self, layer_names, feature_dim: int, hidden_dim: int = 256, out_dim: int = 256,
                 num_patches: int = 256, use_mlp: bool = True):
        super().__init__()
        self.num_patches = num_patches
        self.use_mlp = use_mlp
        # nn.ModuleDict keys can't contain "." (raises KeyError), but layer names
        # like "layers.0" do -- since they're module paths from named_modules(). Store
        # under a sanitized key and translate on lookup in forward().
        if use_mlp:
            self.mlps = nn.ModuleDict({
                self._safe_key(name): nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
                )
                for name in layer_names
            })

    @staticmethod
    def _safe_key(name: str) -> str:
        return name.replace(".", "__")

    @staticmethod
    def _to_blc(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4:  # B, C, H, W -> B, L, C
            return feat.flatten(2).permute(0, 2, 1)
        return feat  # already B, L, C

    def forward(self, feats: dict, patch_ids: dict = None):
        out_feats, out_ids = {}, {}
        for name, feat in feats.items():
            feat = self._to_blc(feat)
            L = feat.shape[1]
            if patch_ids is not None and name in patch_ids:
                ids = patch_ids[name]
            else:
                n = min(self.num_patches, L)
                ids = torch.randperm(L, device=feat.device)[:n]
            sampled = feat[:, ids, :]
            if self.use_mlp:
                sampled = self.mlps[self._safe_key(name)](sampled)
            sampled = F.normalize(sampled, dim=-1)
            out_feats[name] = sampled
            out_ids[name] = ids
        return out_feats, out_ids


class PatchNCELoss(nn.Module):
    """InfoNCE over spatial patches from a single image: patches at the same location
    in feat_q / feat_k are the positive pair, all other sampled locations in that same
    image are negatives (no cross-image negatives -- matches CUT's default).
    """

    def __init__(self, nce_t: float = 0.07):
        super().__init__()
        self.nce_t = nce_t
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        # feat_q, feat_k: (B, N, C), L2-normalized
        B, N, _ = feat_q.shape
        l_pos = torch.bmm(feat_q.reshape(B * N, 1, -1), feat_k.reshape(B * N, -1, 1)).view(B * N, 1)
        l_neg = torch.bmm(feat_q, feat_k.transpose(2, 1))  # (B, N, N)
        diag = torch.eye(N, device=feat_q.device, dtype=torch.bool).unsqueeze(0)
        l_neg = l_neg.masked_fill(diag, -10.0).view(B * N, N)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.nce_t
        target = torch.zeros(B * N, dtype=torch.long, device=feat_q.device)
        return self.ce(logits, target)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = g.t() @ g
    return window.unsqueeze(0).unsqueeze(0)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Standard single-channel windowed SSIM, averaged over the batch."""
    device, dtype = img1.device, img1.dtype
    window = _gaussian_window(window_size, 1.5, device, dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


class SupervisedLoss(nn.Module):
    """L1 + (1 - SSIM) for stage 2, paired supervised finetuning."""

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.l1_weight * F.l1_loss(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * (1 - ssim(pred, target))
        return loss
