import random
from pathlib import Path

from torch.utils.data import Dataset

from . import utils


class UnpairedSliceDataset(Dataset):
    """Stage 1: retrospective, unpaired. Returns an independent 2D slice from the
    source field (A) and a random 2D slice from the target field (B). There is no
    correspondence between A and B beyond both being T2FLAIR brain slices -- this
    is exactly the "unpaired" setting PatchNCE is designed for.
    """

    def __init__(self, root: Path, contrast: str, source_field: str, target_field: str,
                 patch_size: int = 128, slice_axis: int = 2, min_fg_frac: float = 0.05):
        self.patch_size = patch_size
        self.slice_axis = slice_axis
        self.min_fg_frac = min_fg_frac

        src_volumes = utils.find_volumes(root, "Training_retrospective", contrast, source_field, prefix="R")
        tgt_volumes = utils.find_volumes(root, "Training_retrospective", contrast, target_field, prefix="R")
        if not src_volumes or not tgt_volumes:
            raise RuntimeError(f"no retrospective volumes found for {contrast} {source_field}/{target_field}")

        self.src_slices = self._index_slices(src_volumes)
        self.tgt_slices = self._index_slices(tgt_volumes)
        if not self.src_slices or not self.tgt_slices:
            raise RuntimeError("no foreground slices found -- check min_fg_frac / slice_axis")

    def _index_slices(self, volumes: dict):
        index = []
        for _id, path in volumes.items():
            vol = utils.load_volume(path)
            for s in utils.foreground_slice_indices(vol, self.slice_axis, self.min_fg_frac):
                index.append((path, s))
        return index

    def __len__(self):
        return len(self.src_slices)

    def _load_patch(self, index_list, idx):
        path, s = index_list[idx]
        vol = utils.load_volume(path)
        sl = utils.take_slice(vol, self.slice_axis, s)
        sl = utils.random_crop_single(sl, self.patch_size)
        sl = utils.augment_single(sl)
        return utils.to_chw_tensor(sl)

    def __getitem__(self, idx):
        a = self._load_patch(self.src_slices, idx)
        b = self._load_patch(self.tgt_slices, random.randrange(len(self.tgt_slices)))
        return {"A": a, "B": b}


class PairedSliceDataset(Dataset):
    """Stage 2: prospective, paired by volunteer ID. LQ (source field) and GT (target
    field) volumes for the same ID are spatially registered, so the same crop location
    is valid for both.
    """

    def __init__(self, root: Path, contrast: str, source_field: str, target_field: str,
                 subject_ids, patch_size: int = 128, slice_axis: int = 2,
                 min_fg_frac: float = 0.05, train: bool = True):
        self.patch_size = patch_size
        self.slice_axis = slice_axis
        self.train = train

        lq_volumes = utils.find_volumes(root, "Training_prospective", contrast, source_field, prefix="P")
        gt_volumes = utils.find_volumes(root, "Training_prospective", contrast, target_field, prefix="P")

        self.index = []
        for sid in subject_ids:
            if sid not in lq_volumes or sid not in gt_volumes:
                continue
            lq_vol = utils.load_volume(lq_volumes[sid])
            gt_vol = utils.load_volume(gt_volumes[sid])
            if lq_vol.shape != gt_vol.shape:
                raise RuntimeError(
                    f"subject {sid}: {source_field} shape {lq_vol.shape} != {target_field} shape {gt_vol.shape}; "
                    "expected registered, matching-grid volumes")
            fg = set(utils.foreground_slice_indices(lq_vol, slice_axis, min_fg_frac)) & \
                 set(utils.foreground_slice_indices(gt_vol, slice_axis, min_fg_frac))
            for s in sorted(fg):
                self.index.append((lq_volumes[sid], gt_volumes[sid], s))

        if not self.index:
            raise RuntimeError(f"no paired foreground slices found for subjects {subject_ids}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        lq_path, gt_path, s = self.index[idx]
        lq_vol = utils.load_volume(lq_path)
        gt_vol = utils.load_volume(gt_path)
        lq_sl = utils.take_slice(lq_vol, self.slice_axis, s)
        gt_sl = utils.take_slice(gt_vol, self.slice_axis, s)

        if self.train:
            lq_sl, gt_sl = utils.random_crop_pair(lq_sl, gt_sl, self.patch_size)
            lq_sl, gt_sl = utils.sync_augment_pair(lq_sl, gt_sl)
        else:
            lq_sl, gt_sl = utils.center_crop_pair(lq_sl, gt_sl, self.patch_size)

        return {"lq": utils.to_chw_tensor(lq_sl), "gt": utils.to_chw_tensor(gt_sl)}
