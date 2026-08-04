import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm

from . import utils


class UnpairedSliceDataset(Dataset):
    """Stage 1: retrospective, unpaired. Returns an independent 2D slice from the
    source field (A) and a random 2D slice from the target field (B). There is no
    correspondence between A and B beyond both being T2FLAIR brain slices -- this
    is exactly the "unpaired" setting PatchNCE is designed for.
    """

    def __init__(self, root: Path, contrast: str, source_field: str, target_field: str,
                 patch_size: int = 128, slice_axis: int = 2, min_fg_frac: float = 0.05,
                 index_workers: int = 8):
        self.patch_size = patch_size
        self.slice_axis = slice_axis
        self.min_fg_frac = min_fg_frac
        self.index_workers = index_workers

        src_volumes = utils.find_volumes(root, "Training_retrospective", contrast, source_field, prefix="R")
        tgt_volumes = utils.find_volumes(root, "Training_retrospective", contrast, target_field, prefix="R")
        if not src_volumes or not tgt_volumes:
            raise RuntimeError(f"no retrospective volumes found for {contrast} {source_field}/{target_field}")

        self.src_slices = self._index_slices(src_volumes, f"{source_field} (source)")
        self.tgt_slices = self._index_slices(tgt_volumes, f"{target_field} (target)")
        if not self.src_slices or not self.tgt_slices:
            raise RuntimeError("no foreground slices found -- check min_fg_frac / slice_axis")

    def _index_slices(self, volumes: dict, label: str):
        # Scanning every volume for foreground slices is a one-time cost, but on slow
        # or contended shared storage it can take *hours* across a few hundred large
        # volumes -- cache the result to disk so it never has to happen twice for the
        # same file set/params, and parallelize the scan itself across threads (using
        # the uncached raw reader, since lru_cache's lock would otherwise serialize
        # concurrent loads of different volumes and defeat the point).
        cache_path = utils.index_cache_path(volumes, self.slice_axis, self.min_fg_frac)
        cached = utils.load_index_cache(cache_path)
        if cached is not None:
            return [(Path(p), s) for p, s in cached]

        def scan_one(path):
            vol = utils._read_volume(path)
            return [(path, s) for s in utils.foreground_slice_indices(vol, self.slice_axis, self.min_fg_frac)]

        index = []
        paths = list(volumes.values())
        with ThreadPoolExecutor(max_workers=self.index_workers) as ex:
            for result in tqdm(ex.map(scan_one, paths), total=len(paths), desc=f"indexing {label}", unit="volume"):
                index.extend(result)

        utils.save_index_cache(cache_path, [[str(p), s] for p, s in index])
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
        pairs = [(sid, lq_volumes[sid], gt_volumes[sid]) for sid in subject_ids
                 if sid in lq_volumes and sid in gt_volumes]

        cache_key_paths = [p for _, lq, gt in pairs for p in (lq, gt)]
        cache_path = utils.index_cache_path(cache_key_paths, slice_axis, min_fg_frac)
        cached = utils.load_index_cache(cache_path)
        if cached is not None:
            self.index = [(Path(lq), Path(gt), s) for lq, gt, s in cached]
        else:
            self.index = []
            for sid, lq_path, gt_path in tqdm(pairs, desc=f"indexing {source_field}->{target_field} pairs", unit="subject"):
                lq_vol = utils.load_volume(lq_path)
                gt_vol = utils.load_volume(gt_path)
                if lq_vol.shape != gt_vol.shape:
                    raise RuntimeError(
                        f"subject {sid}: {source_field} shape {lq_vol.shape} != {target_field} shape {gt_vol.shape}; "
                        "expected registered, matching-grid volumes")
                fg = set(utils.foreground_slice_indices(lq_vol, slice_axis, min_fg_frac)) & \
                     set(utils.foreground_slice_indices(gt_vol, slice_axis, min_fg_frac))
                for s in sorted(fg):
                    self.index.append((lq_path, gt_path, s))
            utils.save_index_cache(cache_path, [[str(lq), str(gt), s] for lq, gt, s in self.index])

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
