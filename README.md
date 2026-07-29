# MambaIRv2 finetuning for MRIxFields 2026 -- Task 2, T2FLAIR, 0.1T -> 3T

Two-stage finetuning of [MambaIRv2](https://github.com/csguoh/MambaIR) to turn 0.1T
T2FLAIR brain MRI into 3T-quality T2FLAIR, for Task 2 of the
[MRIxFields 2026](https://mrixfields.chihucloud.com/2026/tasks.html) challenge.

- **Stage 1** finetunes on the *unpaired* retrospective pool (different subjects per
  field strength) using a PatchNCE contrastive loss only -- no discriminator, no
  ground-truth 0.1T<->3T correspondence needed.
- **Stage 2** finetunes the stage-1 model on the small *paired* prospective pool
  (same volunteer scanned at 0.1T and 3T) with a standard supervised L1 + SSIM loss.

This is scoped to T2FLAIR, 0.1T->3T only. The code is organized so other
contrasts/target fields are just different `--contrast/--source-field/--target-field`
flags, but only this combination has been set up/reasoned about here.

## What was changed vs. stock MambaIRv2

`MambaIR/` is the upstream repo, vendored unmodified. All MRI-specific logic lives in
`mri/` and `train.py`, on top of it:

1. **No SR/pixel-shuffle head.** `mri/model.py:build_mambairv2()` instantiates
   `MambaIRv2` with `upsampler=''` and `upscale=1`. MambaIRv2 already has a branch for
   this (see the `else` clause in `MambaIRv2.forward()`, `MambaIR/basicsr/archs/mambairv2_arch.py`):
   it skips `conv_before_upsample`/`upsample` entirely and reconstructs the image with
   a single `Conv2d(embed_dim, num_out_ch, kernel_size=3)` plus a global residual. With
   `in_chans=1` that's exactly "conv kernel size 3, output channel 1" replacing the SR
   head.
2. **1-channel input/output.** `mri/model.py:adapt_pretrained_checkpoint()` loads an
   upstream (3-channel RGB) MambaIRv2 checkpoint and:
   - averages `conv_first.weight` over the input-channel dimension
     (`(embed_dim,3,3,3) -> (embed_dim,1,3,3)`) to initialize the 1-channel first conv,
     keeping `conv_first.bias` as-is;
   - never loads `conv_last` (or any pixel-shuffle head weights) -- these stay at
     their random initialization, since the output layer's channel semantics changed
     completely (RGB SR output vs. single-channel MRI output).
   - everything else (patch embed/unembed, the ASSB Mamba blocks, `conv_after_body`,
     norm) is copied directly, since embed_dim/depths/heads are unchanged.

## Data layout expected

```
ChallengeData/
  Training_retrospective/T2FLAIR/{0.1T,1.5T,3T,5T,7T}/R_T2FLAIR_<field>_<id>.nii.gz
  Training_prospective/T2FLAIR/{0.1T,1.5T,3T,5T,7T}/P_T2FLAIR_<field>_<id>.nii.gz
```

matching the challenge's [data page](https://mrixfields.chihucloud.com/2026/data.html):
`R_` = retrospective/unpaired (disjoint subject-ID ranges per field), `P_` =
prospective/paired (same ID = same volunteer across fields). Point `--data-root` at
the directory containing `Training_retrospective/` and `Training_prospective/`.

## 1. Environment setup (remote Linux GPU server)

MambaIRv2's selective-scan op needs `mamba_ssm` + `causal_conv1d`, which need matching
CUDA/PyTorch versions and only build on Linux+CUDA -- this is why it must run on your
GPU server, not locally.

```bash
# on the remote server
cd MambaIR-MRIxFields/MambaIR
conda env create -f environment.yaml   # creates a conda env named `mambair`
conda activate mambair
```

If `environment.yaml` doesn't resolve cleanly for your CUDA version, install by hand
instead (see `MambaIR/README.md` "Installation" for version-matched alternatives):

```bash
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

Then add this project's extra dependencies on top of that env:

```bash
cd MambaIR-MRIxFields
pip install -r requirements.txt
```

Sanity check the env can actually import the arch:

```bash
python -c "from mri.model import build_mambairv2; build_mambairv2().cuda(); print('ok')"
```

## 2. Getting a pretrained checkpoint (recommended, not required)

Training from scratch on this little data will not converge well. Download any
official MambaIRv2 checkpoint from the
[MambaIR model zoo](https://github.com/csguoh/MambaIR#model_summary) -- e.g. the
classic SR x2 model (`MambaIR_SR2`) -- and pass it as `--pretrained`. The config
in `configs/t2flair_0p1T_to_3T.yaml` uses the same `embed_dim=174,
depths=[6]*6, num_heads=[6]*6` as the official base SR/denoising configs, so their
weights (minus the head, see above) load directly with `--pretrained`.

```bash
# from your local machine, if you download it there first:
scp MambaIRv2_SRx2.pth <user>@<gpu-server>:MambaIR-MRIxFields/pretrained/
```

## 3. Running

Both stages, end to end:

```bash
cd MambaIR-MRIxFields
python train.py \
  --config configs/t2flair_0p1T_to_3T.yaml \
  --data-root /path/to/ChallengeData \
  --out-dir runs/t2flair_0p1T_3T \
  --pretrained pretrained/MambaIRv2_SRx2.pth \
  --device cuda
```

Any flag in the config can be overridden on the command line, e.g. to shorten a
smoke-test run:

```bash
python train.py --config configs/t2flair_0p1T_to_3T.yaml \
  --data-root /path/to/ChallengeData --out-dir runs/smoke_test \
  --stage1-iters 200 --stage2-iters 100 --log-every 10
```

Run just one stage (e.g. stage 2 only, resuming from a stage-1 checkpoint):

```bash
python train.py --config configs/t2flair_0p1T_to_3T.yaml \
  --data-root /path/to/ChallengeData --out-dir runs/t2flair_0p1T_3T \
  --stage 2 --init-checkpoint runs/t2flair_0p1T_3T/stage1/stage1_final.pth
```

By default, stage 2 auto-discovers every paired 0.1T/3T subject ID under
`Training_prospective/T2FLAIR/` and trains on all of them with no held-out
validation. To hold one subject out for a sanity-check validation curve
(recommended, given only ~3 paired training volunteers):

```bash
python train.py ... --stage2-subject-ids 0006 0007 0009 --stage2-val-ids 0009
```

Checkpoints and a `log.txt` land in `<out-dir>/stage1/` and `<out-dir>/stage2/`.
`stage2/stage2_best.pth` is the checkpoint with the best validation SSIM (only
written if `--stage2-val-ids` is set).

## 4. Inference / building a Task 2 submission

`infer.py` runs a trained checkpoint over `Validating_prospective` and packages the
output to match the official challenge repo's submission spec
([github.com/mrixfields/MRIxFields2026](https://github.com/mrixfields/MRIxFields2026),
`Submission/README.md`): NIfTI float32, axial slab `(X, Y, 30)` (z in `[150, 180)` of
the full `364x436x364` grid), filenames `P_T2FLAIR_3T_<ID>.nii.gz`, under
`<submission-dir>/task2/T2FLAIR/0.1T_to_3T/pred/`.

```bash
python infer.py \
  --config configs/t2flair_0p1T_to_3T.yaml \
  --checkpoint runs/t2flair_0p1T_3T/stage2/stage2_best.pth \
  --data-root /path/to/ChallengeData \
  --submission-dir submission/ \
  --save-full-volumes debug_full_volumes/ \
  --zip
```

- `--data-root` defaults `--input-dir` to `<data-root>/Validating_prospective/T2FLAIR/0.1T/`;
  pass `--input-dir` directly to run over something else (e.g. a held-out
  `Training_prospective` subject for a sanity check before you have real
  `Validating_prospective` data).
- `--save-full-volumes DIR` also dumps the uncropped `(364, 436, 364)`-ish prediction
  for visual QC -- open it in a NIfTI viewer before trusting the cropped submission slab.
- `--zip` writes `<submission-dir's parent>/task2.zip` with the `task2/` directory at
  its root, as the scorer requires. Sanity check with `unzip -l task2.zip | head -3`.
- **No `seg/` is produced.** Task 2 segmentation needs the separate SynthSeg-based
  tool from the official repo (`Baseline/scripts/segment_predictions.py` or
  `Evaluation/segment.py`) run on top of these predictions. Per the official spec, a
  submission with zero seg files still scores SSIM/nRMSE/LPIPS normally -- only
  Dice/Volume come back `null`.
- Model-architecture flags (`--embed-dim` etc.) must match what the checkpoint was
  trained with; passing the same `--config` you used for `train.py` handles this
  automatically (unrelated `stage1_*`/`stage2_*` keys in the yaml are just ignored).

## Notes / things you may want to tune

- **Stage 1 cost**: each step runs the model forward twice (once on the input patch,
  once on its own output) to compute PatchNCE features at matching spatial locations.
  Set `--idt-weight 0` (the default) to skip the extra identity-domain regularization
  term and keep this at 2 forward passes/step; `--idt-weight > 0` doubles it again.
- **Stage 2 is easy to overfit**: only ~3 paired training volunteers. Keep
  `--stage2-lr` low (default `1e-5`), watch the validation curve if you set
  `--stage2-val-ids`, and don't be afraid to stop early.
- `--slice-axis` (default 2) picks which NIfTI axis 2D training slices are taken
  from; check that this matches the acquisition plane for your downloaded volumes.
- Both `mri/utils.py:load_volume` (training) and `infer.py` (inference) canonicalize
  orientation via `nibabel.as_closest_canonical`, matching the official baseline's
  `mrixfields.data.utils.load_nifti` convention -- this dataset should already be on
  a consistent grid, but this keeps training/inference orientation-consistent even if
  it isn't for some file.
