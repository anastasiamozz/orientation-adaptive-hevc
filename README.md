# Orientation-Adaptive Perceptual HEVC Coding for UAV Video

**Author:** Dr. Anastasia Mozhaeva, Eastern Institute of Technology
**Contact:** anast.mozhaeva@gmail.com

## Overview

This repository accompanies a conference submission on an orientation-
adaptive, perceptually-weighted H.265/HEVC coding pipeline for UAV
(drone) video. The method:

1. Chooses a coding orientation (rotate 90 deg or keep as-is) per clip
   based on the dominant direction of the motion field, using an
   **exact, lossless pixel-coordinate rotation** (not an interpolating
   warp) so the choice carries no resampling penalty of its own.
2. Builds 8-16 adaptive geometry+motion+perceptual zones along the
   coding axis.
3. Applies a region-of-interest QP pattern combining (a) a relative,
   zero-mean redistribution term that decides *where* quality is spent,
   and (b) a genuine, non-zero-mean bias term (`BIAS_QP`) -- distributed
   inversely by a combined motion+eccentricity importance score -- that
   is what actually reduces net bitrate.

Full derivation and formulas: [`docs/METHOD.md`](docs/METHOD.md).

**Note on scope:** this repository is provided for peer-review purposes
alongside the conference submission. See [`LICENSE`](LICENSE) --
**all rights are reserved by the author**; no reuse, modification, or
redistribution rights are granted beyond the review process itself.

## Repository layout

```
src/
  orientation_adaptive_h265.py       Proposed method (current, tuned)
  orientation_adaptive_h265_original.py   Pre-tuning version (relative pattern only, no BIAS_QP)
  orientation_only_h265.py           Ablation: orientation adaptation only, no ROI/zones
  horizontal_baseline.py             Plain H.265/HEVC baseline (no rotation, no ROI)
  vertical_crf_psnr_experiment.py    Simple rotate+CRF+PSNR experiment (no adaptive logic)
results/
  01_ablation_orientation_only_vs_full_original.csv   Pre-tuning: isolates the ORIGINAL
                                                       (zero-mean) spatial component's effect
  02_baseline_vs_proposed_tuned.csv                   H.265/HEVC baseline vs. proposed
                                                       (tuned, BIAS_QP) method
  03_ablation_only_vs_proposed_tuned.csv              Orientation-only vs. proposed (tuned):
                                                       isolates the tuned spatial component
  04_foveated_JOD_baseline_vs_proposed.csv            FovVideoVDP (--foveated) quality scores
docs/
  METHOD.md                          Formulas and parameter table
```

## Requirements

- `ffmpeg` / `ffprobe` with `libx265` and the `addroi` filter compiled in
- Python 3.10+, `pip install av numpy`
- For quality evaluation: `pip install pyfvvdp` + PyTorch (GPU strongly
  recommended); `fvvdp` CLI run with `--ffmpeg-cc --foveated`

## Usage

Each script is configured by editing the constants at its top
(`INPUT_VIDEO`, `OUTPUT_ROOT`/`OUTPUT_DIR`, `CRF`), then run directly:

```bash
python src/orientation_adaptive_h265.py
```

For a fair baseline-vs-proposed comparison, keep `CRF` identical between
`horizontal_baseline.py` and `orientation_adaptive_h265.py` runs on the
same input file.

## Data

Experiments were run on 9 clips reassembled losslessly (H.264 CRF 0,
25 fps) from frame sequences in the [VisDrone2019-VID](https://github.com/VisDrone/VisDrone-Dataset)
training set (not redistributed here -- see the dataset's own license).
The reassembled clips and/or the intermediate experiment outputs can be
made available on request for review purposes -- contact the author.

## Results summary

**Baseline (H.265/HEVC, CRF 25) vs. Proposed method (tuned), 9 clips:**

Average bitrate reduction across all 9 clips: **27.6%** (range 20.9-36.3%),
at an average foveated-JOD cost of **0.025** (range 0.013-0.042, on a
0-10 scale) -- see
[`results/02_baseline_vs_proposed_tuned.csv`](results/02_baseline_vs_proposed_tuned.csv)
and [`results/04_foveated_JOD_baseline_vs_proposed.csv`](results/04_foveated_JOD_baseline_vs_proposed.csv)
for the full per-clip breakdown.

**Ablation -- isolating the spatial component from orientation adaptation:**
since `orientation_only_h265.py` and `orientation_adaptive_h265.py` make
the identical rotate/keep decision, the difference between them isolates
the spatial (ROI) component alone. Orientation adaptation by itself
contributes only ~0-6% bitrate change (and not consistently); the tuned
spatial component contributes the dominant ~20-36% share -- see
[`results/03_ablation_only_vs_proposed_tuned.csv`](results/03_ablation_only_vs_proposed_tuned.csv).

**Measurement note:** PSNR is a pure MSE-based metric and cannot reflect
the benefit of perceptually-targeted bit allocation (it is blind to
*where* distortion is spent). Quality is therefore reported using
FovVideoVDP in `--foveated` mode; non-foveated FovVideoVDP substantially
under-credits the method since it does not model eccentricity-dependent
visual sensitivity at all. See `docs/METHOD.md` for a worked example.

## Citation

If you build on the ideas in this repository, please contact the author
for permission and citation details (see [`LICENSE`](LICENSE)).
