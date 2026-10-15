# Orientation-Adaptive Perceptual HEVC Coding for UAV Video

**Author:** Dr. Anastasia Mozhaeva, Eastern Institute of Technology
**Contact:** anast.mozhaeva@gmail.com

## Overview

This repository accompanies a conference submission on an orientation-adaptive, perceptually-weighted H.265/HEVC coding pipeline for UAV
(drone) video. The method:

1. Chooses a coding orientation per clip
   based on the dominant direction of the motion field, using an
   **exact, lossless pixel-coordinate rotation** (not an interpolating
   warp) so the choice carries no resampling penalty of its own.
2. Builds 8-16 adaptive geometry+motion+perceptual zones along the
   coding axis.
3. Applies a region-of-interest QP pattern combining (a) a relative,
   zero-mean redistribution term that decides *where* quality is spent,
   and (b) a genuine, non-zero-mean bias term (`BIAS_QP`), distributed
   inversely by a combined motion+eccentricity importance score, which
   is what actually reduces net bitrate.

Full derivation and formulas: [`docs/METHOD.md`](docs/METHOD.md).

**Note on scope:** this repository is provided for peer-review purposes
alongside the conference submission. See [`LICENSE`](LICENSE) 
**all rights are reserved by the author**; no reuse, modification, or
redistribution rights are granted beyond the review process itself.


## Requirements

- `ffmpeg` / `ffprobe` with `libx265` and the `addroi` filter compiled in
- Python 3.10+, `pip install av numpy`
- For quality evaluation: `pip install pyfvvdp` + PyTorch (GPU strongly
  recommended); `fvvdp` CLI run with `ffmpeg-cc foveated`

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

**Controlled 4K UAV dataset** (introduced in the paper): 24 sequences recorded with a DJI Air 3S
camera (3840x2160, 60 fps, nadir orientation) at low, medium and high flight speeds over grass,
carpet and linoleum surfaces. Used to compare horizontal and vertical H.265/HEVC coding at CRF 25
and 51. Available upon request for research and educational purposes; contact the author
(anast.mozhaeva@gmail.com).

**Independent validation datasets:** 32 sequences from
[VisDrone-VID2019](https://github.com/VisDrone/VisDrone-Dataset) and 24 sequences from UAVDT
(56 sequences in total; nadir, oblique and lateral views). For the additional validation sequences
the first 150 frames were evaluated. See each dataset's own license.

## Evaluation protocol

- Controlled orientation experiments: PSNR and video-only bitrate, identical content and encoder
  settings for horizontal and vertical coding.
- Independent validation: conventional full-frame H.265/HEVC vs. the proposed framework at a fixed
  CRF of 25; rate-distortion additionally assessed at CRF 22, 25, 28 and 31 with BD-rate at matched
  perceptual quality (FovVideoVDP, PCHIP interpolation).
- The decoded output of the proposed method is transformed back to the original input orientation
  before quality assessment; both methods are evaluated against the same original reference video.
- All configurations use identical x265 settings with native adaptive quantisation (`aq-mode=2`).

## Results

**Table I** - orientation-dependent H.265/HEVC coding on the controlled dataset, all 18
experiments at CRF 25 and 51:
[`results/table1_controlled_orientation.csv`](results/table1_controlled_orientation.csv).
Vertical coding required higher bitrate than horizontal coding in every configuration; at CRF 25
the increase ranged from 9% to 40% with PSNR differences not exceeding 0.35 dB.

**Table II** - fixed-CRF evaluation on the independent datasets (CRF 25, relative to conventional
H.265/HEVC): [`results/table2_fixed_crf_validation.csv`](results/table2_fixed_crf_validation.csv).

| Dataset | Sequences | Bitrate reduction | JOD loss |
|---|---|---|---|
| VisDrone-VID2019 | 32 | 27.1% | 0.026 |
| UAVDT | 24 | 26.9% | 0.049 |
| Overall | 56 | 27.0% | 0.036 |

**Table III** - target scenario (forward UAV flight with an oblique camera): bitrate reduction at
CRF 25 and perceptual BD-rate for ten sequences:
[`results/table3_target_scenario_examples.csv`](results/table3_target_scenario_examples.csv).
Bitrate reductions range from 22.7% to 42.5%.

## Citation

If you use this code or these results, please cite:
> A. Mozhaeva, V. Vinetsky, V. Fedorov, K. A. Chisholm, and P. Delmas, "Motion- and Perception-Adaptive Video Compression for Real-Time UAV Piloting," International Conference on Image and Vision Computing New Zealand (IVCNZ), 2026.
