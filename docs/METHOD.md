# Method

## 1. Orientation adaptation

For a landscape input, motion vectors are extracted from a preliminary
H.264 analysis pass. The horizontal and vertical motion **energy**
(mean squared component) are compared:

- if vertical energy > horizontal energy: the dominant motion runs along
  the frame's current SHORT axis -> rotate 90 deg for coding (puts that
  motion along the new LONG axis, which motion estimation handles more
  efficiently);
- otherwise: keep the original orientation.

The rotation is an **exact pixel-coordinate transpose** (`ffmpeg
transpose`), not an interpolating affine warp -- it is a pure coordinate
permutation, verified bit-exact on round-trip (>360 dB), so it introduces
no resampling penalty of its own. After decoding, the inverse transpose
(if any) restores the target display orientation.

## 2. Geometry + motion + perceptual adaptive zones

On the coding-orientation reference, a second motion-vector pass drives
construction of 5-16 adaptive zones along the coding axis:

- a **motion profile**: mean motion magnitude per spatial bin, smoothed;
- a **geometry profile**: `1/cos^2(phi)` where `phi` is the angular
  position from frame centre given the camera's vertical FOV (`PSI_V_DEG`
  / `PSI_H_DEG`) -- models the increasing ground-sample distance towards
  the edges of a wide-FOV shot;
- zone priority = `GEOMETRY_WEIGHT * geometry + MOTION_WEIGHT * motion`
  (normalised), zone boundaries placed at equal-priority-mass intervals,
  with a minimum zone-size constraint.

## 3. Relative (zero-mean) QP pattern

For each zone `i`, with `P_i = M_i / max(M)` (normalised zone motion),
`w_i` the zone's pixel-width fraction, and dispersion
`D = (max(M)-min(M))/max(M)`:

```
dQP_motion_i = -2 * DELTA_QP * D * (P_i - P_bar),      P_bar = sum(w_i * P_i)
V_i          = perceptual (photoreceptor-density) eccentricity weight
dQP_relative_i = dQP_motion_i + BETA * DELTA_QP * (V_bar - V_i)
```

This pattern is **zero pixel-weighted-mean by construction**
(`sum(w_i * dQP_relative_i) = 0`): it decides *where* quality is spent,
not *how much* is spent overall. Because the bits-vs-QP relationship in
CRF-mode HEVC encoding is convex, a zero-mean QP perturbation cannot
reduce net bitrate (Jensen's inequality) -- verified empirically as well
as derived analytically (see `docs/BIAS_QP.md` history / experiment log
in the conference paper).

## 4. Genuine bitrate-reducing term: BIAS_QP

```
Importance_i    = BIAS_MOTION_WEIGHT * P_i + BIAS_PERCEPTUAL_WEIGHT * normalize01(V_i)
InvImportance_i = 1 - Importance_i
dQP_bias_i      = BIAS_QP * InvImportance_i / sum(w_j * InvImportance_j)

dQP_i     = dQP_relative_i + dQP_bias_i
qoffset_i = clip(dQP_i / 51, -1, 1)
```

`dQP_bias` has pixel-weighted mean **exactly** `BIAS_QP` regardless of
targeting, so it is a real, predictable average-QP increase (genuine
bitrate reduction) -- distributed inversely by a combined motion +
eccentricity importance score, so the cut concentrates on zones that are
both low-motion and peripheral (low visual acuity), rather than uniformly
or by motion alone.

`qoffset_i` is applied per zone via ffmpeg's `addroi` filter at HEVC
encoding time (`aq-mode=2` kept active throughout).

## 5. Evaluation notes

- PSNR is a pure MSE-based metric: it is blind to *where* distortion is
  spent, so it cannot reflect the benefit of perceptual targeting -- it
  only shows the "cost" side of the trade.
- Quality was therefore evaluated with **FovVideoVDP in `--foveated`
  mode** (gaze fixed at frame centre). Non-foveated FovVideoVDP does not
  model eccentricity-dependent sensitivity at all and substantially
  under-credits the method (observed on one clip: -0.099 JOD
  non-foveated vs. -0.018 JOD foveated for the *same* file pair).
- All comparisons use the same encoder settings (x265, matched CRF and
  preset) between the baseline and the proposed method; only the ROI
  pattern differs.

## Parameters (current)

| Parameter | Value | Notes |
|---|---|---|
| `CRF` | 25 | matched with baseline |
| `PRESET` | medium | "slower" was tested and reverted: it raised bitrate ~26% at matched CRF while decoding ~50x slower |
| `DELTA_QP` | 4.0 | relative-pattern spread |
| `BETA` | 0.6 | perceptual weight in relative pattern |
| `BIAS_QP` | 2.0 | genuine average QP increase |
| `BIAS_MOTION_WEIGHT` | 0.5 | |
| `BIAS_PERCEPTUAL_WEIGHT` | 0.5 | |
| `MIN_ZONES` / `MAX_ZONES` | 8 / 16 | |
| `MIN_ZONE_SIZE_RATIO` | 0.03 | |
| `PSI_V_DEG` / `PSI_H_DEG` | 44.0 | placeholder -- no published per-clip camera FOV exists for the VisDrone dataset; recalibrate if a better estimate becomes available |
