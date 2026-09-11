"""
Orientation-adaptive H.265 coding: EXACT pixel-coordinate 90-degree
rotation (ffmpeg transpose -- a pure index permutation, mathematically
lossless, NO interpolation) used to place the frame in whichever
orientation the motion field favours for coding efficiency, combined
with our adaptive geometry+motion+perceptual region method.

==========================================================================
WHY THIS REPLACES THE GMC/AFFINE APPROACH
==========================================================================
The GMC/affine warp-dewarp prototype showed a hard ~36-37 dB PSNR ceiling
regardless of bit allocation, because bilinear resampling in the warp AND
the dewarp step each discard high-frequency detail irrecoverably. A
transpose-based 90-degree rotation has NO such cost: it is an exact
permutation of pixel coordinates (verified: transpose then transpose-back
reproduces the input at >360 dB, i.e. bit-for-bit identical up to
container/codec rounding). So choosing WHICH of the two orthogonal
orientations to encode in -- rather than warping to an arbitrary
continuous angle -- recovers coding-efficiency benefits with none of the
resampling penalty.

==========================================================================
PIPELINE
==========================================================================
1. Analyse motion vectors on the ORIGINAL (un-rotated) input via a
   preliminary H.264 analysis pass.
2. Compute the motion field's horizontal-energy vs vertical-energy
   (mean squared horizontal vs vertical motion-vector component) --
   this is the "dominant motion orientation" signal.
3. Decide the CODING orientation:
     - if vertical-energy > horizontal-energy: the dominant motion runs
       along what is currently the frame's SHORT (height) axis in this
       landscape source; rotating 90 degrees puts that motion along the
       new LONG axis -> rotate for coding.
     - otherwise: motion already runs along the current (wide) axis ->
       keep the original orientation for coding, no rotation.
   This decision, and the underlying statistics, are printed and saved
   so you can verify it against the actual footage rather than trust it
   blindly.
4. Apply the EXACT (lossless) rotation via ffmpeg's transpose filter if
   required -> save as the coding reference.
5. Re-run the H.264 motion analysis on the coding-orientation reference
   (simpler and more robust than manually re-mapping vector coordinates)
   to drive the adaptive-zone construction in the correct axis:
     - coding orientation = original (horizontal): zones run top-to-
       bottom (y-axis), using the native vertical FOV (PSI_V_DEG);
     - coding orientation = rotated (vertical): zones run left-to-right
       (x-axis), using PSI_H_DEG (the camera's ORIGINAL vertical FOV,
       now running along the rotated frame's width axis after rotation
       -- same physical reasoning used throughout this project).
6. Geometry + residual motion -> 5-10 adaptive zones -> perceptual
   weighting -> mean-centred relative dQP pattern -> ROI qoffsets
   (UNCHANGED formulas from the rest of this project).
7. H.265/x265 encode of the coding-orientation reference with the ROI
   pattern (aq-mode=2 kept).
8. Decode, and if a rotation was applied for coding, rotate back
   (EXACT, lossless transpose in the opposite direction) to reach the
   TARGET DISPLAY ORIENTATION (vertical/portrait -- the orientation
   used throughout this project's "vertical" experiments). If the
   coding orientation was already the original (horizontal), ONE
   transpose is still needed here to reach the vertical target; if the
   coding orientation was already rotated, NO further rotation is
   needed. Either way, at most one exact transpose total is applied
   after decoding -- never more, and never an affine warp.
9. PSNR is measured against a vertical reference produced by the SAME
   exact transpose method from the original input (matching every other
   "vertical" experiment in this project), so numbers are directly
   comparable.

Research contribution kept unchanged: motion-vector concentration
analysis, detection of strongest/non-uniform motion regions, 5-10
adaptive geometry+motion zones, human-visual perceptual weighting. The
NEW mechanism is the orientation decision itself, which removes the
large coding-efficiency penalty observed when the frame is forced into
an unfavourable orientation -- and does so with an operation that is
exactly lossless, unlike the earlier GMC affine warp.

Note: as in the direct-application (non-search) version of the adaptive
method, the zero-mean relative pattern is applied directly here (no
empirical strength/offset search this time, per this task's scope). If
you want the additional empirical (S x G) RD-search layer combined with
orientation adaptation, that can be added on top of this script.

NOT EMPIRICALLY TESTED against your real UAV footage -- tested end-to-end
(motion analysis, orientation decision, exact rotation both directions,
zone construction on both axes, encoding, and PSNR) against a synthetic
video in the environment used to write this.

Requires: ffmpeg.exe, ffprobe.exe in PATH, `pip install av numpy`.
"""

import csv
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import av
except ImportError:
    av = None

# --------------------------------------------------------------------------
# CONFIGURATION -- edit these lines for your setup
# --------------------------------------------------------------------------
INPUT_VIDEO = r"C:\path\to\input\original_horizontal.mp4"
OUTPUT_ROOT = r"C:\path\to\results"

CRF = 25
PRESET = "medium"  # tried "slower": measured WORSE, not better -- higher
                   # PSNR but +26% MORE bitrate on clip3 (40.05dB/5.18Mbps
                   # vs baseline 39.49dB/4.11Mbps), and ~50x slower to
                   # encode (0.02x realtime). Reverted.

DELTA_QP = 4.0      # was 3.0 -- relative (zero-pixel-weighted-mean) spread
                    # between zones, unchanged in kind from the original
                    # design: this shapes WHERE quality is protected, it
                    # does not by itself reduce bitrate (a zero-mean QP
                    # perturbation through a convex bits-vs-QP curve can
                    # only ever raise the pixel-weighted-mean bitrate, by
                    # Jensen's inequality -- verified empirically: a v2
                    # attempt that only scaled this term to 15.0 made
                    # bitrate WORSE, not better, e.g. clip6 11.41->13.56
                    # Mbps). Real savings need a genuine non-zero-mean QP
                    # bias -- see BIAS_QP below.
BETA = 0.6          # was 0.5 -- slightly more weight on the perceptual
                    # (photoreceptor density) term in the relative pattern
BIAS_QP = 2.0       # NEW. Genuine average QP increase (in the same units as
                    # dqp, i.e. roughly QP steps out of 51) applied across
                    # the frame -- this is what actually removes bits net,
                    # since it shifts the pixel-weighted-mean QP up, not
                    # just its spread. WHERE it lands is controlled by
                    # BIAS_MOTION_WEIGHT/BIAS_PERCEPTUAL_WEIGHT below (see
                    # adaptation()) -- but the pixel-weighted mean of the
                    # applied bias is always exactly BIAS_QP regardless of
                    # targeting, so the net bitrate effect is real and
                    # predictable, not redistribution.
BIAS_MOTION_WEIGHT = 0.5       # NEW. Share of the bias-targeting importance
BIAS_PERCEPTUAL_WEIGHT = 0.5   # NEW. score that comes from motion (P) vs
                                # eccentricity/visual-acuity (V, photoreceptor
                                # density) -- 0.5/0.5 blends both, so the QP
                                # cut concentrates on zones that are BOTH
                                # low-motion AND peripheral, which is what a
                                # perceptual metric (FovVideoVDP) penalises
                                # least. Was motion-only (1.0/0.0 equivalent).
PSI_V_DEG = 44.0   # native camera vertical FOV, used when coding orientation = original
PSI_H_DEG = 44.0   # same physical angle, used when coding orientation = rotated (see docstring)

MV_ANALYSIS_PRESET = "veryfast"
MV_ANALYSIS_CRF = 23

MIN_ZONES = 8       # was 5 -- finer zone granularity so the QP cut can be
MAX_ZONES = 16      # was 10 -- concentrated on the truly lowest-priority
MIN_ZONE_SIZE_RATIO = 0.03   # was 0.05 -- pixels instead of being diluted
                             # across a coarser band

GEOMETRY_WEIGHT = 0.4
MOTION_WEIGHT = 0.6

ANALYSIS_BINS = 60
SMOOTHING_WINDOW = 7
PEAK_MIN_SEPARATION_BINS = 4
PEAK_PROMINENCE_RATIO = 0.15

# --------------------------------------------------------------------------

EXPORT_MVS = 1 << 28


def run(cmd, description, show_live_output=False):
    print(f"--> {description}")
    print("    $ " + " ".join(str(c) for c in cmd))
    if show_live_output:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("\nERROR: command failed.")
        if hasattr(result, "stderr") and result.stderr:
            print("\n".join(str(result.stderr).strip().splitlines()[-40:]))
        sys.exit(1)
    return result


def require_tools():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ERROR: ffmpeg/ffprobe not found in PATH.")
        sys.exit(1)
    if av is None:
        print("ERROR: PyAV not installed. Run: pip install av")
        sys.exit(1)
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    if "addroi" not in filters:
        print("ERROR: FFmpeg build has no addroi filter.")
        sys.exit(1)


def get_dimensions(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
            f"Reading dimensions of {os.path.basename(str(path))}")
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def get_duration_seconds(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            f"Reading duration of {os.path.basename(str(path))}")
    return float(r.stdout.strip())


def get_filesize_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_psnr(distorted_path, reference_path):
    cmd = ["ffmpeg", "-y", "-i", str(distorted_path), "-i", str(reference_path),
           "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    match = re.search(r"average:(\d+\.?\d*)", result.stderr)
    if not match:
        print("\nERROR: could not find PSNR 'average:' value.")
        sys.exit(1)
    return float(match.group(1))


# --------------------------------------------------------------------------
# EXACT pixel-coordinate rotation -- transpose is a permutation, NOT an
# interpolating warp. transpose=1 (90 CW) and transpose=2 (90 CCW) are
# exact inverses of each other (verified: round trip >360 dB, i.e.
# bit-identical up to codec/container rounding, unlike affine warping).
# --------------------------------------------------------------------------

def exact_rotate(input_path, output_path, clockwise):
    code = "1" if clockwise else "2"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(input_path), "-vf", f"transpose={code}",
           "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
           "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run(cmd, f"Exact pixel rotation ({'clockwise' if clockwise else 'counterclockwise'}, "
              f"transpose -- no interpolation)", show_live_output=False)


def exact_rotate_from_h265(input_path, output_path, clockwise, preset):
    """Same exact transpose, applied directly to a decoded H.265 stream
    (ffmpeg decodes internally; transpose itself remains an exact
    permutation regardless of what codec produced the input pixels)."""
    code = "1" if clockwise else "2"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(input_path), "-vf", f"transpose={code}",
           "-c:v", "libx264", "-crf", "0", "-preset", preset,
           "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run(cmd, f"Exact rotate-back after decode ({'clockwise' if clockwise else 'counterclockwise'})",
        show_live_output=False)


def copy_no_rotate(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(input_path), "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
           "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run(cmd, "Re-encoding losslessly with no rotation (orientation kept as-is)")


# --------------------------------------------------------------------------
# Motion analysis (H.264 analysis pass -- unchanged mechanism used
# throughout this project)
# --------------------------------------------------------------------------

def collect_motion_vectors_full(input_path, analysis_preset, analysis_crf):
    """Returns dst_x, dst_y, dx, dy arrays -- both position and both
    displacement components, needed for the orientation decision AND for
    axis-appropriate zone construction."""
    tmp = Path(input_path).with_name(Path(input_path).stem + "_mvtmp.mp4")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(input_path), "-map", "0:v:0", "-an",
           "-c:v", "libx264", "-preset", analysis_preset, "-crf", str(analysis_crf), str(tmp)]
    subprocess.run(cmd, check=True)

    xs, ys, dxs, dys = [], [], [], []
    container = av.open(str(tmp))
    vstream = container.streams.video[0]
    vstream.codec_context.flags2 |= EXPORT_MVS
    frames_with_mvs = 0
    for frame in container.decode(vstream):
        mvs = frame.side_data.get("MOTION_VECTORS")
        if not mvs:
            continue
        frames_with_mvs += 1
        for mv in mvs:
            xs.append(mv.dst_x)
            ys.append(mv.dst_y)
            dxs.append(mv.motion_x / mv.motion_scale)
            dys.append(mv.motion_y / mv.motion_scale)
    container.close()
    tmp.unlink(missing_ok=True)

    if frames_with_mvs == 0:
        print("ERROR: no motion vectors found in the analysis pass.")
        sys.exit(1)
    return (np.array(xs, float), np.array(ys, float),
            np.array(dxs, float), np.array(dys, float))


def decide_orientation(dxs, dys):
    """Horizontal vs vertical motion ENERGY (mean squared component).
    If vertical energy dominates, the motion currently runs along the
    frame's short (height) axis in this landscape source -- rotating
    puts it along the new long axis. Otherwise keep original orientation."""
    horiz_energy = float(np.mean(dxs ** 2))
    vert_energy = float(np.mean(dys ** 2))
    dominant_angle_deg = float(np.degrees(np.arctan2(np.mean(dys), np.mean(dxs))))
    mean_speed = float(np.mean(np.sqrt(dxs ** 2 + dys ** 2)))
    rotate = vert_energy > horiz_energy
    stats = {
        "horizontal_energy": horiz_energy, "vertical_energy": vert_energy,
        "energy_ratio_v_over_h": vert_energy / horiz_energy if horiz_energy > 0 else float("inf"),
        "dominant_angle_deg": dominant_angle_deg, "mean_speed": mean_speed,
        "n_vectors": len(dxs),
    }
    return rotate, stats


# --------------------------------------------------------------------------
# Adaptive zone construction -- generic over axis ("x" for vertical/
# rotated coding orientation, "y" for horizontal/original), same formulas
# used throughout this project either way.
# --------------------------------------------------------------------------

def build_motion_profile(positions, mags, frame_size, n_bins, smoothing_window):
    bin_edges = np.linspace(0, frame_size, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    idx = np.clip(np.digitize(positions, bin_edges) - 1, 0, n_bins - 1)
    sums, counts = np.zeros(n_bins), np.zeros(n_bins)
    for i, m in zip(idx, mags):
        sums[i] += m
        counts[i] += 1
    profile = np.zeros(n_bins)
    has_data = counts > 0
    profile[has_data] = sums[has_data] / counts[has_data]
    if has_data.any():
        last = profile[has_data][0]
        for i in range(n_bins):
            if has_data[i]:
                last = profile[i]
            else:
                profile[i] = last
    if smoothing_window > 1:
        pad = smoothing_window // 2
        padded = np.pad(profile, pad, mode="edge")
        kernel = np.ones(smoothing_window) / smoothing_window
        profile = np.convolve(padded, kernel, mode="valid")[:n_bins]
    return bin_edges, bin_centers, profile


def build_geometry_profile(bin_centers, frame_size, psi_deg):
    phi = np.arctan((bin_centers - frame_size / 2.0) / (frame_size / 2.0) * np.tan(np.radians(psi_deg / 2.0)))
    return 1.0 / np.cos(phi) ** 2


def normalize01(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def count_significant_peaks(profile, prominence_ratio, min_separation):
    if len(profile) < 3:
        return 0
    threshold = profile.mean() + prominence_ratio * (profile.max() - profile.min())
    peaks = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= threshold and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_separation:
                peaks.append(i)
    return len(peaks)


def equal_mass_boundaries(bin_edges, priority, n_zones):
    bin_widths = np.diff(bin_edges)
    mass = priority * bin_widths
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    cum_norm = cum / cum[-1]
    targets = np.linspace(0.0, 1.0, n_zones + 1)
    boundaries = np.interp(targets, cum_norm, bin_edges)
    boundaries[0], boundaries[-1] = bin_edges[0], bin_edges[-1]
    return boundaries


def enforce_min_zone_size(boundaries, frame_size, min_ratio, min_zones):
    min_size = min_ratio * frame_size
    b = [float(x) for x in boundaries]
    changed = True
    while changed and (len(b) - 1) > min_zones:
        changed = False
        for i in range(len(b) - 1):
            if (b[i + 1] - b[i]) < min_size:
                if i == 0:
                    del b[1]
                elif i == len(b) - 2:
                    del b[-2]
                else:
                    if (b[i] - b[i - 1]) <= (b[i + 2] - b[i + 1]):
                        del b[i]
                    else:
                        b.pop(i + 1)
                changed = True
                break
    b[0], b[-1] = 0.0, frame_size
    return np.round(np.array(b)).astype(int)


def build_adaptive_zones(positions, mags, frame_size, psi_deg):
    bin_edges, bin_centers, motion_profile = build_motion_profile(
        positions, mags, frame_size, ANALYSIS_BINS, SMOOTHING_WINDOW)
    geometry_profile = build_geometry_profile(bin_centers, frame_size, psi_deg)
    priority = GEOMETRY_WEIGHT * normalize01(geometry_profile) + MOTION_WEIGHT * normalize01(motion_profile)
    n_peaks = count_significant_peaks(priority, PEAK_PROMINENCE_RATIO, PEAK_MIN_SEPARATION_BINS)
    n_zones = int(np.clip(MIN_ZONES + max(0, n_peaks - 1), MIN_ZONES, MAX_ZONES))
    raw_boundaries = equal_mass_boundaries(bin_edges, priority, n_zones)
    boundaries = enforce_min_zone_size(raw_boundaries, frame_size, MIN_ZONE_SIZE_RATIO, MIN_ZONES)
    zones = [{"name": f"Z{i+1}", "p0": int(boundaries[i]), "p1": int(boundaries[i + 1])}
              for i in range(len(boundaries) - 1)]
    return zones, n_peaks


def compute_zone_motion(positions, mags, zones):
    M = np.zeros(len(zones))
    for i, z in enumerate(zones):
        mask = (positions >= z["p0"]) & (positions < z["p1"])
        M[i] = mags[mask].mean() if mask.any() else 0.0
    return M


def zone_pixel_weights(zones, frame_size):
    return np.array([z["p1"] - z["p0"] for z in zones], dtype=float) / frame_size


def photoreceptor_density(alpha_deg):
    a = alpha_deg
    return 1.26e5 * np.exp(-0.71 * a) + 1.5e4 - 512 * a + 11 * a ** 2 - 0.08 * a ** 3


def perceptual_weights(zones, frame_size, psi_deg):
    centers = np.array([(z["p0"] + z["p1"]) / 2.0 for z in zones])
    alpha = np.abs((centers - frame_size / 2.0) / (frame_size / 2.0)) * (psi_deg / 2.0)
    A = photoreceptor_density(alpha)
    return np.log(A) / np.log(photoreceptor_density(0.0)), alpha


def adaptation(M, delta_qp, zones, frame_size, psi_deg, beta, bias_qp=0.0,
               bias_motion_weight=0.5, bias_perceptual_weight=0.5):
    w = zone_pixel_weights(zones, frame_size)
    max_m, min_m = float(np.max(M)), float(np.min(M))
    if max_m > 0:
        P, dispersion = M / max_m, (max_m - min_m) / max_m
    else:
        P, dispersion = np.zeros_like(M), 0.0
    P_bar = float(np.sum(w * P))
    # Relative (zero-pixel-weighted-mean) pattern -- protects/penalises
    # zones RELATIVE to each other. On its own this cannot reduce net
    # bitrate (bits-vs-QP is convex, so a zero-mean QP perturbation raises
    # the weighted-mean bitrate by Jensen's inequality) -- it only decides
    # WHERE quality is spent once a bitrate target is chosen.
    dqp_motion = -2 * delta_qp * dispersion * (P - P_bar)
    V, alpha = perceptual_weights(zones, frame_size, psi_deg)
    V_bar = float(np.sum(w * V))
    dqp_relative = dqp_motion + beta * delta_qp * (V_bar - V)

    # Genuine bitrate-reducing term: a pixel-weighted-mean QP increase of
    # exactly bias_qp. WHERE it lands is now driven by a MOTION+PERCEPTUAL
    # combined importance score, not motion alone: V (photoreceptor-density
    # eccentricity weight) is high at the frame centre/fovea and falls off
    # toward the periphery, so blending it in targets the cut specifically
    # at zones that are BOTH low-motion AND low visual acuity -- exactly
    # where a perceptual metric (FovVideoVDP) discounts distortion most,
    # rather than motion-only targeting which says nothing about WHERE in
    # the visual field a low-motion zone sits. The area-weighted average of
    # what's actually applied is still exactly bias_qp regardless of how
    # it's targeted, so this changes WHICH pixels absorb the cut without
    # changing the net bitrate outcome much.
    V_norm = normalize01(V)
    importance = bias_motion_weight * P + bias_perceptual_weight * V_norm
    inv_importance = 1.0 - importance
    inv_importance_bar = float(np.sum(w * inv_importance))
    if bias_qp != 0.0 and inv_importance_bar > 1e-9:
        dqp_bias = bias_qp * inv_importance / inv_importance_bar
    else:
        dqp_bias = np.full_like(P, bias_qp)

    dqp = dqp_relative + dqp_bias
    return P, dqp, np.clip(dqp / 51.0, -1.0, 1.0), dispersion, V, alpha


def rat(v):
    f = Fraction(float(v)).limit_denominator(10000)
    return f"{f.numerator}/{f.denominator}"


def roi_filter(zones, qoffset, axis):
    """axis='x' -> vertical strips (left-right); axis='y' -> horizontal
    bands (top-bottom)."""
    parts = []
    for i, (z, q) in enumerate(zip(zones, qoffset)):
        if axis == "x":
            geom = f"x={int(z['p0'])}:y=0:w={int(z['p1']-z['p0'])}:h=ih"
        else:
            geom = f"x=0:y={int(z['p0'])}:w=iw:h={int(z['p1']-z['p0'])}"
        parts.append(f"addroi={geom}:qoffset={rat(q)}:clear={1 if i == 0 else 0}")
    return ",".join(parts)


def encode_h265_with_roi(input_path, output_path, crf, preset, vf):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-stats",
           "-i", str(input_path), "-map", "0:v:0", "-an",
           "-vf", vf, "-c:v", "libx265", "-preset", preset, "-crf", str(crf),
           "-x265-params", "aq-mode=2", str(output_path)]
    run(cmd, f"Encoding H.265 with adaptive ROI, CRF {crf}", show_live_output=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    input_path = Path(INPUT_VIDEO)
    print("=" * 70)
    print("Orientation-adaptive H.265 (exact rotation, no interpolation)")
    print("=" * 70)

    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}")
        sys.exit(1)

    require_tools()
    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Step 1-2: motion analysis + orientation decision, on ORIGINAL ---
    print("\n--- Step 1: motion analysis on the ORIGINAL (un-rotated) input ---")
    xs0, ys0, dxs0, dys0 = collect_motion_vectors_full(input_path, MV_ANALYSIS_PRESET, MV_ANALYSIS_CRF)
    rotate_for_coding, orient_stats = decide_orientation(dxs0, dys0)
    print(f"Horizontal motion energy = {orient_stats['horizontal_energy']:.4f}")
    print(f"Vertical motion energy   = {orient_stats['vertical_energy']:.4f}")
    print(f"Energy ratio (V/H)       = {orient_stats['energy_ratio_v_over_h']:.3f}")
    print(f"Dominant motion angle    = {orient_stats['dominant_angle_deg']:+.2f} deg "
          f"(0=horizontal, +-90=vertical)")
    print(f"Mean motion speed        = {orient_stats['mean_speed']:.4f} px/frame")
    print(f"N motion vectors         = {orient_stats['n_vectors']}")
    print(f"\n==> DECISION: {'ROTATE for coding' if rotate_for_coding else 'KEEP original orientation for coding'}")

    # ---- Step 3-4: EXACT rotation (if needed) for the CODING reference ---
    coding_ref_path = out / "coding_reference.mp4"
    if rotate_for_coding:
        exact_rotate(input_path, coding_ref_path, clockwise=True)
        coding_axis = "x"          # vertical strips, left-to-right
        coding_psi = PSI_H_DEG
    else:
        copy_no_rotate(input_path, coding_ref_path)
        coding_axis = "y"          # horizontal bands, top-to-bottom
        coding_psi = PSI_V_DEG
    coding_w, coding_h = get_dimensions(str(coding_ref_path))
    coding_frame_size = coding_w if coding_axis == "x" else coding_h
    print(f"    Coding reference: {coding_w}x{coding_h}, orientation "
          f"{'ROTATED' if rotate_for_coding else 'ORIGINAL'}")

    # ---- Step 5: re-run motion analysis on the CODING-orientation ref ----
    print("\n--- Step 2: motion analysis on the CODING-orientation reference ---")
    xs, ys, dxs, dys = collect_motion_vectors_full(coding_ref_path, MV_ANALYSIS_PRESET, MV_ANALYSIS_CRF)
    positions = xs if coding_axis == "x" else ys
    mags = np.sqrt(dxs ** 2 + dys ** 2)

    # ---- Step 6: adaptive zones + perceptual weighting (unchanged) -------
    zones, n_peaks = build_adaptive_zones(positions, mags, coding_frame_size, coding_psi)
    M = compute_zone_motion(positions, mags, zones)
    P, dqp, qoffset, dispersion, V, alpha = adaptation(
        M, DELTA_QP, zones, coding_frame_size, coding_psi, BETA, BIAS_QP,
        BIAS_MOTION_WEIGHT, BIAS_PERCEPTUAL_WEIGHT)
    print(f"\nAdaptive zones ({coding_axis}-axis): {len(zones)} (peaks: {n_peaks}), "
          f"dispersion D={dispersion:.4f}")
    for i, z in enumerate(zones):
        print(f"  {z['name']}: p0={z['p0']} p1={z['p1']} M={M[i]:.3f} "
              f"P={P[i]:.3f} V={V[i]:.4f} dQP={dqp[i]:+.3f} qoffset={qoffset[i]:+.5f}")
    vf = roi_filter(zones, qoffset, coding_axis)

    # ---- Step 7: H.265 encode --------------------------------------------
    print(f"\n--- Step 3: H.265 encode (CRF {CRF}) ---")
    compressed_path = out / "compressed_h265.mp4"
    encode_h265_with_roi(coding_ref_path, compressed_path, CRF, PRESET, vf)
    bitrate = (get_filesize_mb(compressed_path) * 1024 * 1024 * 8) / get_duration_seconds(compressed_path) / 1_000_000
    size_mb = get_filesize_mb(compressed_path)

    # ---- Step 8: decode + rotate back (EXACT) to the TARGET DISPLAY ------
    # ---- orientation (vertical/portrait, matching this project's other --
    # ---- "vertical" experiments) ------------------------------------------
    print(f"\n--- Step 4: reconstruct to target display orientation (vertical) ---")
    reconstructed_path = out / "reconstructed_display_orientation.mp4"
    if rotate_for_coding:
        # already vertical after coding-time rotation -- no further rotation
        copy_no_rotate(compressed_path, reconstructed_path)
        print("    Coding orientation was already vertical -- no further rotation needed.")
    else:
        # coding orientation was original (horizontal) -- one exact
        # rotation needed to reach the vertical display target
        exact_rotate_from_h265(compressed_path, reconstructed_path, clockwise=True, preset=PRESET)
        print("    Applied one exact rotation to reach the vertical display target.")

    # ---- PSNR reference: SAME exact-transpose method, from the original --
    vertical_reference_path = out / "vertical_reference.mp4"
    exact_rotate(input_path, vertical_reference_path, clockwise=True)

    psnr = measure_psnr(reconstructed_path, vertical_reference_path)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Selected coding orientation = {'ROTATED (vertical)' if rotate_for_coding else 'ORIGINAL (horizontal)'}")
    print(f"PSNR = {psnr:.3f} dB")
    print(f"Video-only bitrate = {bitrate:.4f} Mbps")
    print(f"File size = {size_mb:.3f} MB")
    print(f"\nMotion orientation statistics:")
    for k, v in orient_stats.items():
        print(f"  {k} = {v}")
    print(f"\nAdaptive zone boundaries ({coding_axis}-axis, {len(zones)} zones):")
    for i, z in enumerate(zones):
        print(f"  {z['name']}: [{z['p0']}, {z['p1']}]  M={M[i]:.3f}  qoffset={qoffset[i]:+.5f}")
    print(f"\nCoding reference saved to: {coding_ref_path}")
    print(f"Compressed H.265 stream saved to: {compressed_path}")
    print(f"Reconstructed (display orientation) saved to: {reconstructed_path}")
    print(f"Vertical PSNR reference saved to: {vertical_reference_path}")

    csv_path = out / "orientation_adaptive_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rotated_for_coding", "PSNR_dB", "Bitrate_Mbps", "FileSize_MB",
                          "N_zones", "Horizontal_energy", "Vertical_energy",
                          "Dominant_angle_deg", "Mean_speed", "N_motion_vectors"])
        writer.writerow([rotate_for_coding, round(psnr, 3), round(bitrate, 4), round(size_mb, 3),
                          len(zones), round(orient_stats["horizontal_energy"], 4),
                          round(orient_stats["vertical_energy"], 4),
                          round(orient_stats["dominant_angle_deg"], 2),
                          round(orient_stats["mean_speed"], 4), orient_stats["n_vectors"]])
    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
