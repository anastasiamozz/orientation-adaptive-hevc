"""
ABLATION VARIANT: orientation-adaptation ONLY -- NO spatial/perceptual
component (no adaptive zones, no ROI qoffsets, no eccentricity/
photoreceptor weighting). Everything else is IDENTICAL to
orientation_adaptive_h265.py:

  - same input-orientation detection;
  - same motion-vector orientation/energy analysis and decision rule;
  - same EXACT (lossless transpose, no interpolation) rotation for
    coding, with the EXACT inverse applied after decoding so output
    orientation always matches input orientation;
  - same H.265/x265 settings (preset, CRF, aq-mode=2).

The ONLY difference: the coding reference is encoded with PLAIN
`-c:v libx265 ... -x265-params aq-mode=2` -- no addroi filter, no
zone-specific qoffsets. x265's own native adaptive quantisation is still
active (aq-mode=2 kept, exactly as in the full method), but nothing from
this project's geometry+motion+perceptual zone mechanism is applied.

PURPOSE: run this AND orientation_adaptive_h265.py on the SAME input,
same CRF, and compare their PSNR/bitrate. Since the orientation
mechanism is identical in both, any difference between the two results
isolates the contribution of the spatial/perceptual (adaptive-zone)
component specifically -- the same logic as the earlier
vertical_adaptive_motion_only.py ablation, now applied on top of
orientation-adaptation instead of a fixed vertical baseline.

NOT EMPIRICALLY TESTED against your real UAV footage -- tested end-to-end
(orientation detection for both landscape and portrait inputs, exact
rotation in both directions, plain encoding, and PSNR) against synthetic
videos in the environment used to write this.

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
PRESET = "medium"

MV_ANALYSIS_PRESET = "veryfast"
MV_ANALYSIS_CRF = 23
# (No zone/perceptual parameters here -- this is the no-spatial-component
# ablation. See orientation_adaptive_h265.py for the full method's
# DELTA_QP, BETA, PSI_V_DEG/PSI_H_DEG, zone-count and geometry/motion
# weighting parameters.)

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


def decide_orientation(dxs, dys, input_is_landscape):
    """Horizontal vs vertical motion ENERGY (mean squared component).
    CRITICAL: the decision depends on the INPUT's CURRENT orientation --
    "rotate helps" means "the dominant motion currently runs along the
    frame's SHORT axis; rotating would put it along the new LONG axis."
    This is the opposite test depending on whether the input is currently
    landscape (long axis = width) or portrait (long axis = height)."""
    horiz_energy = float(np.mean(dxs ** 2))
    vert_energy = float(np.mean(dys ** 2))
    dominant_angle_deg = float(np.degrees(np.arctan2(np.mean(dys), np.mean(dxs))))
    mean_speed = float(np.mean(np.sqrt(dxs ** 2 + dys ** 2)))
    if input_is_landscape:
        # long axis = width (x). Rotating helps if motion favours the
        # currently-SHORT axis (y, height).
        rotate = vert_energy > horiz_energy
    else:
        # long axis = height (y). Rotating helps if motion favours the
        # currently-SHORT axis (x, width).
        rotate = horiz_energy > vert_energy
    stats = {
        "horizontal_energy": horiz_energy, "vertical_energy": vert_energy,
        "energy_ratio_v_over_h": vert_energy / horiz_energy if horiz_energy > 0 else float("inf"),
        "dominant_angle_deg": dominant_angle_deg, "mean_speed": mean_speed,
        "n_vectors": len(dxs), "input_is_landscape": input_is_landscape,
    }
    return rotate, stats


# --------------------------------------------------------------------------
# Adaptive zone construction -- generic over axis ("x" for vertical/
# rotated coding orientation, "y" for horizontal/original), same formulas
# used throughout this project either way.
# --------------------------------------------------------------------------

def encode_h265_plain(input_path, output_path, crf, preset):
    """Plain H.265 encode -- NO addroi, NO zone-specific offsets. Only
    x265's own native adaptive quantisation (aq-mode=2) is active, same
    as in the full method, but nothing from the zone/perceptual layer."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-stats",
           "-i", str(input_path), "-map", "0:v:0", "-an",
           "-c:v", "libx265", "-preset", preset, "-crf", str(crf),
           "-x265-params", "aq-mode=2", str(output_path)]
    run(cmd, f"Encoding PLAIN H.265 (no zones, aq-mode=2 only), CRF {crf}",
        show_live_output=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    input_path = Path(INPUT_VIDEO)
    print("=" * 70)
    print("ABLATION: orientation-adaptation ONLY (no spatial/perceptual component)")
    print("Output orientation ALWAYS matches input orientation.")
    print("=" * 70)

    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}")
        sys.exit(1)

    require_tools()
    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Detect the INPUT's own current orientation first -- everything
    # ---- else is relative to this, not to a fixed "vertical" target. -----
    input_w, input_h = get_dimensions(str(input_path))
    input_is_landscape = input_w >= input_h
    print(f"\nInput video: {input_w}x{input_h} "
          f"({'landscape/horizontal' if input_is_landscape else 'portrait/vertical'})")

    # ---- Step 1-2: motion analysis + orientation decision, on ORIGINAL ---
    print("\n--- Step 1: motion analysis on the ORIGINAL (as-is) input ---")
    xs0, ys0, dxs0, dys0 = collect_motion_vectors_full(input_path, MV_ANALYSIS_PRESET, MV_ANALYSIS_CRF)
    rotate_for_coding, orient_stats = decide_orientation(dxs0, dys0, input_is_landscape)
    print(f"Horizontal motion energy = {orient_stats['horizontal_energy']:.4f}")
    print(f"Vertical motion energy   = {orient_stats['vertical_energy']:.4f}")
    print(f"Energy ratio (V/H)       = {orient_stats['energy_ratio_v_over_h']:.3f}")
    print(f"Dominant motion angle    = {orient_stats['dominant_angle_deg']:+.2f} deg "
          f"(0=horizontal, +-90=vertical)")
    print(f"Mean motion speed        = {orient_stats['mean_speed']:.4f} px/frame")
    print(f"N motion vectors         = {orient_stats['n_vectors']}")
    print(f"\n==> DECISION: {'ROTATE for coding, then rotate BACK to original orientation after decode' if rotate_for_coding else 'KEEP original orientation throughout -- no rotation at all'}")

    # ---- Step 3-4: EXACT rotation (if needed) for the CODING reference ---
    # ---- Direction: transpose=1 (clockwise) forward. The inverse used ----
    # ---- later is transpose=2 (counterclockwise) -- NOT the same code ---
    # ---- again, which would land 180 degrees off instead of undoing it. -
    coding_ref_path = out / "coding_reference.mp4"
    if rotate_for_coding:
        exact_rotate(input_path, coding_ref_path, clockwise=True)
    else:
        copy_no_rotate(input_path, coding_ref_path)
    coding_w, coding_h = get_dimensions(str(coding_ref_path))
    print(f"    Coding reference: {coding_w}x{coding_h} "
          f"({'ROTATED' if rotate_for_coding else 'ORIGINAL, unrotated'})")

    # ---- Step 5 (ablation: SKIPPED): no re-analysis, no zone construction,
    # ---- no perceptual weighting, no addroi -- this is the whole point of
    # ---- this ablation variant. ------------------------------------------
    print("\n--- Spatial/perceptual component: SKIPPED (ablation) ---")
    print("    No adaptive zones, no ROI qoffsets, no perceptual weighting.")
    print("    Only x265's own native aq-mode=2 is active.")

    # ---- H.265 encode, PLAIN (no addroi) ----------------------------------
    print(f"\n--- H.265 encode (CRF {CRF}), plain, no zones ---")
    compressed_path = out / "compressed_h265.mp4"
    encode_h265_plain(coding_ref_path, compressed_path, CRF, PRESET)
    bitrate = (get_filesize_mb(compressed_path) * 1024 * 1024 * 8) / get_duration_seconds(compressed_path) / 1_000_000
    size_mb = get_filesize_mb(compressed_path)

    # ---- decode + rotate BACK (EXACT, opposite direction) to the ---------
    # ---- ORIGINAL INPUT orientation -- NOT a fixed "vertical" target. ----
    print(f"\n--- Reconstruct to ORIGINAL input orientation ---")
    reconstructed_path = out / "reconstructed_original_orientation.mp4"
    if rotate_for_coding:
        # undo the forward rotation with the OPPOSITE transpose direction
        exact_rotate_from_h265(compressed_path, reconstructed_path, clockwise=False, preset=PRESET)
        print("    Applied the inverse exact rotation to return to the "
              "original input orientation.")
    else:
        copy_no_rotate(compressed_path, reconstructed_path)
        print("    No rotation was applied for coding -- output is already "
              "in the original input orientation.")

    # ---- PSNR reference: a lossless copy of the INPUT in ITS OWN, -------
    # ---- unmodified orientation -- since output must match input, the ---
    # ---- reference must too. No forced rotation here either. -----------
    reference_path = out / "reference_original_orientation.mp4"
    copy_no_rotate(input_path, reference_path)

    psnr = measure_psnr(reconstructed_path, reference_path)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Input orientation  = {input_w}x{input_h} "
          f"({'landscape' if input_is_landscape else 'portrait'})")
    print(f"Output orientation = {input_w}x{input_h} (MATCHES input, as required)")
    print(f"Coding orientation used = {'ROTATED' if rotate_for_coding else 'SAME as input'}")
    print(f"PSNR = {psnr:.3f} dB")
    print(f"Video-only bitrate = {bitrate:.4f} Mbps")
    print(f"File size = {size_mb:.3f} MB")
    print(f"\nMotion orientation statistics:")
    for k, v in orient_stats.items():
        print(f"  {k} = {v}")
    print(f"\n(No adaptive zones in this ablation -- spatial/perceptual component skipped.)")
    print(f"\nCoding reference saved to: {coding_ref_path}")
    print(f"Compressed H.265 stream saved to: {compressed_path}")
    print(f"Reconstructed (original orientation) saved to: {reconstructed_path}")
    print(f"PSNR reference (original orientation) saved to: {reference_path}")

    csv_path = out / "orientation_only_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Input_orientation", "Rotated_for_coding", "PSNR_dB", "Bitrate_Mbps",
                          "FileSize_MB", "Horizontal_energy", "Vertical_energy",
                          "Dominant_angle_deg", "Mean_speed", "N_motion_vectors"])
        writer.writerow(["landscape" if input_is_landscape else "portrait",
                          rotate_for_coding, round(psnr, 3), round(bitrate, 4), round(size_mb, 3),
                          round(orient_stats["horizontal_energy"], 4),
                          round(orient_stats["vertical_energy"], 4),
                          round(orient_stats["dominant_angle_deg"], 2),
                          round(orient_stats["mean_speed"], 4), orient_stats["n_vectors"]])
    print(f"\nResults written to: {csv_path}")
    print(f"\nCompare this PSNR/bitrate against orientation_adaptive_h265.py's result")
    print(f"on the SAME input/CRF -- the difference isolates the contribution of")
    print(f"the spatial/perceptual (adaptive-zone) component.")


if __name__ == "__main__":
    main()
