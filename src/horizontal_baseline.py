"""
Script 2: HORIZONTAL BASELINE

Uses the ORIGINAL horizontal video directly (no rotation, no crop, no
rescale) as the reference, and encodes it with H.265/HEVC at CRF 25 and
CRF 51 using the same encoder settings as vertical_baseline.py.

Completely independent script -- does not import anything from the other
two scripts.

Requires: ffmpeg.exe and ffprobe.exe available in PATH.
"""

import os
import re
import csv
import sys
import subprocess

# --------------------------------------------------------------------------
# CONFIGURATION -- edit these lines for your setup
# --------------------------------------------------------------------------
INPUT_VIDEO = r"C:\path\to\input.mp4"   # same source file used by vertical_baseline.py
OUTPUT_DIR = r"C:\path\to\results"

PRESET = "medium"        # same preset as vertical_baseline.py
CRF_VALUES = [25, 51]

# --------------------------------------------------------------------------


def run(cmd, description, show_live_output=False):
    """Run a subprocess command, stop with a clear error if it fails.

    If show_live_output is True, ffmpeg's own progress is streamed straight
    to the terminal as it runs (used for the slow encoding step), instead
    of being captured silently until the command finishes.
    """
    print(f"--> {description}")
    print("    $ " + " ".join(cmd))
    if show_live_output:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: command failed (exit code {result.returncode}). "
                  f"See ffmpeg output above for details.")
            sys.exit(1)
        return result
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("\nERROR: command failed.")
        print("----- ffmpeg/ffprobe stderr (last 40 lines) -----")
        print("\n".join(result.stderr.strip().splitlines()[-40:]))
        sys.exit(1)
    return result


def check_tools_available():
    for tool in ("ffmpeg", "ffprobe"):
        result = subprocess.run([tool, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"ERROR: '{tool}' was not found or did not run correctly. "
                  f"Make sure {tool}.exe is available in PATH.")
            sys.exit(1)


def get_duration_seconds(path):
    result = run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        f"Reading duration of {os.path.basename(path)}",
    )
    return float(result.stdout.strip())


def get_dimensions(path):
    result = run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", path],
        f"Reading dimensions of {os.path.basename(path)}",
    )
    w_str, h_str = result.stdout.strip().split(",")
    return int(w_str), int(h_str)


def encode_crf(input_path, output_path, crf, preset):
    """Plain H.265/HEVC encode of the ORIGINAL horizontal video -- no
    rotation, crop, or rescale. Same settings as vertical_baseline.py."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-stats",
        "-i", input_path, "-map", "0:v:0", "-an",
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    run(cmd, f"Encoding CRF {crf} (H.265/HEVC, preset={preset}, horizontal, no rotation)",
        show_live_output=True)


def measure_psnr(distorted_path, reference_path):
    """
    Runs ffmpeg's psnr filter comparing distorted_path (main) against
    reference_path (reference), and returns the overall average PSNR
    reported by the filter.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", distorted_path,
        "-i", reference_path,
        "-lavfi", "[0:v][1:v]psnr",
        "-f", "null", "-",
    ]
    print(f"--> Measuring PSNR: {os.path.basename(distorted_path)} vs "
          f"{os.path.basename(reference_path)}")
    print("    $ " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("\nERROR: PSNR calculation failed.")
        print("----- ffmpeg stderr (last 40 lines) -----")
        print("\n".join(result.stderr.strip().splitlines()[-40:]))
        sys.exit(1)

    match = re.search(r"average:(\d+\.?\d*)", result.stderr)
    if not match:
        print("\nERROR: could not find PSNR 'average:' value in ffmpeg output.")
        print(result.stderr[-2000:])
        sys.exit(1)
    return float(match.group(1))


def get_filesize_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    print("=" * 70)
    print("Horizontal baseline CRF/PSNR experiment (no rotation)")
    print("=" * 70)

    if not os.path.isfile(INPUT_VIDEO):
        print(f"ERROR: input file not found: {INPUT_VIDEO}")
        sys.exit(1)

    check_tools_available()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The ORIGINAL horizontal file is used directly as the PSNR reference --
    # no rotation, crop or rescale, and no lossless intermediate copy is
    # made (unlike vertical_baseline.py's reference_vertical.mp4). This
    # matches the source file exactly, but note it means the horizontal
    # reference and the vertical reference are not the same kind of file
    # (original vs. re-encoded-lossless) -- see the accompanying note.
    ref_w, ref_h = get_dimensions(INPUT_VIDEO)
    print(f"    Original horizontal resolution (reference): {ref_w}x{ref_h}")

    rows = []
    for crf in CRF_VALUES:
        out_path = os.path.join(OUTPUT_DIR, f"horizontal_crf{crf}.mp4")
        encode_crf(INPUT_VIDEO, out_path, crf, PRESET)

        out_w, out_h = get_dimensions(out_path)
        if (out_w, out_h) != (ref_w, ref_h):
            print(f"WARNING: CRF {crf} output resolution {out_w}x{out_h} "
                  f"does not match reference {ref_w}x{ref_h}.")

        psnr_db = measure_psnr(out_path, INPUT_VIDEO)
        duration = get_duration_seconds(out_path)
        size_mb = get_filesize_mb(out_path)
        bitrate_mbps = (size_mb * 1024 * 1024 * 8) / duration / 1_000_000

        rows.append({
            "CRF": crf,
            "PSNR_dB": round(psnr_db, 2),
            "Bitrate_Mbps": round(bitrate_mbps, 2),
            "FileSize_MB": round(size_mb, 2),
        })

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for row in rows:
        print(f"CRF {row['CRF']}:")
        print(f"  PSNR      = {row['PSNR_dB']:.2f} dB")
        print(f"  Bitrate   = {row['Bitrate_Mbps']:.2f} Mbps")
        print(f"  File size = {row['FileSize_MB']:.2f} MB")
        print()

    csv_path = os.path.join(OUTPUT_DIR, "horizontal_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["CRF", "PSNR_dB", "Bitrate_Mbps", "FileSize_MB"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to: {csv_path}")

    print("\nOutput files:")
    print(f"  Reference (original, horizontal, unmodified): {INPUT_VIDEO}")
    for crf in CRF_VALUES:
        print(f"  CRF {crf}: {os.path.join(OUTPUT_DIR, f'horizontal_crf{crf}.mp4')}")
    print(f"  CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
