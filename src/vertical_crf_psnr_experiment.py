"""
Simple video compression experiment: rotate a landscape video to portrait,
encode it at two CRF levels with libx265, and measure PSNR against the
rotated reference. No adaptive/ROI/motion-vector processing of any kind --
just rotate, encode at two CRF values, measure PSNR, print/save results.

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
INPUT_VIDEO = r"C:\path\to\input.mp4"
OUTPUT_DIR = r"C:\path\to\results"
ROTATION = "clockwise"          # "clockwise" or "counterclockwise"

PRESET = "medium"               # x265 preset used for both CRF encodes
CRF_VALUES = [25, 51]

# --------------------------------------------------------------------------

TRANSPOSE_CODES = {
    "clockwise": "1",
    "counterclockwise": "2",
}


def run(cmd, description, show_live_output=False):
    """Run a subprocess command, stop with a clear error if it fails.

    If show_live_output is True, ffmpeg's own progress is streamed straight
    to the terminal as it runs (used for the slow encoding steps), instead
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
        stderr_lines = result.stderr.strip().splitlines()
        print("\n".join(stderr_lines[-40:]))
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


def rotate_to_reference(input_path, output_path, rotation):
    """
    Rotates the input 90 degrees using ffmpeg's transpose filter. This swaps
    width and height (e.g. 3840x2160 -> 2160x3840) with no cropping and no
    rescaling. The result is re-encoded losslessly (libx264 -crf 0) purely
    as an intermediate container -- this is the exact reference both CRF
    outputs will be compared against for PSNR.
    """
    if rotation not in TRANSPOSE_CODES:
        print(f"ERROR: ROTATION must be 'clockwise' or 'counterclockwise', got '{rotation}'.")
        sys.exit(1)
    transpose_code = TRANSPOSE_CODES[rotation]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-stats",
        "-i", input_path,
        "-vf", f"transpose={transpose_code}",
        "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    run(cmd, "Rotating input video 90 degrees to create the vertical reference "
             "(mathematically lossless intermediate, CRF 0)",
        show_live_output=True)


def encode_crf(reference_path, output_path, crf, preset):
    """Plain H.265/HEVC encode of the reference at a single CRF value."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-stats",
        "-i", reference_path,
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    run(cmd, f"Encoding CRF {crf} (H.265/HEVC, preset={preset})",
        show_live_output=True)


def measure_psnr(distorted_path, reference_path):
    """
    Runs ffmpeg's psnr filter comparing distorted_path (main) against
    reference_path (reference), and returns the overall average PSNR in dB
    reported by the filter. Both files share resolution, pixel format and
    frame rate by construction, since the CRF outputs are encoded directly
    from this exact reference with no scaling.
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
    print("Simple vertical-video CRF/PSNR experiment")
    print("=" * 70)

    if not os.path.isfile(INPUT_VIDEO):
        print(f"ERROR: input file not found: {INPUT_VIDEO}")
        sys.exit(1)

    check_tools_available()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reference_path = os.path.join(OUTPUT_DIR, "reference_vertical.mp4")
    rotate_to_reference(INPUT_VIDEO, reference_path, ROTATION)

    ref_w, ref_h = get_dimensions(reference_path)
    print(f"    Reference vertical resolution: {ref_w}x{ref_h}")

    rows = []
    for crf in CRF_VALUES:
        out_path = os.path.join(OUTPUT_DIR, f"vertical_crf{crf}.mp4")
        encode_crf(reference_path, out_path, crf, PRESET)

        out_w, out_h = get_dimensions(out_path)
        if (out_w, out_h) != (ref_w, ref_h):
            print(f"WARNING: CRF {crf} output resolution {out_w}x{out_h} "
                  f"does not match reference {ref_w}x{ref_h}.")

        psnr_db = measure_psnr(out_path, reference_path)
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

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["CRF", "PSNR_dB", "Bitrate_Mbps", "FileSize_MB"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to: {csv_path}")

    print("\nOutput files:")
    print(f"  Reference (lossless, vertical): {reference_path}")
    for crf in CRF_VALUES:
        print(f"  CRF {crf}: {os.path.join(OUTPUT_DIR, f'vertical_crf{crf}.mp4')}")
    print(f"  CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
