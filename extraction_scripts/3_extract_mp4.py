#!/usr/bin/env python3
"""
Extract RGB MP4 video from an Aria VRS file using projectaria_tools.

Usage:
    python 3_extract_mp4.py --vrs recording.vrs --output output.mp4
    python 3_extract_mp4.py --vrs recording.vrs --output output.mp4 --duration 300
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from projectaria_tools.tools.vrs_to_mp4.vrs_to_mp4_utils import convert_vrs_to_mp4


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB MP4 from Aria VRS file"
    )
    parser.add_argument("--vrs", required=True, help="Path to .vrs file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output .mp4 file path")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max duration in seconds (trims output)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor (default: 1)")

    args = parser.parse_args()

    if not os.path.exists(args.vrs):
        print(f"ERROR: VRS file not found: {args.vrs}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # If we need to trim, extract to a temp file first
    if args.duration is not None:
        tmp_output = args.output + ".tmp_full.mp4"
    else:
        tmp_output = args.output

    print(f"Extracting MP4 from VRS...")
    print(f"  VRS:    {args.vrs}")
    print(f"  Output: {args.output}")
    sys.stdout.flush()

    convert_vrs_to_mp4(args.vrs, tmp_output, down_sample_factor=args.downsample)

    # Trim to duration if requested
    if args.duration is not None and os.path.exists(tmp_output):
        print(f"  Trimming to {args.duration}s ({args.duration/60:.1f} min)...")
        sys.stdout.flush()
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_output,
            "-t", str(args.duration),
            "-c", "copy",       # fast copy, no re-encoding
            args.output,
        ], capture_output=True, text=True)

        os.remove(tmp_output)

        if result.returncode != 0:
            print(f"  ffmpeg trim error: {result.stderr[-300:]}")
            sys.exit(1)

    if os.path.exists(args.output):
        size_mb = Path(args.output).stat().st_size / 1024 / 1024
        print(f"  ✓ Saved: {args.output} ({size_mb:.1f} MB)")
    else:
        print("  ✗ Failed to create output")
        sys.exit(1)


if __name__ == "__main__":
    main()