#!/usr/bin/env python3
"""
Aria VRS Data Extraction Pipeline
==================================

Orchestrator that calls the existing extraction scripts to extract all data
from a VRS file into a single organized directory.

Steps:
  1. 1_extract_eye_video.py   → ET camera video (60 fps)
  2. 3_extract_mp4.py         → RGB video with audio (30 fps, via convert_vrs_to_mp4)
  3. 2_extract_pupil_metrics.py → Pupil metrics from ET video
  4. 4_extract_eye_gaze_metrics.py → Eye gaze yaw/pitch (optional, needs model)

All timestamps use the device clock (nanoseconds) for cross-stream sync.

Usage:
    python run_pipeline.py \\
        --vrs /data2/aria/02-17-SecretHitler/Videep-secret-02-17.vrs \\
        --duration 300
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Always use the aria_extraction conda env Python so all packages are available
PYTHON = "/home/videep/miniconda3/envs/aria_extraction/bin/python"
if not os.path.exists(PYTHON):
    PYTHON = sys.executable  # fallback


def run_step(description, cmd):
    """Run a pipeline step as a subprocess, streaming output."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  $ {' '.join(cmd)}\n")
    sys.stdout.flush()

    result = subprocess.run(cmd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Print output
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
    sys.stdout.flush()

    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit code {result.returncode})")
        return False
    print(f"\n  ✓ Done")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Aria VRS Data Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python run_pipeline.py \\
        --vrs /data2/aria/02-17-SecretHitler/Videep-secret-02-17.vrs \\
        --duration 300

Output:
    output/Videep-secret-02-17/
    ├── et_video.mp4                # Eye tracking camera video (60 fps)
    ├── et_timestamps.csv           # ET frame timestamps (device_timestamp_ns)
    ├── rgb_video.mp4               # RGB video with audio (30 fps)
    ├── et_video_pupil_metrics.csv  # Per-frame pupil measurements
    ├── et_video_pupil_summary.json # Pupil summary stats
    ├── et_video_pupil_output.mp4   # Annotated pupil video
    ├── eye_gaze.csv                # Eye gaze yaw/pitch (optional)
    └── pipeline_info.json          # Run metadata
        """)

    parser.add_argument("--vrs", required=True, help="Path to .vrs file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ./output/<vrs_stem>)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max duration in seconds (default: all)")
    parser.add_argument("--eye-side", default="right",
                        choices=["left", "right", "both"],
                        help="Eye for pupil detection (default: right)")
    parser.add_argument("--skip-pupil", action="store_true",
                        help="Skip pupil metric extraction")
    parser.add_argument("--skip-gaze", action="store_true",
                        help="Skip eye gaze inference")
    parser.add_argument("--skip-rgb", action="store_true",
                        help="Skip RGB video extraction")
    parser.add_argument("--skip-et", action="store_true",
                        help="Skip ET video extraction")
    parser.add_argument("--fps", type=int, default=60,
                        help="ET video frame rate (default: 60, use 20 for glasses 3)")

    args = parser.parse_args()

    vrs_path = os.path.abspath(args.vrs)
    if not os.path.exists(vrs_path):
        print(f"ERROR: VRS file not found: {vrs_path}")
        sys.exit(1)

    # Default output dir
    if args.output_dir is None:
        vrs_name = Path(vrs_path).stem
        args.output_dir = os.path.join(SCRIPT_DIR, "output", vrs_name)
    os.makedirs(args.output_dir, exist_ok=True)

    dur_str = (f"{args.duration}s ({args.duration/60:.1f} min)"
               if args.duration else "full recording")

    print("=" * 60)
    print("  ARIA VRS DATA EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"  VRS      : {vrs_path}")
    print(f"  Output   : {args.output_dir}")
    print(f"  Duration : {dur_str}")
    print(f"  Eye side : {args.eye_side}")
    print(f"  ET FPS   : {args.fps}")
    sys.stdout.flush()

    results = {}

    # ── Step 1: ET video ─────────────────────────────────────────────────
    et_video_path = os.path.join(args.output_dir, "et_video.mp4")
    if not args.skip_et:
        cmd = [
            PYTHON, os.path.join(SCRIPT_DIR, "1_extract_eye_video.py"),
            "--vrs", vrs_path,
            "--output-dir", args.output_dir,
            "--fps", str(args.fps),
        ]
        if args.duration is not None:
            cmd += ["--duration", str(args.duration)]

        results["et_video"] = run_step("STEP 1: Extract Eye Tracking Video", cmd)
    else:
        print("\n  Skipping ET video extraction")
        results["et_video"] = False

    # ── Step 2: RGB video + audio (via convert_vrs_to_mp4) ───────────────
    rgb_video_path = os.path.join(args.output_dir, "rgb_video.mp4")
    if not args.skip_rgb:
        cmd = [
            PYTHON, os.path.join(SCRIPT_DIR, "3_extract_mp4.py"),
            "--vrs", vrs_path,
            "--output", rgb_video_path,
        ]
        # No --duration: keep full RGB video (convert_vrs_to_mp4 processes
        # the full VRS anyway, so trimming just wastes the effort).

        results["rgb_video"] = run_step("STEP 2: Extract RGB Video + Audio (full)", cmd)
    else:
        print("\n  Skipping RGB video extraction")
        results["rgb_video"] = False

    # ── Step 3: Pupil metrics ────────────────────────────────────────────
    if not args.skip_pupil and os.path.exists(et_video_path):
        cmd = [
            PYTHON, os.path.join(SCRIPT_DIR, "2_extract_pupil_metrics.py"),
            "--video", et_video_path,
            "--vrs", vrs_path,
            "--output", args.output_dir,
            "--eye-side", args.eye_side,
            "--headless",
        ]
        results["pupil"] = run_step("STEP 3: Extract Pupil Metrics", cmd)
    else:
        print("\n  Skipping pupil metrics")
        results["pupil"] = False

    # ── Step 4: Eye gaze inference ───────────────────────────────────────
    if not args.skip_gaze:
        gaze_output = os.path.join(args.output_dir, "eye_gaze.csv")
        cmd = [
            PYTHON, os.path.join(SCRIPT_DIR, "4_extract_eye_gaze_metrics.py"),
            "--vrs", vrs_path,
            "--device", "cuda",
            "--output_file", gaze_output,
            "--console_only",
        ]
        results["eye_gaze"] = run_step("STEP 4: Extract Eye Gaze Metrics", cmd)
    else:
        print("\n  Skipping eye gaze inference")
        results["eye_gaze"] = False

    # ── Save pipeline metadata ───────────────────────────────────────────
    info = {
        "vrs_file": vrs_path,
        "output_dir": args.output_dir,
        "duration_seconds": args.duration,
        "eye_side": args.eye_side,
        "results": {k: v for k, v in results.items()},
        "timestamp": datetime.now().isoformat(),
    }
    info_path = os.path.join(args.output_dir, "pipeline_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ✓  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {args.output_dir}\n")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                print(f"    {fname:45s} {size/1024/1024:8.1f} MB")
            else:
                print(f"    {fname:45s} {size/1024:8.1f} KB")
    print()


if __name__ == "__main__":
    main()
