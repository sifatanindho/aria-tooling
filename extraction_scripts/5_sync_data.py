#!/usr/bin/env python3
"""
Manual Sync Tool for Aria Recordings

Workflow:
  1. Generate previews for all recordings:
     python 5_sync_data.py preview-all --game Game1

  2. Browse the preview folders, find the frame where stopwatch shows your target time
     Note the frame number for each recording

  3. Edit sync_config_Game1.json with your frame numbers

  4. Apply sync to all:
     python 5_sync_data.py sync-all --config sync_config_Game1.json

Single recording commands:
  python 5_sync_data.py preview --vrs recording.vrs --output preview_dir
  python 5_sync_data.py sync --input output_dir --sync-seconds 6.5
"""

import argparse
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np


def extract_preview_frames(vrs_path: str, output_dir: str, seconds: float = 300, fps: int = 5):
    """Extract initial RGB frames from VRS file for sync point identification."""

    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.sensor_data import TimeDomain
    from PIL import Image

    RGB_STREAM = StreamId("214-1")  # RGB camera stream

    print(f"Extracting preview frames from: {os.path.basename(vrs_path)}")

    provider = data_provider.create_vrs_data_provider(vrs_path)
    if provider is None:
        print(f"ERROR: Failed to open VRS file: {vrs_path}")
        return False

    num_total = provider.get_num_data(RGB_STREAM)
    if num_total == 0:
        print("ERROR: No RGB data found in VRS file")
        return False

    # Get timestamps
    timestamps = list(provider.get_timestamps_ns(RGB_STREAM, TimeDomain.DEVICE_TIME))

    start_ns = timestamps[0]
    end_ns = start_ns + int(seconds * 1e9)

    os.makedirs(output_dir, exist_ok=True)

    # Save timestamp info
    sync_info = {
        "vrs_file": os.path.abspath(vrs_path),
        "start_timestamp_ns": start_ns,
        "preview_fps": fps,
        "rgb_native_fps": 30,  # Aria RGB is typically 30fps
        "frames": []
    }

    frame_count = 0
    # Sample every N frames to get desired fps from ~30fps source
    sample_interval = max(1, round(30 / fps))

    for i, ts in enumerate(timestamps):
        if ts > end_ns:
            break

        if i % sample_interval == 0:
            img_data = provider.get_image_data_by_index(RGB_STREAM, i)
            arr = img_data[0].to_numpy_array()

            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)

            img = Image.fromarray(arr)

            time_sec = (ts - start_ns) / 1e9
            # Name frames with time for easy identification
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}_t{time_sec:.2f}s.jpg")
            img.save(frame_path, quality=85)

            sync_info["frames"].append({
                "preview_frame": frame_count,
                "vrs_frame_index": i,
                "timestamp_ns": ts,
                "time_seconds": round(time_sec, 3)
            })

            frame_count += 1

    # Save sync info
    info_path = os.path.join(output_dir, "frame_info.json")
    with open(info_path, "w") as f:
        json.dump(sync_info, f, indent=2)

    print(f"  → {frame_count} frames saved to {output_dir}/")
    return True


def preview_all_game(game_name: str, base_data_dir: str, output_base: str, seconds: float = 300):
    """Generate preview frames for all recordings in a game folder."""

    game_dir = os.path.join(base_data_dir, game_name)
    if not os.path.exists(game_dir):
        print(f"ERROR: Game directory not found: {game_dir}")
        sys.exit(1)

    # Find all VRS files
    vrs_files = []
    for root, dirs, files in os.walk(game_dir):
        for f in files:
            if f.endswith('.vrs'):
                vrs_files.append(os.path.join(root, f))

    if not vrs_files:
        print(f"No VRS files found in {game_dir}")
        sys.exit(1)

    print(f"Found {len(vrs_files)} recordings in {game_name}")
    print("=" * 60)

    # Create config template
    config = {
        "game": game_name,
        "target_sync_time": "SET_THIS (e.g., '00:00:06' on stopwatch)",
        "recordings": []
    }

    preview_base = os.path.join(output_base, f"previews_{game_name}")
    os.makedirs(preview_base, exist_ok=True)

    for vrs_path in sorted(vrs_files):
        # Extract participant ID from path (e.g., Game1/1/xxx.vrs -> "1")
        rel_path = os.path.relpath(vrs_path, game_dir)
        participant = rel_path.split(os.sep)[0]
        vrs_id = Path(vrs_path).stem[:8]

        preview_dir = os.path.join(preview_base, f"{participant}_{vrs_id}")

        print(f"\n[{participant}] {vrs_id}")
        extract_preview_frames(vrs_path, preview_dir, seconds=seconds)

        config["recordings"].append({
            "participant": participant,
            "vrs_id": vrs_id,
            "vrs_path": vrs_path,
            "preview_dir": preview_dir,
            "sync_seconds": None,  # USER FILLS THIS IN
            "notes": ""
        })

    # Save config template
    config_path = os.path.join(output_base, f"sync_config_{game_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Preview extraction complete!")
    print(f"\nPreviews saved to: {preview_base}/")
    print(f"Config template:   {config_path}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Browse each preview folder")
    print(f"  2. Find the frame showing your target stopwatch time")
    print(f"     (Frame names include time, e.g., frame_0025_t5.00s.jpg = 5 seconds into video)")
    print(f"  3. Edit {config_path}")
    print(f"     - Set 'sync_seconds' for each recording")
    print(f"  4. Run: python 5_sync_data.py sync-all --config {config_path}")


def sync_single(input_dir: str, sync_seconds: float, output_dir: str = None):
    """Trim all data in input_dir to start from sync_seconds."""

    input_dir = os.path.abspath(input_dir)
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return False

    if output_dir is None:
        output_dir = input_dir.rstrip("/") + "_synced"

    os.makedirs(output_dir, exist_ok=True)

    # Load ET timestamps
    et_ts_path = os.path.join(input_dir, "et_timestamps.csv")
    if not os.path.exists(et_ts_path):
        print(f"  ERROR: et_timestamps.csv not found")
        return False

    et_df = pd.read_csv(et_ts_path)
    et_start_ns = et_df["capture_timestamp_ns"].iloc[0]
    sync_ns = et_start_ns + int(sync_seconds * 1e9)

    # 1. Sync ET timestamps
    et_synced = et_df[et_df["capture_timestamp_ns"] >= sync_ns].copy()
    if len(et_synced) == 0:
        print(f"  ERROR: sync_seconds ({sync_seconds}) is beyond recording length")
        return False

    first_synced_idx = et_synced.iloc[0]["frame_index"]
    et_synced["capture_timestamp_ns"] = et_synced["capture_timestamp_ns"] - sync_ns
    et_synced["frame_index"] = range(len(et_synced))
    et_synced.to_csv(os.path.join(output_dir, "et_timestamps.csv"), index=False)

    # 2. Sync pupil metrics
    pupil_path = os.path.join(input_dir, "et_video_pupil_metrics.csv")
    if os.path.exists(pupil_path):
        pupil_df = pd.read_csv(pupil_path)
        pupil_synced = pupil_df[pupil_df["frame"] >= first_synced_idx].copy()
        pupil_synced["frame"] = pupil_synced["frame"] - first_synced_idx
        pupil_synced.to_csv(os.path.join(output_dir, "et_video_pupil_metrics.csv"), index=False)

    # 3. Sync eye gaze metrics (if extracted)
    # Eye gaze uses tracking_timestamp_us (microseconds)
    sync_us = int(sync_ns / 1000)  # Convert ns to us
    for gaze_file in ["general_eye_gaze.csv", "eye_gaze.csv"]:
        gaze_path = os.path.join(input_dir, gaze_file)
        if os.path.exists(gaze_path):
            gaze_df = pd.read_csv(gaze_path)
            if "tracking_timestamp_us" in gaze_df.columns:
                # Filter to keep only data after sync point
                gaze_start_us = gaze_df["tracking_timestamp_us"].iloc[0]
                sync_threshold_us = gaze_start_us + int(sync_seconds * 1e6)
                gaze_synced = gaze_df[gaze_df["tracking_timestamp_us"] >= sync_threshold_us].copy()
                # Rebase timestamps to start from 0
                gaze_synced["tracking_timestamp_us"] = gaze_synced["tracking_timestamp_us"] - sync_threshold_us
                gaze_synced.to_csv(os.path.join(output_dir, gaze_file), index=False)
                print(f"    Synced {gaze_file}: {len(gaze_synced)}/{len(gaze_df)} rows")

    # 4. Trim videos
    for video_name in ["et_video.mp4", "rgb_video.mp4", "et_video_pupil_output.mp4"]:
        video_path = os.path.join(input_dir, video_name)
        if os.path.exists(video_path):
            output_video = os.path.join(output_dir, video_name)
            _ffmpeg = next((p for p in ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg'] if __import__('os').path.exists(p)), 'ffmpeg')
            cmd = [
                _ffmpeg, "-y", "-loglevel", "error",
                "-ss", str(sync_seconds),
                "-i", video_path,
                "-c", "copy",
                output_video
            ]
            subprocess.run(cmd, capture_output=True)

    # 5. Sync metadata
    sync_metadata = {
        "original_dir": input_dir,
        "sync_seconds": sync_seconds,
        "sync_timestamp_ns": int(sync_ns),
        "frames_trimmed": int(first_synced_idx),
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "sync_metadata.json"), "w") as f:
        json.dump(sync_metadata, f, indent=2)

    # Copy other files
    for f in ["et_video_pupil_summary.json", "pipeline_info.json"]:
        src = os.path.join(input_dir, f)
        if os.path.exists(src):
            shutil.copy(src, output_dir)

    return True


def sync_all_from_config(config_path: str, extracted_data_dir: str):
    """Apply sync to all recordings based on config file."""

    with open(config_path) as f:
        config = json.load(f)

    print(f"Syncing {len(config['recordings'])} recordings from {config['game']}")
    print("=" * 60)

    success_count = 0
    for rec in config["recordings"]:
        participant = rec["participant"]
        vrs_id = rec["vrs_id"]

        # Find the ORIGINAL extracted data directory (not _synced versions)
        # Pattern: Game1_1_69566477 (game_participant_vrsid) - exact match, no _synced suffix
        base_name = f"{config['game']}_{participant}_{vrs_id[:8]}"
        
        # Look for exact match first (original folder)
        candidates = glob(os.path.join(extracted_data_dir, f"{base_name}*"))
        # Filter out _synced folders to find the original
        original_matches = [c for c in candidates if not c.rstrip("/").endswith("_synced")]

        if not original_matches:
            print(f"\n[{participant}] ✗ No original extracted data found matching {base_name}")
            continue

        input_dir = original_matches[0]

        # Get sync time
        sync_seconds = rec.get("sync_seconds")
        if sync_seconds is None:
            print(f"\n[{participant}] ✗ No sync_seconds specified - skipping")
            continue

        print(f"\n[{participant}] Syncing at {sync_seconds:.2f}s...")
        output_dir = input_dir.rstrip("/") + "_synced"
        
        # Remove existing synced folder to overwrite
        if os.path.exists(output_dir):
            print(f"    Overwriting existing {os.path.basename(output_dir)}/")
            shutil.rmtree(output_dir)

        if sync_single(input_dir, sync_seconds, output_dir):
            print(f"  → {output_dir}")
            success_count += 1
        else:
            print(f"  ✗ Failed")

    print("\n" + "=" * 60)
    print(f"✓ Synced {success_count}/{len(config['recordings'])} recordings")


def main():
    parser = argparse.ArgumentParser(description="Manual sync tool for Aria recordings")
    subparsers = parser.add_subparsers(dest="command")

    # preview - single VRS
    p1 = subparsers.add_parser("preview", help="Extract preview frames from one VRS")
    p1.add_argument("--vrs", required=True, help="Path to VRS file")
    p1.add_argument("--output", "-o", required=True, help="Output directory")
    p1.add_argument("--seconds", type=float, default=60, help="Seconds to extract (default: 60)")
    p1.add_argument("--fps", type=int, default=5, help="Frames per second (default: 5)")

    # preview-all - all VRS in a game folder
    p2 = subparsers.add_parser("preview-all", help="Extract previews for all recordings in a game")
    p2.add_argument("--game", required=True, help="Game folder name (e.g., Game1)")
    p2.add_argument("--data-dir", default="/data2/aria_data/aria_data",
                    help="Base data directory")
    p2.add_argument("--output", "-o", default="/data2/aria_data/extraction_scripts/output",
                    help="Output base directory")
    p2.add_argument("--seconds", type=float, default=60, help="Seconds to extract (default: 60)")

    # sync - single recording
    p3 = subparsers.add_parser("sync", help="Sync one extracted recording")
    p3.add_argument("--input", "-i", required=True, help="Input directory with extracted data")
    p3.add_argument("--sync-seconds", type=float, required=True,
                    help="Seconds from start where sync point is")
    p3.add_argument("--output", "-o", help="Output directory (default: input_synced)")

    # sync-all - batch sync from config
    p4 = subparsers.add_parser("sync-all", help="Sync all recordings from config file")
    p4.add_argument("--config", "-c", required=True, help="Path to sync config JSON")
    p4.add_argument("--data-dir", default="/data2/aria_data/extraction_scripts/output",
                    help="Directory containing extracted data")

    args = parser.parse_args()

    if args.command == "preview":
        extract_preview_frames(args.vrs, args.output, args.seconds, args.fps)
    elif args.command == "preview-all":
        preview_all_game(args.game, args.data_dir, args.output, args.seconds)
    elif args.command == "sync":
        if sync_single(args.input, args.sync_seconds, args.output):
            print(f"✓ Synced to: {args.output or args.input + '_synced'}")
    elif args.command == "sync-all":
        sync_all_from_config(args.config, args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
