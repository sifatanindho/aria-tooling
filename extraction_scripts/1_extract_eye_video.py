"""
Extract Eye Tracking (ET) camera video from an Aria VRS file.

Outputs:
  - Individual ET frames as PNGs (optional, for downstream pupil detection)
  - MP4 video of the ET stream

Usage:
    python extract_et_video.py --vrs recording.vrs --fps 60
    python extract_et_video.py --vrs recording.vrs --fps 10 --save-frames

Requirements:
    pip install projectaria-tools Pillow numpy
    ffmpeg must be installed (sudo apt install ffmpeg)
"""

import argparse
import os
import subprocess
import csv
import numpy as np
from pathlib import Path
from PIL import Image

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId

ET_STREAM = StreamId("211-1")

# Use system ffmpeg (conda's may have broken shared libs)
FFMPEG = next((p for p in ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg'] if __import__('os').path.exists(p)), 'ffmpeg')


def extract_et(vrs_path: str, output_dir: str, fps: int, save_frames: bool, duration: float = None):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    if provider is None:
        raise RuntimeError(f"Failed to open: {vrs_path}")

    num_frames = provider.get_num_data(ET_STREAM)
    if num_frames == 0:
        print("No ET data found in VRS file.")
        return

    # If duration is specified, compute how many frames to extract
    if duration is not None:
        ts_ns = list(provider.get_timestamps_ns(ET_STREAM, TimeDomain.DEVICE_TIME))
        start_ns = ts_ns[0]
        end_ns = start_ns + int(duration * 1e9)
        num_frames = sum(1 for t in ts_ns if t <= end_ns)
        print(f"Duration limit: {duration}s → extracting {num_frames} frames")

    print(f"Extracting {num_frames} ET frames")
    print(f"At {fps} fps that's ~{num_frames / fps:.1f} seconds of data")

    frames_dir = os.path.join(output_dir, "et_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Collect timestamps for downstream alignment
    timestamps = []

    for i in range(num_frames):
        img_data = provider.get_image_data_by_index(ET_STREAM, i)
        arr = img_data[0].to_numpy_array()
        record_info = img_data[1]
        ts_ns = record_info.capture_timestamp_ns

        timestamps.append((i, ts_ns))

        # Save frame as PNG
        img = Image.fromarray(arr)
        img.save(os.path.join(frames_dir, f"frame_{i:06d}.png"))

        if i % 100 == 0 and i > 0:
            print(f"  Extracted {i}/{num_frames} frames...")

    print(f"Extracted {num_frames} frames to {frames_dir}/")

    # Save timestamps CSV
    ts_path = os.path.join(output_dir, "et_timestamps.csv")
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "capture_timestamp_ns"])
        writer.writerows(timestamps)
    print(f"Saved timestamps to {ts_path}")

    # Encode to MP4
    video_path = os.path.join(output_dir, "et_video.mp4")
    cmd = [
        FFMPEG, "-y",
        "-r", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # high quality
        video_path
    ]
    print(f"\nEncoding video at {fps} fps...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Saved ET video to {video_path}")
    else:
        print(f"ffmpeg error: {result.stderr}")

    # Clean up frames if not needed
    if not save_frames:
        import shutil
        shutil.rmtree(frames_dir)
        print("Cleaned up frame PNGs (use --save-frames to keep them)")

    print(f"\nDone. Output in {output_dir}/")
    if save_frames:
        print(f"  Frames: {frames_dir}/frame_XXXXXX.png")
    print(f"  Video:  {video_path}")
    print(f"  Timestamps: {ts_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ET camera video from Aria VRS"
    )
    parser.add_argument("--vrs", required=True, help="Path to .vrs file")
    parser.add_argument("--fps", type=int, default=10,
                        help="ET camera frame rate (default: 10 for Gen 1, use 60 for 60Hz profile)")
    parser.add_argument("--output-dir", default="et_output",
                        help="Output directory (default: et_output)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Keep individual PNG frames (for pupil detection pipeline)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max duration in seconds to extract (default: all)")

    args = parser.parse_args()
    extract_et(args.vrs, args.output_dir, args.fps, args.save_frames, args.duration)


if __name__ == "__main__":
    main()
