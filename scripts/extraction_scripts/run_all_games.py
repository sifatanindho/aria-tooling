#!/usr/bin/env python3
"""
Batch extraction script for Game1-Game6 VRS files.

Runs run_pipeline.py on all VRS files found in Game1-Game6 directories.
Extracts FULL recordings (no duration limit).

Special handling:
  - Glasses 3 uses 20 fps (custom recording profile)
  - All other glasses use 60 fps

Usage:
    conda activate aria_extract
    python run_all_games.py

Output structure:
    output/
        Game1_1_69566477/
        Game1_6_96e68a76/
        ...
        Game2_1_41d5123c/
        ...
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PIPELINE_SCRIPT = SCRIPT_DIR / "run_pipeline.py"

# Always use the aria_extraction conda env Python
PYTHON = "/home/videep/miniconda3/envs/aria_extraction/bin/python"
if not Path(PYTHON).exists():
    PYTHON = sys.executable  # fallback

# Game directories to process (Game6 through Game11)
GAME_NAMES = ["Game5"]

# Glasses 3 uses a custom recording profile with 20 fps ET camera
GLASSES_FPS = {
    "3": 20,  # Custom profile - 20 fps
    # All others default to 60 fps
}
DEFAULT_FPS = 60


def find_vrs_files():
    """Find all VRS files in Game6-Game11 directories."""
    vrs_files = []
    
    for game_name in GAME_NAMES:
        game_dir = DATA_DIR / game_name
        
        if not game_dir.exists():
            print(f"Warning: {game_dir} does not exist, skipping")
            continue
        
        # Each subdirectory is a glasses ID (1, 2, 3, 6, 9, l7, red2)
        for glasses_dir in sorted(game_dir.iterdir()):
            if not glasses_dir.is_dir():
                continue
            
            glasses_id = glasses_dir.name
            fps = GLASSES_FPS.get(glasses_id, DEFAULT_FPS)
            
            # Find VRS files in this glasses directory
            for vrs_file in glasses_dir.glob("*.vrs"):
                # Create output name: Game1_1_<uuid_prefix>
                uuid_prefix = vrs_file.stem[:8]
                output_name = f"{game_name}_{glasses_id}_{uuid_prefix}"
                
                vrs_files.append({
                    "path": vrs_file,
                    "game": game_name,
                    "glasses": glasses_id,
                    "output_name": output_name,
                    "fps": fps,
                })
    
    return vrs_files


def run_pipeline(vrs_info, output_base_dir):
    """Run the extraction pipeline on a single VRS file."""
    vrs_path = vrs_info["path"]
    output_dir = output_base_dir / vrs_info["output_name"]
    fps = vrs_info["fps"]

    # Skip if already completed (both videos present)
    if (output_dir / "et_video.mp4").exists() and (output_dir / "rgb_video.mp4").exists():
        print(f"\n  ⏭  Skipping {vrs_info['output_name']} (already done)")
        return True

    cmd = [
        PYTHON,
        str(PIPELINE_SCRIPT),
        "--vrs", str(vrs_path),
        "--output-dir", str(output_dir),
        "--fps", str(fps),
        "--skip-gaze",   # Skip eye gaze inference for batch run
        # No --duration flag = extract ALL data
    ]
    
    print(f"\n{'=' * 70}")
    print(f"  Processing: {vrs_info['game']} / glasses {vrs_info['glasses']}")
    print(f"  VRS: {vrs_path.name}")
    print(f"  Output: {output_dir}")
    print(f"  FPS: {fps}")
    print(f"{'=' * 70}")
    sys.stdout.flush()
    
    result = subprocess.run(cmd, text=True)
    
    return result.returncode == 0


def main():
    print("=" * 70)
    print("  BATCH VRS EXTRACTION - Game1 through Game6")
    print("=" * 70)
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Python: {sys.executable}")
    print(f"  Data dir: {DATA_DIR}")
    sys.stdout.flush()
    
    # Find all VRS files
    vrs_files = find_vrs_files()
    
    if not vrs_files:
        print("\nNo VRS files found in Game1-Game6 directories!")
        sys.exit(1)
    
    print(f"\nFound {len(vrs_files)} VRS files to process:")
    for vrs in vrs_files:
        fps_note = f" (custom {vrs['fps']}fps)" if vrs['fps'] != DEFAULT_FPS else ""
        print(f"  - {vrs['game']}/glasses {vrs['glasses']}{fps_note}: {vrs['path'].name}")
    
    # Output directory
    output_base = DATA_DIR / "output"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each VRS file
    results = {}
    for i, vrs_info in enumerate(vrs_files, 1):
        print(f"\n\n>>> [{i}/{len(vrs_files)}] Starting extraction...")
        success = run_pipeline(vrs_info, output_base)
        results[vrs_info["output_name"]] = success
    
    # Summary
    print(f"\n\n{'=' * 70}")
    print("  BATCH EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"\nResults:")
    
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {succeeded} succeeded, {failed} failed")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
