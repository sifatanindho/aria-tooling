#!/usr/bin/env python3
"""
Multi-source Audio Transcription for ARIA Glasses

This script creates merged transcripts from multiple ARIA glasses recordings.
The key insight: speech is clearest on the speaker's own glasses.

Strategy:
1. Extract audio from each video source
2. Transcribe each audio separately with word-level timestamps
3. Compute audio energy/RMS for each source at each time window
4. For overlapping speech segments, pick transcription from highest-energy source
5. Merge into single coherent transcript with speaker labels



Usage examples

Manual file list:
cd /data2/aria_data/extraction_scripts && python 6_create_transcript.py \
    output/Game2_1_41d5123c_synced/rgb_video.mp4 \
    output/Game2_6_4ff26da3_synced/rgb_video.mp4 \
    output/Game2_9_a4b33882_synced/rgb_video.mp4 \
    output/Game2_l7_c390a5f4_synced/rgb_video.mp4 \
    output/Game2_red2_b2e8cfba_synced/rgb_video.mp4 \
    -o output/Game2_transcript \
    -m base

Auto-discover by game name:
cd /data2/aria_data/extraction_scripts && python 6_create_transcript.py \
    --game Game2 \
    -o output/Game2_transcript \
    -m large
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

# Check for required packages
try:
    import whisper
except ImportError:
    print("Please install openai-whisper: pip install openai-whisper")
    sys.exit(1)

try:
    from scipy.io import wavfile
    from scipy.signal import resample
except ImportError:
    print("Please install scipy: pip install scipy")
    sys.exit(1)


@dataclass
class Segment:
    """A transcribed speech segment"""
    start: float  # seconds
    end: float    # seconds
    text: str
    source: str   # which glasses/video
    speaker: str  # speaker label
    confidence: float  # audio energy indicator
    words: List[dict] = None  # word-level timestamps


@dataclass
class MergedSegment:
    """A segment in the final merged transcript"""
    start: float
    end: float
    text: str
    speaker: str
    source: str


def get_ffmpeg() -> str:
    """Return a working ffmpeg binary path."""
    import shutil, subprocess
    # Prefer system ffmpeg over conda's (which may have broken shared libs)
    for candidate in ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg']:
        try:
            r = subprocess.run([candidate, '-version'], capture_output=True)
            if r.returncode == 0:
                return candidate
        except FileNotFoundError:
            pass
    found = shutil.which('ffmpeg')
    if found:
        return found
    raise RuntimeError("ffmpeg not found. Install with: sudo apt install ffmpeg")

FFMPEG = get_ffmpeg()


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg"""
    cmd = [
        FFMPEG, '-y', '-i', video_path,
        '-vn',  # no video
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', str(sample_rate),  # sample rate
        '-ac', '1',  # mono
        output_path
    ]
    print(f"Extracting audio from {Path(video_path).name}...")
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def compute_audio_energy(audio_path: str, window_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS energy over time windows.
    Returns: (time_points, energy_values)
    """
    sample_rate, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    
    window_samples = int(window_size * sample_rate)
    hop_samples = window_samples // 2  # 50% overlap
    
    n_windows = (len(audio) - window_samples) // hop_samples + 1
    energy = np.zeros(n_windows)
    times = np.zeros(n_windows)
    
    for i in range(n_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples
        window = audio[start_idx:end_idx]
        energy[i] = np.sqrt(np.mean(window ** 2))  # RMS
        times[i] = (start_idx + end_idx) / 2 / sample_rate
    
    return times, energy


def transcribe_audio(audio_path: str, model, source_name: str) -> List[Segment]:
    """Transcribe audio using Whisper with word timestamps"""
    print(f"Transcribing {source_name}...")
    
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False
    )
    
    segments = []
    for seg in result['segments']:
        words = None
        if 'words' in seg:
            words = [{'word': w['word'], 'start': w['start'], 'end': w['end']} 
                     for w in seg['words']]
        
        segments.append(Segment(
            start=seg['start'],
            end=seg['end'],
            text=seg['text'].strip(),
            source=source_name,
            speaker=source_name,  # Initial speaker is source name
            confidence=0.0,  # Will be filled with energy
            words=words
        ))
    
    return segments


def get_energy_for_segment(times: np.ndarray, energy: np.ndarray, 
                           start: float, end: float) -> float:
    """Get average energy for a time segment"""
    mask = (times >= start) & (times <= end)
    if np.any(mask):
        return float(np.mean(energy[mask]))
    return 0.0


def merge_transcripts(all_segments: Dict[str, List[Segment]], 
                      all_energy: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      overlap_threshold: float = 0.5) -> List[MergedSegment]:
    """
    Merge transcripts from multiple sources.
    For overlapping segments, pick the one from highest-energy source.
    """
    
    # Add energy confidence to each segment
    for source, segments in all_segments.items():
        times, energy = all_energy[source]
        for seg in segments:
            seg.confidence = get_energy_for_segment(times, energy, seg.start, seg.end)
    
    # Collect all segments with their timing
    all_segs = []
    for source, segments in all_segments.items():
        for seg in segments:
            all_segs.append(seg)
    
    # Sort by start time
    all_segs.sort(key=lambda s: s.start)
    
    if not all_segs:
        return []
    
    # Merge overlapping segments
    merged = []
    used = set()
    
    for i, seg in enumerate(all_segs):
        if i in used:
            continue
        
        # Find all segments that overlap with this one
        overlapping = [seg]
        overlapping_indices = [i]
        
        for j, other in enumerate(all_segs[i+1:], start=i+1):
            if j in used:
                continue
            
            # Check overlap
            overlap_start = max(seg.start, other.start)
            overlap_end = min(seg.end, other.end)
            overlap_duration = overlap_end - overlap_start
            
            min_duration = min(seg.end - seg.start, other.end - other.start)
            if min_duration > 0 and overlap_duration / min_duration > overlap_threshold:
                overlapping.append(other)
                overlapping_indices.append(j)
        
        # Pick the segment with highest energy (clearest audio)
        best_seg = max(overlapping, key=lambda s: s.confidence)
        
        merged.append(MergedSegment(
            start=best_seg.start,
            end=best_seg.end,
            text=best_seg.text,
            speaker=best_seg.source,
            source=best_seg.source
        ))
        
        # Mark all overlapping segments as used
        for idx in overlapping_indices:
            used.add(idx)
    
    # Sort final result by time
    merged.sort(key=lambda s: s.start)
    
    return merged


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def save_transcript(merged: List[MergedSegment], output_path: str, 
                    format: str = 'all'):
    """Save transcript in various formats"""
    base_path = Path(output_path).with_suffix('')
    
    # JSON format with full details
    if format in ['all', 'json']:
        json_path = str(base_path) + '.json'
        with open(json_path, 'w') as f:
            json.dump([asdict(s) for s in merged], f, indent=2)
        print(f"Saved JSON transcript to {json_path}")
    
    # SRT format
    if format in ['all', 'srt']:
        srt_path = str(base_path) + '.srt'
        with open(srt_path, 'w') as f:
            for i, seg in enumerate(merged, 1):
                start_ts = format_timestamp(seg.start).replace('.', ',')
                end_ts = format_timestamp(seg.end).replace('.', ',')
                f.write(f"{i}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"[{seg.speaker}] {seg.text}\n\n")
        print(f"Saved SRT transcript to {srt_path}")
    
    # Plain text with timestamps
    if format in ['all', 'txt']:
        txt_path = str(base_path) + '.txt'
        with open(txt_path, 'w') as f:
            for seg in merged:
                f.write(f"[{format_timestamp(seg.start)}] [{seg.speaker}] {seg.text}\n")
        print(f"Saved TXT transcript to {txt_path}")
    
    # CSV format
    if format in ['all', 'csv']:
        csv_path = str(base_path) + '.csv'
        with open(csv_path, 'w') as f:
            f.write("start_time,end_time,speaker,text\n")
            for seg in merged:
                # Escape quotes in text
                text = seg.text.replace('"', '""')
                f.write(f'{seg.start:.3f},{seg.end:.3f},"{seg.speaker}","{text}"\n')
        print(f"Saved CSV transcript to {csv_path}")


def create_speaker_map(sources: List[str]) -> Dict[str, str]:
    """Create a mapping from source names to speaker labels"""
    # Try to extract meaningful names from source paths
    speaker_map = {}
    for i, source in enumerate(sources):
        # Extract the folder name pattern like "Game1_1_69566477"
        name = Path(source).parent.name
        # Use the identifier part (e.g., "1", "9", "l7", "red2")
        parts = name.split('_')
        if len(parts) >= 2:
            speaker_id = parts[1]  # e.g., "1", "9", "l7", "red2"
            speaker_map[source] = f"Player_{speaker_id}"
        else:
            speaker_map[source] = f"Speaker_{i+1}"
    return speaker_map


def discover_game_videos(game: str, input_root: str) -> List[str]:
    """Find synced RGB videos for a game under input_root."""
    root = Path(input_root)
    pattern = f"{game}_*_synced/rgb_video.mp4"
    return sorted(str(p) for p in root.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description='Create merged transcript from multiple ARIA glasses recordings'
    )
    parser.add_argument('videos', nargs='*',
                        help='Video files to process (manual mode)')
    parser.add_argument('--game',
                        help='Game identifier for auto-discovery (e.g., Game2)')
    parser.add_argument('--input-root', default='output',
                        help='Root directory to search for synced folders in game mode')
    parser.add_argument('-o', '--output', default='transcript',
                        help='Output filename (without extension)')
    parser.add_argument('-m', '--model', default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    parser.add_argument('--temp-dir', default='/tmp/aria_transcribe',
                        help='Directory for temporary audio files')
    parser.add_argument('--keep-audio', action='store_true',
                        help='Keep extracted audio files')
    parser.add_argument('--format', default='all',
                        choices=['all', 'json', 'srt', 'txt', 'csv'],
                        help='Output format(s)')
    
    args = parser.parse_args()

    if not args.videos and not args.game:
        parser.error('Provide video files or use --game <GameName> for auto-discovery.')
    if args.videos and args.game:
        parser.error('Use either manual video files or --game mode, not both.')
    
    # Create temp directory
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect video files (manual mode or auto-discovery mode)
    if args.game:
        discovered = discover_game_videos(args.game, args.input_root)
        print(f"Discovered {len(discovered)} synced RGB videos for {args.game} in {args.input_root}")
        videos = discovered
    else:
        videos = args.videos

    # Verify all video files exist
    valid_videos = []
    for v in videos:
        if not os.path.exists(v):
            print(f"Warning: Video file not found: {v}")
            continue
        valid_videos.append(v)
    videos = valid_videos
    
    if not videos:
        print("No valid video files found!")
        sys.exit(1)
    
    print(f"Processing {len(videos)} video files...")
    print(f"Using Whisper model: {args.model}")
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model(args.model)
    
    # Process each video
    all_segments = {}
    all_energy = {}
    audio_files = []
    
    for video_path in videos:
        source_name = video_path  # Use full path as source identifier
        audio_path = temp_dir / f"{Path(video_path).parent.name}_audio.wav"
        audio_files.append(str(audio_path))
        
        # Extract audio
        extract_audio(video_path, str(audio_path))
        
        # Compute energy
        times, energy = compute_audio_energy(str(audio_path))
        all_energy[source_name] = (times, energy)
        
        # Transcribe
        segments = transcribe_audio(str(audio_path), model, source_name)
        all_segments[source_name] = segments
        
        print(f"  Found {len(segments)} segments in {Path(video_path).parent.name}")
    
    # Create speaker map
    speaker_map = create_speaker_map(videos)
    
    # Update speaker names in segments
    for source, segments in all_segments.items():
        for seg in segments:
            seg.speaker = speaker_map.get(source, seg.source)
    
    # Merge transcripts
    print("\nMerging transcripts...")
    merged = merge_transcripts(all_segments, all_energy)
    
    # Update speaker names in merged segments
    for seg in merged:
        seg.speaker = speaker_map.get(seg.source, seg.speaker)
    
    print(f"Created {len(merged)} merged segments")
    
    # Save output
    save_transcript(merged, args.output, args.format)
    
    # Cleanup
    if not args.keep_audio:
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        print("\nCleaned up temporary audio files")
    
    # Print summary
    print("\n=== Transcript Summary ===")
    print(f"Total segments: {len(merged)}")
    
    speaker_counts = defaultdict(int)
    speaker_duration = defaultdict(float)
    for seg in merged:
        speaker_counts[seg.speaker] += 1
        speaker_duration[seg.speaker] += seg.end - seg.start
    
    print("\nSpeaker breakdown:")
    for speaker in sorted(speaker_counts.keys()):
        count = speaker_counts[speaker]
        duration = speaker_duration[speaker]
        print(f"  {speaker}: {count} segments, {duration:.1f}s total")
    
    if merged:
        total_duration = merged[-1].end - merged[0].start
        print(f"\nTotal duration: {format_timestamp(total_duration)}")


if __name__ == '__main__':
    main()
