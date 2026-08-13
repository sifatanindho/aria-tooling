# aria-tooling

Utilities for extracting, synchronizing, transcribing, and annotating data from **Project Aria** `.vrs` recordings.

This repository is organized around an end-to-end workflow:

1. Extract ET and RGB streams from VRS files
2. Compute pupil and eye-gaze metrics
3. Manually synchronize multi-glasses recordings
4. Build merged multi-speaker transcripts
5. (Optional) Annotate transcripts with LLM-based social/deliberation schemas

---

## Repository structure

```text
aria-tooling/
├── extraction_scripts/
│   ├── run_pipeline.py
│   ├── run_all_games.py
│   ├── 1_extract_eye_video.py
│   ├── 2_extract_pupil_metrics.py
│   ├── 3_extract_mp4.py
│   ├── 4_extract_eye_gaze_metrics.py
│   ├── 5_sync_data.py
│   ├── 6_create_transcript.py
│   └── 7_annotate_gpt.py
├── projectaria_eyetracking/
│   ├── model_inference_demo.py
│   └── inference/
└── aria_data/
    ├── analyze_vrs.py
    └── check_games.py
```

---

## What each component does

### `extraction_scripts/`

- **`run_pipeline.py`**: orchestrates extraction for one VRS recording.
  - ET video + ET timestamps
  - RGB video with audio
  - pupil metrics
  - optional eye-gaze inference
- **`run_all_games.py`**: batch runner across game folders/glasses directories.
- **`1_extract_eye_video.py`**: extracts ET frames/video (`211-1`) and timestamp CSV.
- **`2_extract_pupil_metrics.py`**: estimates pupil metrics from ET video, writes annotated video + CSV + JSON summary.
- **`3_extract_mp4.py`**: converts VRS RGB stream to MP4 using `projectaria_tools`.
- **`4_extract_eye_gaze_metrics.py`**: runs SocialEye eye-gaze inference and exports CSV.
- **`5_sync_data.py`**: preview + manual sync workflow for aligning multiple recordings.
- **`6_create_transcript.py`**: creates merged transcript from multiple synced RGB videos using Whisper + energy-based source selection.
- **`7_annotate_gpt.py`**: annotates transcript utterances via OpenAI API for:
  - Werewolf persuasion strategy schema
  - DeliData deliberation schema

### `projectaria_eyetracking/`

Local eye-gaze inference package and pretrained assets used by the extraction pipeline.

### `aria_data/`

Data-inspection helper scripts for checking recording organization/timing across games.

---

## Requirements

### System dependencies

- Python 3.10+ recommended
- `ffmpeg` (required by multiple scripts)

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Python dependencies

Install the packages used across scripts:

```bash
pip install \
  projectaria-tools \
  pillow \
  numpy \
  pandas \
  opencv-python \
  matplotlib \
  scipy \
  tqdm \
  torch \
  rerun-sdk \
  openai-whisper \
  openai \
  pyyaml
```

> Notes
> - `7_annotate_gpt.py` requires `OPENAI_API_KEY`.
> - `4_extract_eye_gaze_metrics.py` / `projectaria_eyetracking` require PyTorch and model weights already present in this repo.

---

## Quick start

### 1) Run full extraction for a single `.vrs`

```bash
cd extraction_scripts
python run_pipeline.py --vrs /path/to/recording.vrs --duration 300
```

Common options:

- `--output-dir /path/to/output_dir`
- `--fps 60` (or `20` for glasses using 20Hz ET profile)
- `--skip-et`, `--skip-rgb`, `--skip-pupil`, `--skip-gaze`

## 2) Sync multiple recordings in a game

```bash
python 5_sync_data.py preview-all --game Game1 --data-dir /path/to/aria_data --output ./output
# edit generated sync_config_Game1.json with per-recording sync_seconds
python 5_sync_data.py sync-all --config ./output/sync_config_Game1.json --data-dir ./output
```

## 3) Build merged transcript from synced RGB videos

```bash
python 6_create_transcript.py --game Game1 --input-root ./output -o ./output/Game1_transcript -m base
```

## 4) Annotate transcript with GPT schemas (optional)

```bash
export OPENAI_API_KEY=your_key_here
python 7_annotate_gpt.py --input ./output/Game1_transcript.csv --output ./output/Game1_transcript_annotated.csv --schema both
```

---

## `run_pipeline.py` output

Typical output directory contents:

```text
output/<recording_name>/
├── et_video.mp4
├── et_timestamps.csv
├── rgb_video.mp4
├── et_video_pupil_metrics.csv
├── et_video_pupil_summary.json
├── et_video_pupil_output.mp4
├── eye_gaze.csv                 # if gaze step enabled
└── pipeline_info.json
```

All streams are tracked with device-clock-based timestamps for downstream alignment.

---

## Script reference

### Extraction

- `1_extract_eye_video.py`
  - `--vrs`, `--fps`, `--output-dir`, `--save-frames`, `--duration`
- `2_extract_pupil_metrics.py`
  - `--video` or `--webcam`
  - optional `--vrs` for Aria calibration
  - `--output`, `--headless`, `--eye`, `--eye-side`
- `3_extract_mp4.py`
  - `--vrs`, `--output`, optional `--duration`, `--downsample`
- `4_extract_eye_gaze_metrics.py`
  - `--vrs`, `--model_checkpoint_path`, `--model_config_path`, `--output_file`, `--device`, `--console_only`

### Synchronization and batch orchestration

- `run_pipeline.py`: all-in-one single-recording pipeline
- `run_all_games.py`: batch runner over predefined game folders
- `5_sync_data.py` subcommands:
  - `preview`
  - `preview-all`
  - `sync`
  - `sync-all`

### Transcript and annotation

- `6_create_transcript.py`
  - manual mode: pass video files directly
  - game mode: `--game` + `--input-root`
  - outputs: `json`, `srt`, `txt`, `csv`, or `all`
- `7_annotate_gpt.py`
  - input CSV/TSV with required columns:
    - `speaker`
    - `text`
  - auto-generates `utterance_id` if missing
  - output adds `ww_*` and/or `dd_*` annotation columns

---

## Important path/config notes

- Several scripts include environment-specific defaults (for example hardcoded `/data2/...` paths and conda Python paths).
- Before running in a new environment, review:
  - `extraction_scripts/run_pipeline.py`
  - `extraction_scripts/run_all_games.py`
  - `extraction_scripts/4_extract_eye_gaze_metrics.py`
  - `extraction_scripts/5_sync_data.py`
  - `extraction_scripts/6_create_transcript.py`
- Update default paths or pass explicit CLI arguments as needed.

---

## Troubleshooting

- **`ffmpeg` not found / video trim fails**
  - Install system `ffmpeg` and confirm `ffmpeg -version`.
- **`projectaria_tools` import errors**
  - Ensure the active Python environment has `projectaria-tools`.
- **Eye-gaze inference import/path issues**
  - Verify `projectaria_eyetracking` paths and model weight/config paths.
- **No OpenAI key for annotation**
  - Set `OPENAI_API_KEY` in shell environment before running `7_annotate_gpt.py`.
- **No data found in batch scripts**
  - Check game/data directory defaults and folder naming conventions.

---

## License

This project is primarily licensed under the [MIT License](./LICENSE). Portions of the repository (notably `projectaria_eyetracking/` and `extraction_scripts/4_extract_eye_gaze_metrics.py`) include Apache-2.0 licensed code; see the headers in those files for details.
