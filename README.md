# CS6384 Project: Real-Time Facial Expression Music Generator

This project is an interactive computer vision and audio system that:

- detects facial landmarks from a webcam stream,
- classifies facial emotion with a fine-tuned FER model, and
- maps emotion and facial geometry to expressive real-time music.

It was built for a computer vision course project and combines perception, temporal smoothing, and multimodal output (visual + audio).

## Project Overview

The pipeline runs in real time:

1. `vision.py` captures webcam frames and extracts facial landmarks with MediaPipe.
2. The same module classifies emotion (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) using a PyTorch `efficientnetv2_s` model.
3. `mapping.py` maps emotion + facial features to musical parameters (tempo, instrument, scale, note, velocity, sustain, pan, etc.).
4. `audio.py` plays music through either:
   - system MIDI (preferred, low latency), or
   - a software synth fallback.
5. `main.py` renders the HUD, handles keyboard controls, and optionally records evaluation stats.

## Repository Structure

- `main.py` - main real-time application
- `launcher.py` - Tkinter launcher UI for selecting modes/settings
- `vision.py` - webcam capture, landmarks, FER inference, feature extraction
- `mapping.py` - facial expression to music mapping logic
- `audio.py` - MIDI/synth audio backends
- `evaluation.py` - emotion metrics and latency profiler utilities
- `test_audio.py` - audio-only demo (no camera required)
- `test_offline.py` - offline FER evaluation script
- `models/` - model config and class mapping (`best_model.pth` expected here)
- `face_landmarker.task` - MediaPipe face landmarker model asset
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+ recommended
- Webcam (for real-time vision modes)
- Audio output device
- Optional: system MIDI synthesizer/device for best latency

### Python Packages

Install project dependencies:

```bash
pip install -r requirements.txt
```

If not already installed in your environment, install PyTorch + torchvision for your system:

```bash
pip install torch torchvision
```

## Setup

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
pip install torch torchvision
```

## Running the Project

### Option A: Launcher UI (recommended)

```bash
python launcher.py
```

The launcher lets you choose:

- full mode (vision + audio),
- audio-only test mode,
- vision-only mode,
- evaluation mode,
- camera/FPS/smoothing settings, and audio backend.

### Option B: Command line

```bash
python main.py
```

Useful flags:

- `--synth` - force software synth backend
- `--eval` - enable ground-truth labeling/evaluation mode
- `--latency` - print latency summary every 5 seconds
- `--no-audio` - vision-only debug mode
- `--camera <idx>` - camera index (default `0`)
- `--fps <n>` - target FPS cap (default `30`)
- `--smooth <0..1>` - temporal smoothing alpha (default `0.3`)
- `--interval <seconds>` - note trigger interval (default `0.4`)

Example:

```bash
python main.py --eval --latency --camera 0 --fps 30
```

## Runtime Controls (Main Window)

- `Q` / `ESC` - quit
- `P` - pause/resume audio
- `R` - reset evaluator/profiler
- `S` - print latency summary
- `E` - print emotion evaluation report
- `L` - cycle ground-truth label in eval mode
- `M` - toggle landmark dots
- `L` (non-eval) - toggle mesh lines
- `C` - toggle confidence bars
- `F` - toggle fullscreen
- `TAB` - recalibrate baseline facial measurements

## Evaluation and Testing

### 1) Audio-only functional test

```bash
python test_audio.py
```

Use this to validate mapping + playback without webcam dependencies.

### 2) Offline FER evaluation

Smoke test (no dataset required):

```bash
python test_offline.py --smoke
```

FER2013 CSV mode:

```bash
python test_offline.py --fer2013 /path/to/fer2013.csv --split PublicTest --samples 1000
```

Folder dataset mode:

```bash
python test_offline.py --testdir /path/to/test_dataset --samples 1000
```

Expected FER2013 labels:

- `0 angry`
- `1 disgust`
- `2 fear`
- `3 happy`
- `4 sad`
- `5 surprise`
- `6 neutral`

## Outputs and Artifacts

- `session_stats.json` - written after sessions (used by launcher statistics popup)
- `fer2013_eval_results.csv` - offline FER evaluation records
- `folder_eval_results.csv` - folder-based offline evaluation records

## Troubleshooting

- **Camera not opening**: verify camera index (`--camera 0`, `--camera 1`, ...).
- **No audio output**: switch to software synth (`--synth`).
- **Missing model file**: ensure `models/best_model.pth` exists.
- **Slow runtime**: reduce FPS (`--fps 15` or `24`) and/or increase smoothing.
- **Missing PyTorch errors**: install `torch torchvision` in the active environment.

## Credits

CS6384 Computer Vision course project by the repository author(s), combining:

- MediaPipe face landmarks,
- PyTorch FER classification, and
- real-time generative music mapping.
