# Meetings2

Local real-time transcription that captures **microphone** and **system audio (loopback)** simultaneously, mixes them, and transcribes using **NVIDIA Parakeet** (NeMo ASR) in a sliding-window fashion.

## Features

- **WASAPI capture**: Microphone + system loopback (no "Stereo Mix" required) via [SoundCard](https://github.com/bastibe/SoundCard).
- **Stereo trick**: Left channel = mic, Right channel = loopback (kept for potential diarization; current pipeline mixes to mono for Parakeet).
- **Gap filling**: When the system is silent, loopback provides no data; the app fills with silence so streams stay in sync.
- **Sliding window**: Records in 5-second chunks, writes each to a temp WAV, transcribes with Parakeet, then deletes the file.
- **GUI**: Start / Stop and a scrollable text area that updates live with transcribed text.

## Requirements

- **Windows** (WASAPI loopback is used).
- **Python 3.10+**.
- **NVIDIA GPU** recommended for Parakeet (runs on CPU otherwise).
- **CUDA + PyTorch** if you want GPU acceleration for NeMo.

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install PyTorch (with CUDA if you have an NVIDIA GPU):

   - See [pytorch.org](https://pytorch.org) for the right command, e.g.:
   - CPU: `pip install torch`
   - GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If `nemo_toolkit[asr]` fails, try:

   ```bash
   pip install Cython
   pip install nemo_toolkit[asr]
   ```

   On **Python 3.12+**, if you see `TypeError: object.__init__() takes exactly one argument` when transcribing, upgrade Lhotse: `pip install -U "lhotse>=1.22"`.

4. Run the app:

   ```bash
   python app.py
   ```

   **If the app crashes when you press Start**, use the PyAudioWPatch backend (see "If the app crashes on Start" below).

## Usage

1. Click **Start** to begin capturing mic + system audio and transcribing.
2. Text appears in the scrollable area as each 5-second chunk is transcribed.
3. Click **Stop** to stop capture and transcription.

## If the app crashes when you press Start

If the app closes as soon as you press Start (often due to SoundCard/WASAPI on some drivers), use the **PyAudioWPatch** backend instead:

```bash
pip install pyaudiowpatch
set MEETINGS2_USE_PYAUDIOWPATCH=1
python app.py
```

Or in PowerShell: `$env:MEETINGS2_USE_PYAUDIOWPATCH="1"; python app.py`

This uses PyAudioWPatch for mic + loopback capture instead of SoundCard, with the same behavior.

## Configuration

Edit `config.py` to change:

- `SAMPLE_RATE` (default 16 kHz)
- `CHUNK_DURATION_SEC` (default 5 s)
- `MIXER_GAIN` (default 0.7)
- `PARAKEET_MODEL`: e.g. `nvidia/parakeet-tdt_ctc-110m` (lighter) or `nvidia/parakeet-tdt-0.6b-v2`

Temp WAVs are written to `%TEMP%\Meetings2Chunks` and deleted after transcription.

## Architecture

1. **AudioMixer** (`audio_capture.py`): Two capture streams (mic + loopback) at the same sample rate; outputs stereo (L=mic, R=loopback), filling loopback gaps with silence.
2. **ChunkRecorder** (`chunk_recorder.py`): Buffers stereo, writes 5 s mono WAVs to temp, calls back with the file path.
3. **ParakeetTranscriber** (`transcriber.py`): Queue of WAV paths; loads NeMo Parakeet once, transcribes each file, deletes it, and pushes text to the GUI.
4. **App** (`app.py`): Tkinter GUI; Start/Stop and live transcript text area.

## License

Use and modify as you like.
