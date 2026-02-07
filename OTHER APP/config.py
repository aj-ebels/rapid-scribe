"""App configuration: sample rate, chunk duration, paths."""

import os

# Capture at 48 kHz to avoid WASAPI/driver crashes (many Windows devices dislike 16 kHz)
SAMPLE_RATE = 48000

# Rate at which we save WAVs for Parakeet (ASR expects 16 kHz)
ASR_SAMPLE_RATE = 16000

# Sliding window: record this many seconds per chunk, then transcribe
CHUNK_DURATION_SEC = 5.0

# Frames per chunk for capture (larger = more stable on some WASAPI drivers)
FRAMES_PER_READ = 4096

# Mixer gain to avoid clipping when both sources are loud (0.0–1.0)
MIXER_GAIN = 0.7

# Temp folder for chunk WAVs (under system temp)
TEMP_DIR = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "Meetings2Chunks")
os.makedirs(TEMP_DIR, exist_ok=True)

# NeMo Parakeet model (~660MB = 110M params; 0.6b-v2 is ~2.5GB)
# Options: "nvidia/parakeet-tdt_ctc-110m" (smaller/faster), "nvidia/parakeet-tdt-0.6b-v2" (larger/more accurate)
PARAKEET_MODEL = "nvidia/parakeet-tdt_ctc-110m"

# Use PyAudioWPatch for capture instead of SoundCard (set True if SoundCard crashes on your system)
# Requires: pip install pyaudiowpatch
USE_PYAUDIOWPATCH = os.environ.get("MEETINGS2_USE_PYAUDIOWPATCH", "").strip().lower() in ("1", "true", "yes")
