#!/usr/bin/env python3
"""
Real-time system audio capture and transcription.
Uses PulseAudio Monitor / RDPSource on WSL2, NVIDIA Parakeet (onnx-asr), and CustomTkinter GUI.
Same Parakeet/ONNX stack as Meetily: https://github.com/Zackriya-Solutions/meeting-minutes
"""

import os
import sys
import json
import subprocess
import tempfile

# Subprocess entry for frozen build (PyInstaller): mic capture only — no pyaudiowpatch/GUI in this process
if len(sys.argv) >= 2 and sys.argv[1] == "--mic-capture":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: --mic-capture <device_index> [sample_rate] [chunk_samples]\n")
        sys.exit(1)
    import numpy as _np
    import sounddevice as _sd
    _di = int(sys.argv[2])
    _sr = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    _cs = int(sys.argv[4]) if len(sys.argv) > 4 else 80000
    _out = sys.stdout.buffer
    try:
        while True:
            _chunk = _sd.rec(_cs, samplerate=_sr, channels=1, dtype=_np.float32, device=_di, blocking=True)
            if _chunk is not None and _chunk.size > 0:
                _out.write(_np.asarray(_chunk, dtype=_np.float32).flatten().tobytes())
                _out.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception as e:
        sys.stderr.write(f"mic_capture error: {e}\n")
        sys.exit(1)
    sys.exit(0)
# Loopback subprocess entry (frozen build): pyaudiowpatch only, write float32 PCM to stdout
if len(sys.argv) >= 2 and sys.argv[1] == "--loopback-capture":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: --loopback-capture <device_index|default> [sample_rate] [chunk_samples]\n")
        sys.exit(1)
    if sys.platform != "win32":
        sys.stderr.write("Loopback is Windows-only.\n")
        sys.exit(1)
    import numpy as _np
    import pyaudiowpatch as _pyaudio
    _dev_arg = sys.argv[2]
    _dev_idx = None if _dev_arg.lower() == "default" else int(_dev_arg)
    _sr = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    _cs = int(sys.argv[4]) if len(sys.argv) > 4 else 80000
    _dur = _cs / _sr
    _out = sys.stdout.buffer
    try:
        with _pyaudio.PyAudio() as _p:
            if _dev_idx is not None:
                _dev = _p.get_device_info_by_index(_dev_idx)
            else:
                _wi = _p.get_host_api_info_by_type(_pyaudio.paWASAPI)
                _dev = _p.get_device_info_by_index(_wi["defaultOutputDevice"])
                if not _dev.get("isLoopbackDevice"):
                    for _lb in _p.get_loopback_device_info_generator():
                        if _dev["name"] in _lb["name"]:
                            _dev = _lb
                            break
            _rate = int(_dev["defaultSampleRate"])
            _ch = int(_dev["maxInputChannels"])
            _nf = int(_rate * _dur) * _ch
            while True:
                _buf = []
                with _p.open(format=_pyaudio.paInt16, channels=_ch, rate=_rate, frames_per_buffer=1024,
                             input=True, input_device_index=_dev["index"]) as _stream:
                    while len(_buf) * 1024 < _nf:
                        try:
                            _buf.append(_np.frombuffer(_stream.read(1024, exception_on_overflow=False), dtype=_np.int16))
                        except Exception:
                            break
                if not _buf:
                    continue
                _raw = _np.concatenate(_buf)[:_nf].reshape(-1, _ch).astype(_np.float64) / 32768.0
                _mono = _np.mean(_raw, axis=1).astype(_np.float32)
                if _rate != _sr:
                    _nout = int(round(len(_mono) * _sr / _rate))
                    _mono = _np.array(_np.interp(_np.linspace(0, len(_mono) - 1, _nout), _np.arange(len(_mono)), _mono), dtype=_np.float32)
                _mono = _mono[:_cs]
                if len(_mono) < _cs:
                    _mono = _np.pad(_mono, (0, _cs - len(_mono)))
                _out.write(_mono.tobytes())
                _out.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception as _e:
        sys.stderr.write(f"loopback_capture error: {_e}\n")
        sys.exit(1)
    sys.exit(0)
import threading
import time
import queue
import uuid
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass  # no-op if python-dotenv not installed

# Ensure WSLg can display the GUI (PulseAudio + X11/Wayland). No-op on Windows.
try:
    release = os.uname().release
    if "WSL" in release or "microsoft" in release.lower():
        if "DISPLAY" not in os.environ or not os.environ["DISPLAY"].strip():
            os.environ.setdefault("DISPLAY", ":0")
except AttributeError:
    pass  # os.uname() not available (e.g. Windows)

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
import customtkinter as ctk

# In-process mic+loopback (Meeting mode on Windows)
if sys.platform == "win32":
    from audio_mixer import AudioMixer
    from chunk_recorder import ChunkRecorder
from tkinter import messagebox, filedialog

# Lazy load Parakeet via onnx-asr (same stack as Meetily: istupakov ONNX Parakeet)
# Model: nemo-parakeet-tdt-0.6b-v2 (en) or nemo-parakeet-tdt-0.6b-v3 (multilingual)
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"
_parakeet_model = None
_parakeet_model_id = None

def get_transcription_model(model_id=None):
    """Load and return the ONNX ASR model. model_id defaults to PARAKEET_MODEL. Cached per model_id."""
    global _parakeet_model, _parakeet_model_id
    model_id = model_id or PARAKEET_MODEL
    if _parakeet_model is None or _parakeet_model_id != model_id:
        if _parakeet_model is not None:
            _parakeet_model = None
            _parakeet_model_id = None
        import onnx_asr
        _parakeet_model = onnx_asr.load_model(
            model_id,
            quantization="int8",  # faster on CPU, similar to previous Whisper int8
        )
        _parakeet_model_id = model_id
    return _parakeet_model


def clear_transcription_model_cache():
    """Clear the cached model so the next transcription uses the currently selected model."""
    global _parakeet_model, _parakeet_model_id
    _parakeet_model = None
    _parakeet_model_id = None


# -----------------------------------------------------------------------------
# Installed transcription models (Hugging Face cache)
# -----------------------------------------------------------------------------

# Substrings in repo_id that we treat as "transcription" models (ASR, Whisper, Parakeet, etc.)
ASR_REPO_PATTERNS = (
    "parakeet", "whisper", "asr", "speech", "stt", "vosk", "gigaam", "canary",
    "conformer", "transcribe",
)


def _format_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"


def list_installed_transcription_models():
    """List cached Hugging Face models that look like ASR/transcription. Returns list of dicts."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return [], "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
    except Exception as e:
        return [], str(e)
    out = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        rid = (repo.repo_id or "").lower()
        if not any(p in rid for p in ASR_REPO_PATTERNS):
            continue
        out.append({
            "repo_id": repo.repo_id,
            "size_on_disk": repo.size_on_disk,
            "size_str": _format_size(repo.size_on_disk),
            "revision_hashes": [r.commit_hash for r in repo.revisions],
        })
    out.sort(key=lambda x: x["repo_id"])
    return out, None


def uninstall_transcription_model(repo_id, revision_hashes):
    """Remove a cached model from the Hugging Face cache. Returns (success, error_message)."""
    if not revision_hashes:
        return False, "No revisions to delete"
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False, "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
        strategy = cache.delete_revisions(*revision_hashes)
        strategy.execute()
        return True, None
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
# Settings (audio mode, devices)
# -----------------------------------------------------------------------------

SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
MIC_CAPTURE_SCRIPT = Path(__file__).resolve().parent / "mic_capture_subprocess.py"
LOOPBACK_CAPTURE_SCRIPT = Path(__file__).resolve().parent / "loopback_capture_subprocess.py"
AUDIO_MODE_DEFAULT = "default"
AUDIO_MODE_MEETING_FFMPEG = "meeting_ffmpeg"
AUDIO_MODE_LOOPBACK = "loopback"
AUDIO_MODE_MEETING = "meeting"


def load_settings():
    """Load settings from JSON. Returns dict with audio_mode, meeting_mic_device, loopback_device_index, transcription_model."""
    out = {"audio_mode": AUDIO_MODE_DEFAULT, "meeting_mic_device": None, "loopback_device_index": None,
           "ffmpeg_mic_name": None, "ffmpeg_loopback_name": None, "transcription_model": PARAKEET_MODEL}
    if not SETTINGS_FILE.exists():
        return out
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out["audio_mode"] = data.get("audio_mode", out["audio_mode"])
            out["meeting_mic_device"] = data.get("meeting_mic_device")
            out["loopback_device_index"] = data.get("loopback_device_index")
            out["ffmpeg_mic_name"] = data.get("ffmpeg_mic_name")
            out["ffmpeg_loopback_name"] = data.get("ffmpeg_loopback_name")
            out["transcription_model"] = data.get("transcription_model", PARAKEET_MODEL) or PARAKEET_MODEL
    except Exception:
        pass
    return out


def save_settings(settings):
    """Write settings to JSON."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def list_loopback_devices():
    """On Windows with pyaudiowpatch: list WASAPI loopback devices. Otherwise empty."""
    if sys.platform != "win32":
        return [], None
    try:
        import pyaudiowpatch as pyaudio
        with pyaudio.PyAudio() as p:
            devices = []
            for loopback in p.get_loopback_device_info_generator():
                devices.append({
                    "index": loopback["index"],
                    "name": loopback["name"],
                    "default_samplerate": loopback.get("defaultSampleRate", 48000),
                    "max_input_channels": loopback.get("maxInputChannels", 2),
                })
            return devices, None
    except ImportError:
        return [], "pyaudiowpatch not installed"
    except Exception as e:
        return [], str(e)


# -----------------------------------------------------------------------------
# Audio device helpers (WSL / PulseAudio Monitor)
# -----------------------------------------------------------------------------

def list_audio_devices():
    """List all audio devices with index, name, and input channels."""
    devices = []
    try:
        all_devices = sd.query_devices()
        for dev in all_devices:
            devices.append({
                "index": dev.get("index", len(devices)),
                "name": dev.get("name", "Unknown"),
                "max_input_channels": dev.get("max_input_channels", 0),
                "default_samplerate": dev.get("default_samplerate", 0),
            })
    except Exception as e:
        return [], str(e)
    return devices, None


def get_default_monitor_device():
    """
    Select default input device: first input device whose name contains
    'monitor' or 'RDPSource' (case-insensitive for 'monitor').
    Falls back to system default input if none found.
    """
    devices, err = list_audio_devices()
    if err:
        return None, err
    keywords = ("monitor", "RDPSource")
    for d in devices:
        if d["max_input_channels"] <= 0:
            continue
        name = (d.get("name") or "").lower()
        if any(kw.lower() in name for kw in keywords):
            return d["index"], None
    # Fallback: default input device
    try:
        default = sd.query_devices(kind="input")
        return default["index"], None
    except Exception as e:
        return None, str(e)


def get_effective_audio_device(app):
    """
    Resolve capture mode and device indices from settings.
    Returns (mode, mic_device_index, loopback_device_index).
    - mode: AUDIO_MODE_DEFAULT | AUDIO_MODE_LOOPBACK | AUDIO_MODE_MEETING
    - mic_device_index: for default mode or meeting mic; None means use get_default_monitor_device()
    - loopback_device_index: for loopback/meeting; None means default loopback
    """
    settings = load_settings() if app is None else getattr(app, "settings", load_settings())
    mode = settings.get("audio_mode") or AUDIO_MODE_DEFAULT
    if mode == AUDIO_MODE_LOOPBACK:
        loopback_idx = settings.get("loopback_device_index")
        return (AUDIO_MODE_LOOPBACK, None, loopback_idx)
    if mode == AUDIO_MODE_MEETING:
        return (AUDIO_MODE_MEETING, settings.get("meeting_mic_device"), settings.get("loopback_device_index"))
    if mode == AUDIO_MODE_MEETING_FFMPEG:
        return (AUDIO_MODE_MEETING_FFMPEG, settings.get("ffmpeg_mic_name"), settings.get("ffmpeg_loopback_name"))
    return (AUDIO_MODE_DEFAULT, None, None)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_SEC = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
# In-process Meeting (mic+loopback): capture at 48k to avoid WASAPI issues, then resample to 16k for ASR
CAPTURE_SAMPLE_RATE_MEETING = 48000
FRAMES_PER_READ_MEETING = 4096
MIXER_GAIN_MEETING = 0.7
# Use OS temp dir so it works on Windows (no /tmp) and Linux/WSL
CHUNK_PATH = Path(tempfile.gettempdir()) / "meetings_chunk.wav"

# Skip transcribing chunks that are effectively silent (avoids Whisper "hallucinations"
# where it invents phrases like "Thanks for watching" when there's no real speech).
SILENCE_RMS_THRESHOLD = 0.01  # float32 RMS; increase if real speech is being skipped


def _is_silent(chunk: np.ndarray) -> bool:
    """True if chunk has very low energy (silence or negligible noise)."""
    rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
    return rms < SILENCE_RMS_THRESHOLD


# -----------------------------------------------------------------------------
# Audio capture thread
# -----------------------------------------------------------------------------

def capture_worker(device_index, chunk_queue, stop_event):
    """Record chunks; save to temp chunk file only when not silent, put path in queue."""
    while not stop_event.is_set():
        try:
            chunk = sd.rec(
                CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                device=device_index,
                blocking=True,
            )
            if stop_event.is_set():
                break
            if _is_silent(chunk):
                continue  # Skip silent chunks — don't send to Whisper (avoids hallucinations)
            chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, chunk_int16)
            try:
                chunk_queue.put(str(CHUNK_PATH), timeout=1.0)  # block until transcription took it
            except queue.Full:
                pass
        except Exception as e:
            if not stop_event.is_set():
                try:
                    chunk_queue.put_nowait(("error", str(e)))
                except queue.Full:
                    pass
            break


def _ensure_float32_mono_16k(samples, in_rate, in_channels):
    """Convert to float32 mono 16 kHz. samples: ndarray (frames,) or (frames, ch)."""
    if samples.dtype != np.float32:
        samples = samples.astype(np.float64) / (np.iinfo(samples.dtype).max if np.issubdtype(samples.dtype, np.integer) else 1.0)
        samples = samples.astype(np.float32)
    if len(samples.shape) > 1 and samples.shape[1] > 1:
        samples = np.mean(samples, axis=1)
    if in_rate != SAMPLE_RATE:
        num_out = int(round(len(samples) * SAMPLE_RATE / in_rate))
        samples = resample(samples, num_out).astype(np.float32)
    return samples


def capture_worker_loopback(loopback_device_index, chunk_queue, stop_event):
    """Record from WASAPI loopback only (PyAudioWPatch). Puts chunk paths into chunk_queue."""
    if sys.platform != "win32":
        try:
            chunk_queue.put_nowait(("error", "Loopback is only supported on Windows with pyaudiowpatch."))
        except queue.Full:
            pass
        return
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        try:
            chunk_queue.put_nowait(("error", "pyaudiowpatch not installed. pip install pyaudiowpatch"))
        except queue.Full:
            pass
        return
    try:
        with pyaudio.PyAudio() as p:
            if loopback_device_index is not None:
                dev = p.get_device_info_by_index(loopback_device_index)
            else:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                if not default_speakers.get("isLoopbackDevice"):
                    for loopback in p.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            default_speakers = loopback
                            break
                dev = default_speakers
            rate = int(dev["defaultSampleRate"])
            ch = int(dev["maxInputChannels"])
            chunk_frames_loopback = int(rate * CHUNK_DURATION_SEC) * ch
            while not stop_event.is_set():
                buf = []
                with p.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    frames_per_buffer=1024,
                    input=True,
                    input_device_index=dev["index"],
                ) as stream:
                    while len(buf) * 1024 < chunk_frames_loopback and not stop_event.is_set():
                        try:
                            data = stream.read(1024, exception_on_overflow=False)
                            buf.append(np.frombuffer(data, dtype=np.int16))
                        except Exception:
                            break
                if stop_event.is_set():
                    break
                if not buf:
                    continue
                raw = np.concatenate(buf)[:chunk_frames_loopback]
                raw = raw.reshape(-1, ch).astype(np.float64) / 32768.0
                mono = np.mean(raw, axis=1).astype(np.float32)
                if rate != SAMPLE_RATE:
                    num_out = int(round(len(mono) * SAMPLE_RATE / rate))
                    mono = resample(mono, num_out).astype(np.float32)
                mono = mono[:CHUNK_SAMPLES]
                if len(mono) < CHUNK_SAMPLES:
                    mono = np.pad(mono, (0, CHUNK_SAMPLES - len(mono)))
                if _is_silent(mono):
                    continue
                chunk_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
                wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, chunk_int16)
                try:
                    chunk_queue.put(str(CHUNK_PATH), timeout=1.0)
                except queue.Full:
                    pass
    except Exception as e:
        if not stop_event.is_set():
            try:
                chunk_queue.put_nowait(("error", str(e)))
            except queue.Full:
                pass


def capture_worker_mic_sounddevice(device_index, out_queue, stop_event):
    """Capture mic with sounddevice only; put float32 mono chunks (CHUNK_SAMPLES) into out_queue."""
    try:
        default_sr = sd.query_devices(device_index, "input")["default_samplerate"]
        rate = int(default_sr)
    except Exception:
        rate = SAMPLE_RATE
    while not stop_event.is_set():
        try:
            chunk = sd.rec(
                CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                device=device_index,
                blocking=True,
            )
            if stop_event.is_set():
                break
            out_queue.put(chunk.copy())
        except Exception as e:
            if not stop_event.is_set():
                try:
                    out_queue.put(("error", str(e)))
                except queue.Full:
                    pass
            break


def capture_worker_loopback_for_meeting(loopback_device_index, out_queue, stop_event):
    """Capture loopback with PyAudioWPatch; put float32 mono 16k chunks (CHUNK_SAMPLES) into out_queue."""
    if sys.platform != "win32":
        return
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return
    try:
        with pyaudio.PyAudio() as p:
            if loopback_device_index is not None:
                dev = p.get_device_info_by_index(loopback_device_index)
            else:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                if not default_speakers.get("isLoopbackDevice"):
                    for loopback in p.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            default_speakers = loopback
                            break
                dev = default_speakers
            rate = int(dev["defaultSampleRate"])
            ch = int(dev["maxInputChannels"])
            chunk_frames_loopback = int(rate * CHUNK_DURATION_SEC) * ch
            while not stop_event.is_set():
                buf = []
                with p.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    frames_per_buffer=1024,
                    input=True,
                    input_device_index=dev["index"],
                ) as stream:
                    while len(buf) * 1024 < chunk_frames_loopback and not stop_event.is_set():
                        try:
                            data = stream.read(1024, exception_on_overflow=False)
                            buf.append(np.frombuffer(data, dtype=np.int16))
                        except Exception:
                            break
                if stop_event.is_set():
                    break
                if not buf:
                    continue
                raw = np.concatenate(buf)[:chunk_frames_loopback]
                raw = raw.reshape(-1, ch).astype(np.float64) / 32768.0
                mono = np.mean(raw, axis=1).astype(np.float32)
                if rate != SAMPLE_RATE:
                    num_out = int(round(len(mono) * SAMPLE_RATE / rate))
                    mono = resample(mono, num_out).astype(np.float32)
                mono = mono[:CHUNK_SAMPLES]
                if len(mono) < CHUNK_SAMPLES:
                    mono = np.pad(mono, (0, CHUNK_SAMPLES - len(mono)))
                out_queue.put(mono.copy())
    except Exception:
        if not stop_event.is_set():
            try:
                out_queue.put(("error", "Loopback failed"))
            except queue.Full:
                pass


def _meeting_chunk_ready(app, wav_path: str):
    """Callback from ChunkRecorder (in-process Meeting mode): queue WAV path for transcription."""
    try:
        app.chunk_queue.put_nowait(wav_path)
    except queue.Full:
        pass


def capture_mixer_worker(mic_queue, loopback_queue, chunk_queue, stop_event):
    """
    Read one chunk from mic and one from loopback (sounddevice + pyaudiowpatch);
    mix 50/50 and push to chunk_queue as WAV path. This avoids two WASAPI streams
    in the same library so the mic stays active.
    """
    while not stop_event.is_set():
        try:
            mic_chunk = mic_queue.get(timeout=0.5)
            if isinstance(mic_chunk, tuple) and mic_chunk[0] == "error":
                chunk_queue.put_nowait(("error", mic_chunk[1]))
                continue
        except queue.Empty:
            continue
        try:
            loopback_chunk = loopback_queue.get(timeout=CHUNK_DURATION_SEC + 1)
            if isinstance(loopback_chunk, tuple) and loopback_chunk[0] == "error":
                chunk_queue.put_nowait(("error", loopback_chunk[1]))
                continue
        except queue.Empty:
            continue
        if stop_event.is_set():
            break
        # Align lengths
        m = np.asarray(mic_chunk, dtype=np.float32).flatten()[:CHUNK_SAMPLES]
        l = np.asarray(loopback_chunk, dtype=np.float32).flatten()[:CHUNK_SAMPLES]
        if len(m) < CHUNK_SAMPLES:
            m = np.pad(m, (0, CHUNK_SAMPLES - len(m)))
        if len(l) < CHUNK_SAMPLES:
            l = np.pad(l, (0, CHUNK_SAMPLES - len(l)))
        mixed = (0.5 * m + 0.5 * l).astype(np.float32)
        if _is_silent(mixed):
            continue
        chunk_int16 = (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, chunk_int16)
        try:
            chunk_queue.put(str(CHUNK_PATH), timeout=1.0)
        except queue.Full:
            pass


def mic_pipe_reader_thread(proc, mic_queue, stop_event):
    """
    Read raw float32 PCM from subprocess stdout (mic_capture_subprocess.py).
    Puts numpy float32 arrays of shape (CHUNK_SAMPLES,) into mic_queue.
    """
    chunk_bytes = CHUNK_SAMPLES * 4
    try:
        while not stop_event.is_set():
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            if len(data) < chunk_bytes:
                data = data + b"\x00" * (chunk_bytes - len(data))
            arr = np.frombuffer(data, dtype=np.float32).copy()
            try:
                mic_queue.put(arr, timeout=1.0)
            except queue.Full:
                pass
    except (BrokenPipeError, ValueError, OSError):
        pass
    except Exception:
        if not stop_event.is_set():
            try:
                mic_queue.put_nowait(("error", "Mic pipe read failed"))
            except queue.Full:
                pass


def loopback_pipe_reader_thread(proc, loopback_queue, stop_event):
    """Read float32 PCM from loopback subprocess stdout; put arrays into loopback_queue."""
    chunk_bytes = CHUNK_SAMPLES * 4
    try:
        while not stop_event.is_set():
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            if len(data) < chunk_bytes:
                data = data + b"\x00" * (chunk_bytes - len(data))
            arr = np.frombuffer(data, dtype=np.float32).copy()
            try:
                loopback_queue.put(arr, timeout=1.0)
            except queue.Full:
                pass
    except (BrokenPipeError, ValueError, OSError):
        pass
    except Exception:
        if not stop_event.is_set():
            try:
                loopback_queue.put_nowait(("error", "Loopback pipe read failed"))
            except queue.Full:
                pass


def capture_worker_ffmpeg_meeting(mic_dshow_name, loopback_dshow_name, chunk_queue, stop_event, app=None):
    """
    Run FFmpeg to capture mic + loopback (dshow), mix with amix, output s16le 16kHz mono to pipe.
    Requires ffmpeg on PATH. Device names from: ffmpeg -list_devices true -f dshow -i dummy
    """
    if not (mic_dshow_name and loopback_dshow_name):
        try:
            chunk_queue.put_nowait(("error", "Set ffmpeg_mic_name and ffmpeg_loopback_name in Settings."))
        except queue.Full:
            pass
        return
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "dshow", "-i", f"audio={mic_dshow_name}",
        "-f", "dshow", "-i", f"audio={loopback_dshow_name}",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    except FileNotFoundError:
        try:
            chunk_queue.put_nowait(("error", "FFmpeg not found. Install FFmpeg and add it to PATH."))
        except queue.Full:
            pass
        return
    except Exception as e:
        try:
            chunk_queue.put_nowait(("error", f"FFmpeg start failed: {e}"))
        except queue.Full:
            pass
        return
    if app is not None:
        app.ffmpeg_process = proc
    chunk_bytes = CHUNK_SAMPLES * 2  # s16le
    try:
        while not stop_event.is_set():
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            if len(data) < chunk_bytes:
                data = data + b"\x00" * (chunk_bytes - len(data))
            arr = np.frombuffer(data, dtype=np.int16)
            arr_float = (arr.astype(np.float64) / 32768.0).astype(np.float32)
            if _is_silent(arr_float):
                continue
            wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, arr)
            try:
                chunk_queue.put(str(CHUNK_PATH), timeout=1.0)
            except queue.Full:
                pass
    except (BrokenPipeError, OSError):
        pass
    except Exception as e:
        if not stop_event.is_set():
            try:
                chunk_queue.put_nowait(("error", f"FFmpeg read: {e}"))
            except queue.Full:
                pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            pass
        if app is not None:
            app.ffmpeg_process = None


# -----------------------------------------------------------------------------
# Transcription thread
# -----------------------------------------------------------------------------

def transcription_worker(chunk_queue, text_queue, stop_event, model_id=None):
    """Take WAV paths from chunk_queue, transcribe with selected model, push text to text_queue, delete file."""
    model = get_transcription_model(model_id or PARAKEET_MODEL)
    while not stop_event.is_set():
        try:
            item = chunk_queue.get(timeout=0.5)
            if isinstance(item, tuple) and item[0] == "error":
                text_queue.put_nowait(("[Error] " + item[1] + "\n"))
                continue
            path = item
            if not Path(path).exists():
                continue
            try:
                result = model.recognize(path)
                text = result if isinstance(result, str) else getattr(result, "text", str(result))
                if text and isinstance(text, str) and text.strip():
                    text_queue.put_nowait(text.strip() + "\n")
            finally:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        except queue.Empty:
            continue
        except Exception as e:
            if not stop_event.is_set():
                text_queue.put_nowait(f"[Transcribe error] {e}\n")
            break


# -----------------------------------------------------------------------------
# AI prompts storage and OpenAI summary
# -----------------------------------------------------------------------------

PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.json"
TRANSCRIPT_PLACEHOLDER = "{{transcript}}"


def load_prompts():
    """Load prompt templates from JSON file. Returns list of dicts with id, name, prompt."""
    if not PROMPTS_FILE.exists():
        return []
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_prompts(prompts):
    """Save prompt templates to JSON file."""
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def add_prompt(name, prompt_text):
    """Add a new prompt template. Returns the new prompt dict."""
    prompts = load_prompts()
    p = {"id": str(uuid.uuid4()), "name": name, "prompt": prompt_text}
    prompts.append(p)
    save_prompts(prompts)
    return p


def update_prompt(prompt_id, name, prompt_text):
    """Update an existing prompt. Returns True if found."""
    prompts = load_prompts()
    for p in prompts:
        if p.get("id") == prompt_id:
            p["name"] = name
            p["prompt"] = prompt_text
            save_prompts(prompts)
            return True
    return False


def delete_prompt(prompt_id):
    """Delete a prompt by id. Returns True if found and deleted."""
    prompts = load_prompts()
    new_list = [p for p in prompts if p.get("id") != prompt_id]
    if len(new_list) == len(prompts):
        return False
    save_prompts(new_list)
    return True


def get_prompt_by_id(prompt_id):
    """Return prompt dict by id or None."""
    for p in load_prompts():
        if p.get("id") == prompt_id:
            return p
    return None


def generate_ai_summary(api_key, prompt_template, transcript):
    """
    Call OpenAI API to generate summary. Runs in thread.
    Returns (success, result_text_or_error).
    """
    if not (prompt_template or "").strip():
        return False, "Prompt template is empty."
    if not (transcript or "").strip():
        return False, "Transcript is empty."
    text = prompt_template.replace(TRANSCRIPT_PLACEHOLDER, transcript)
    try:
        from openai import OpenAI
    except ImportError:
        return False, "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
        )
        content = response.choices[0].message.content
        return True, (content or "").strip()
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

def poll_text_queue(app):
    """Called periodically from main thread to append new transcript text."""
    try:
        while True:
            line = app.text_queue.get_nowait()
            app.log.insert("end", line)
            app.log.see("end")
    except queue.Empty:
        pass
    if app.running:
        app.root.after(200, lambda: poll_text_queue(app))


def start_stop(app):
    if app.running:
        app.running = False
        app.stop_event.set()
        # Terminate FFmpeg if used
        fp = getattr(app, "ffmpeg_process", None)
        if fp is not None and fp.poll() is None:
            try:
                fp.terminate()
                fp.wait(timeout=2)
            except Exception:
                pass
            app.ffmpeg_process = None
        # In-process Meeting: stop mixer and flush recorder
        if getattr(app, "mixer", None) is not None:
            try:
                app.mixer.stop()
            except Exception:
                pass
            app.mixer = None
        if getattr(app, "recorder", None) is not None:
            try:
                app.recorder.flush()
            except Exception:
                pass
            app.recorder = None
        # Terminate subprocesses so pipe reader threads can exit (used only for legacy/other modes)
        for name in ("mic_subprocess", "loopback_subprocess"):
            proc = getattr(app, name, None)
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception:
                    pass
                setattr(app, name, None)
        for t in getattr(app, "capture_threads", []) or ([app.capture_thread] if app.capture_thread else []):
            if t and t.is_alive():
                t.join(timeout=CHUNK_DURATION_SEC + 2)
        if app.transcription_thread and app.transcription_thread.is_alive():
            app.transcription_thread.join(timeout=10)
        app.start_btn.configure(state="normal")
        app.stop_btn.configure(state="disabled")
        app.status_var.set("Stopped")
        return
    # Start
    app.stop_event.clear()
    mode, mic_idx, loopback_idx = get_effective_audio_device(app)
    settings = load_settings()

    if mode == AUDIO_MODE_DEFAULT:
        dev_idx, err = get_default_monitor_device()
        if err or dev_idx is None:
            app.log.insert("end", f"[No input device] {err or 'No monitor device found'}\n")
            app.log.see("end")
            return
        app.capture_thread = threading.Thread(
            target=capture_worker,
            args=(dev_idx, app.chunk_queue, app.stop_event),
            daemon=True,
        )
        app.capture_threads = [app.capture_thread]
        app.status_var.set("Recording & transcribing…")
    elif mode == AUDIO_MODE_LOOPBACK:
        if sys.platform != "win32":
            app.log.insert("end", "[Loopback] Only supported on Windows with pyaudiowpatch.\n")
            app.log.see("end")
            return
        app.capture_thread = threading.Thread(
            target=capture_worker_loopback,
            args=(loopback_idx, app.chunk_queue, app.stop_event),
            daemon=True,
        )
        app.capture_threads = [app.capture_thread]
        app.status_var.set("Recording loopback & transcribing…")
    elif mode == AUDIO_MODE_MEETING:
        if sys.platform != "win32":
            app.log.insert("end", "[Meeting] Mic+loopback only supported on Windows.\n")
            app.log.see("end")
            return
        # In-process: single PyAudioWPatch thread, mic blocking + loopback non-blocking (get_read_available)
        try:
            app.recorder = ChunkRecorder(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                chunk_duration_sec=CHUNK_DURATION_SEC,
                asr_sample_rate=SAMPLE_RATE,
                on_chunk_ready=lambda wav_path: _meeting_chunk_ready(app, wav_path),
            )
            app.mixer = AudioMixer(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                frames_per_read=FRAMES_PER_READ_MEETING,
                gain=MIXER_GAIN_MEETING,
            )
            app.mixer.set_stereo_callback(app.recorder.push_stereo)
            # Use default mic (PyAudio index); loopback_idx is from list_loopback_devices (PyAudio)
            app.mixer.start(loopback_device_index=loopback_idx, mic_device_index=None)
            app.capture_thread = None
            app.capture_threads = []
            app.status_var.set("Meeting (mic + loopback) — recording…")
        except Exception as e:
            app.log.insert("end", f"[Meeting] Start failed: {e}\n")
            app.log.see("end")
            if getattr(app, "mixer", None) is not None:
                try:
                    app.mixer.stop()
                except Exception:
                    pass
                app.mixer = None
            app.recorder = None
            return
    elif mode == AUDIO_MODE_MEETING_FFMPEG:
        # FFmpeg captures both sources (dshow), mixes, we read pipe — no Python audio APIs
        mic_name = (mic_idx or "").strip() if isinstance(mic_idx, str) else None
        lb_name = (loopback_idx or "").strip() if isinstance(loopback_idx, str) else None
        if not mic_name or not lb_name:
            app.log.insert("end", "[Meeting FFmpeg] Set FFmpeg mic and loopback device names in Settings. Run: ffmpeg -list_devices true -f dshow -i dummy\n")
            app.log.see("end")
            return
        app.ffmpeg_process = None
        app.capture_thread = threading.Thread(
            target=capture_worker_ffmpeg_meeting,
            args=(mic_name, lb_name, app.chunk_queue, app.stop_event, app),
            daemon=True,
        )
        app.capture_threads = [app.capture_thread]
        app.status_var.set("Meeting (FFmpeg) — recording…")
    else:
        app.log.insert("end", f"[Unknown mode] {mode}\n")
        app.log.see("end")
        return

    if app.capture_thread:
        app.capture_thread.start()
    model_id = app.settings.get("transcription_model") or PARAKEET_MODEL
    app.transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(app.chunk_queue, app.text_queue, app.stop_event, model_id),
        daemon=True,
    )
    app.transcription_thread.start()
    app.running = True
    app.start_btn.configure(state="disabled")
    app.stop_btn.configure(state="normal")
    poll_text_queue(app)


def _open_edit_prompt_dialog(parent, prompt_id, on_saved, ui_pad, ui_radius, font_family, font_sizes, colors):
    """Open dialog to add or edit a prompt template (name + prompt text with {{transcript}})."""
    prompt = get_prompt_by_id(prompt_id) if prompt_id else None
    is_new = prompt is None

    win = ctk.CTkToplevel(parent)
    win.title("Add prompt" if is_new else "Edit prompt")
    win.geometry("520x380")
    win.transient(parent)
    # Defer grab until window is viewable (avoids TclError: grab failed: window not viewable)
    def _set_grab():
        try:
            win.grab_set()
        except Exception:
            pass
    win.after(100, _set_grab)

    ctk.CTkLabel(
        win,
        text="Name",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
    ).pack(anchor="w", padx=ui_pad, pady=(ui_pad, 2))
    name_var = ctk.StringVar(value=prompt.get("name", "") if prompt else "")
    name_entry = ctk.CTkEntry(win, width=400, height=32, font=ctk.CTkFont(family=font_family, size=font_sizes.body))
    name_entry.pack(anchor="w", padx=ui_pad, pady=(0, ui_pad))
    name_entry.insert(0, name_var.get())

    ctk.CTkLabel(
        win,
        text="Prompt (use {{transcript}} where the transcript should go)",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
    ).pack(anchor="w", padx=ui_pad, pady=(ui_pad, 2))
    prompt_text = ctk.CTkTextbox(win, width=500, height=180, font=ctk.CTkFont(family=font_family, size=font_sizes.small))
    prompt_text.pack(anchor="w", padx=ui_pad, pady=(0, ui_pad))
    if prompt:
        prompt_text.insert("1.0", prompt.get("prompt", ""))

    def save():
        name = name_entry.get().strip()
        text = prompt_text.get("1.0", "end").strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a name for the prompt.", parent=win)
            return
        if not text:
            messagebox.showwarning("Missing prompt", "Please enter the prompt text.", parent=win)
            return
        if TRANSCRIPT_PLACEHOLDER not in text:
            if not messagebox.askyesno("No placeholder", f"Prompt does not contain '{TRANSCRIPT_PLACEHOLDER}'. Add it so the transcript is inserted?", parent=win):
                return
        if is_new:
            add_prompt(name, text)
        else:
            update_prompt(prompt_id, name, text)
        on_saved()
        win.destroy()

    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill="x", padx=ui_pad, pady=ui_pad)
    ctk.CTkButton(
        btn_frame,
        text="Save",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
        width=80,
        height=32,
        corner_radius=ui_radius,
        fg_color=colors["primary_fg"],
        hover_color=colors["primary_hover"],
        command=save,
    ).pack(side="left", padx=(0, ui_pad))
    ctk.CTkButton(
        btn_frame,
        text="Cancel",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
        width=80,
        height=32,
        corner_radius=ui_radius,
        fg_color=colors["secondary_fg"],
        hover_color=colors["secondary_hover"],
        command=win.destroy,
    ).pack(side="left")


# -----------------------------------------------------------------------------
# DPI scaling for high-DPI displays (used for CustomTkinter scaling)
# -----------------------------------------------------------------------------


def _get_dpi_scale():
    """Return scale factor for high-DPI displays. Aggressive scaling for laptop readability."""
    scale = 1.0
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        _root.update_idletasks()
        dpi = _root.winfo_fpixels("1i")
        if dpi and dpi > 0:
            scale = max(1.0, min(2.0, dpi / 96.0))
        # Windows: Tk often reports 96 on high-DPI; force big scale for laptops
        if scale <= 1.0 and sys.platform == "win32":
            h = _root.winfo_screenheight()
            if h >= 900:
                scale = 1.85  # big fonts on laptop screens
        _root.destroy()
    except Exception:
        pass
    return scale


def main():
    # Load transcription model before starting the GUI (uses saved preference if set)
    _saved_model = load_settings().get("transcription_model") or PARAKEET_MODEL
    print(f"Loading transcription model '{_saved_model}' (first run may download from Hugging Face)...")
    get_transcription_model(_saved_model)
    print("Model ready. Opening window...")

    # CustomTkinter theme and scaling (before creating window)
    ctk.set_appearance_mode("dark")
    theme_path = Path(__file__).resolve().parent / "themes" / "meetings-dark.json"
    if theme_path.exists():
        ctk.set_default_color_theme(str(theme_path))
    else:
        ctk.set_default_color_theme("dark-blue")
    scale = _get_dpi_scale()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    # Font sizes: clean app-like scale (cap 12–22)
    _fs = lambda base: max(12, min(22, round(base * scale)))
    F = type("F", (), {"title": _fs(18), "header": _fs(16), "body": _fs(14), "small": _fs(13), "tiny": _fs(12)})()

    # UI theme constants (spacing, radius, colors)
    UI_RADIUS = 10
    UI_PAD = 12
    UI_PAD_LG = 16
    COLORS = {
        "sidebar": ("gray88", "gray18"),
        "card": ("gray92", "gray18"),
        "header": ("gray92", "gray18"),
        "primary_fg": ("#3B76FB", "#3B76FB"),
        "primary_hover": ("#2d65e8", "#2d65e8"),
        "danger_fg": ("#c62828", "#b71c1c"),
        "danger_hover": ("#d32f2f", "#c62828"),
        "secondary_fg": ("gray70", "gray35"),
        "secondary_hover": ("gray60", "gray45"),
        "textbox_bg": ("gray97", "gray14"),
        "error_text": ("red", "#f7768e"),
    }
    # Modern UI font per platform (mono kept for transcript only)
    if sys.platform == "win32":
        UI_FONT_FAMILY = "Segoe UI"
        MONO_FONT_FAMILY = "Consolas"
    elif sys.platform == "darwin":
        UI_FONT_FAMILY = "SF Pro Display"
        MONO_FONT_FAMILY = "SF Mono"
    else:
        UI_FONT_FAMILY = "Ubuntu"
        MONO_FONT_FAMILY = "Ubuntu Mono"

    root = ctk.CTk()
    root.title("Blue Bridge Meeting Companion")
    # Custom icon: title bar and Windows taskbar (icon.ico in app folder)
    _base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    _icon = _base / "icon.ico"
    if _icon.exists():
        try:
            root.iconbitmap(str(_icon))
        except Exception:
            pass
    root.geometry("960x480")
    root.minsize(720, 380)

    app = type("App", (), {})()
    app.root = root
    app.running = False
    app.stop_event = threading.Event()
    app.chunk_queue = queue.Queue(maxsize=1)
    app.text_queue = queue.Queue()
    app.capture_thread = None
    app.transcription_thread = None
    app.capture_threads = []
    app.settings = load_settings()

    # Main layout: content only (no sidebar)
    content_frame = ctk.CTkFrame(root, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=UI_PAD, pady=UI_PAD)

    main_content = ctk.CTkFrame(content_frame, fg_color="transparent")
    main_content.pack(fill="both", expand=True)

    # Header bar
    header = ctk.CTkFrame(main_content, fg_color=COLORS["header"], corner_radius=UI_RADIUS, height=52)
    header.pack(fill="x", pady=(0, UI_PAD))
    header.pack_propagate(False)
    app.status_var = ctk.StringVar(value="Ready — click Start to begin")
    ctk.CTkLabel(
        header,
        textvariable=app.status_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(side="left", padx=UI_PAD_LG, pady=UI_PAD)
    btn_frame = ctk.CTkFrame(header, fg_color="transparent")
    btn_frame.pack(side="right", padx=UI_PAD_LG, pady=UI_PAD)
    app.start_btn = ctk.CTkButton(
        btn_frame,
        text="Start",
        command=lambda: start_stop(app),
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
        width=100,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["primary_fg"],
        hover_color=COLORS["primary_hover"],
    )
    app.start_btn.pack(side="left", padx=(0, UI_PAD))
    app.stop_btn = ctk.CTkButton(
        btn_frame,
        text="Stop",
        command=lambda: start_stop(app),
        state="disabled",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
        width=100,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["danger_fg"],
        hover_color=COLORS["danger_hover"],
    )
    app.stop_btn.pack(side="left")

    # Tabview: Transcript | AI Prompts | Settings | Models
    tabview = ctk.CTkTabview(main_content, fg_color=COLORS["card"], corner_radius=UI_RADIUS)
    tabview.pack(fill="both", expand=True, pady=(0, UI_PAD))
    tab_transcript = tabview.add("Transcript")
    tab_prompts = tabview.add("AI Prompts")
    tab_settings = tabview.add("Settings")
    tab_models = tabview.add("Models")

    # ---- Models tab: select model for transcription + installed models list ----
    models_card = ctk.CTkFrame(tab_models, fg_color="transparent")
    models_card.pack(fill="both", expand=True)
    ctk.CTkLabel(
        models_card,
        text="Installed models",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.title, weight="bold"),
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 6))
    # Row: Transcription model selector
    model_selector_row = ctk.CTkFrame(models_card, fg_color="transparent")
    model_selector_row.pack(fill="x", padx=UI_PAD_LG, pady=(0, UI_PAD))
    ctk.CTkLabel(
        model_selector_row,
        text="Transcription model:",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    ).pack(side="left", padx=(0, UI_PAD))
    app.model_selector_var = ctk.StringVar(value=app.settings.get("transcription_model") or PARAKEET_MODEL)

    def on_model_selected(choice):
        app.settings["transcription_model"] = choice
        save_settings(app.settings)
        clear_transcription_model_cache()

    app.model_selector = ctk.CTkOptionMenu(
        model_selector_row,
        variable=app.model_selector_var,
        values=[],  # populated in refresh_models_tab
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=320,
        command=on_model_selected,
    )
    app.model_selector.pack(side="left")
    models_scroll = ctk.CTkScrollableFrame(models_card, fg_color="transparent")
    models_scroll.pack(fill="both", expand=True)

    def refresh_models_tab():
        for w in models_scroll.winfo_children():
            w.destroy()
        models, err = list_installed_transcription_models()
        # Update transcription model dropdown
        repo_ids = [m["repo_id"] for m in models] if models else []
        current = app.settings.get("transcription_model") or PARAKEET_MODEL
        if current not in repo_ids and repo_ids:
            repo_ids.insert(0, current)  # show saved model even if not in cache (e.g. not yet downloaded)
        elif not repo_ids:
            repo_ids = [current] if current else [PARAKEET_MODEL]
        app.model_selector.configure(values=repo_ids)
        chosen = current if current in repo_ids else (repo_ids[0] if repo_ids else PARAKEET_MODEL)
        app.model_selector_var.set(chosen)
        if chosen != current:
            app.settings["transcription_model"] = chosen
            save_settings(app.settings)
            clear_transcription_model_cache()
        if err:
            ctk.CTkLabel(
                models_scroll,
                text=f"Error: {err[:50]}…" if len(err) > 50 else err,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                text_color=COLORS["error_text"],
                wraplength=400,
            ).pack(anchor="w", padx=UI_PAD, pady=4)
            return
        if not models:
            ctk.CTkLabel(
                models_scroll,
                text="No transcription models in cache.",
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                text_color="gray",
                wraplength=400,
            ).pack(anchor="w", padx=UI_PAD, pady=4)
            return
        for m in models:
            row = ctk.CTkFrame(models_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(
                row,
                text=f"{m['repo_id']}\n{m['size_str']}",
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                wraplength=400,
                anchor="w",
            ).pack(side="left", padx=(UI_PAD, 4))
            def _uninstall(repo_id=m["repo_id"], hashes=m["revision_hashes"]):
                if not messagebox.askyesno("Uninstall model", f"Delete cached model '{repo_id}'? This frees disk space; you can re-download later."):
                    return
                ok, err = uninstall_transcription_model(repo_id, hashes)
                if ok:
                    messagebox.showinfo("Uninstalled", f"Removed {repo_id} from cache.")
                    refresh_models_tab()
                else:
                    messagebox.showerror("Error", err or "Failed to delete.")
            ctk.CTkButton(
                row,
                text="Uninstall",
                width=70,
                height=28,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
                corner_radius=UI_RADIUS,
                fg_color=COLORS["danger_fg"],
                hover_color=COLORS["danger_hover"],
                command=_uninstall,
            ).pack(side="right", padx=(0, UI_PAD))

    refresh_models_tab()
    ctk.CTkButton(
        models_card,
        text="Refresh list",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=refresh_models_tab,
    ).pack(pady=(4, UI_PAD))

    # ---- Transcript tab: Transcript (left) | Summary (right) ----
    card = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    card.pack(fill="both", expand=True)

    # Left panel: Transcript
    transcript_panel = ctk.CTkFrame(card, fg_color="transparent")
    transcript_panel.pack(side="left", fill="both", expand=True, padx=(UI_PAD_LG, UI_PAD))
    card_header = ctk.CTkFrame(transcript_panel, fg_color="transparent")
    card_header.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(
        card_header,
        text="Transcript",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(side="left")
    def copy_transcript():
        text = app.log.get("1.0", "end")
        text = text.rstrip()
        if text:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()  # keep clipboard content after window loses focus
    def clear_transcript():
        app.log.delete("1.0", "end")
    ctk.CTkButton(
        card_header,
        text="Copy transcript",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=140,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=copy_transcript,
    ).pack(side="left", padx=(UI_PAD, 0), pady=4)
    ctk.CTkButton(
        card_header,
        text="Clear",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=80,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=clear_transcript,
    ).pack(side="right", padx=UI_PAD, pady=4)
    app.log = ctk.CTkTextbox(
        transcript_panel,
        wrap="word",
        font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body),
        corner_radius=8,
        border_width=0,
        fg_color=COLORS["textbox_bg"],
        border_spacing=UI_PAD,
    )
    app.log.pack(fill="both", expand=True, pady=(0, UI_PAD))

    # Right panel: AI Summary
    summary_panel = ctk.CTkFrame(card, fg_color="transparent")
    summary_panel.pack(side="left", fill="both", expand=True, padx=(UI_PAD, UI_PAD_LG))
    summary_header = ctk.CTkFrame(summary_panel, fg_color="transparent")
    summary_header.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(
        summary_header,
        text="AI Summary",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(side="left")
    # Prompt template selector (refresh list from prompts file)
    _prompts_for_summary = load_prompts()
    _prompt_names = [p.get("name", "Unnamed") for p in _prompts_for_summary] or ["(No prompts — add in AI Prompts tab)"]
    app.summary_prompt_var = ctk.StringVar(value=_prompt_names[0])
    app.summary_prompt_menu = ctk.CTkOptionMenu(
        summary_header,
        values=_prompt_names,
        variable=app.summary_prompt_var,
        width=220,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    )
    app.summary_prompt_menu.pack(side="left", padx=(UI_PAD, 0), pady=4)
    app.summary_status_var = ctk.StringVar(value="")
    ctk.CTkLabel(
        summary_header,
        textvariable=app.summary_status_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
    ).pack(side="left", padx=(UI_PAD, 0), pady=4)

    def _refresh_summary_prompt_menu():
        prompts = load_prompts()
        names = [p.get("name", "Unnamed") for p in prompts] or ["(No prompts — add in AI Prompts tab)"]
        app.summary_prompt_menu.configure(values=names)
        if names and (app.summary_prompt_var.get() not in names):
            app.summary_prompt_var.set(names[0])

    def _do_ai_summary():
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            messagebox.showerror("API key required", "Set OPENAI_API_KEY in your environment or .env file.", parent=root)
            return
        prompts = load_prompts()
        if not prompts:
            messagebox.showinfo("No prompts", "Create at least one prompt in the 'AI Prompts' tab first.", parent=root)
            return
        name = app.summary_prompt_var.get()
        prompt_obj = next((p for p in prompts if p.get("name") == name), None)
        if not prompt_obj or name.startswith("("):
            messagebox.showwarning("Select a prompt", "Choose a prompt template from the dropdown.", parent=root)
            return
        transcript = app.log.get("1.0", "end").strip()
        if not transcript:
            messagebox.showwarning("Empty transcript", "Transcript is empty. Record or paste some text first.", parent=root)
            return
        app.summary_generate_btn.configure(state="disabled")
        app.summary_status_var.set("Generating…")
        result_holder = []

        def worker():
            ok, out = generate_ai_summary(api_key, prompt_obj["prompt"], transcript)
            result_holder.append((ok, out))

        def check_done():
            if not result_holder:
                root.after(200, check_done)
                return
            ok, out = result_holder[0]
            app.summary_generate_btn.configure(state="normal")
            app.summary_status_var.set("")
            if ok:
                app.summary_text.delete("1.0", "end")
                app.summary_text.insert("1.0", out)
            else:
                messagebox.showerror("AI Summary failed", out, parent=root)

        threading.Thread(target=worker, daemon=True).start()
        root.after(200, check_done)

    app.summary_generate_btn = ctk.CTkButton(
        summary_header,
        text="Generate",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=100,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["primary_fg"],
        hover_color=COLORS["primary_hover"],
        command=_do_ai_summary,
    )
    app.summary_generate_btn.pack(side="left", padx=(UI_PAD, 0), pady=4)

    def _export_markdown():
        summary = app.summary_text.get("1.0", "end").strip()
        transcript = app.log.get("1.0", "end").strip()
        if not summary and not transcript:
            messagebox.showwarning("Nothing to export", "Add an AI summary and/or transcript first.", parent=root)
            return
        name_part = (app.export_name_var.get() or "").strip()
        # Sanitize for filename: replace spaces and invalid chars with hyphen
        if name_part:
            name_part = "".join(c if c.isalnum() or c in "._- " else "-" for c in name_part)
            name_part = name_part.replace(" ", "-").strip("-") or "export"
        else:
            name_part = "export"
        default_name = f"{date.today().isoformat()} {name_part}.md"
        path = filedialog.asksaveasfilename(
            parent=root,
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return
        parts = []
        if summary:
            parts.append("## Summary\n\n" + summary)
        if transcript:
            parts.append("## Transcript\n\n" + transcript)
        content = "\n\n---\n\n".join(parts) + "\n"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Exported", f"Saved to {path}", parent=root)
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=root)

    app.summary_text = ctk.CTkTextbox(
        summary_panel,
        wrap="word",
        font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body),
        corner_radius=8,
        border_width=0,
        fg_color=COLORS["textbox_bg"],
        border_spacing=UI_PAD,
    )
    app.summary_text.pack(fill="both", expand=True, pady=(0, UI_PAD))

    # Full-width row below Transcript and Summary: Export
    export_row = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    export_row.pack(fill="x", pady=(UI_PAD, UI_PAD_LG), padx=UI_PAD_LG)
    ctk.CTkLabel(
        export_row,
        text="Export:",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    ).pack(side="left", padx=(UI_PAD, 4), pady=4)
    app.export_name_var = ctk.StringVar(value="")
    app.export_name_entry = ctk.CTkEntry(
        export_row,
        textvariable=app.export_name_var,
        width=160,
        height=28,
        placeholder_text="e.g. meeting-notes",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    )
    app.export_name_entry.pack(side="left", padx=4, pady=4)
    ctk.CTkButton(
        export_row,
        text="Export",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=80,
        height=28,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=_export_markdown,
    ).pack(side="left", padx=(0, 0), pady=4)

    # ---- AI Prompts tab: list, add, edit, delete ----
    ctk.CTkLabel(
        tab_prompts,
        text="Custom prompt templates for AI Summary. Use {{transcript}} where the transcript should be inserted.",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
        wraplength=500,
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    prompts_scroll = ctk.CTkScrollableFrame(tab_prompts, fg_color="transparent")
    prompts_scroll.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=UI_PAD)

    def refresh_prompts_list():
        for w in prompts_scroll.winfo_children():
            w.destroy()
        prompts = load_prompts()
        _refresh_summary_prompt_menu()  # keep Transcript tab prompt dropdown in sync
        if not prompts:
            ctk.CTkLabel(
                prompts_scroll,
                text="No prompts yet. Click 'Add prompt' to create one.",
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                text_color="gray",
            ).pack(anchor="w", padx=UI_PAD, pady=8)
        for p in prompts:
            row = ctk.CTkFrame(prompts_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(
                row,
                text=p.get("name", "Unnamed"),
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.body, weight="bold"),
                anchor="w",
            ).pack(side="left", padx=(UI_PAD, 8))
            def _edit(pid=p["id"]):
                _open_edit_prompt_dialog(root, pid, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)
            def _delete(pid=p["id"], pname=p.get("name", "?")):
                if not messagebox.askyesno("Delete prompt", f"Delete prompt '{pname}'?"):
                    return
                if delete_prompt(pid):
                    refresh_prompts_list()
                    messagebox.showinfo("Deleted", "Prompt deleted.")
            ctk.CTkButton(
                row,
                text="Edit",
                width=60,
                height=28,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
                corner_radius=UI_RADIUS,
                fg_color=COLORS["secondary_fg"],
                hover_color=COLORS["secondary_hover"],
                command=_edit,
            ).pack(side="right", padx=4)
            ctk.CTkButton(
                row,
                text="Delete",
                width=60,
                height=28,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
                corner_radius=UI_RADIUS,
                fg_color=COLORS["danger_fg"],
                hover_color=COLORS["danger_hover"],
                command=_delete,
            ).pack(side="right", padx=(0, UI_PAD))

    def add_new_prompt():
        _open_edit_prompt_dialog(root, None, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)

    refresh_prompts_list()
    ctk.CTkButton(
        tab_prompts,
        text="Add prompt",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        corner_radius=UI_RADIUS,
        fg_color=COLORS["primary_fg"],
        hover_color=COLORS["primary_hover"],
        command=add_new_prompt,
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(4, UI_PAD))

    # ---- Settings tab: capture mode, meeting mic, loopback device ----
    settings_card = ctk.CTkFrame(tab_settings, fg_color="transparent")
    settings_card.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=UI_PAD)
    ctk.CTkLabel(
        settings_card,
        text="Capture mode",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(anchor="w", pady=(0, 4))
    ctk.CTkLabel(
        settings_card,
        text="Meeting = in-process mic + loopback (single PyAudioWPatch thread; loopback read only when data available). Meeting (FFmpeg) = FFmpeg captures and mixes both. Loopback device below applies to Meeting mode.",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
        wraplength=520,
    ).pack(anchor="w", pady=(0, UI_PAD))
    mode_values = ["Default input", "Loopback (system audio)", "Meeting (mic + loopback)", "Meeting (FFmpeg)"]
    mode_to_val = {AUDIO_MODE_DEFAULT: mode_values[0], AUDIO_MODE_LOOPBACK: mode_values[1], AUDIO_MODE_MEETING: mode_values[2], AUDIO_MODE_MEETING_FFMPEG: mode_values[3]}
    val_to_mode = {v: k for k, v in mode_to_val.items()}
    app.audio_mode_var = ctk.StringVar(value=mode_to_val.get(app.settings.get("audio_mode"), mode_values[0]))
    app.audio_mode_menu = ctk.CTkOptionMenu(
        settings_card,
        values=mode_values,
        variable=app.audio_mode_var,
        width=320,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        command=lambda v: _apply_settings(app),
    )
    app.audio_mode_menu.pack(anchor="w", pady=(0, UI_PAD_LG))

    ctk.CTkLabel(
        settings_card,
        text="Meeting microphone (used when Capture mode = Meeting)",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small, weight="bold"),
    ).pack(anchor="w", pady=(UI_PAD, 4))
    input_devices, _ = list_audio_devices()
    input_options = ["System Default"]
    input_options += [f"{d['index']}: {d['name']}" for d in input_devices if d.get("max_input_channels", 0) > 0]
    app.meeting_mic_var = ctk.StringVar(value="System Default")
    meeting_mic_idx = app.settings.get("meeting_mic_device")
    if meeting_mic_idx is not None:
        for d in input_devices:
            if d["index"] == meeting_mic_idx:
                app.meeting_mic_var.set(f"{d['index']}: {d['name']}")
                break
    app.meeting_mic_menu = ctk.CTkOptionMenu(
        settings_card,
        values=input_options,
        variable=app.meeting_mic_var,
        width=400,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        command=lambda v: _apply_settings(app),
    )
    app.meeting_mic_menu.pack(anchor="w", pady=(0, UI_PAD_LG))

    ctk.CTkLabel(
        settings_card,
        text="Loopback device (used for Loopback mode and for Meeting mode)",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small, weight="bold"),
    ).pack(anchor="w", pady=(UI_PAD, 4))
    loopback_devices, lb_err = list_loopback_devices()
    loopback_options = ["System Default"]
    loopback_options += [f"{d['index']}: {d['name']}" for d in loopback_devices]
    if lb_err:
        loopback_options = ["System Default", f"(Error: {lb_err})"]
    app.loopback_device_var = ctk.StringVar(value="System Default")
    lb_idx = app.settings.get("loopback_device_index")
    if lb_idx is not None and loopback_devices:
        for d in loopback_devices:
            if d["index"] == lb_idx:
                app.loopback_device_var.set(f"{d['index']}: {d['name']}")
                break
    app.loopback_device_menu = ctk.CTkOptionMenu(
        settings_card,
        values=loopback_options,
        variable=app.loopback_device_var,
        width=400,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        command=lambda v: _apply_settings(app),
    )
    app.loopback_device_menu.pack(anchor="w", pady=(0, UI_PAD))

    ctk.CTkLabel(
        settings_card,
        text="Meeting (FFmpeg): dshow device names (run: ffmpeg -list_devices true -f dshow -i dummy)",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small, weight="bold"),
    ).pack(anchor="w", pady=(UI_PAD, 4))
    app.ffmpeg_mic_var = ctk.StringVar(value=app.settings.get("ffmpeg_mic_name") or "")
    app.ffmpeg_loopback_var = ctk.StringVar(value=app.settings.get("ffmpeg_loopback_name") or "")
    app.ffmpeg_mic_entry = ctk.CTkEntry(settings_card, textvariable=app.ffmpeg_mic_var, width=420, placeholder_text="e.g. Microphone (Realtek)")
    app.ffmpeg_mic_entry.pack(anchor="w", pady=(0, 4))
    app.ffmpeg_loopback_entry = ctk.CTkEntry(settings_card, textvariable=app.ffmpeg_loopback_var, width=420, placeholder_text="e.g. Stereo Mix (Realtek) or VoiceMeeter Output")
    app.ffmpeg_loopback_entry.pack(anchor="w", pady=(0, UI_PAD))
    for _e in (app.ffmpeg_mic_entry, app.ffmpeg_loopback_entry):
        _e.bind("<FocusOut>", lambda _a, _app=app: _apply_settings(_app))

    def _apply_settings(app):
        mode_str = app.audio_mode_var.get()
        mode = val_to_mode.get(mode_str, AUDIO_MODE_DEFAULT)
        meeting_val = app.meeting_mic_var.get()
        meeting_idx = None
        if meeting_val and meeting_val != "System Default" and ":" in meeting_val:
            try:
                meeting_idx = int(meeting_val.split(":")[0].strip())
            except ValueError:
                pass
        loopback_val = app.loopback_device_var.get()
        loopback_idx = None
        if loopback_val and loopback_val != "System Default" and ":" in loopback_val and not loopback_val.startswith("(Error:"):
            try:
                loopback_idx = int(loopback_val.split(":")[0].strip())
            except ValueError:
                pass
        ffmpeg_mic = (app.ffmpeg_mic_var.get() or "").strip()
        ffmpeg_loopback = (app.ffmpeg_loopback_var.get() or "").strip()
        app.settings = {"audio_mode": mode, "meeting_mic_device": meeting_idx, "loopback_device_index": loopback_idx, "ffmpeg_mic_name": ffmpeg_mic or None, "ffmpeg_loopback_name": ffmpeg_loopback or None}
        save_settings(app.settings)

    # Footer: device info and capture mode
    mode = app.settings.get("audio_mode") or AUDIO_MODE_DEFAULT
    mode_label = {"default": "Default input", "loopback": "Loopback", "meeting": "Meeting (mic + loopback)", "meeting_ffmpeg": "Meeting (FFmpeg)"}.get(mode, "Default input")
    devices, _ = list_audio_devices()
    dev_idx, dev_err = get_default_monitor_device()
    if dev_err:
        dev_info = f"Capture: {mode_label} — {dev_err}"
    elif dev_idx is not None and devices:
        name = next((d["name"] for d in devices if d["index"] == dev_idx), f"Device {dev_idx}")
        dev_info = f"Capture: {mode_label} · Input: {name}"
    else:
        dev_info = f"Capture: {mode_label}"
    ctk.CTkLabel(
        main_content,
        text=dev_info,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(0, UI_PAD))

    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_event.set(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
