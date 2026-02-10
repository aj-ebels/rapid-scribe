"""
App settings: JSON file path, audio mode constants, load/save.
When frozen (built app), use per-user app data so settings are writable (e.g. from Program Files).
"""
import json
import os
import sys
from pathlib import Path

# Project root (parent of app/) for settings.json in dev
_BASE = Path(__file__).resolve().parent.parent


def _get_settings_path():
    """Path to settings.json; use app data when running as built exe so writes always succeed."""
    if getattr(sys, "frozen", False):
        if os.name == "nt":
            base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = str(Path.home() / "Library" / "Application Support")
        else:
            base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(base) / "Meetings" / "settings.json"
    return _BASE / "settings.json"


SETTINGS_FILE = _get_settings_path()

# Default transcription model (Hugging Face repo id)
DEFAULT_TRANSCRIPTION_MODEL = "istupakov/parakeet-tdt-0.6b-v2-onnx"

AUDIO_MODE_DEFAULT = "default"
AUDIO_MODE_LOOPBACK = "loopback"
AUDIO_MODE_MEETING = "meeting"


def load_settings(default_model=None):
    """Load settings from JSON. Returns dict with audio_mode, meeting_mic_device, loopback_device_index, transcription_model."""
    model = default_model or DEFAULT_TRANSCRIPTION_MODEL
    out = {
        "audio_mode": AUDIO_MODE_MEETING,
        "meeting_mic_device": None,
        "loopback_device_index": None,
        "transcription_model": model,
        "chunk_duration_sec": 5.0,
        "use_silence_chunking": True,
        "min_chunk_sec": 1.5,
        "max_chunk_sec": 8.0,
        "silence_duration_sec": 0.5,
        "min_rms_transcribe": 0.005,
        "auto_generate_export_name": True,
        "export_prepend_date": True,
        "auto_generate_summary_when_stopping": False,
        "auto_save_delay_seconds": 1,
        "ui_scale": None,  # None = auto (DPI-based); else float 0.5–2.0 for manual scale
    }
    if not SETTINGS_FILE.exists():
        return out
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out["audio_mode"] = data.get("audio_mode", out["audio_mode"])
            out["meeting_mic_device"] = data.get("meeting_mic_device")
            out["loopback_device_index"] = data.get("loopback_device_index")
            out["transcription_model"] = data.get("transcription_model", model) or model
            if "chunk_duration_sec" in data and isinstance(data["chunk_duration_sec"], (int, float)):
                out["chunk_duration_sec"] = max(3.0, min(30.0, float(data["chunk_duration_sec"])))
            if "use_silence_chunking" in data:
                out["use_silence_chunking"] = bool(data["use_silence_chunking"])
            if "min_chunk_sec" in data and isinstance(data["min_chunk_sec"], (int, float)):
                out["min_chunk_sec"] = max(0.5, min(10.0, float(data["min_chunk_sec"])))
            if "max_chunk_sec" in data and isinstance(data["max_chunk_sec"], (int, float)):
                out["max_chunk_sec"] = max(3.0, min(60.0, float(data["max_chunk_sec"])))
            if "silence_duration_sec" in data and isinstance(data["silence_duration_sec"], (int, float)):
                out["silence_duration_sec"] = max(0.2, min(2.0, float(data["silence_duration_sec"])))
            if "min_rms_transcribe" in data and isinstance(data["min_rms_transcribe"], (int, float)):
                out["min_rms_transcribe"] = max(0.001, min(0.05, float(data["min_rms_transcribe"])))
            if "auto_generate_export_name" in data:
                out["auto_generate_export_name"] = bool(data["auto_generate_export_name"])
            if "export_prepend_date" in data:
                out["export_prepend_date"] = bool(data["export_prepend_date"])
            if "auto_generate_summary_when_stopping" in data:
                out["auto_generate_summary_when_stopping"] = bool(data["auto_generate_summary_when_stopping"])
            if "auto_save_delay_seconds" in data and isinstance(data["auto_save_delay_seconds"], (int, float)):
                out["auto_save_delay_seconds"] = max(1, min(30, int(float(data["auto_save_delay_seconds"]))))
            if "ui_scale" in data:
                v = data["ui_scale"]
                if v is None:
                    out["ui_scale"] = None
                elif isinstance(v, (int, float)):
                    out["ui_scale"] = max(0.5, min(2.0, float(v)))
    except Exception:
        pass
    return out


def save_settings(settings):
    """Write settings to JSON."""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass
