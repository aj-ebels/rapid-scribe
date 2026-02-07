"""
App settings: JSON file path, audio mode constants, load/save.
When frozen (built app), use per-user app data so settings are writable (e.g. from Program Files).
"""
import json
import os
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent


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
