"""
Secure storage for OpenAI API key: per-user app data directory, owner-only file.
Key is never stored in settings.json or in the app folder.
"""
import os
import sys
from pathlib import Path


def _get_app_data_dir():
    """Return platform-specific app data directory for this app (e.g. for installed users)."""
    if os.name == "nt":  # Windows
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":  # macOS
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings"


def _get_key_path():
    return _get_app_data_dir() / "openai_api_key"


def get_openai_api_key():
    """
    Read OpenAI API key from the app's secure storage (per-user file).
    Returns the key string or empty string if not set or on read error.
    """
    path = _get_key_path()
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (f.read() or "").strip()
    except Exception:
        return ""


def set_openai_api_key(api_key):
    """
    Store OpenAI API key in the app's secure storage.
    Creates the app data directory if needed and sets file to owner-only where supported.
    Returns True on success, False on error.
    """
    key = (api_key or "").strip()
    dir_path = _get_key_path().parent
    file_path = _get_key_path()
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(key)
        # Restrict to owner read/write on Unix (no effect on Windows)
        try:
            os.chmod(file_path, 0o600)
        except Exception:
            pass
        return True
    except Exception:
        return False


def clear_openai_api_key():
    """Remove the stored API key file. Returns True if removed or not present."""
    path = _get_key_path()
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False
