"""
Lightweight diagnostic log for the built (frozen) app.
Writes to %APPDATA%\\Meetings\\diagnostic.log when running as exe or when MEETINGS_DEBUG=1.
Use this to see where the capture/transcription pipeline fails when there is no console.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

_log_file = None


def _get_log_path():
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings" / "diagnostic.log"


def _enabled():
    if getattr(sys, "frozen", False):
        return True
    return os.environ.get("MEETINGS_DEBUG", "").strip().lower() in ("1", "true", "yes")


def init():
    """Call once at startup. Enables file logging when frozen or MEETINGS_DEBUG=1."""
    global _log_file
    if _log_file is not None or not _enabled():
        return
    try:
        log_path = _get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(log_path, "a", encoding="utf-8")
        write("---", "start", frozen=getattr(sys, "frozen", False), cwd=os.getcwd())
    except Exception:
        _log_file = None


def write(*parts, **kwargs):
    """Append one line to the diagnostic log (when enabled). Flush so user can tail the file."""
    global _log_file
    if _log_file is None:
        if _enabled():
            init()
        if _log_file is None:
            return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        extra = " ".join(f"{k}={v!r}" for k, v in kwargs.items()) if kwargs else ""
        line = " ".join(str(p) for p in parts) + (" " + extra if extra else "") + "\n"
        _log_file.write(f"{ts} | {line}")
        _log_file.flush()
    except Exception:
        pass
