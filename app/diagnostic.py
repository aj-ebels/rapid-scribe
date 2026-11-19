"""
Lightweight diagnostic log for the built (frozen) app.
Writes to %APPDATA%\\Meetings\\diagnostic.log when running as exe or when MEETINGS_DEBUG=1.
Use this to see where the capture/transcription pipeline fails when there is no console.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Rotate diagnostic.log so a long webinar cannot grow it without bound (field log was 16 MB).
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 3
_ready = False


def _get_log_path():
    override = os.environ.get("MEETINGS_DIAGNOSTIC_LOG", "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings" / "diagnostic.log"


def log_file_path() -> Path:
    """Path to diagnostic.log (whether file logging is active)."""
    return _get_log_path()


def _enabled():
    if getattr(sys, "frozen", False):
        return True
    return os.environ.get("MEETINGS_DEBUG", "").strip().lower() in ("1", "true", "yes")


def rotate_log(path: Path, *, max_bytes: int = _MAX_BYTES, backups: int = _BACKUPS) -> None:
    """Rename path to path.1 .. path.N when it is already over max_bytes. Keeps `backups` old files."""
    try:
        if path.stat().st_size <= max_bytes:
            return
    except OSError:
        return
    oldest = Path(f"{path}.{backups}")
    if oldest.exists():
        oldest.unlink()
    for i in range(backups - 1, 0, -1):
        src = Path(f"{path}.{i}")
        if src.exists():
            src.replace(Path(f"{path}.{i + 1}"))
    if path.exists():
        path.replace(Path(f"{path}.1"))


def init():
    """Call once at startup. Enables file logging when frozen or MEETINGS_DEBUG=1."""
    global _ready
    if _ready or not _enabled():
        return
    _ready = True
    write("---", "start", frozen=getattr(sys, "frozen", False), cwd=os.getcwd())


def write(*parts, **kwargs):
    """Append one line to the diagnostic log (when enabled)."""
    global _ready
    if not _enabled():
        return
    if not _ready:
        init()
        if not _ready:
            return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        extra = " ".join(f"{k}={v!r}" for k, v in kwargs.items()) if kwargs else ""
        line = " ".join(str(p) for p in parts) + (" " + extra if extra else "") + "\n"
        path = _get_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        rotate_log(path)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{ts} | {line}")
    except Exception:
        pass
