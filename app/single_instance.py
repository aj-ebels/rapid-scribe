"""
Single-instance lock so only one app process runs at a time.
Prevents double-launch (e.g. user clicking the icon again) which can cause
mic/loopback capture to fail when multiple processes compete for the same devices.
"""
import os
import sys
from pathlib import Path

# Hold reference so the lock file stays open for the life of the process
_lock_file = None


def _get_lock_dir():
    """Directory for the lock file; same convention as settings (frozen vs dev)."""
    if getattr(sys, "frozen", False):
        if os.name == "nt":
            base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = str(Path.home() / "Library" / "Application Support")
        else:
            base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(base) / "Meetings"
    return Path(__file__).resolve().parent.parent


def acquire_single_instance_lock():
    """
    Try to acquire the single-instance lock. Only one process can hold it.
    Returns True if this process got the lock (first instance), False if another
    instance is already running. When True, the lock is held until process exit.
    """
    global _lock_file
    lock_dir = _get_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "single_instance.lock"

    try:
        # Open for write; create if not exist. We need a real file for locking.
        fd = os.open(
            str(lock_path),
            os.O_RDWR | os.O_CREAT,
            0o644,
        )
    except OSError:
        return False

    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                os.close(fd)
                return False
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError):
                os.close(fd)
                return False
    except ImportError:
        # No msvcrt/fcntl (unusual); fallback: close and allow run (no lock)
        os.close(fd)
        return True

    # We got the lock. Keep the file open so the lock is held.
    _lock_file = fd
    return True
