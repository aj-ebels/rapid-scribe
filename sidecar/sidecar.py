#!/usr/bin/env python3
"""
Headless Rapid Scribe sidecar: JSON-line IPC on stdin/stdout.
Spawned by Electron; replaces the CustomTkinter GUI entry point.
"""
import multiprocessing

# Required before any Process spawn (Windows).
multiprocessing.freeze_support()

import os
import sys

# Project root on path when running unfrozen
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _redirect_stderr():
    """Log stderr to %APPDATA%\\Meetings\\sidecar.log (plan: Rapid Scribe; keep Meetings path)."""
    if getattr(sys, "frozen", False) or os.environ.get("MEETINGS_SIDECAR_LOG", "").strip():
        try:
            from pathlib import Path

            if os.name == "nt":
                base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
            else:
                base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
            log_dir = Path(base) / "Meetings"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "sidecar.log"
            sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            pass


def main():
    if multiprocessing.current_process().name != "MainProcess":
        return
    _redirect_stderr()
    from app.ipc import IpcServer

    server = IpcServer()
    server.run()


if __name__ == "__main__":
    main()
