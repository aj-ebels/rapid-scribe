#!/usr/bin/env python3
"""
Real-time system audio capture and transcription.
Uses PulseAudio Monitor / RDPSource on WSL2, NVIDIA Parakeet (onnx-asr), and CustomTkinter GUI.
Same Parakeet/ONNX stack as Meetily: https://github.com/Zackriya-Solutions/meeting-minutes
"""
import os
import sys

# WSL: ensure DISPLAY for GUI (no-op on Windows)
try:
    release = os.uname().release
    if "WSL" in release or "microsoft" in release.lower():
        if "DISPLAY" not in os.environ or not os.environ["DISPLAY"].strip():
            os.environ.setdefault("DISPLAY", ":0")
except AttributeError:
    pass

if __name__ == "__main__":
    import multiprocessing
    # Required for frozen (PyInstaller) apps on Windows so multiprocessing spawn works correctly.
    multiprocessing.freeze_support()
    # On Windows, spawn re-imports this module in the transcription subprocess. Only run the GUI
    # in the main process so we don't open a second window when record is pressed.
    if multiprocessing.current_process().name == "MainProcess":
        from app.diagnostic import init as init_diagnostic
        from app.gui import main
        init_diagnostic()
        main()
