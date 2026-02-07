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

from gui import main

if __name__ == "__main__":
    main()
