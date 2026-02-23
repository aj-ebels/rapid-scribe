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
        from app.single_instance import acquire_single_instance_lock
        if not acquire_single_instance_lock():
            from tkinter import Tk, messagebox
            root = Tk()
            root.withdraw()
            messagebox.showinfo(
                "Rapid Scribe already running",
                "Another instance of Rapid Scribe is already running.\n\n"
                "Only one instance can run at a time so that microphone and system audio "
                "capture work correctly. Please use the existing window, or close it and try again."
            )
            root.destroy()
            sys.exit(0)
        # Show a loading splash immediately so the user sees the app has launched
        from tkinter import Tk, Label, font as tkfont
        _splash = Tk()
        _splash.title("Rapid Scribe")
        _splash.resizable(False, False)
        _splash.geometry("320x100")
        _splash.configure(bg="#2b2b2b")
        try:
            _splash.attributes("-topmost", True)
        except Exception:
            pass
        _f = tkfont.nametofont("TkDefaultFont")
        _title_font = (_f.actual()["family"], 14, "bold")
        _splash.option_add("*Font", _f)
        Label(
            _splash, text="Rapid Scribe",
            bg="#2b2b2b", fg="white", font=_title_font
        ).pack(pady=(16, 4))
        Label(
            _splash, text="Loading…",
            bg="#2b2b2b", fg="#b0b0b0", font=(_f.actual()["family"], 10)
        ).pack(pady=(0, 16))
        _splash.update_idletasks()
        _x = (_splash.winfo_screenwidth() - 320) // 2
        _y = (_splash.winfo_screenheight() - 100) // 2
        _splash.geometry(f"+{_x}+{_y}")
        _splash.update()
        from app.diagnostic import init as init_diagnostic
        from app.gui import main
        init_diagnostic()
        try:
            main(splash_window=_splash)
        finally:
            try:
                _splash.destroy()
            except Exception:
                pass
