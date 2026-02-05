#!/usr/bin/env python3
"""
Real-time system audio capture and transcription.
Uses PulseAudio Monitor / RDPSource on WSL2, faster-whisper, and Tkinter GUI.
"""

import os
import sys
import threading
import queue
from pathlib import Path

# Ensure WSLg can display the GUI (PulseAudio + X11/Wayland)
if "WSL" in os.uname().release or "microsoft" in os.uname().release.lower():
    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"].strip():
        os.environ.setdefault("DISPLAY", ":0")
    # Optional: ensure Wayland/Xwayland is used if available
    if "WAYLAND_DISPLAY" not in os.environ and "DISPLAY" in os.environ:
        pass  # DISPLAY already set

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import tkinter as tk
from tkinter import scrolledtext, font as tkfont, ttk

# Lazy import to avoid loading model until needed
_faster_whisper = None

def get_whisper_model():
    global _faster_whisper
    if _faster_whisper is None:
        from faster_whisper import WhisperModel
        _faster_whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _faster_whisper


# -----------------------------------------------------------------------------
# Audio device helpers (WSL / PulseAudio Monitor)
# -----------------------------------------------------------------------------

def list_audio_devices():
    """List all audio devices with index, name, and input channels."""
    devices = []
    try:
        all_devices = sd.query_devices()
        for dev in all_devices:
            devices.append({
                "index": dev.get("index", len(devices)),
                "name": dev.get("name", "Unknown"),
                "max_input_channels": dev.get("max_input_channels", 0),
                "default_samplerate": dev.get("default_samplerate", 0),
            })
    except Exception as e:
        return [], str(e)
    return devices, None


def get_default_monitor_device():
    """
    Select default input device: first input device whose name contains
    'monitor' or 'RDPSource' (case-insensitive for 'monitor').
    Falls back to system default input if none found.
    """
    devices, err = list_audio_devices()
    if err:
        return None, err
    keywords = ("monitor", "RDPSource")
    for d in devices:
        if d["max_input_channels"] <= 0:
            continue
        name = (d.get("name") or "").lower()
        if any(kw.lower() in name for kw in keywords):
            return d["index"], None
    # Fallback: default input device
    try:
        default = sd.query_devices(kind="input")
        return default["index"], None
    except Exception as e:
        return None, str(e)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_SEC = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
CHUNK_PATH = Path("/tmp/chunk.wav")


# -----------------------------------------------------------------------------
# Audio capture thread
# -----------------------------------------------------------------------------

def capture_worker(device_index, chunk_queue, stop_event):
    """Record 5-second chunks; save to /tmp/chunk.wav, put path in queue (blocking so no overwrite)."""
    while not stop_event.is_set():
        try:
            chunk = sd.rec(
                CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                device=device_index,
                blocking=True,
            )
            if stop_event.is_set():
                break
            chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, chunk_int16)
            try:
                chunk_queue.put(str(CHUNK_PATH), timeout=1.0)  # block until transcription took it
            except queue.Full:
                pass
        except Exception as e:
            if not stop_event.is_set():
                try:
                    chunk_queue.put_nowait(("error", str(e)))
                except queue.Full:
                    pass
            break


# -----------------------------------------------------------------------------
# Transcription thread
# -----------------------------------------------------------------------------

def transcription_worker(chunk_queue, text_queue, stop_event):
    """Take WAV paths from chunk_queue, transcribe, push text to text_queue, delete file."""
    model = get_whisper_model()
    while not stop_event.is_set():
        try:
            item = chunk_queue.get(timeout=0.5)
            if isinstance(item, tuple) and item[0] == "error":
                text_queue.put_nowait(("[Error] " + item[1] + "\n"))
                continue
            path = item
            if not Path(path).exists():
                continue
            try:
                segments, _ = model.transcribe(path, language="en", beam_size=1)
                text = " ".join(s.text.strip() for s in segments if s.text).strip()
                if text:
                    text_queue.put_nowait(text + "\n")
            finally:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        except queue.Empty:
            continue
        except Exception as e:
            if not stop_event.is_set():
                text_queue.put_nowait(f"[Transcribe error] {e}\n")
            break


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

def poll_text_queue(app):
    """Called periodically from main thread to append new transcript text."""
    try:
        while True:
            line = app.text_queue.get_nowait()
            app.log.insert(tk.END, line)
            app.log.see(tk.END)
    except queue.Empty:
        pass
    if app.running:
        app.root.after(200, lambda: poll_text_queue(app))


def start_stop(app):
    if app.running:
        app.running = False
        app.stop_event.set()
        if app.capture_thread and app.capture_thread.is_alive():
            app.capture_thread.join(timeout=CHUNK_DURATION_SEC + 2)
        if app.transcription_thread and app.transcription_thread.is_alive():
            app.transcription_thread.join(timeout=10)
        app.start_btn.config(state=tk.NORMAL)
        app.stop_btn.config(state=tk.DISABLED)
        app.status_var.set("Stopped")
        return
    # Start
    app.stop_event.clear()
    dev_idx, err = get_default_monitor_device()
    if err or dev_idx is None:
        app.log.insert(tk.END, f"[No input device] {err or 'No monitor device found'}\n")
        app.log.see(tk.END)
        return
    app.capture_thread = threading.Thread(
        target=capture_worker,
        args=(dev_idx, app.chunk_queue, app.stop_event),
        daemon=True,
    )
    app.transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(app.chunk_queue, app.text_queue, app.stop_event),
        daemon=True,
    )
    app.capture_thread.start()
    app.transcription_thread.start()
    app.running = True
    app.start_btn.config(state=tk.DISABLED)
    app.stop_btn.config(state=tk.NORMAL)
    app.status_var.set("Recording & transcribing…")
    poll_text_queue(app)


# -----------------------------------------------------------------------------
# Modern GUI theme (dark, minimal)
# -----------------------------------------------------------------------------

COLORS = {
    "bg": "#1a1b26",
    "bg_card": "#24283b",
    "bg_input": "#16161e",
    "fg": "#c0caf5",
    "fg_muted": "#565f89",
    "accent_start": "#9ece6a",
    "accent_stop": "#f7768e",
    "border": "#3b4261",
    "font_sans": ("Ubuntu", 11),
    "font_mono": ("Ubuntu Mono", 11),
}


def setup_modern_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=COLORS["bg"])

    style.configure(
        "Card.TFrame",
        background=COLORS["bg_card"],
    )
    style.configure(
        "Header.TFrame",
        background=COLORS["bg_card"],
    )
    style.configure(
        "TLabel",
        background=COLORS["bg_card"],
        foreground=COLORS["fg"],
        font=COLORS["font_sans"],
        padding=(0, 4),
    )
    style.configure(
        "Muted.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["fg_muted"],
        font=(COLORS["font_sans"][0], 9),
    )
    style.configure(
        "Start.TButton",
        background=COLORS["accent_start"],
        foreground=COLORS["bg"],
        font=COLORS["font_sans"],
        padding=(20, 10),
    )
    style.map("Start.TButton", background=[("active", "#b9f27c"), ("disabled", COLORS["border"])])
    style.configure(
        "Stop.TButton",
        background=COLORS["accent_stop"],
        foreground=COLORS["bg"],
        font=COLORS["font_sans"],
        padding=(20, 10),
    )
    style.map("Stop.TButton", background=[("active", "#ff9db2"), ("disabled", COLORS["border"])])


def main():
    root = tk.Tk()
    root.title("System Audio → Real-time Transcription")
    root.geometry("720x440")
    root.minsize(520, 340)
    root.option_add("*Font", COLORS["font_sans"])

    setup_modern_theme(root)

    app = type("App", (), {})()
    app.root = root
    app.running = False
    app.stop_event = threading.Event()
    app.chunk_queue = queue.Queue(maxsize=1)
    app.text_queue = queue.Queue()
    app.capture_thread = None
    app.transcription_thread = None

    # Header bar
    header = tk.Frame(root, bg=COLORS["bg_card"], padx=16, pady=12)
    header.pack(fill=tk.X)
    app.status_var = tk.StringVar(value="Ready — click Start to begin")
    status_lbl = tk.Label(
        header,
        textvariable=app.status_var,
        font=COLORS["font_sans"],
        fg=COLORS["fg"],
        bg=COLORS["bg_card"],
    )
    status_lbl.pack(side=tk.LEFT)
    btn_frame = tk.Frame(header, bg=COLORS["bg_card"])
    btn_frame.pack(side=tk.RIGHT)
    app.start_btn = tk.Button(
        btn_frame,
        text="Start",
        command=lambda: start_stop(app),
        font=COLORS["font_sans"],
        fg=COLORS["bg"],
        bg=COLORS["accent_start"],
        activebackground="#b9f27c",
        activeforeground=COLORS["bg"],
        relief=tk.FLAT,
        padx=20,
        pady=10,
        cursor="hand2",
        borderwidth=0,
    )
    app.start_btn.pack(side=tk.LEFT, padx=(0, 8))
    app.stop_btn = tk.Button(
        btn_frame,
        text="Stop",
        command=lambda: start_stop(app),
        state=tk.DISABLED,
        font=COLORS["font_sans"],
        fg=COLORS["bg"],
        bg=COLORS["accent_stop"],
        activebackground="#ff9db2",
        activeforeground=COLORS["bg"],
        relief=tk.FLAT,
        padx=20,
        pady=10,
        cursor="hand2",
        borderwidth=0,
    )
    app.stop_btn.pack(side=tk.LEFT)

    # Transcript card
    card = tk.Frame(root, bg=COLORS["bg_card"], padx=0, pady=0)
    card.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
    tk.Label(
        card,
        text="Transcript",
        font=COLORS["font_sans"],
        fg=COLORS["fg"],
        bg=COLORS["bg_card"],
    ).pack(anchor="w", padx=16, pady=(12, 4))
    transcript_inner = tk.Frame(card, bg=COLORS["bg_card"])
    transcript_inner.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
    app.log = scrolledtext.ScrolledText(
        transcript_inner,
        wrap=tk.WORD,
        font=COLORS["font_mono"],
        height=18,
        bg=COLORS["bg_input"],
        fg=COLORS["fg"],
        insertbackground=COLORS["fg"],
        selectbackground=COLORS["border"],
        selectforeground=COLORS["fg"],
        relief=tk.FLAT,
        padx=12,
        pady=12,
        borderwidth=0,
    )
    app.log.pack(fill=tk.BOTH, expand=True)

    # Footer: device info
    devices, _ = list_audio_devices()
    dev_idx, dev_err = get_default_monitor_device()
    if dev_err:
        dev_info = f"Input: (default) — {dev_err}"
    elif dev_idx is not None and devices:
        name = next((d["name"] for d in devices if d["index"] == dev_idx), f"Device {dev_idx}")
        dev_info = f"Input: {name}"
    else:
        dev_info = "Input: (default)"
    footer = tk.Label(
        root,
        text=dev_info,
        font=(COLORS["font_sans"][0], 9),
        fg=COLORS["fg_muted"],
        bg=COLORS["bg"],
    )
    footer.pack(anchor="w", padx=16, pady=(0, 12))

    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_event.set(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
