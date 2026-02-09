"""
GUI and app controller: CustomTkinter window, tabs, start/stop recording, prompts dialog.
"""
import math
import multiprocessing
import sys
import queue
import threading
from datetime import date, datetime
from pathlib import Path

import numpy as np
import customtkinter as ctk
import sounddevice as sd
from tkinter import messagebox, filedialog, Canvas, Scrollbar

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

from . import __version__ as app_version
from .settings import load_settings, save_settings, AUDIO_MODE_DEFAULT, AUDIO_MODE_LOOPBACK, AUDIO_MODE_MEETING
from .devices import list_audio_devices, list_loopback_devices, get_default_monitor_device, get_effective_audio_device
from .prompts import load_prompts, add_prompt, update_prompt, delete_prompt, get_prompt_by_id, TRANSCRIPT_PLACEHOLDER, MANUAL_NOTES_PLACEHOLDER
from .transcription import (
    STANDARD_TRANSCRIPTION_MODEL,
    get_transcription_model,
    is_transcription_model_loaded,
    clear_transcription_model_cache,
    list_installed_transcription_models,
    uninstall_transcription_model,
    download_transcription_model,
    start_transcription_subprocess,
)
from .capture import (
    capture_worker,
    capture_worker_loopback,
    meeting_chunk_ready,
    CHUNK_DURATION_SEC,
    CAPTURE_SAMPLE_RATE_MEETING,
    FRAMES_PER_READ_MEETING,
    MIXER_GAIN_MEETING,
    SAMPLE_RATE,
)
from .ai_summary import generate_ai_summary, generate_export_name
from .api_key_storage import get_openai_api_key, set_openai_api_key, clear_openai_api_key
from .diagnostic import write as diag
from .meetings_storage import (
    load_meetings,
    save_meetings,
    create_meeting,
    get_meeting_by_id,
    update_meeting_fields,
    delete_meeting_by_id,
    ensure_at_least_one_meeting,
)

if sys.platform == "win32":
    from .audio_mixer import AudioMixer
    from .chunk_recorder import ChunkRecorder


def poll_text_queue(app):
    """Called periodically from main thread to append new transcript text.
    Drains all available lines then does a single insert/scroll (batched update)
    to reduce UI work when multiple chunks complete in one poll tick."""
    lines = []
    try:
        while True:
            line = app.text_queue.get_nowait()
            lines.append(line)
    except queue.Empty:
        pass
    if lines:
        app.log.insert("end", "".join(lines))
        app.log.see("end")
    if lines and getattr(app, "schedule_save_meeting", None) is not None:
        app.schedule_save_meeting()
    # Drain live level updates and show latest
    if getattr(app, "level_queue", None) is not None:
        level = 0.0
        try:
            while True:
                level = app.level_queue.get_nowait()
        except queue.Empty:
            pass
        if getattr(app, "volume_bar", None) is not None:
            try:
                # Map RMS to 0..1 (speech roughly 0.005–0.3)
                p = min(1.0, max(0.0, level / 0.3))
                app.volume_bar.set(p)
            except Exception:
                pass
    if app.running:
        app.root.after(200, lambda: poll_text_queue(app))
    else:
        if getattr(app, "volume_bar", None) is not None:
            try:
                app.volume_bar.set(0)
            except Exception:
                pass


def _level_monitor_worker(device_index, level_queue, stop_event):
    """Run a lightweight input stream and push RMS levels to level_queue."""
    try:
        def callback(indata, _frames, _time, status):
            if status:
                return
            rms = float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
            try:
                level_queue.put_nowait(rms)
            except queue.Full:
                pass

        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=1024,
            dtype=np.float32,
            callback=callback,
        ):
            while not stop_event.wait(timeout=0.1):
                pass
    except Exception:
        pass


def _is_transcription_model_installed(app):
    """Return True if the supported transcription model is installed (in cache)."""
    models, err = list_installed_transcription_models()
    if err or not models:
        return False
    repo_ids = [m["repo_id"] for m in models]
    return STANDARD_TRANSCRIPTION_MODEL in repo_ids


def _on_model_load_done(app):
    """Called from main thread after background model load finishes."""
    app._model_load_in_progress = False
    update_model_status(app)


def update_model_status(app):
    """Update status bar, Start button, and model status label. Installed + loaded = ready to record."""
    installed = _is_transcription_model_installed(app)
    current_model = STANDARD_TRANSCRIPTION_MODEL
    loaded = installed and is_transcription_model_loaded(current_model)
    ready = installed and loaded

    if getattr(app, "model_status_var", None) is not None:
        if not installed:
            app.model_status_var.set("Transcription model: Not installed — open the Model tab and click \"Download & install\" before recording.")
            if getattr(app, "model_status_label", None) is not None:
                try:
                    app.model_status_label.configure(text_color=getattr(app, "model_status_warning_color", "#f7768e"))
                except Exception:
                    pass
        elif not loaded:
            app.model_status_var.set("Transcription model: Loading…")
            if getattr(app, "model_status_label", None) is not None:
                app.model_status_label.configure(text_color="gray")
            if not getattr(app, "_model_load_in_progress", False):
                app._model_load_in_progress = True
                def do_load():
                    try:
                        get_transcription_model(current_model)
                    finally:
                        app.root.after(0, lambda: _on_model_load_done(app))
                threading.Thread(target=do_load, daemon=True).start()
        else:
            app.model_status_var.set("Transcription model: Ready — you can start recording.")
            if getattr(app, "model_status_label", None) is not None:
                app.model_status_label.configure(text_color="gray")
    if getattr(app, "status_var", None) is not None:
        if not installed:
            app.status_var.set("Install the transcription model first (Model tab → Download & install)")
        elif not loaded:
            app.status_var.set("Loading transcription model…")
        else:
            app.status_var.set("Ready to record & transcribe")
    if getattr(app, "start_btn", None) is not None:
        app.start_btn.configure(state="normal" if ready else "disabled")


def start_stop(app):
    if app.running:
        app.running = False
        app.stop_event.set()
        if getattr(app, "mixer", None) is not None:
            try:
                app.mixer.stop()
            except Exception:
                pass
            app.mixer = None
        if getattr(app, "recorder", None) is not None:
            try:
                app.recorder.flush()
            except Exception:
                pass
            app.recorder = None
        for t in getattr(app, "capture_threads", []) or ([app.capture_thread] if app.capture_thread else []):
            if t and t.is_alive():
                t.join(timeout=CHUNK_DURATION_SEC + 2)
        if app.transcription_process and app.transcription_process.is_alive():
            app.transcription_process.join(timeout=10)
            if app.transcription_process.is_alive():
                app.transcription_process.terminate()
                app.transcription_process.join(timeout=2)
        if getattr(app, "stop_btn", None) is not None:
            app.start_btn.configure(state="normal")
            app.stop_btn.configure(state="disabled")
        elif getattr(app, "_record_ctk", None) is not None:
            app.start_btn.configure(image=app._record_ctk)
        _clear = getattr(app, "_clear_transcript_pulse", None)
        if _clear is not None:
            _clear(app)
        app.status_var.set("Stopped")
        if getattr(app, "auto_generate_summary_when_stopping_var", None) and app.auto_generate_summary_when_stopping_var.get():
            if getattr(app, "_do_ai_summary", None):
                app.root.after(500, app._do_ai_summary)
        return
    app.stop_event.clear()
    if not _is_transcription_model_installed(app):
        messagebox.showinfo(
            "Transcription model required",
            "Before your first recording, you need to install the transcription model.\n\n"
            "Go to the Models tab and click \"Download & install\" to download the recommended model (one-time setup, about 250 MB).",
            parent=app.root,
        )
        return
    mode, mic_idx, loopback_idx = get_effective_audio_device(app)
    diag("start_recording", mode=mode, mic_idx=mic_idx, loopback_idx=loopback_idx)

    if mode == AUDIO_MODE_DEFAULT:
        dev_idx, err = get_default_monitor_device()
        if err or dev_idx is None:
            app.log.insert("end", f"[No input device] {err or 'No monitor device found'}\n")
            app.log.see("end")
            return
        app.capture_thread = threading.Thread(
            target=capture_worker,
            args=(dev_idx, app.chunk_queue, app.stop_event),
            daemon=True,
        )
        app.capture_threads = [app.capture_thread]
        app.status_var.set("Recording & transcribing…")
        app.level_thread = threading.Thread(
            target=_level_monitor_worker,
            args=(dev_idx, app.level_queue, app.stop_event),
            daemon=True,
        )
        app.level_thread.start()
    elif mode == AUDIO_MODE_LOOPBACK:
        if sys.platform != "win32":
            app.log.insert("end", "[Loopback] Only supported on Windows with pyaudiowpatch.\n")
            app.log.see("end")
            return
        app.capture_thread = threading.Thread(
            target=capture_worker_loopback,
            args=(loopback_idx, app.chunk_queue, app.stop_event, app.level_queue),
            daemon=True,
        )
        app.capture_threads = [app.capture_thread]
        app.status_var.set("Recording loopback & transcribing…")
    elif mode == AUDIO_MODE_MEETING:
        if sys.platform != "win32":
            app.log.insert("end", "[Meeting] Mic+loopback only supported on Windows.\n")
            app.log.see("end")
            return
        try:
            use_silence = app.settings.get("use_silence_chunking", True)
            min_sec = app.settings.get("min_chunk_sec", 1.5)
            max_sec = app.settings.get("max_chunk_sec", 8.0)
            silence_sec = app.settings.get("silence_duration_sec", 0.5)
            if not isinstance(min_sec, (int, float)) or min_sec < 0.5 or min_sec > 10:
                min_sec = 1.5
            if not isinstance(max_sec, (int, float)) or max_sec < 3 or max_sec > 60:
                max_sec = 8.0
            if not isinstance(silence_sec, (int, float)) or silence_sec < 0.2 or silence_sec > 2:
                silence_sec = 0.5
            if min_sec > max_sec:
                max_sec = min_sec
            chunk_sec = app.settings.get("chunk_duration_sec", 5.0)  # fallback for fixed mode
            if not isinstance(chunk_sec, (int, float)) or chunk_sec < 3 or chunk_sec > 30:
                chunk_sec = CHUNK_DURATION_SEC
            app.mixer = AudioMixer(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                frames_per_read=FRAMES_PER_READ_MEETING,
                gain=MIXER_GAIN_MEETING,
            )
            def _push_level(rms):
                try:
                    app.level_queue.put_nowait(rms)
                except queue.Full:
                    pass
            app.mixer.set_level_callback(_push_level)
            app.mixer.start(loopback_device_index=loopback_idx, mic_device_index=mic_idx)
            app.recorder = ChunkRecorder(
                sample_rate=app.mixer.sample_rate,
                chunk_duration_sec=float(chunk_sec),
                asr_sample_rate=SAMPLE_RATE,
                on_chunk_ready=lambda wav_path, rms: meeting_chunk_ready(app, wav_path, rms),
                use_silence_chunking=use_silence,
                min_chunk_sec=float(min_sec),
                max_chunk_sec=float(max_sec),
                silence_duration_sec=float(silence_sec),
            )
            app.mixer.set_stereo_callback(app.recorder.push_stereo)
            app.capture_thread = None
            app.capture_threads = []
            app.status_var.set("Meeting (mic + loopback) — recording…")
            diag("meeting_started", mixer_ok=True)
        except Exception as e:
            diag("meeting_start_failed", error=str(e))
            app.log.insert("end", f"[Meeting] Start failed: {e}\n")
            app.log.see("end")
            if getattr(app, "mixer", None) is not None:
                try:
                    app.mixer.stop()
                except Exception:
                    pass
                app.mixer = None
            app.recorder = None
            return
    else:
        app.log.insert("end", f"[Unknown mode] {mode}\n")
        app.log.see("end")
        return

    if app.capture_thread:
        app.capture_thread.start()
    model_id = STANDARD_TRANSCRIPTION_MODEL
    app.transcription_process = start_transcription_subprocess(
        app.chunk_queue, app.text_queue, app.stop_event, model_id
    )
    app.running = True
    _start = getattr(app, "_start_transcript_pulse", None)
    if _start is not None:
        _start(app)
    if getattr(app, "stop_btn", None) is not None:
        app.start_btn.configure(state="disabled")
        app.stop_btn.configure(state="normal")
    elif getattr(app, "_stop_ctk", None) is not None:
        app.start_btn.configure(image=app._stop_ctk)
    poll_text_queue(app)


def _open_edit_prompt_dialog(parent, prompt_id, on_saved, ui_pad, ui_radius, font_family, font_sizes, colors):
    """Open dialog to add or edit a prompt template (name + prompt text with {{transcript}})."""
    prompt = get_prompt_by_id(prompt_id) if prompt_id else None
    is_new = prompt is None

    win = ctk.CTkToplevel(parent)
    win.title("Add prompt" if is_new else "Edit prompt")
    win.geometry("520x520")
    win.transient(parent)
    def _set_grab():
        try:
            win.grab_set()
        except Exception:
            pass
    win.after(100, _set_grab)

    ctk.CTkLabel(win, text="Name", font=ctk.CTkFont(family=font_family, size=font_sizes.small)).pack(anchor="w", padx=ui_pad, pady=(ui_pad, 2))
    name_var = ctk.StringVar(value=prompt.get("name", "") if prompt else "")
    name_entry = ctk.CTkEntry(win, width=400, height=32, font=ctk.CTkFont(family=font_family, size=font_sizes.body))
    name_entry.pack(anchor="w", padx=ui_pad, pady=(0, ui_pad))
    name_entry.insert(0, name_var.get())

    ctk.CTkLabel(win, text=f"Prompt: use {TRANSCRIPT_PLACEHOLDER} for the transcript, {MANUAL_NOTES_PLACEHOLDER} for manual notes from the Transcript tab.", font=ctk.CTkFont(family=font_family, size=font_sizes.small), wraplength=400, anchor="center").pack(anchor="center", padx=ui_pad, pady=(ui_pad, 2))
    prompt_text = ctk.CTkTextbox(win, width=500, height=320, font=ctk.CTkFont(family=font_family, size=font_sizes.small))
    prompt_text.pack(anchor="w", padx=ui_pad, pady=(0, ui_pad))
    if prompt:
        prompt_text.insert("1.0", prompt.get("prompt", ""))

    def save():
        name = name_entry.get().strip()
        text = prompt_text.get("1.0", "end").strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a name for the prompt.", parent=win)
            return
        if not text:
            messagebox.showwarning("Missing prompt", "Please enter the prompt text.", parent=win)
            return
        if TRANSCRIPT_PLACEHOLDER not in text and MANUAL_NOTES_PLACEHOLDER not in text:
            if not messagebox.askyesno("No placeholder", f"Prompt does not contain '{TRANSCRIPT_PLACEHOLDER}' or '{MANUAL_NOTES_PLACEHOLDER}'. Add at least one so content is inserted?", parent=win):
                return
        if is_new:
            add_prompt(name, text)
        else:
            update_prompt(prompt_id, name, text)
        on_saved()
        win.destroy()

    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill="x", padx=ui_pad, pady=ui_pad)
    ctk.CTkButton(btn_frame, text="Save", font=ctk.CTkFont(family=font_family, size=font_sizes.small), width=80, height=32, corner_radius=ui_radius, fg_color=colors["primary_fg"], hover_color=colors["primary_hover"], command=save).pack(side="left", padx=(0, ui_pad))
    ctk.CTkButton(btn_frame, text="Cancel", font=ctk.CTkFont(family=font_family, size=font_sizes.small), width=80, height=32, corner_radius=ui_radius, fg_color=colors["secondary_fg"], hover_color=colors["secondary_hover"], command=win.destroy).pack(side="left")


def _get_dpi_scale():
    """Return scale factor for high-DPI displays."""
    scale = 1.0
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        _root.update_idletasks()
        dpi = _root.winfo_fpixels("1i")
        if dpi and dpi > 0:
            scale = max(1.0, min(2.0, dpi / 96.0))
        if scale <= 1.0 and sys.platform == "win32":
            h = _root.winfo_screenheight()
            if h >= 900:
                scale = 1.85
        _root.destroy()
    except Exception:
        pass
    return scale


def main():
    # Theme and scaling (model loads on first Start or when installing from Models tab)
    ctk.set_appearance_mode("dark")
    # When frozen (PyInstaller), use bundle root so themes/assets are found (onefile: _MEIPASS; onedir: exe dir)
    if getattr(sys, "frozen", False):
        _base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        _base = Path(__file__).resolve().parent.parent  # project root (themes/, icon.ico)
    theme_path = _base / "themes" / "meetings-dark.json"
    if theme_path.exists():
        ctk.set_default_color_theme(str(theme_path))
    else:
        ctk.set_default_color_theme("dark-blue")
    # Use saved UI scale if set; otherwise auto-detect from DPI (can be large on high-DPI/multi-monitor)
    _initial_settings = load_settings()
    _saved_scale = _initial_settings.get("ui_scale")
    if _saved_scale is not None and isinstance(_saved_scale, (int, float)):
        scale = max(0.5, min(2.0, float(_saved_scale)))
    else:
        scale = _get_dpi_scale()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    _fs = lambda base: max(12, min(22, round(base * scale)))
    F = type("F", (), {"title": _fs(18), "header": _fs(16), "body": _fs(14), "small": _fs(13), "tiny": _fs(12)})()

    def _format_iso_date(iso_str):
        """Format ISO timestamp for display (e.g. 'Feb 9, 2025')."""
        if not iso_str:
            return ""
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)  # show local-style date
            return dt.strftime("%b ") + str(dt.day) + dt.strftime(", %Y")
        except Exception:
            return str(iso_str)[:16]

    UI_RADIUS = 10
    UI_PAD = 12
    UI_PAD_LG = 16
    COLORS = {
        "sidebar": ("gray88", "gray18"),
        "card": ("gray92", "gray18"),
        "header": ("gray92", "gray18"),
        "primary_fg": ("#3B76FB", "#3B76FB"),
        "primary_hover": ("#2d65e8", "#2d65e8"),
        "danger_fg": ("#c62828", "#b71c1c"),
        "danger_hover": ("#d32f2f", "#c62828"),
        "secondary_fg": ("gray70", "gray35"),
        "secondary_hover": ("gray60", "gray45"),
        "textbox_bg": ("gray97", "gray14"),
        "error_text": ("red", "#f7768e"),
        "prompt_item_bg": ("gray90", "gray22"),
    }
    # Subtle pulsing outline for recording (red) / generating (blue) states (smooth color transition)
    PULSE_BORDER_RED_DIM = "#b44"
    PULSE_BORDER_RED_BRIGHT = "#e88"
    PULSE_BORDER_BLUE_DIM = "#48a"
    PULSE_BORDER_BLUE_BRIGHT = "#8cf"
    PULSE_BORDER_WIDTH = 2
    PULSE_TICK_MS = 40
    PULSE_CYCLE_SEC = 1.8

    def _hex_to_rgb(hex_str):
        h = hex_str.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

    _pulse_rgb_red_dim = _hex_to_rgb(PULSE_BORDER_RED_DIM)
    _pulse_rgb_red_bright = _hex_to_rgb(PULSE_BORDER_RED_BRIGHT)
    _pulse_rgb_blue_dim = _hex_to_rgb(PULSE_BORDER_BLUE_DIM)
    _pulse_rgb_blue_bright = _hex_to_rgb(PULSE_BORDER_BLUE_BRIGHT)

    def _pulse_color(phase_rad, dim_rgb, bright_rgb):
        """Phase in radians; 0.5 + 0.5*sin(phase) gives smooth 0..1 blend."""
        blend = 0.5 + 0.5 * math.sin(phase_rad)
        r = dim_rgb[0] + (bright_rgb[0] - dim_rgb[0]) * blend
        g = dim_rgb[1] + (bright_rgb[1] - dim_rgb[1]) * blend
        b = dim_rgb[2] + (bright_rgb[2] - dim_rgb[2]) * blend
        return _rgb_to_hex(r, g, b)

    def _clear_transcript_pulse(app):
        if getattr(app, "_transcript_pulse_id", None) is not None:
            try:
                app.root.after_cancel(app._transcript_pulse_id)
            except Exception:
                pass
            app._transcript_pulse_id = None
        if getattr(app, "log", None) is not None:
            try:
                app.log.configure(border_width=0)
            except Exception:
                pass

    def _tick_transcript_pulse(app):
        app._transcript_pulse_id = None
        if not getattr(app, "running", False):
            _clear_transcript_pulse(app)
            return
        phase = getattr(app, "_transcript_pulse_phase", 0.0)
        try:
            app.log.configure(border_width=PULSE_BORDER_WIDTH, border_color=_pulse_color(phase, _pulse_rgb_red_dim, _pulse_rgb_red_bright))
        except Exception:
            pass
        app._transcript_pulse_phase = phase + (2 * math.pi * PULSE_TICK_MS / 1000.0 / PULSE_CYCLE_SEC)
        app._transcript_pulse_id = app.root.after(PULSE_TICK_MS, lambda: _tick_transcript_pulse(app))

    def _start_transcript_pulse(app):
        app._transcript_pulse_phase = 0.0
        _tick_transcript_pulse(app)

    def _clear_summary_pulse(app):
        if getattr(app, "_summary_pulse_id", None) is not None:
            try:
                app.root.after_cancel(app._summary_pulse_id)
            except Exception:
                pass
            app._summary_pulse_id = None
        if getattr(app, "summary_text", None) is not None:
            try:
                app.summary_text.configure(border_width=0)
            except Exception:
                pass

    def _tick_summary_pulse(app):
        app._summary_pulse_id = None
        if not getattr(app, "_summary_generating", False):
            _clear_summary_pulse(app)
            return
        phase = getattr(app, "_summary_pulse_phase", 0.0)
        try:
            app.summary_text.configure(border_width=PULSE_BORDER_WIDTH, border_color=_pulse_color(phase, _pulse_rgb_blue_dim, _pulse_rgb_blue_bright))
        except Exception:
            pass
        app._summary_pulse_phase = phase + (2 * math.pi * PULSE_TICK_MS / 1000.0 / PULSE_CYCLE_SEC)
        app._summary_pulse_id = app.root.after(PULSE_TICK_MS, lambda: _tick_summary_pulse(app))

    def _start_summary_pulse(app):
        app._summary_pulse_phase = 0.0
        _tick_summary_pulse(app)

    if sys.platform == "win32":
        UI_FONT_FAMILY = "Segoe UI"
        MONO_FONT_FAMILY = "Consolas"
    elif sys.platform == "darwin":
        UI_FONT_FAMILY = "SF Pro Display"
        MONO_FONT_FAMILY = "SF Mono"
    else:
        UI_FONT_FAMILY = "Ubuntu"
        MONO_FONT_FAMILY = "Ubuntu Mono"

    root = ctk.CTk()
    root.title(f"Blue Bridge Meeting Companion v{app_version}")
    _icon = _base / "icon.ico"
    if _icon.exists():
        try:
            root.iconbitmap(str(_icon))
        except Exception:
            pass
    root.geometry("960x480")
    root.minsize(720, 380)

    app = type("App", (), {})()
    app.root = root
    app.running = False
    # Use multiprocessing primitives so transcription can run in a subprocess (avoids GIL contention with GUI/Teams).
    app.stop_event = multiprocessing.Event()
    app.chunk_queue = multiprocessing.Queue(maxsize=1)
    app.text_queue = multiprocessing.Queue()
    app.level_queue = queue.Queue(maxsize=32)
    app.capture_thread = None
    app.transcription_process = None
    app.capture_threads = []
    app.settings = load_settings()

    # Meeting instances: load list, ensure at least one, select first
    app.meetings = load_meetings()
    new_one = ensure_at_least_one_meeting(app.meetings)
    if new_one is not None:
        save_meetings(app.meetings)
    app.current_meeting_id = app.meetings[0]["id"] if app.meetings else None
    app._save_meeting_after_id = None
    AUTO_SAVE_DELAY_MS = 1000

    # Attach pulse helpers to app so module-level start_stop() can call them
    app._start_transcript_pulse = _start_transcript_pulse
    app._clear_transcript_pulse = _clear_transcript_pulse
    app._start_summary_pulse = _start_summary_pulse
    app._clear_summary_pulse = _clear_summary_pulse

    content_frame = ctk.CTkFrame(root, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=UI_PAD, pady=UI_PAD)
    content_frame.grid_rowconfigure(0, weight=1)
    content_frame.grid_columnconfigure(0, weight=0)
    content_frame.grid_columnconfigure(1, weight=1)

    SIDEBAR_WIDTH = 220
    SIDEBAR_COLLAPSED_WIDTH = 40
    app.sidebar_collapsed = False
    app.refresh_sidebar_meetings_list = lambda: None  # set after transcript tab

    sidebar_frame = ctk.CTkFrame(content_frame, width=SIDEBAR_WIDTH, fg_color=COLORS["sidebar"], corner_radius=UI_RADIUS)
    sidebar_frame.grid(row=0, column=0, sticky="nswe", padx=(0, UI_PAD))
    sidebar_frame.grid_propagate(False)
    sidebar_header = ctk.CTkFrame(sidebar_frame, fg_color="transparent")
    sidebar_header.pack(fill="x", padx=UI_PAD, pady=(UI_PAD, 4))
    ctk.CTkLabel(sidebar_header, text="Meetings", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left")
    def _toggle_sidebar():
        app.sidebar_collapsed = not app.sidebar_collapsed
        if app.sidebar_collapsed:
            sidebar_frame.configure(width=SIDEBAR_COLLAPSED_WIDTH)
            sidebar_header.pack_forget()
            app.sidebar_scroll_container.pack_forget()
            app.new_meeting_btn.pack_forget()
            sidebar_expand_btn.pack(fill="x", padx=4, pady=UI_PAD)
        else:
            sidebar_expand_btn.pack_forget()
            sidebar_frame.configure(width=SIDEBAR_WIDTH)
            sidebar_header.pack(fill="x", padx=UI_PAD, pady=(UI_PAD, 4))
            app.sidebar_scroll_container.pack(fill="both", expand=True, padx=UI_PAD, pady=(0, UI_PAD))
            app.new_meeting_btn.pack(fill="x", padx=UI_PAD, pady=(0, UI_PAD))
            app.refresh_sidebar_meetings_list()
    sidebar_collapse_btn = ctk.CTkButton(sidebar_header, text="\u00AB", width=28, height=28, font=ctk.CTkFont(size=F.small), corner_radius=6, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_toggle_sidebar)
    sidebar_collapse_btn.pack(side="right")
    # Scrollable list with scrollbar only when content overflows
    app.sidebar_scroll_container = ctk.CTkFrame(sidebar_frame, fg_color="transparent")
    app.sidebar_scroll_container.pack(fill="both", expand=True, padx=UI_PAD, pady=(0, UI_PAD))
    _sidebar_bg = COLORS["sidebar"][1] if ctk.get_appearance_mode() == "Dark" else COLORS["sidebar"][0]
    app._sidebar_canvas = Canvas(app.sidebar_scroll_container, bg=_sidebar_bg, highlightthickness=0)
    app._sidebar_scrollbar = Scrollbar(app.sidebar_scroll_container)
    app._sidebar_canvas.pack(side="left", fill="both", expand=True)
    app.meetings_scroll = ctk.CTkFrame(app._sidebar_canvas, fg_color="transparent")
    app._sidebar_canvas_window = app._sidebar_canvas.create_window(0, 0, window=app.meetings_scroll, anchor="nw")
    app._sidebar_canvas.configure(yscrollcommand=app._sidebar_scrollbar.set)
    app._sidebar_scrollbar.configure(command=app._sidebar_canvas.yview)

    def _on_sidebar_frame_configure(event, inner=app.meetings_scroll):
        app._sidebar_canvas.itemconfig(app._sidebar_canvas_window, width=event.width)
        inner.update_idletasks()
        app._sidebar_canvas.configure(scrollregion=app._sidebar_canvas.bbox("all"))
        _update_sidebar_scrollbar_visibility()

    def _update_sidebar_scrollbar_visibility():
        app.meetings_scroll.update_idletasks()
        app._sidebar_canvas.update_idletasks()
        req_h = app.meetings_scroll.winfo_reqheight()
        vis_h = app._sidebar_canvas.winfo_height()
        if req_h > vis_h and vis_h > 0:
            app._sidebar_scrollbar.pack(side="right", fill="y")
        else:
            app._sidebar_scrollbar.pack_forget()

    app._sidebar_canvas.bind("<Configure>", _on_sidebar_frame_configure)

    def _on_sidebar_inner_configure(event, c=app._sidebar_canvas):
        c.configure(scrollregion=c.bbox("all"))
        root.after_idle(_update_sidebar_scrollbar_visibility)

    app.meetings_scroll.bind("<Configure>", _on_sidebar_inner_configure)
    app._update_sidebar_scrollbar_visibility = _update_sidebar_scrollbar_visibility
    app.new_meeting_btn = ctk.CTkButton(sidebar_frame, text="New Meeting", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), height=32, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"])
    app.new_meeting_btn.pack(fill="x", padx=UI_PAD, pady=(0, UI_PAD))
    sidebar_expand_btn = ctk.CTkButton(sidebar_frame, text="\u00BB", width=32, height=32, font=ctk.CTkFont(size=F.small), corner_radius=6, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_toggle_sidebar)

    main_content = ctk.CTkFrame(content_frame, fg_color="transparent")
    main_content.grid(row=0, column=1, sticky="nswe")

    _icons_dir = _base / "assets" / "icons"
    _record_path = _icons_dir / "record.png"
    _stop_path = _icons_dir / "stop recording.png"
    _record_img = _stop_img = None
    if PILImage is not None:
        try:
            if _record_path.exists():
                _record_img = PILImage.open(_record_path).convert("RGBA")
            if _stop_path.exists():
                _stop_img = PILImage.open(_stop_path).convert("RGBA")
        except Exception:
            pass
    _use_image_buttons = _record_img is not None and _stop_img is not None

    _btn_max_w, _btn_max_h = 120, 48
    _header_height = 52
    if _use_image_buttons:
        _rw, _rh = _record_img.size
        _sw, _sh = _stop_img.size
        _r_scale = min(_btn_max_w / _rw, _btn_max_h / _rh, 1.0)
        _s_scale = min(_btn_max_w / _sw, _btn_max_h / _sh, 1.0)
        _disp_rw = max(1, int(_rw * _r_scale))
        _disp_rh = max(1, int(_rh * _r_scale))
        _disp_sw = max(1, int(_sw * _s_scale))
        _disp_sh = max(1, int(_sh * _s_scale))
        _header_height = max(52, 2 * UI_PAD + max(_disp_rh, _disp_sh) + 12)
    header = ctk.CTkFrame(main_content, fg_color=COLORS["header"], corner_radius=UI_RADIUS, height=_header_height)
    header.pack(fill="x", pady=(0, UI_PAD))
    header.pack_propagate(False)
    app.status_var = ctk.StringVar(value="Loading transcription model…")
    ctk.CTkLabel(header, textvariable=app.status_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left", padx=UI_PAD_LG, pady=UI_PAD)
    app.model_status_var = ctk.StringVar(value="")
    # Live input volume indicator (small bar, right of status)
    vol_frame = ctk.CTkFrame(header, fg_color="transparent")
    vol_frame.pack(side="right", padx=(0, UI_PAD), pady=UI_PAD)
    app.volume_bar = ctk.CTkProgressBar(vol_frame, width=72, height=5, corner_radius=3, progress_color=COLORS["primary_fg"], fg_color=("gray88", "gray28"))
    app.volume_bar.set(0)
    app.volume_bar.pack(side="right")
    btn_frame = ctk.CTkFrame(header, fg_color="transparent")
    btn_frame.pack(side="right", padx=UI_PAD_LG, pady=UI_PAD)
    if _use_image_buttons:
        _btn_w = max(_disp_rw, _disp_sw)
        _btn_h = max(_disp_rh, _disp_sh)
        app._record_ctk = ctk.CTkImage(light_image=_record_img, dark_image=_record_img, size=(_btn_w, _btn_h))
        app._stop_ctk = ctk.CTkImage(light_image=_stop_img, dark_image=_stop_img, size=(_btn_w, _btn_h))
        app.start_btn = ctk.CTkButton(btn_frame, image=app._record_ctk, text="", command=lambda: start_stop(app), width=_btn_w, height=_btn_h, fg_color="transparent", hover_color=("gray85", "gray25"), state="disabled")
        app.start_btn.pack(side="left")
        app.stop_btn = None
    else:
        app.start_btn = ctk.CTkButton(btn_frame, text="Start", command=lambda: start_stop(app), font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"), width=100, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], state="disabled")
        app.start_btn.pack(side="left", padx=(0, UI_PAD))
        app.stop_btn = ctk.CTkButton(btn_frame, text="Stop", command=lambda: start_stop(app), state="disabled", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"), width=100, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"])
        app.stop_btn.pack(side="left")

    tabview = ctk.CTkTabview(main_content, fg_color=COLORS["card"], corner_radius=UI_RADIUS)
    tabview.pack(fill="both", expand=True, pady=(0, UI_PAD))
    tab_transcript = tabview.add("Transcript")
    tab_prompts = tabview.add("AI Prompts")
    tab_settings = tabview.add("Settings")
    tab_models = tabview.add("Model")

    # Model tab: single supported model — install if missing, or show installed size + uninstall
    models_card = ctk.CTkFrame(tab_models, fg_color="transparent")
    models_card.pack(fill="both", expand=True)

    # Install block (shown when model is not installed)
    install_card = ctk.CTkFrame(models_card, fg_color=COLORS["card"], corner_radius=UI_RADIUS, border_width=1)
    install_card.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, UI_PAD_LG))
    ctk.CTkLabel(install_card, text="Install the transcription model (required before first recording)", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    ctk.CTkLabel(install_card, text="Recording and live transcription require a speech-to-text model. Download the model below once; after that you can start recording. Size: ~650 MB. Est. download time: 5-20 seconds.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=1000, anchor="w").pack(anchor="w", padx=UI_PAD_LG, pady=(0, 4))
    ctk.CTkLabel(install_card, text=f"Model: {STANDARD_TRANSCRIPTION_MODEL}", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), text_color="gray", wraplength=600, anchor="w").pack(anchor="w", padx=UI_PAD_LG, pady=(0, UI_PAD))
    install_row = ctk.CTkFrame(install_card, fg_color="transparent")
    install_row.pack(fill="x", padx=UI_PAD_LG, pady=(0, UI_PAD))
    def _do_install_standard_model():
        app.install_model_btn.configure(state="disabled")
        if getattr(app, "model_status_var", None) is not None:
            app.model_status_var.set("Transcription model: Downloading… (one-time setup)")
        if getattr(app, "model_status_label", None) is not None:
            app.model_status_label.configure(text_color="gray")
        result_holder = []

        def worker():
            ok, err = download_transcription_model(STANDARD_TRANSCRIPTION_MODEL)
            result_holder.append((ok, err))

        def check_done():
            if not result_holder:
                app.root.after(200, check_done)
                return
            ok, err = result_holder[0]
            app.install_model_btn.configure(state="normal")
            if ok:
                app.settings["transcription_model"] = STANDARD_TRANSCRIPTION_MODEL
                save_settings(app.settings)
                refresh_models_tab()
                update_model_status(app)
            else:
                messagebox.showerror("Install failed", err or "Download failed.", parent=app.root)
        threading.Thread(target=worker, daemon=True).start()
        app.root.after(200, check_done)

    app.install_model_btn = ctk.CTkButton(install_row, text="Download & install", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=140, height=32, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], command=_do_install_standard_model)
    app.install_model_btn.pack(side="left")

    # Installed block (shown when model is installed): status, size, uninstall
    installed_card = ctk.CTkFrame(models_card, fg_color=COLORS["card"], corner_radius=UI_RADIUS, border_width=1)
    app.installed_model_size_var = ctk.StringVar(value="")
    ctk.CTkLabel(installed_card, text="The transcription model is installed.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    ctk.CTkLabel(installed_card, textvariable=app.installed_model_size_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", anchor="w").pack(anchor="w", padx=UI_PAD_LG, pady=(0, UI_PAD))
    installed_btn_row = ctk.CTkFrame(installed_card, fg_color="transparent")
    installed_btn_row.pack(fill="x", padx=UI_PAD_LG, pady=(0, UI_PAD))
    def _do_uninstall_model():
        if not messagebox.askyesno("Uninstall model", f"Delete cached model '{STANDARD_TRANSCRIPTION_MODEL}'? This frees disk space; you can re-download from this tab later."):
            return
        models, _ = list_installed_transcription_models()
        m = next((x for x in (models or []) if x["repo_id"] == STANDARD_TRANSCRIPTION_MODEL), None)
        if not m:
            messagebox.showerror("Error", "Model not found in cache.", parent=app.root)
            return
        ok, err = uninstall_transcription_model(STANDARD_TRANSCRIPTION_MODEL, m["revision_hashes"])
        if ok:
            messagebox.showinfo("Uninstalled", "Model removed from cache.")
            refresh_models_tab()
            update_model_status(app)
        else:
            messagebox.showerror("Error", err or "Failed to delete.", parent=app.root)
    ctk.CTkButton(installed_btn_row, text="Uninstall", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=84, height=32, corner_radius=UI_RADIUS, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"], command=_do_uninstall_model).pack(side="left")

    def refresh_models_tab(models_err=None):
        """Refresh Model tab. If models_err is provided, use (models, err) instead of calling list_installed_transcription_models() on this thread."""
        if models_err is not None:
            models, err = models_err
        else:
            models, err = list_installed_transcription_models()
        installed_repo_ids = [m["repo_id"] for m in models] if models else []
        standard_installed = STANDARD_TRANSCRIPTION_MODEL in installed_repo_ids
        app.settings["transcription_model"] = STANDARD_TRANSCRIPTION_MODEL
        save_settings(app.settings)
        if standard_installed:
            install_card.pack_forget()
            m = next((x for x in models if x["repo_id"] == STANDARD_TRANSCRIPTION_MODEL), None)
            app.installed_model_size_var.set(f"Model: {STANDARD_TRANSCRIPTION_MODEL}  ·  Size on disk: {m['size_str']}" if m else f"Model: {STANDARD_TRANSCRIPTION_MODEL}")
            installed_card.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, UI_PAD_LG))
        else:
            installed_card.pack_forget()
            install_card.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, UI_PAD_LG))
            app.install_model_btn.configure(state="normal")
        update_model_status(app)

    def _startup_check_done(models_err):
        """Called on main thread after background startup check; refreshes Model tab and enables Record when ready."""
        refresh_models_tab(models_err=models_err)

    def _run_startup_check():
        """Run in background: scan cache then schedule UI update on main thread so app stays responsive."""
        models_err = list_installed_transcription_models()
        app.root.after(0, lambda: _startup_check_done(models_err))

    # Defer model check to background so the window appears immediately and stays responsive.
    # Record button stays disabled until update_model_status() runs (after check and optional model load).
    app.model_status_var.set("Transcription model: Loading…")
    threading.Thread(target=_run_startup_check, daemon=True).start()

    # Transcript tab — meeting name and dates at top, then three equal-width columns: Manual Notes | Transcript | AI Summary
    meeting_name_row = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    meeting_name_row.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, 0))
    meeting_name_row.grid_columnconfigure(0, weight=1)
    name_line = ctk.CTkFrame(meeting_name_row, fg_color="transparent")
    name_line.grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(name_line, text="Meeting name:", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small)).pack(side="left", padx=(0, 4), pady=4)
    app.meeting_name_var = ctk.StringVar(value="New Meeting")
    app.meeting_name_entry = ctk.CTkEntry(name_line, textvariable=app.meeting_name_var, width=280, height=28, placeholder_text="e.g. Team standup", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small))
    app.meeting_name_entry.pack(side="left", padx=4, pady=4)
    def _do_generate_meeting_name():
        api_key = get_openai_api_key()
        if not api_key:
            messagebox.showerror("API key required", "Add your OpenAI API key in Settings.", parent=root)
            return
        summary = app.summary_text.get("1.0", "end").strip()
        if not summary:
            return
        app.generate_name_btn.configure(state="disabled")
        result_holder = []
        def worker():
            name_ok, name_result = generate_export_name(api_key, summary[:250])
            result_holder.append((name_ok, name_result))
        def check_done():
            if not result_holder:
                root.after(200, check_done)
                return
            name_ok, name_result = result_holder[0]
            app.generate_name_btn.configure(state="normal" if (app.summary_text.get("1.0", "end").strip() != "") else "disabled")
            if name_ok and name_result:
                safe = "".join(c if c.isalnum() or c in "._- " else "-" for c in name_result)
                safe = safe.replace(" ", "-").strip("-") or name_result
                app.meeting_name_var.set(safe)
            elif not name_ok:
                messagebox.showerror("Generate name failed", name_result, parent=root)
        threading.Thread(target=worker, daemon=True).start()
        root.after(200, check_done)
    app.generate_name_btn = ctk.CTkButton(name_line, text="Generate name", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=100, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_do_generate_meeting_name, state="disabled")
    app.generate_name_btn.pack(side="left", padx=4, pady=4)
    def _update_generate_name_btn_state():
        has_summary = bool(getattr(app, "summary_text", None) and app.summary_text.get("1.0", "end").strip())
        app.generate_name_btn.configure(state="normal" if has_summary else "disabled")
    app._update_generate_name_btn_state = _update_generate_name_btn_state
    app.meeting_dates_var = ctk.StringVar(value="")
    meeting_dates_lbl = ctk.CTkLabel(meeting_name_row, textvariable=app.meeting_dates_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), text_color="gray", anchor="w")
    meeting_dates_lbl.grid(row=1, column=0, sticky="w", padx=0, pady=(0, 4))

    card = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    card.pack(fill="both", expand=True)
    for col in (0, 1, 2):
        card.grid_columnconfigure(col, weight=1, uniform="transcript_cols")
    card.grid_rowconfigure(0, weight=0)
    card.grid_rowconfigure(1, weight=0)
    card.grid_rowconfigure(2, weight=1)

    PAD_BELOW_SUBHEADER = 8   # padding below Transcript (and Manual Notes) sub-header
    PAD_BELOW_AUTO_CHECKBOX = 4   # smaller padding below auto-generate checkbox

    # Manual Notes (left column)
    notes_header = ctk.CTkFrame(card, fg_color="transparent")
    notes_header.grid(row=0, column=0, sticky="ew", padx=(UI_PAD_LG, UI_PAD))
    ctk.CTkLabel(notes_header, text="Manual Notes", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left")
    notes_sub = ctk.CTkFrame(card, fg_color="transparent")
    notes_sub.grid(row=1, column=0, sticky="ew", padx=(UI_PAD_LG, UI_PAD), pady=(0, PAD_BELOW_SUBHEADER))
    ctk.CTkLabel(notes_sub, text="Notes will be included in the AI summary.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), text_color="gray", wraplength=400, anchor="w").pack(anchor="w")
    app.manual_notes = ctk.CTkTextbox(card, wrap="word", font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body), corner_radius=8, border_width=0, fg_color=COLORS["textbox_bg"], border_spacing=UI_PAD)
    app.manual_notes.grid(row=2, column=0, sticky="nsew", padx=(UI_PAD_LG, UI_PAD), pady=(0, UI_PAD))

    # Transcript (middle column)
    card_header = ctk.CTkFrame(card, fg_color="transparent")
    card_header.grid(row=0, column=1, sticky="ew", padx=UI_PAD)
    ctk.CTkLabel(card_header, text="Transcript", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left")
    def copy_transcript():
        text = app.log.get("1.0", "end").rstrip()
        if text:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
    def clear_transcript():
        app.log.delete("1.0", "end")
    ctk.CTkButton(card_header, text="Copy transcript", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=100, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=copy_transcript).pack(side="left", padx=(UI_PAD, 0), pady=4)
    ctk.CTkButton(card_header, text="Clear", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=50, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=clear_transcript).pack(side="right", padx=UI_PAD, pady=4)
    transcript_sub = ctk.CTkFrame(card, fg_color="transparent")
    transcript_sub.grid(row=1, column=1, sticky="ew", padx=UI_PAD, pady=(0, PAD_BELOW_SUBHEADER))
    ctk.CTkLabel(transcript_sub, text="Transcript will be included in the AI summary.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), text_color="gray", wraplength=400, anchor="w").pack(anchor="w")
    app.log = ctk.CTkTextbox(card, wrap="word", font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body), corner_radius=8, border_width=0, fg_color=COLORS["textbox_bg"], border_spacing=UI_PAD)
    app.log.grid(row=2, column=1, sticky="nsew", padx=UI_PAD, pady=(0, UI_PAD))

    # AI Summary (right column)
    summary_header = ctk.CTkFrame(card, fg_color="transparent")
    summary_header.grid(row=0, column=2, sticky="ew", padx=(UI_PAD, UI_PAD_LG))
    ctk.CTkLabel(summary_header, text="AI Summary", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left")
    _prompts_for_summary = load_prompts()
    _prompt_names = [p.get("name", "Unnamed") for p in _prompts_for_summary] or ["(No prompts — add in AI Prompts tab)"]
    app.summary_prompt_var = ctk.StringVar(value=_prompt_names[0])
    app.summary_prompt_menu = ctk.CTkOptionMenu(summary_header, values=_prompt_names, variable=app.summary_prompt_var, width=150, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small))
    app.summary_prompt_menu.pack(side="left", padx=(UI_PAD, 0), pady=4)
    app.summary_status_var = ctk.StringVar(value="")
    ctk.CTkLabel(summary_header, textvariable=app.summary_status_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray").pack(side="left", padx=(UI_PAD, 0), pady=4)

    app.auto_generate_summary_when_stopping_var = ctk.BooleanVar(value=app.settings.get("auto_generate_summary_when_stopping", False))
    def _on_auto_summary_when_stopping_changed():
        app.settings["auto_generate_summary_when_stopping"] = app.auto_generate_summary_when_stopping_var.get()
        save_settings(app.settings)
    summary_options_row = ctk.CTkFrame(card, fg_color="transparent")
    summary_options_row.grid(row=1, column=2, sticky="ew", padx=(UI_PAD, UI_PAD_LG), pady=(0, PAD_BELOW_AUTO_CHECKBOX))
    ctk.CTkCheckBox(
        summary_options_row, text="Auto-generate summary when recording stops", variable=app.auto_generate_summary_when_stopping_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=_on_auto_summary_when_stopping_changed,
        border_width=1, corner_radius=3, checkbox_width=18, checkbox_height=18
    ).pack(side="left", padx=0, pady=0)

    def _refresh_summary_prompt_menu():
        prompts_list = load_prompts()
        names = [p.get("name", "Unnamed") for p in prompts_list] or ["(No prompts — add in AI Prompts tab)"]
        app.summary_prompt_menu.configure(values=names)
        if names and (app.summary_prompt_var.get() not in names):
            app.summary_prompt_var.set(names[0])

    def _do_ai_summary():
        # key stored in app (Settings tab)
        api_key = get_openai_api_key()
        if not api_key:
            messagebox.showerror(
                "API key required",
                "Add your OpenAI API key in Settings.",
                parent=root,
            )
            return
        prompts_list = load_prompts()
        if not prompts_list:
            messagebox.showinfo("No prompts", "Create at least one prompt in the 'AI Prompts' tab first.", parent=root)
            return
        name = app.summary_prompt_var.get()
        prompt_obj = next((p for p in prompts_list if p.get("name") == name), None)
        if not prompt_obj or name.startswith("("):
            messagebox.showwarning("Select a prompt", "Choose a prompt template from the dropdown.", parent=root)
            return
        transcript = app.log.get("1.0", "end").strip()
        if not transcript:
            messagebox.showwarning("Empty transcript", "Transcript is empty. Record or paste some text first.", parent=root)
            return
        manual_notes = app.manual_notes.get("1.0", "end").strip()
        app.summary_generate_btn.configure(state="disabled")
        app.summary_status_var.set("Generating…")
        app._summary_generating = True
        _start_summary_pulse(app)
        result_holder = []
        def worker():
            ok, out = generate_ai_summary(api_key, prompt_obj["prompt"], transcript, manual_notes=manual_notes)
            result_holder.append((ok, out))
        def check_done():
            if not result_holder:
                root.after(200, check_done)
                return
            ok, out = result_holder[0]
            app._summary_generating = False
            _clear_summary_pulse(app)
            app.summary_generate_btn.configure(state="normal")
            app.summary_status_var.set("")
            if ok:
                app.summary_text.delete("1.0", "end")
                app.summary_text.insert("1.0", out)
                app._update_generate_name_btn_state()
            else:
                messagebox.showerror("AI Summary failed", out, parent=root)
        threading.Thread(target=worker, daemon=True).start()
        root.after(200, check_done)

    app._do_ai_summary = _do_ai_summary
    app.summary_generate_btn = ctk.CTkButton(summary_header, text="Generate", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=74, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], command=_do_ai_summary)
    app.summary_generate_btn.pack(side="left", padx=(UI_PAD, 0), pady=4)

    def _export_markdown():
        summary = app.summary_text.get("1.0", "end").strip()
        transcript = app.log.get("1.0", "end").strip()
        manual_notes = app.manual_notes.get("1.0", "end").strip()
        if not summary and not transcript and not manual_notes:
            messagebox.showwarning("Nothing to export", "Add an AI summary, transcript, and/or manual notes first.", parent=root)
            return
        name_part = (app.meeting_name_var.get() or "").strip()
        if name_part:
            name_part = "".join(c if c.isalnum() or c in "._- " else "-" for c in name_part)
            name_part = name_part.replace(" ", "-").strip("-") or "export"
        else:
            name_part = "export"
        if app.export_prepend_date_var.get():
            default_name = f"{date.today().isoformat()} {name_part}.md"
        else:
            default_name = f"{name_part}.md"
        path = filedialog.asksaveasfilename(parent=root, defaultextension=".md", filetypes=[("Markdown", "*.md"), ("All files", "*.*")], initialfile=default_name)
        if not path:
            return
        # Order: AI Summary → Manual Notes → Full Transcript
        parts = []
        if summary:
            parts.append("# AI Summary\n\n" + summary)
        if manual_notes:
            parts.append("# Manual Notes\n\n" + manual_notes)
        if transcript:
            parts.append("# Full Transcript\n\n" + transcript)
        content = "\n\n---\n\n".join(parts) + "\n"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=root)

    app.summary_text = ctk.CTkTextbox(card, wrap="word", font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body), corner_radius=8, border_width=0, fg_color=COLORS["textbox_bg"], border_spacing=UI_PAD)
    app.summary_text.grid(row=2, column=2, sticky="nsew", padx=(UI_PAD, UI_PAD_LG), pady=(0, UI_PAD))
    app.summary_text.bind("<KeyRelease>", lambda e: app._update_generate_name_btn_state())
    app._update_generate_name_btn_state()

    export_row = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    export_row.pack(fill="x", pady=(UI_PAD, UI_PAD_LG), padx=UI_PAD_LG)
    app.export_prepend_date_var = ctk.BooleanVar(value=app.settings.get("export_prepend_date", True))
    def _on_prepend_date_changed():
        app.settings["export_prepend_date"] = app.export_prepend_date_var.get()
        save_settings(app.settings)
    ctk.CTkCheckBox(
        export_row, text="Prepend today's date to export file name", variable=app.export_prepend_date_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=_on_prepend_date_changed,
        border_width=1, corner_radius=3, checkbox_width=18, checkbox_height=18
    ).pack(side="left", padx=8, pady=4)
    ctk.CTkButton(export_row, text="Export", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=64, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_export_markdown).pack(side="left", padx=(0, 0), pady=4)

    # ----- Meeting instances: sync UI <-> current meeting, debounced save, sidebar list -----
    app._loading_meeting = False

    def sync_ui_to_current_meeting():
        """Read UI into current meeting dict; only update updated_at and re-sort when something changed."""
        mid = app.current_meeting_id
        if not mid:
            return
        meeting = get_meeting_by_id(app.meetings, mid)
        if not meeting:
            return
        new_name = (app.meeting_name_var.get() or "").strip() or "New Meeting"
        new_notes = app.manual_notes.get("1.0", "end").strip()
        new_transcript = app.log.get("1.0", "end").strip()
        new_summary = app.summary_text.get("1.0", "end").strip()
        if (
            meeting.get("meeting_name") == new_name
            and meeting.get("manual_notes") == new_notes
            and meeting.get("transcript") == new_transcript
            and meeting.get("ai_summary") == new_summary
        ):
            return
        update_meeting_fields(
            meeting,
            meeting_name=new_name,
            manual_notes=new_notes,
            transcript=new_transcript,
            ai_summary=new_summary,
        )
        app.meetings.sort(key=lambda m: m.get("updated_at") or m.get("created_at") or "", reverse=True)
        save_meetings(app.meetings)

    def _do_save_meeting():
        if app._save_meeting_after_id is not None:
            try:
                root.after_cancel(app._save_meeting_after_id)
            except Exception:
                pass
            app._save_meeting_after_id = None
        sync_ui_to_current_meeting()
        meeting = get_meeting_by_id(app.meetings, app.current_meeting_id) if app.current_meeting_id else None
        if meeting:
            _set_meeting_dates_var(meeting)
        app.refresh_sidebar_meetings_list()

    def schedule_save_meeting():
        if app._loading_meeting:
            return
        if app._save_meeting_after_id is not None:
            try:
                root.after_cancel(app._save_meeting_after_id)
            except Exception:
                pass
        def run():
            app._save_meeting_after_id = None
            sync_ui_to_current_meeting()
            app.refresh_sidebar_meetings_list()
        app._save_meeting_after_id = root.after(AUTO_SAVE_DELAY_MS, run)

    app.schedule_save_meeting = schedule_save_meeting

    def _set_meeting_dates_var(meeting):
        """Update the Created / Updated line from meeting dict."""
        if not meeting:
            app.meeting_dates_var.set("")
            return
        created = _format_iso_date(meeting.get("created_at"))
        updated = _format_iso_date(meeting.get("updated_at"))
        parts = [f"Created: {created}"] if created else []
        if updated:
            parts.append(f"Updated: {updated}")
        app.meeting_dates_var.set("  ·  ".join(parts))

    def load_meeting_to_ui(meeting):
        if not meeting:
            return
        app._loading_meeting = True
        try:
            app.current_meeting_id = meeting["id"]
            app.meeting_name_var.set(meeting.get("meeting_name") or "New Meeting")
            _set_meeting_dates_var(meeting)
            app.manual_notes.delete("1.0", "end")
            app.manual_notes.insert("1.0", meeting.get("manual_notes") or "")
            app.log.delete("1.0", "end")
            app.log.insert("1.0", meeting.get("transcript") or "")
            app.summary_text.delete("1.0", "end")
            app.summary_text.insert("1.0", meeting.get("ai_summary") or "")
            app._update_generate_name_btn_state()
        finally:
            app._loading_meeting = False

    def _select_meeting(meeting_id):
        _do_save_meeting()
        m = get_meeting_by_id(app.meetings, meeting_id)
        if m:
            load_meeting_to_ui(m)
        app.refresh_sidebar_meetings_list()

    def _new_meeting():
        _do_save_meeting()
        new_m = create_meeting("New Meeting")
        app.meetings.insert(0, new_m)
        save_meetings(app.meetings)
        load_meeting_to_ui(new_m)
        app.refresh_sidebar_meetings_list()

    def _delete_meeting(meeting_id, event=None):
        if not messagebox.askyesno("Delete meeting", "Delete this meeting? This cannot be undone.", parent=root):
            return
        _do_save_meeting()
        delete_meeting_by_id(app.meetings, meeting_id)
        save_meetings(app.meetings)
        if not app.meetings:
            ensure_at_least_one_meeting(app.meetings)
            save_meetings(app.meetings)
        next_m = app.meetings[0] if app.meetings else None
        if next_m:
            load_meeting_to_ui(next_m)
        app.refresh_sidebar_meetings_list()

    def refresh_sidebar_meetings_list():
        for w in app.meetings_scroll.winfo_children():
            w.destroy()
        # List already sorted by updated_at desc from load_meetings / after save
        for m in app.meetings:
            mid = m.get("id")
            name = (m.get("meeting_name") or "New Meeting").strip()
            if len(name) > 28:
                name = name[:25] + "..."
            updated_str = _format_iso_date(m.get("updated_at"))
            row = ctk.CTkFrame(app.meetings_scroll, fg_color=COLORS["prompt_item_bg"], corner_radius=6, cursor="hand2")
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(0, weight=1, minsize=0)
            row.grid_columnconfigure(1, weight=0, minsize=32)
            row.grid_rowconfigure(0, weight=0)
            row.grid_rowconfigure(1, weight=0)
            is_current = mid == app.current_meeting_id
            if is_current:
                row.configure(border_width=2, border_color=COLORS["primary_fg"])
            lbl = ctk.CTkLabel(row, text=name, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), anchor="w")
            lbl.grid(row=0, column=0, sticky="ew", padx=(UI_PAD, 4), pady=(4, 0))
            date_lbl = ctk.CTkLabel(row, text=f"Updated {updated_str}" if updated_str else "", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), anchor="w", text_color="gray")
            date_lbl.grid(row=1, column=0, sticky="ew", padx=(UI_PAD, 4), pady=(0, 4))
            del_btn = ctk.CTkButton(row, text="\u2715", width=28, height=28, font=ctk.CTkFont(size=F.tiny), corner_radius=4, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"])
            del_btn.grid(row=0, column=1, rowspan=2, padx=(0, 4), pady=4)
            del_btn.configure(command=lambda mid=mid: _delete_meeting(mid))
            row.bind("<Button-1>", lambda e, mid=mid: _select_meeting(mid))
            lbl.bind("<Button-1>", lambda e, mid=mid: _select_meeting(mid))
            date_lbl.bind("<Button-1>", lambda e, mid=mid: _select_meeting(mid))
        app.new_meeting_btn.configure(command=_new_meeting)
        if getattr(app, "_update_sidebar_scrollbar_visibility", None) is not None:
            root.after_idle(app._update_sidebar_scrollbar_visibility)

    app.refresh_sidebar_meetings_list = refresh_sidebar_meetings_list
    refresh_sidebar_meetings_list()
    load_meeting_to_ui(get_meeting_by_id(app.meetings, app.current_meeting_id))

    def _on_meeting_name_changed(*args):
        schedule_save_meeting()
    app.meeting_name_var.trace_add("write", _on_meeting_name_changed)

    def _on_manual_notes_changed(event=None):
        schedule_save_meeting()
    app.manual_notes.bind("<KeyRelease>", _on_manual_notes_changed)
    def _on_transcript_changed(event=None):
        schedule_save_meeting()
    app.log.bind("<KeyRelease>", _on_transcript_changed)
    def _on_summary_changed(event=None):
        schedule_save_meeting()
    app.summary_text.bind("<KeyRelease>", _on_summary_changed)

    # AI Prompts tab
    prompts_header = ctk.CTkFrame(tab_prompts, fg_color="transparent")
    prompts_header.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    ctk.CTkLabel(prompts_header, text=f"Custom prompt templates for AI Summary. Use {TRANSCRIPT_PLACEHOLDER} for the transcript and {MANUAL_NOTES_PLACEHOLDER} for manual notes from the Transcript tab.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=1000, anchor="w").pack(anchor="w", side="left", fill="x", expand=True, padx=(0, UI_PAD))
    prompts_scroll = ctk.CTkScrollableFrame(tab_prompts, fg_color="transparent")
    prompts_scroll.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=UI_PAD)

    def refresh_prompts_list():
        for w in prompts_scroll.winfo_children():
            w.destroy()
        prompts_list = load_prompts()
        _refresh_summary_prompt_menu()
        if not prompts_list:
            ctk.CTkLabel(prompts_scroll, text="No prompts yet. Click 'Add prompt' to create one.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray").pack(anchor="w", padx=UI_PAD, pady=8)
        for p in prompts_list:
            row = ctk.CTkFrame(prompts_scroll, fg_color=COLORS["prompt_item_bg"], corner_radius=UI_RADIUS)
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=p.get("name", "Unnamed"), font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.body, weight="bold"), anchor="w").pack(side="left", padx=(UI_PAD_LG, 8), pady=UI_PAD)
            def _edit(pid=p["id"]):
                _open_edit_prompt_dialog(root, pid, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)
            def _delete(pid=p["id"], pname=p.get("name", "?")):
                if not messagebox.askyesno("Delete prompt", f"Delete prompt '{pname}'?"):
                    return
                if delete_prompt(pid):
                    refresh_prompts_list()
                    messagebox.showinfo("Deleted", "Prompt deleted.")
            ctk.CTkButton(row, text="Edit", width=60, height=28, font=ctk.CTkFont(family=UI_FONT_FAMILY), corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_edit).pack(side="right", padx=4, pady=UI_PAD)
            ctk.CTkButton(row, text="Delete", width=60, height=28, font=ctk.CTkFont(family=UI_FONT_FAMILY), corner_radius=UI_RADIUS, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"], command=_delete).pack(side="right", padx=(0, UI_PAD_LG), pady=UI_PAD)

    def add_new_prompt():
        _open_edit_prompt_dialog(root, None, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)

    ctk.CTkButton(prompts_header, text="Add prompt", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], command=add_new_prompt, width=92).pack(side="right")
    refresh_prompts_list()

    # Settings tab
    settings_card = ctk.CTkFrame(tab_settings, fg_color="transparent")
    settings_card.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=UI_PAD)

    # Interface scale (backup when auto DPI makes UI too large or too small)
    UI_SCALE_OPTIONS = [
        ("Auto (recommended)", None),
        ("70%", 0.7),
        ("80%", 0.8),
        ("90%", 0.9),
        ("100%", 1.0),
        ("110%", 1.1),
        ("125%", 1.25),
        ("150%", 1.5),
    ]
    scale_labels = [t[0] for t in UI_SCALE_OPTIONS]
    label_to_scale = {t[0]: t[1] for t in UI_SCALE_OPTIONS}
    _ui_scale_val = app.settings.get("ui_scale")
    if _ui_scale_val is None:
        _initial_scale_label = "Auto (recommended)"
    else:
        _closest = min(UI_SCALE_OPTIONS[1:], key=lambda t: abs((t[1] or 1.0) - _ui_scale_val))
        _initial_scale_label = _closest[0]
    ctk.CTkLabel(settings_card, text="Interface scale", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(anchor="w", pady=(0, 4))
    ctk.CTkLabel(settings_card, text="If the app looks too large or too small on your display, choose a fixed scale. Restart the app for changes to take effect.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=520, anchor="w").pack(anchor="w", pady=(0, 6))
    app.ui_scale_var = ctk.StringVar(value=_initial_scale_label)
    def _on_ui_scale_change(choice):
        val = label_to_scale.get(choice)
        app.settings["ui_scale"] = val
        save_settings(app.settings)
        messagebox.showinfo("Scale saved", "Restart the app for the new scale to take effect.", parent=root)
    app.ui_scale_menu = ctk.CTkOptionMenu(settings_card, values=scale_labels, variable=app.ui_scale_var, width=220, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=_on_ui_scale_change)
    app.ui_scale_menu.pack(anchor="w", pady=(0, UI_PAD_LG))

    # OpenAI API key (stored securely in user app data)
    ctk.CTkLabel(settings_card, text="OpenAI API key", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(anchor="w", pady=(0, 4))
    ctk.CTkLabel(settings_card, text="Required for AI Summary. Stored on this device only.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=520, anchor="w").pack(anchor="w", pady=(0, 6))
    api_key_row = ctk.CTkFrame(settings_card, fg_color="transparent")
    api_key_row.pack(fill="x", pady=(0, UI_PAD_LG))
    app.openai_key_entry = ctk.CTkEntry(api_key_row, width=400, height=32, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), show="•", placeholder_text="Enter key to save, or leave blank to keep existing")
    app.openai_key_entry.pack(side="left", padx=(0, UI_PAD))
    def _save_openai_key():
        key = app.openai_key_entry.get().strip()
        if set_openai_api_key(key):
            app.openai_key_entry.delete(0, "end")
            app.openai_key_entry.configure(placeholder_text="Saved. Key is stored on this device only.")
            app.openai_key_status_label.configure(text="A key is already saved. Enter a new key and click Save key to replace it.")
        else:
            messagebox.showerror("Save failed", "Could not save API key. Check write access to app data folder.", parent=root)
    def _clear_openai_key():
        if not messagebox.askyesno("Clear API key", "Remove the stored API key from this device? You can add it again later.", parent=root):
            return
        if clear_openai_api_key():
            app.openai_key_entry.delete(0, "end")
            app.openai_key_entry.configure(placeholder_text="Enter key to save, or leave blank to keep existing")
            app.openai_key_status_label.configure(text="No key saved yet. Get an API key from platform.openai.com and paste it above.")
        else:
            messagebox.showerror("Clear failed", "Could not remove the API key file.", parent=root)
    ctk.CTkButton(api_key_row, text="Save key", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=84, height=32, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], command=_save_openai_key).pack(side="left")
    ctk.CTkButton(api_key_row, text="Clear key", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=84, height=32, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_clear_openai_key).pack(side="left", padx=(UI_PAD, 0))
    status_text = "A key is already saved. Enter a new key and click Save key to replace it." if get_openai_api_key() else "No key saved yet. Get an API key from platform.openai.com and paste it above."
    app.openai_key_status_label = ctk.CTkLabel(settings_card, text=status_text, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), text_color="gray", wraplength=520, anchor="w")
    app.openai_key_status_label.pack(anchor="w", pady=(0, UI_PAD_LG))

    ctk.CTkLabel(settings_card, text="Capture mode", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(anchor="w", pady=(0, 4))
    ctk.CTkLabel(settings_card, text="Meeting = in-process mic + loopback (PyAudioWPatch; loopback read only when data available). Loopback device below applies to Meeting mode.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=520, anchor="w").pack(anchor="w", pady=(0, UI_PAD))
    mode_values = ["Default input", "Loopback (system audio)", "Meeting (mic + loopback)"]
    mode_to_val = {AUDIO_MODE_DEFAULT: mode_values[0], AUDIO_MODE_LOOPBACK: mode_values[1], AUDIO_MODE_MEETING: mode_values[2]}
    val_to_mode = {v: k for k, v in mode_to_val.items()}
    app.audio_mode_var = ctk.StringVar(value=mode_to_val.get(app.settings.get("audio_mode")) or ("Meeting (mic + loopback)" if app.settings.get("audio_mode") == "meeting_ffmpeg" else mode_values[0]))
    app.audio_mode_menu = ctk.CTkOptionMenu(settings_card, values=mode_values, variable=app.audio_mode_var, width=320, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=lambda v: _apply_settings(app))
    app.audio_mode_menu.pack(anchor="w", pady=(0, UI_PAD_LG))

    ctk.CTkLabel(settings_card, text="Meeting microphone (used when Capture mode = Meeting)", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small, weight="bold")).pack(anchor="w", pady=(UI_PAD, 4))
    input_devices, _ = list_audio_devices()
    input_options = ["System Default"] + [f"{d['index']}: {d['name']}" for d in input_devices if d.get("max_input_channels", 0) > 0]
    app.meeting_mic_var = ctk.StringVar(value="System Default")
    meeting_mic_idx = app.settings.get("meeting_mic_device")
    if meeting_mic_idx is not None:
        for d in input_devices:
            if d["index"] == meeting_mic_idx:
                app.meeting_mic_var.set(f"{d['index']}: {d['name']}")
                break
    app.meeting_mic_menu = ctk.CTkOptionMenu(settings_card, values=input_options, variable=app.meeting_mic_var, width=400, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=lambda v: _apply_settings(app))
    app.meeting_mic_menu.pack(anchor="w", pady=(0, UI_PAD_LG))

    ctk.CTkLabel(settings_card, text="Loopback device (used for Loopback mode and for Meeting mode)", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small, weight="bold")).pack(anchor="w", pady=(UI_PAD, 4))
    loopback_devices, lb_err = list_loopback_devices()
    loopback_options = ["System Default"] + [f"{d['index']}: {d['name']}" for d in loopback_devices]
    if lb_err:
        loopback_options = ["System Default", f"(Error: {lb_err})"]
    app.loopback_device_var = ctk.StringVar(value="System Default")
    lb_idx = app.settings.get("loopback_device_index")
    if lb_idx is not None and loopback_devices:
        for d in loopback_devices:
            if d["index"] == lb_idx:
                app.loopback_device_var.set(f"{d['index']}: {d['name']}")
                break
    app.loopback_device_menu = ctk.CTkOptionMenu(settings_card, values=loopback_options, variable=app.loopback_device_var, width=400, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=lambda v: _apply_settings(app))
    app.loopback_device_menu.pack(anchor="w", pady=(0, UI_PAD))

    def _apply_settings(app):
        mode_str = app.audio_mode_var.get()
        mode = val_to_mode.get(mode_str, AUDIO_MODE_DEFAULT)
        meeting_val = app.meeting_mic_var.get()
        meeting_idx = None
        if meeting_val and meeting_val != "System Default" and ":" in meeting_val:
            try:
                meeting_idx = int(meeting_val.split(":")[0].strip())
            except ValueError:
                pass
        loopback_val = app.loopback_device_var.get()
        loopback_idx = None
        if loopback_val and loopback_val != "System Default" and ":" in loopback_val and not loopback_val.startswith("(Error:"):
            try:
                loopback_idx = int(loopback_val.split(":")[0].strip())
            except ValueError:
                pass
        app.settings = {**app.settings, "audio_mode": mode, "meeting_mic_device": meeting_idx, "loopback_device_index": loopback_idx}
        save_settings(app.settings)

    mode = app.settings.get("audio_mode") or AUDIO_MODE_DEFAULT
    mode_label = {"default": "Default input", "loopback": "Loopback", "meeting": "Meeting (mic + loopback)", "meeting_ffmpeg": "Meeting (mic + loopback)"}.get(mode, "Default input")
    devices_list, _ = list_audio_devices()
    dev_idx, dev_err = get_default_monitor_device()
    if dev_err:
        dev_info = f"Capture: {mode_label} — {dev_err}"
    elif dev_idx is not None and devices_list:
        name = next((d["name"] for d in devices_list if d["index"] == dev_idx), f"Device {dev_idx}")
        dev_info = f"Capture: {mode_label} · Input: {name}"
    else:
        dev_info = f"Capture: {mode_label}"
    ctk.CTkLabel(main_content, text=dev_info, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray").pack(anchor="w", padx=UI_PAD_LG, pady=(0, 4))
    app.model_status_warning_color = COLORS["error_text"][1]
    app.model_status_label = ctk.CTkLabel(main_content, textvariable=app.model_status_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=700)
    app.model_status_label.pack(anchor="w", padx=UI_PAD_LG, pady=(0, UI_PAD))
    # Model status and Record button are updated by _startup_check_done after background check (and optional load).

    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_event.set(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
