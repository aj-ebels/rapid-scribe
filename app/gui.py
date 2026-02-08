"""
GUI and app controller: CustomTkinter window, tabs, start/stop recording, prompts dialog.
"""
import math
import sys
import queue
import threading
from datetime import date
from pathlib import Path

import numpy as np
import customtkinter as ctk
import sounddevice as sd
from tkinter import messagebox, filedialog

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

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
    transcription_worker,
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

if sys.platform == "win32":
    from .audio_mixer import AudioMixer
    from .chunk_recorder import ChunkRecorder


def poll_text_queue(app):
    """Called periodically from main thread to append new transcript text."""
    try:
        while True:
            line = app.text_queue.get_nowait()
            app.log.insert("end", line)
            app.log.see("end")
    except queue.Empty:
        pass
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
                # Map RMS to 0..1 (speech roughly 0.004–0.3)
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
        if app.transcription_thread and app.transcription_thread.is_alive():
            app.transcription_thread.join(timeout=10)
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
            app.recorder = ChunkRecorder(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                chunk_duration_sec=float(chunk_sec),
                asr_sample_rate=SAMPLE_RATE,
                on_chunk_ready=lambda wav_path, rms: meeting_chunk_ready(app, wav_path, rms),
                use_silence_chunking=use_silence,
                min_chunk_sec=float(min_sec),
                max_chunk_sec=float(max_sec),
                silence_duration_sec=float(silence_sec),
            )
            app.mixer = AudioMixer(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                frames_per_read=FRAMES_PER_READ_MEETING,
                gain=MIXER_GAIN_MEETING,
            )
            app.mixer.set_stereo_callback(app.recorder.push_stereo)
            def _push_level(rms):
                try:
                    app.level_queue.put_nowait(rms)
                except queue.Full:
                    pass
            app.mixer.set_level_callback(_push_level)
            app.mixer.start(loopback_device_index=loopback_idx, mic_device_index=None)
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
    app.transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(app.chunk_queue, app.text_queue, app.stop_event, model_id),
        daemon=True,
    )
    app.transcription_thread.start()
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
    scale = _get_dpi_scale()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    _fs = lambda base: max(12, min(22, round(base * scale)))
    F = type("F", (), {"title": _fs(18), "header": _fs(16), "body": _fs(14), "small": _fs(13), "tiny": _fs(12)})()

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
    root.title("Blue Bridge Meeting Companion")
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
    app.stop_event = threading.Event()
    app.chunk_queue = queue.Queue(maxsize=1)
    app.text_queue = queue.Queue()
    app.level_queue = queue.Queue(maxsize=32)
    app.capture_thread = None
    app.transcription_thread = None
    app.capture_threads = []
    app.settings = load_settings()

    # Attach pulse helpers to app so module-level start_stop() can call them
    app._start_transcript_pulse = _start_transcript_pulse
    app._clear_transcript_pulse = _clear_transcript_pulse
    app._start_summary_pulse = _start_summary_pulse
    app._clear_summary_pulse = _clear_summary_pulse

    content_frame = ctk.CTkFrame(root, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=UI_PAD, pady=UI_PAD)
    main_content = ctk.CTkFrame(content_frame, fg_color="transparent")
    main_content.pack(fill="both", expand=True)

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

    # Transcript tab — three equal-width columns: Manual Notes | Transcript | AI Summary
    # Use a 3-row grid so the large text boxes align across columns (row 2).
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
        want_auto_name = (
            not (app.export_name_var.get() or "").strip()
            and app.auto_generate_export_name_var.get()
        )
        def worker():
            ok, out = generate_ai_summary(api_key, prompt_obj["prompt"], transcript, manual_notes=manual_notes)
            generated_name = None
            if ok and want_auto_name and (out or "").strip():
                name_ok, name_result = generate_export_name(api_key, (out or "")[:250])
                if name_ok and name_result:
                    generated_name = name_result
            result_holder.append((ok, out, generated_name))
        def check_done():
            if not result_holder:
                root.after(200, check_done)
                return
            ok, out, generated_name = result_holder[0]
            app._summary_generating = False
            _clear_summary_pulse(app)
            app.summary_generate_btn.configure(state="normal")
            app.summary_status_var.set("")
            if ok:
                app.summary_text.delete("1.0", "end")
                app.summary_text.insert("1.0", out)
                if generated_name:
                    safe = "".join(c if c.isalnum() or c in "._- " else "-" for c in generated_name)
                    safe = safe.replace(" ", "-").strip("-") or generated_name
                    app.export_name_var.set(safe)
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
        name_part = (app.export_name_var.get() or "").strip()
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

    export_row = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    export_row.pack(fill="x", pady=(UI_PAD, UI_PAD_LG), padx=UI_PAD_LG)
    ctk.CTkLabel(export_row, text="Export file name:", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small)).pack(side="left", padx=(UI_PAD, 4), pady=4)
    app.export_name_var = ctk.StringVar(value="")
    app.export_name_entry = ctk.CTkEntry(export_row, textvariable=app.export_name_var, width=280, height=28, placeholder_text="e.g. meeting-notes", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small))
    app.export_name_entry.pack(side="left", padx=4, pady=4)
    app.export_prepend_date_var = ctk.BooleanVar(value=app.settings.get("export_prepend_date", True))
    def _on_prepend_date_changed():
        app.settings["export_prepend_date"] = app.export_prepend_date_var.get()
        save_settings(app.settings)
    ctk.CTkCheckBox(
        export_row, text="Prepend today's date", variable=app.export_prepend_date_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=_on_prepend_date_changed,
        border_width=1, corner_radius=3, checkbox_width=18, checkbox_height=18
    ).pack(side="left", padx=8, pady=4)
    app.auto_generate_export_name_var = ctk.BooleanVar(value=app.settings.get("auto_generate_export_name", True))
    def _on_auto_generate_name_changed():
        app.settings["auto_generate_export_name"] = app.auto_generate_export_name_var.get()
        save_settings(app.settings)
    app.auto_generate_export_name_cb = ctk.CTkCheckBox(
        export_row, text="Also AI-generate file name if blank", variable=app.auto_generate_export_name_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), command=_on_auto_generate_name_changed,
        border_width=1, corner_radius=3, checkbox_width=18, checkbox_height=18
    )
    app.auto_generate_export_name_cb.pack(side="left", padx=8, pady=4)
    ctk.CTkButton(export_row, text="Export", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=64, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_export_markdown).pack(side="left", padx=(0, 0), pady=4)

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
