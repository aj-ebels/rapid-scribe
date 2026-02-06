#!/usr/bin/env python3
"""
Real-time system audio capture and transcription.
Uses PulseAudio Monitor / RDPSource on WSL2, NVIDIA Parakeet (onnx-asr), and CustomTkinter GUI.
Same Parakeet/ONNX stack as Meetily: https://github.com/Zackriya-Solutions/meeting-minutes
"""

import os
import sys
import json
import tempfile
import threading
import queue
import uuid
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass  # no-op if python-dotenv not installed

# Ensure WSLg can display the GUI (PulseAudio + X11/Wayland). No-op on Windows.
try:
    release = os.uname().release
    if "WSL" in release or "microsoft" in release.lower():
        if "DISPLAY" not in os.environ or not os.environ["DISPLAY"].strip():
            os.environ.setdefault("DISPLAY", ":0")
except AttributeError:
    pass  # os.uname() not available (e.g. Windows)

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import customtkinter as ctk
from tkinter import messagebox, filedialog

# Lazy load Parakeet via onnx-asr (same stack as Meetily: istupakov ONNX Parakeet)
# Model: nemo-parakeet-tdt-0.6b-v2 (en) or nemo-parakeet-tdt-0.6b-v3 (multilingual)
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"
_parakeet_model = None

def get_transcription_model():
    global _parakeet_model
    if _parakeet_model is None:
        import onnx_asr
        _parakeet_model = onnx_asr.load_model(
            PARAKEET_MODEL,
            quantization="int8",  # faster on CPU, similar to previous Whisper int8
        )
    return _parakeet_model


# -----------------------------------------------------------------------------
# Installed transcription models (Hugging Face cache)
# -----------------------------------------------------------------------------

# Substrings in repo_id that we treat as "transcription" models (ASR, Whisper, Parakeet, etc.)
ASR_REPO_PATTERNS = (
    "parakeet", "whisper", "asr", "speech", "stt", "vosk", "gigaam", "canary",
    "conformer", "transcribe",
)


def _format_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"


def list_installed_transcription_models():
    """List cached Hugging Face models that look like ASR/transcription. Returns list of dicts."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return [], "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
    except Exception as e:
        return [], str(e)
    out = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        rid = (repo.repo_id or "").lower()
        if not any(p in rid for p in ASR_REPO_PATTERNS):
            continue
        out.append({
            "repo_id": repo.repo_id,
            "size_on_disk": repo.size_on_disk,
            "size_str": _format_size(repo.size_on_disk),
            "revision_hashes": [r.commit_hash for r in repo.revisions],
        })
    out.sort(key=lambda x: x["repo_id"])
    return out, None


def uninstall_transcription_model(repo_id, revision_hashes):
    """Remove a cached model from the Hugging Face cache. Returns (success, error_message)."""
    if not revision_hashes:
        return False, "No revisions to delete"
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False, "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
        strategy = cache.delete_revisions(*revision_hashes)
        strategy.execute()
        return True, None
    except Exception as e:
        return False, str(e)


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
# Use OS temp dir so it works on Windows (no /tmp) and Linux/WSL
CHUNK_PATH = Path(tempfile.gettempdir()) / "meetings_chunk.wav"

# Skip transcribing chunks that are effectively silent (avoids Whisper "hallucinations"
# where it invents phrases like "Thanks for watching" when there's no real speech).
SILENCE_RMS_THRESHOLD = 0.01  # float32 RMS; increase if real speech is being skipped


def _is_silent(chunk: np.ndarray) -> bool:
    """True if chunk has very low energy (silence or negligible noise)."""
    rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
    return rms < SILENCE_RMS_THRESHOLD


# -----------------------------------------------------------------------------
# Audio capture thread
# -----------------------------------------------------------------------------

def capture_worker(device_index, chunk_queue, stop_event):
    """Record chunks; save to temp chunk file only when not silent, put path in queue."""
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
            if _is_silent(chunk):
                continue  # Skip silent chunks — don't send to Whisper (avoids hallucinations)
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
    """Take WAV paths from chunk_queue, transcribe with Parakeet, push text to text_queue, delete file."""
    model = get_transcription_model()
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
                result = model.recognize(path)
                text = result if isinstance(result, str) else getattr(result, "text", str(result))
                if text and isinstance(text, str) and text.strip():
                    text_queue.put_nowait(text.strip() + "\n")
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
# AI prompts storage and OpenAI summary
# -----------------------------------------------------------------------------

PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.json"
TRANSCRIPT_PLACEHOLDER = "{{transcript}}"


def load_prompts():
    """Load prompt templates from JSON file. Returns list of dicts with id, name, prompt."""
    if not PROMPTS_FILE.exists():
        return []
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_prompts(prompts):
    """Save prompt templates to JSON file."""
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def add_prompt(name, prompt_text):
    """Add a new prompt template. Returns the new prompt dict."""
    prompts = load_prompts()
    p = {"id": str(uuid.uuid4()), "name": name, "prompt": prompt_text}
    prompts.append(p)
    save_prompts(prompts)
    return p


def update_prompt(prompt_id, name, prompt_text):
    """Update an existing prompt. Returns True if found."""
    prompts = load_prompts()
    for p in prompts:
        if p.get("id") == prompt_id:
            p["name"] = name
            p["prompt"] = prompt_text
            save_prompts(prompts)
            return True
    return False


def delete_prompt(prompt_id):
    """Delete a prompt by id. Returns True if found and deleted."""
    prompts = load_prompts()
    new_list = [p for p in prompts if p.get("id") != prompt_id]
    if len(new_list) == len(prompts):
        return False
    save_prompts(new_list)
    return True


def get_prompt_by_id(prompt_id):
    """Return prompt dict by id or None."""
    for p in load_prompts():
        if p.get("id") == prompt_id:
            return p
    return None


def generate_ai_summary(api_key, prompt_template, transcript):
    """
    Call OpenAI API to generate summary. Runs in thread.
    Returns (success, result_text_or_error).
    """
    if not (prompt_template or "").strip():
        return False, "Prompt template is empty."
    if not (transcript or "").strip():
        return False, "Transcript is empty."
    text = prompt_template.replace(TRANSCRIPT_PLACEHOLDER, transcript)
    try:
        from openai import OpenAI
    except ImportError:
        return False, "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
        )
        content = response.choices[0].message.content
        return True, (content or "").strip()
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

def poll_text_queue(app):
    """Called periodically from main thread to append new transcript text."""
    try:
        while True:
            line = app.text_queue.get_nowait()
            app.log.insert("end", line)
            app.log.see("end")
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
        app.start_btn.configure(state="normal")
        app.stop_btn.configure(state="disabled")
        app.status_var.set("Stopped")
        return
    # Start
    app.stop_event.clear()
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
    app.transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(app.chunk_queue, app.text_queue, app.stop_event),
        daemon=True,
    )
    app.capture_thread.start()
    app.transcription_thread.start()
    app.running = True
    app.start_btn.configure(state="disabled")
    app.stop_btn.configure(state="normal")
    app.status_var.set("Recording & transcribing…")
    poll_text_queue(app)


def _open_edit_prompt_dialog(parent, prompt_id, on_saved, ui_pad, ui_radius, font_family, font_sizes, colors):
    """Open dialog to add or edit a prompt template (name + prompt text with {{transcript}})."""
    prompt = get_prompt_by_id(prompt_id) if prompt_id else None
    is_new = prompt is None

    win = ctk.CTkToplevel(parent)
    win.title("Add prompt" if is_new else "Edit prompt")
    win.geometry("520x380")
    win.transient(parent)
    # Defer grab until window is viewable (avoids TclError: grab failed: window not viewable)
    def _set_grab():
        try:
            win.grab_set()
        except Exception:
            pass
    win.after(100, _set_grab)

    ctk.CTkLabel(
        win,
        text="Name",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
    ).pack(anchor="w", padx=ui_pad, pady=(ui_pad, 2))
    name_var = ctk.StringVar(value=prompt.get("name", "") if prompt else "")
    name_entry = ctk.CTkEntry(win, width=400, height=32, font=ctk.CTkFont(family=font_family, size=font_sizes.body))
    name_entry.pack(anchor="w", padx=ui_pad, pady=(0, ui_pad))
    name_entry.insert(0, name_var.get())

    ctk.CTkLabel(
        win,
        text="Prompt (use {{transcript}} where the transcript should go)",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
    ).pack(anchor="w", padx=ui_pad, pady=(ui_pad, 2))
    prompt_text = ctk.CTkTextbox(win, width=500, height=180, font=ctk.CTkFont(family=font_family, size=font_sizes.small))
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
        if TRANSCRIPT_PLACEHOLDER not in text:
            if not messagebox.askyesno("No placeholder", f"Prompt does not contain '{TRANSCRIPT_PLACEHOLDER}'. Add it so the transcript is inserted?", parent=win):
                return
        if is_new:
            add_prompt(name, text)
        else:
            update_prompt(prompt_id, name, text)
        on_saved()
        win.destroy()

    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill="x", padx=ui_pad, pady=ui_pad)
    ctk.CTkButton(
        btn_frame,
        text="Save",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
        width=80,
        height=32,
        corner_radius=ui_radius,
        fg_color=colors["primary_fg"],
        hover_color=colors["primary_hover"],
        command=save,
    ).pack(side="left", padx=(0, ui_pad))
    ctk.CTkButton(
        btn_frame,
        text="Cancel",
        font=ctk.CTkFont(family=font_family, size=font_sizes.small),
        width=80,
        height=32,
        corner_radius=ui_radius,
        fg_color=colors["secondary_fg"],
        hover_color=colors["secondary_hover"],
        command=win.destroy,
    ).pack(side="left")


# -----------------------------------------------------------------------------
# DPI scaling for high-DPI displays (used for CustomTkinter scaling)
# -----------------------------------------------------------------------------


def _get_dpi_scale():
    """Return scale factor for high-DPI displays. Aggressive scaling for laptop readability."""
    scale = 1.0
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        _root.update_idletasks()
        dpi = _root.winfo_fpixels("1i")
        if dpi and dpi > 0:
            scale = max(1.0, min(2.0, dpi / 96.0))
        # Windows: Tk often reports 96 on high-DPI; force big scale for laptops
        if scale <= 1.0 and sys.platform == "win32":
            h = _root.winfo_screenheight()
            if h >= 900:
                scale = 1.85  # big fonts on laptop screens
        _root.destroy()
    except Exception:
        pass
    return scale


def main():
    # Load Parakeet model before starting the GUI
    print("Loading Parakeet model (first run may download ~660MB from Hugging Face)...")
    get_transcription_model()
    print("Model ready. Opening window...")

    # CustomTkinter theme and scaling (before creating window)
    ctk.set_appearance_mode("dark")
    theme_path = Path(__file__).resolve().parent / "themes" / "meetings-dark.json"
    if theme_path.exists():
        ctk.set_default_color_theme(str(theme_path))
    else:
        ctk.set_default_color_theme("dark-blue")
    scale = _get_dpi_scale()
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    # Font sizes: clean app-like scale (cap 12–22)
    _fs = lambda base: max(12, min(22, round(base * scale)))
    F = type("F", (), {"title": _fs(18), "header": _fs(16), "body": _fs(14), "small": _fs(13), "tiny": _fs(12)})()

    # UI theme constants (spacing, radius, colors)
    UI_RADIUS = 10
    UI_PAD = 12
    UI_PAD_LG = 16
    COLORS = {
        "sidebar": ("gray88", "gray18"),
        "card": ("gray92", "gray18"),
        "header": ("gray92", "gray18"),
        "primary_fg": ("#2e7d32", "#1b5e20"),
        "primary_hover": ("#388e3c", "#2e7d32"),
        "danger_fg": ("#c62828", "#b71c1c"),
        "danger_hover": ("#d32f2f", "#c62828"),
        "secondary_fg": ("gray70", "gray35"),
        "secondary_hover": ("gray60", "gray45"),
        "textbox_bg": ("gray97", "gray14"),
        "error_text": ("red", "#f7768e"),
    }
    # Modern UI font per platform (mono kept for transcript only)
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
    root.title("System Audio → Real-time Transcription")
    root.geometry("960x480")
    root.minsize(720, 380)

    app = type("App", (), {})()
    app.root = root
    app.running = False
    app.stop_event = threading.Event()
    app.chunk_queue = queue.Queue(maxsize=1)
    app.text_queue = queue.Queue()
    app.capture_thread = None
    app.transcription_thread = None

    # Main horizontal layout: sidebar | content
    content_frame = ctk.CTkFrame(root, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=UI_PAD, pady=UI_PAD)

    # ---- Left sidebar: installed transcription models ----
    sidebar = ctk.CTkFrame(content_frame, width=260, corner_radius=UI_RADIUS, fg_color=COLORS["sidebar"])
    sidebar.pack(side="left", fill="y", padx=(0, UI_PAD))
    sidebar.pack_propagate(False)
    ctk.CTkLabel(
        sidebar,
        text="Installed models",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.title, weight="bold"),
    ).pack(anchor="w", padx=UI_PAD, pady=(UI_PAD, 6))
    sidebar_scroll = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
    sidebar_scroll.pack(fill="both", expand=True)

    def refresh_sidebar_models():
        for w in sidebar_scroll.winfo_children():
            w.destroy()
        models, err = list_installed_transcription_models()
        if err:
            ctk.CTkLabel(
                sidebar_scroll,
                text=f"Error: {err[:50]}…" if len(err) > 50 else err,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                text_color=COLORS["error_text"],
                wraplength=220,
            ).pack(anchor="w", padx=UI_PAD, pady=4)
            return
        if not models:
            ctk.CTkLabel(
                sidebar_scroll,
                text="No transcription models in cache.",
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                text_color="gray",
                wraplength=220,
            ).pack(anchor="w", padx=UI_PAD, pady=4)
            return
        for m in models:
            row = ctk.CTkFrame(sidebar_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(
                row,
                text=f"{m['repo_id']}\n{m['size_str']}",
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                wraplength=160,
                anchor="w",
            ).pack(side="left", padx=(UI_PAD, 4))
            def _uninstall(repo_id=m["repo_id"], hashes=m["revision_hashes"]):
                if not messagebox.askyesno("Uninstall model", f"Delete cached model '{repo_id}'? This frees disk space; you can re-download later."):
                    return
                ok, err = uninstall_transcription_model(repo_id, hashes)
                if ok:
                    messagebox.showinfo("Uninstalled", f"Removed {repo_id} from cache.")
                    refresh_sidebar_models()
                else:
                    messagebox.showerror("Error", err or "Failed to delete.")
            ctk.CTkButton(
                row,
                text="Uninstall",
                width=70,
                height=28,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
                corner_radius=UI_RADIUS,
                fg_color=COLORS["danger_fg"],
                hover_color=COLORS["danger_hover"],
                command=_uninstall,
            ).pack(side="right", padx=(0, UI_PAD))

    refresh_sidebar_models()
    ctk.CTkButton(
        sidebar,
        text="Refresh list",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=refresh_sidebar_models,
    ).pack(pady=(4, UI_PAD))

    # ---- Right: main content (header + tabview + footer) ----
    main_content = ctk.CTkFrame(content_frame, fg_color="transparent")
    main_content.pack(side="left", fill="both", expand=True)

    # Header bar
    header = ctk.CTkFrame(main_content, fg_color=COLORS["header"], corner_radius=UI_RADIUS, height=52)
    header.pack(fill="x", pady=(0, UI_PAD))
    header.pack_propagate(False)
    app.status_var = ctk.StringVar(value="Ready — click Start to begin")
    ctk.CTkLabel(
        header,
        textvariable=app.status_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(side="left", padx=UI_PAD_LG, pady=UI_PAD)
    btn_frame = ctk.CTkFrame(header, fg_color="transparent")
    btn_frame.pack(side="right", padx=UI_PAD_LG, pady=UI_PAD)
    app.start_btn = ctk.CTkButton(
        btn_frame,
        text="Start",
        command=lambda: start_stop(app),
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
        width=100,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["primary_fg"],
        hover_color=COLORS["primary_hover"],
    )
    app.start_btn.pack(side="left", padx=(0, UI_PAD))
    app.stop_btn = ctk.CTkButton(
        btn_frame,
        text="Stop",
        command=lambda: start_stop(app),
        state="disabled",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
        width=100,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["danger_fg"],
        hover_color=COLORS["danger_hover"],
    )
    app.stop_btn.pack(side="left")

    # Tabview: Transcript | AI Prompts
    tabview = ctk.CTkTabview(main_content, fg_color=COLORS["card"], corner_radius=UI_RADIUS)
    tabview.pack(fill="both", expand=True, pady=(0, UI_PAD))
    tab_transcript = tabview.add("Transcript")
    tab_prompts = tabview.add("AI Prompts")

    # ---- Transcript tab ----
    card = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    card.pack(fill="both", expand=True)
    card_header = ctk.CTkFrame(card, fg_color="transparent")
    card_header.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    ctk.CTkLabel(
        card_header,
        text="Transcript",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(side="left")
    def copy_transcript():
        text = app.log.get("1.0", "end")
        text = text.rstrip()
        if text:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()  # keep clipboard content after window loses focus
    def clear_transcript():
        app.log.delete("1.0", "end")
    ctk.CTkButton(
        card_header,
        text="Copy transcript",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=140,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=copy_transcript,
    ).pack(side="left", padx=(UI_PAD, 0), pady=4)
    ctk.CTkButton(
        card_header,
        text="Clear",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=80,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=clear_transcript,
    ).pack(side="right", padx=UI_PAD, pady=4)
    app.log = ctk.CTkTextbox(
        card,
        wrap="word",
        font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body),
        corner_radius=8,
        border_width=0,
        fg_color=COLORS["textbox_bg"],
        border_spacing=UI_PAD,
    )
    app.log.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=(0, UI_PAD))

    # AI Summary section: header with prompt selector and Generate button
    summary_header = ctk.CTkFrame(card, fg_color="transparent")
    summary_header.pack(fill="x", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    ctk.CTkLabel(
        summary_header,
        text="AI Summary",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"),
    ).pack(side="left")
    # Prompt template selector (refresh list from prompts file)
    _prompts_for_summary = load_prompts()
    _prompt_names = [p.get("name", "Unnamed") for p in _prompts_for_summary] or ["(No prompts — add in AI Prompts tab)"]
    app.summary_prompt_var = ctk.StringVar(value=_prompt_names[0])
    app.summary_prompt_menu = ctk.CTkOptionMenu(
        summary_header,
        values=_prompt_names,
        variable=app.summary_prompt_var,
        width=220,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    )
    app.summary_prompt_menu.pack(side="left", padx=(UI_PAD, 0), pady=4)
    app.summary_status_var = ctk.StringVar(value="")
    ctk.CTkLabel(
        summary_header,
        textvariable=app.summary_status_var,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
    ).pack(side="left", padx=(UI_PAD, 0), pady=4)

    def _refresh_summary_prompt_menu():
        prompts = load_prompts()
        names = [p.get("name", "Unnamed") for p in prompts] or ["(No prompts — add in AI Prompts tab)"]
        app.summary_prompt_menu.configure(values=names)
        if names and (app.summary_prompt_var.get() not in names):
            app.summary_prompt_var.set(names[0])

    def _do_ai_summary():
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            messagebox.showerror("API key required", "Set OPENAI_API_KEY in your environment or .env file.", parent=root)
            return
        prompts = load_prompts()
        if not prompts:
            messagebox.showinfo("No prompts", "Create at least one prompt in the 'AI Prompts' tab first.", parent=root)
            return
        name = app.summary_prompt_var.get()
        prompt_obj = next((p for p in prompts if p.get("name") == name), None)
        if not prompt_obj or name.startswith("("):
            messagebox.showwarning("Select a prompt", "Choose a prompt template from the dropdown.", parent=root)
            return
        transcript = app.log.get("1.0", "end").strip()
        if not transcript:
            messagebox.showwarning("Empty transcript", "Transcript is empty. Record or paste some text first.", parent=root)
            return
        app.summary_generate_btn.configure(state="disabled")
        app.summary_status_var.set("Generating…")
        result_holder = []

        def worker():
            ok, out = generate_ai_summary(api_key, prompt_obj["prompt"], transcript)
            result_holder.append((ok, out))

        def check_done():
            if not result_holder:
                root.after(200, check_done)
                return
            ok, out = result_holder[0]
            app.summary_generate_btn.configure(state="normal")
            app.summary_status_var.set("")
            if ok:
                app.summary_text.delete("1.0", "end")
                app.summary_text.insert("1.0", out)
            else:
                messagebox.showerror("AI Summary failed", out, parent=root)

        threading.Thread(target=worker, daemon=True).start()
        root.after(200, check_done)

    app.summary_generate_btn = ctk.CTkButton(
        summary_header,
        text="Generate",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=100,
        height=36,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["primary_fg"],
        hover_color=COLORS["primary_hover"],
        command=_do_ai_summary,
    )
    app.summary_generate_btn.pack(side="left", padx=(UI_PAD, 0), pady=4)
    ctk.CTkButton(
        summary_header,
        text="Refresh list",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
        width=80,
        height=28,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=_refresh_summary_prompt_menu,
    ).pack(side="left", padx=(UI_PAD, 0), pady=4)
    # Export: naming field + Export button (saves summary + transcript as markdown)
    ctk.CTkLabel(
        summary_header,
        text="Export:",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    ).pack(side="left", padx=(UI_PAD_LG, 4), pady=4)
    app.export_name_var = ctk.StringVar(value="")
    app.export_name_entry = ctk.CTkEntry(
        summary_header,
        textvariable=app.export_name_var,
        width=160,
        height=28,
        placeholder_text="e.g. meeting-notes",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
    )
    app.export_name_entry.pack(side="left", padx=4, pady=4)

    def _export_markdown():
        summary = app.summary_text.get("1.0", "end").strip()
        transcript = app.log.get("1.0", "end").strip()
        if not summary and not transcript:
            messagebox.showwarning("Nothing to export", "Add an AI summary and/or transcript first.", parent=root)
            return
        name_part = (app.export_name_var.get() or "").strip()
        # Sanitize for filename: replace spaces and invalid chars with hyphen
        if name_part:
            name_part = "".join(c if c.isalnum() or c in "._- " else "-" for c in name_part)
            name_part = name_part.replace(" ", "-").strip("-") or "export"
        else:
            name_part = "export"
        default_name = f"{date.today().isoformat()} {name_part}.md"
        path = filedialog.asksaveasfilename(
            parent=root,
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return
        parts = []
        if summary:
            parts.append("## Summary\n\n" + summary)
        if transcript:
            parts.append("## Transcript\n\n" + transcript)
        content = "\n\n---\n\n".join(parts) + "\n"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Exported", f"Saved to {path}", parent=root)
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=root)

    ctk.CTkButton(
        summary_header,
        text="Export",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        width=80,
        height=28,
        corner_radius=UI_RADIUS,
        fg_color=COLORS["secondary_fg"],
        hover_color=COLORS["secondary_hover"],
        command=_export_markdown,
    ).pack(side="left", padx=(UI_PAD, 0), pady=4)

    app.summary_text = ctk.CTkTextbox(
        card,
        wrap="word",
        font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body),
        corner_radius=8,
        border_width=0,
        fg_color=COLORS["textbox_bg"],
        border_spacing=UI_PAD,
    )
    app.summary_text.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=(0, UI_PAD_LG))

    # ---- AI Prompts tab: list, add, edit, delete ----
    ctk.CTkLabel(
        tab_prompts,
        text="Custom prompt templates for AI Summary. Use {{transcript}} where the transcript should be inserted.",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
        wraplength=500,
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 4))
    prompts_scroll = ctk.CTkScrollableFrame(tab_prompts, fg_color="transparent")
    prompts_scroll.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=UI_PAD)

    def refresh_prompts_list():
        for w in prompts_scroll.winfo_children():
            w.destroy()
        prompts = load_prompts()
        if not prompts:
            ctk.CTkLabel(
                prompts_scroll,
                text="No prompts yet. Click 'Add prompt' to create one.",
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
                text_color="gray",
            ).pack(anchor="w", padx=UI_PAD, pady=8)
        for p in prompts:
            row = ctk.CTkFrame(prompts_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(
                row,
                text=p.get("name", "Unnamed"),
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.body, weight="bold"),
                anchor="w",
            ).pack(side="left", padx=(UI_PAD, 8))
            def _edit(pid=p["id"]):
                _open_edit_prompt_dialog(root, pid, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)
            def _delete(pid=p["id"], pname=p.get("name", "?")):
                if not messagebox.askyesno("Delete prompt", f"Delete prompt '{pname}'?"):
                    return
                if delete_prompt(pid):
                    refresh_prompts_list()
                    messagebox.showinfo("Deleted", "Prompt deleted.")
            ctk.CTkButton(
                row,
                text="Edit",
                width=60,
                height=28,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
                corner_radius=UI_RADIUS,
                fg_color=COLORS["secondary_fg"],
                hover_color=COLORS["secondary_hover"],
                command=_edit,
            ).pack(side="right", padx=4)
            ctk.CTkButton(
                row,
                text="Delete",
                width=60,
                height=28,
                font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny),
                corner_radius=UI_RADIUS,
                fg_color=COLORS["danger_fg"],
                hover_color=COLORS["danger_hover"],
                command=_delete,
            ).pack(side="right", padx=(0, UI_PAD))

    def add_new_prompt():
        _open_edit_prompt_dialog(root, None, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)

    refresh_prompts_list()
    ctk.CTkButton(
        tab_prompts,
        text="Add prompt",
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        corner_radius=UI_RADIUS,
        fg_color=COLORS["primary_fg"],
        hover_color=COLORS["primary_hover"],
        command=add_new_prompt,
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(4, UI_PAD))

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
    ctk.CTkLabel(
        main_content,
        text=dev_info,
        font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small),
        text_color="gray",
    ).pack(anchor="w", padx=UI_PAD_LG, pady=(0, UI_PAD))

    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_event.set(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
