"""
GUI and app controller: CustomTkinter window, tabs, start/stop recording, prompts dialog.
"""
import os
import sys
import queue
import threading
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

import customtkinter as ctk
from tkinter import messagebox, filedialog

from settings import load_settings, save_settings, AUDIO_MODE_DEFAULT, AUDIO_MODE_LOOPBACK, AUDIO_MODE_MEETING
from devices import list_audio_devices, list_loopback_devices, get_default_monitor_device, get_effective_audio_device
from prompts import load_prompts, add_prompt, update_prompt, delete_prompt, get_prompt_by_id, TRANSCRIPT_PLACEHOLDER
from transcription import (
    PARAKEET_MODEL,
    get_transcription_model,
    clear_transcription_model_cache,
    list_installed_transcription_models,
    uninstall_transcription_model,
    transcription_worker,
)
from capture import (
    capture_worker,
    capture_worker_loopback,
    meeting_chunk_ready,
    CHUNK_DURATION_SEC,
    CAPTURE_SAMPLE_RATE_MEETING,
    FRAMES_PER_READ_MEETING,
    MIXER_GAIN_MEETING,
    SAMPLE_RATE,
)
from ai_summary import generate_ai_summary

if sys.platform == "win32":
    from audio_mixer import AudioMixer
    from chunk_recorder import ChunkRecorder


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
        app.start_btn.configure(state="normal")
        app.stop_btn.configure(state="disabled")
        app.status_var.set("Stopped")
        return
    app.stop_event.clear()
    mode, mic_idx, loopback_idx = get_effective_audio_device(app)

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
    elif mode == AUDIO_MODE_LOOPBACK:
        if sys.platform != "win32":
            app.log.insert("end", "[Loopback] Only supported on Windows with pyaudiowpatch.\n")
            app.log.see("end")
            return
        app.capture_thread = threading.Thread(
            target=capture_worker_loopback,
            args=(loopback_idx, app.chunk_queue, app.stop_event),
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
            app.recorder = ChunkRecorder(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                chunk_duration_sec=CHUNK_DURATION_SEC,
                asr_sample_rate=SAMPLE_RATE,
                on_chunk_ready=lambda wav_path: meeting_chunk_ready(app, wav_path),
            )
            app.mixer = AudioMixer(
                sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                frames_per_read=FRAMES_PER_READ_MEETING,
                gain=MIXER_GAIN_MEETING,
            )
            app.mixer.set_stereo_callback(app.recorder.push_stereo)
            app.mixer.start(loopback_device_index=loopback_idx, mic_device_index=None)
            app.capture_thread = None
            app.capture_threads = []
            app.status_var.set("Meeting (mic + loopback) — recording…")
        except Exception as e:
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
    model_id = app.settings.get("transcription_model") or PARAKEET_MODEL
    app.transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(app.chunk_queue, app.text_queue, app.stop_event, model_id),
        daemon=True,
    )
    app.transcription_thread.start()
    app.running = True
    app.start_btn.configure(state="disabled")
    app.stop_btn.configure(state="normal")
    poll_text_queue(app)


def _open_edit_prompt_dialog(parent, prompt_id, on_saved, ui_pad, ui_radius, font_family, font_sizes, colors):
    """Open dialog to add or edit a prompt template (name + prompt text with {{transcript}})."""
    prompt = get_prompt_by_id(prompt_id) if prompt_id else None
    is_new = prompt is None

    win = ctk.CTkToplevel(parent)
    win.title("Add prompt" if is_new else "Edit prompt")
    win.geometry("520x380")
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

    ctk.CTkLabel(win, text=f"Prompt (use {TRANSCRIPT_PLACEHOLDER} where the transcript should go)", font=ctk.CTkFont(family=font_family, size=font_sizes.small)).pack(anchor="w", padx=ui_pad, pady=(ui_pad, 2))
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
    _saved_model = load_settings().get("transcription_model") or PARAKEET_MODEL
    print(f"Loading transcription model '{_saved_model}' (first run may download from Hugging Face)...")
    get_transcription_model(_saved_model)
    print("Model ready. Opening window...")

    ctk.set_appearance_mode("dark")
    _base = Path(__file__).resolve().parent
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
    }
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
    icon_base = Path(sys.executable).parent if getattr(sys, "frozen", False) else _base
    _icon = icon_base / "icon.ico"
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
    app.capture_thread = None
    app.transcription_thread = None
    app.capture_threads = []
    app.settings = load_settings()

    content_frame = ctk.CTkFrame(root, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=UI_PAD, pady=UI_PAD)
    main_content = ctk.CTkFrame(content_frame, fg_color="transparent")
    main_content.pack(fill="both", expand=True)

    header = ctk.CTkFrame(main_content, fg_color=COLORS["header"], corner_radius=UI_RADIUS, height=52)
    header.pack(fill="x", pady=(0, UI_PAD))
    header.pack_propagate(False)
    app.status_var = ctk.StringVar(value="Ready — click Start to begin")
    ctk.CTkLabel(header, textvariable=app.status_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left", padx=UI_PAD_LG, pady=UI_PAD)
    btn_frame = ctk.CTkFrame(header, fg_color="transparent")
    btn_frame.pack(side="right", padx=UI_PAD_LG, pady=UI_PAD)
    app.start_btn = ctk.CTkButton(btn_frame, text="Start", command=lambda: start_stop(app), font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"), width=100, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"])
    app.start_btn.pack(side="left", padx=(0, UI_PAD))
    app.stop_btn = ctk.CTkButton(btn_frame, text="Stop", command=lambda: start_stop(app), state="disabled", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold"), width=100, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"])
    app.stop_btn.pack(side="left")

    tabview = ctk.CTkTabview(main_content, fg_color=COLORS["card"], corner_radius=UI_RADIUS)
    tabview.pack(fill="both", expand=True, pady=(0, UI_PAD))
    tab_transcript = tabview.add("Transcript")
    tab_prompts = tabview.add("AI Prompts")
    tab_settings = tabview.add("Settings")
    tab_models = tabview.add("Models")

    # Models tab
    models_card = ctk.CTkFrame(tab_models, fg_color="transparent")
    models_card.pack(fill="both", expand=True)
    ctk.CTkLabel(models_card, text="Installed models", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.title, weight="bold")).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 6))
    model_selector_row = ctk.CTkFrame(models_card, fg_color="transparent")
    model_selector_row.pack(fill="x", padx=UI_PAD_LG, pady=(0, UI_PAD))
    ctk.CTkLabel(model_selector_row, text="Transcription model:", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small)).pack(side="left", padx=(0, UI_PAD))
    app.model_selector_var = ctk.StringVar(value=app.settings.get("transcription_model") or PARAKEET_MODEL)

    def on_model_selected(choice):
        app.settings["transcription_model"] = choice
        save_settings(app.settings)
        clear_transcription_model_cache()

    app.model_selector = ctk.CTkOptionMenu(model_selector_row, variable=app.model_selector_var, values=[], font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=320, command=on_model_selected)
    app.model_selector.pack(side="left")
    models_scroll = ctk.CTkScrollableFrame(models_card, fg_color="transparent")
    models_scroll.pack(fill="both", expand=True)

    def refresh_models_tab():
        for w in models_scroll.winfo_children():
            w.destroy()
        models, err = list_installed_transcription_models()
        repo_ids = [m["repo_id"] for m in models] if models else []
        current = app.settings.get("transcription_model") or PARAKEET_MODEL
        if current not in repo_ids and repo_ids:
            repo_ids.insert(0, current)
        elif not repo_ids:
            repo_ids = [current] if current else [PARAKEET_MODEL]
        app.model_selector.configure(values=repo_ids)
        chosen = current if current in repo_ids else (repo_ids[0] if repo_ids else PARAKEET_MODEL)
        app.model_selector_var.set(chosen)
        if chosen != current:
            app.settings["transcription_model"] = chosen
            save_settings(app.settings)
            clear_transcription_model_cache()
        if err:
            ctk.CTkLabel(models_scroll, text=f"Error: {err[:50]}…" if len(err) > 50 else err, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color=COLORS["error_text"], wraplength=400).pack(anchor="w", padx=UI_PAD, pady=4)
            return
        if not models:
            ctk.CTkLabel(models_scroll, text="No transcription models in cache.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=400).pack(anchor="w", padx=UI_PAD, pady=4)
            return
        for m in models:
            row = ctk.CTkFrame(models_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=f"{m['repo_id']}\n{m['size_str']}", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), wraplength=400, anchor="w").pack(side="left", padx=(UI_PAD, 4))
            def _uninstall(repo_id=m["repo_id"], hashes=m["revision_hashes"]):
                if not messagebox.askyesno("Uninstall model", f"Delete cached model '{repo_id}'? This frees disk space; you can re-download later."):
                    return
                ok, err = uninstall_transcription_model(repo_id, hashes)
                if ok:
                    messagebox.showinfo("Uninstalled", f"Removed {repo_id} from cache.")
                    refresh_models_tab()
                else:
                    messagebox.showerror("Error", err or "Failed to delete.")
            ctk.CTkButton(row, text="Uninstall", width=70, height=28, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), corner_radius=UI_RADIUS, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"], command=_uninstall).pack(side="right", padx=(0, UI_PAD))

    refresh_models_tab()
    ctk.CTkButton(models_card, text="Refresh list", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=refresh_models_tab).pack(pady=(4, UI_PAD))

    # Transcript tab
    card = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    card.pack(fill="both", expand=True)
    transcript_panel = ctk.CTkFrame(card, fg_color="transparent")
    transcript_panel.pack(side="left", fill="both", expand=True, padx=(UI_PAD_LG, UI_PAD))
    card_header = ctk.CTkFrame(transcript_panel, fg_color="transparent")
    card_header.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(card_header, text="Transcript", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left")
    def copy_transcript():
        text = app.log.get("1.0", "end").rstrip()
        if text:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
    def clear_transcript():
        app.log.delete("1.0", "end")
    ctk.CTkButton(card_header, text="Copy transcript", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=140, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=copy_transcript).pack(side="left", padx=(UI_PAD, 0), pady=4)
    ctk.CTkButton(card_header, text="Clear", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=80, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=clear_transcript).pack(side="right", padx=UI_PAD, pady=4)
    app.log = ctk.CTkTextbox(transcript_panel, wrap="word", font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body), corner_radius=8, border_width=0, fg_color=COLORS["textbox_bg"], border_spacing=UI_PAD)
    app.log.pack(fill="both", expand=True, pady=(0, UI_PAD))

    summary_panel = ctk.CTkFrame(card, fg_color="transparent")
    summary_panel.pack(side="left", fill="both", expand=True, padx=(UI_PAD, UI_PAD_LG))
    summary_header = ctk.CTkFrame(summary_panel, fg_color="transparent")
    summary_header.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(summary_header, text="AI Summary", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(side="left")
    _prompts_for_summary = load_prompts()
    _prompt_names = [p.get("name", "Unnamed") for p in _prompts_for_summary] or ["(No prompts — add in AI Prompts tab)"]
    app.summary_prompt_var = ctk.StringVar(value=_prompt_names[0])
    app.summary_prompt_menu = ctk.CTkOptionMenu(summary_header, values=_prompt_names, variable=app.summary_prompt_var, width=220, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small))
    app.summary_prompt_menu.pack(side="left", padx=(UI_PAD, 0), pady=4)
    app.summary_status_var = ctk.StringVar(value="")
    ctk.CTkLabel(summary_header, textvariable=app.summary_status_var, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray").pack(side="left", padx=(UI_PAD, 0), pady=4)

    def _refresh_summary_prompt_menu():
        prompts_list = load_prompts()
        names = [p.get("name", "Unnamed") for p in prompts_list] or ["(No prompts — add in AI Prompts tab)"]
        app.summary_prompt_menu.configure(values=names)
        if names and (app.summary_prompt_var.get() not in names):
            app.summary_prompt_var.set(names[0])

    def _do_ai_summary():
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            messagebox.showerror("API key required", "Set OPENAI_API_KEY in your environment or .env file.", parent=root)
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

    app.summary_generate_btn = ctk.CTkButton(summary_header, text="Generate", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=100, height=36, corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], command=_do_ai_summary)
    app.summary_generate_btn.pack(side="left", padx=(UI_PAD, 0), pady=4)

    def _export_markdown():
        summary = app.summary_text.get("1.0", "end").strip()
        transcript = app.log.get("1.0", "end").strip()
        if not summary and not transcript:
            messagebox.showwarning("Nothing to export", "Add an AI summary and/or transcript first.", parent=root)
            return
        name_part = (app.export_name_var.get() or "").strip()
        if name_part:
            name_part = "".join(c if c.isalnum() or c in "._- " else "-" for c in name_part)
            name_part = name_part.replace(" ", "-").strip("-") or "export"
        else:
            name_part = "export"
        default_name = f"{date.today().isoformat()} {name_part}.md"
        path = filedialog.asksaveasfilename(parent=root, defaultextension=".md", filetypes=[("Markdown", "*.md"), ("All files", "*.*")], initialfile=default_name)
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

    app.summary_text = ctk.CTkTextbox(summary_panel, wrap="word", font=ctk.CTkFont(family=MONO_FONT_FAMILY, size=F.body), corner_radius=8, border_width=0, fg_color=COLORS["textbox_bg"], border_spacing=UI_PAD)
    app.summary_text.pack(fill="both", expand=True, pady=(0, UI_PAD))

    export_row = ctk.CTkFrame(tab_transcript, fg_color="transparent")
    export_row.pack(fill="x", pady=(UI_PAD, UI_PAD_LG), padx=UI_PAD_LG)
    ctk.CTkLabel(export_row, text="Export:", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small)).pack(side="left", padx=(UI_PAD, 4), pady=4)
    app.export_name_var = ctk.StringVar(value="")
    app.export_name_entry = ctk.CTkEntry(export_row, textvariable=app.export_name_var, width=160, height=28, placeholder_text="e.g. meeting-notes", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small))
    app.export_name_entry.pack(side="left", padx=4, pady=4)
    ctk.CTkButton(export_row, text="Export", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), width=80, height=28, corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_export_markdown).pack(side="left", padx=(0, 0), pady=4)

    # AI Prompts tab
    ctk.CTkLabel(tab_prompts, text=f"Custom prompt templates for AI Summary. Use {TRANSCRIPT_PLACEHOLDER} where the transcript should be inserted.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=500).pack(anchor="w", padx=UI_PAD_LG, pady=(UI_PAD, 4))
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
            row = ctk.CTkFrame(prompts_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=p.get("name", "Unnamed"), font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.body, weight="bold"), anchor="w").pack(side="left", padx=(UI_PAD, 8))
            def _edit(pid=p["id"]):
                _open_edit_prompt_dialog(root, pid, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)
            def _delete(pid=p["id"], pname=p.get("name", "?")):
                if not messagebox.askyesno("Delete prompt", f"Delete prompt '{pname}'?"):
                    return
                if delete_prompt(pid):
                    refresh_prompts_list()
                    messagebox.showinfo("Deleted", "Prompt deleted.")
            ctk.CTkButton(row, text="Edit", width=60, height=28, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), corner_radius=UI_RADIUS, fg_color=COLORS["secondary_fg"], hover_color=COLORS["secondary_hover"], command=_edit).pack(side="right", padx=4)
            ctk.CTkButton(row, text="Delete", width=60, height=28, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.tiny), corner_radius=UI_RADIUS, fg_color=COLORS["danger_fg"], hover_color=COLORS["danger_hover"], command=_delete).pack(side="right", padx=(0, UI_PAD))

    def add_new_prompt():
        _open_edit_prompt_dialog(root, None, refresh_prompts_list, UI_PAD, UI_RADIUS, UI_FONT_FAMILY, F, COLORS)

    refresh_prompts_list()
    ctk.CTkButton(tab_prompts, text="Add prompt", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), corner_radius=UI_RADIUS, fg_color=COLORS["primary_fg"], hover_color=COLORS["primary_hover"], command=add_new_prompt).pack(anchor="w", padx=UI_PAD_LG, pady=(4, UI_PAD))

    # Settings tab
    settings_card = ctk.CTkFrame(tab_settings, fg_color="transparent")
    settings_card.pack(fill="both", expand=True, padx=UI_PAD_LG, pady=UI_PAD)
    ctk.CTkLabel(settings_card, text="Capture mode", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.header, weight="bold")).pack(anchor="w", pady=(0, 4))
    ctk.CTkLabel(settings_card, text="Meeting = in-process mic + loopback (PyAudioWPatch; loopback read only when data available). Loopback device below applies to Meeting mode.", font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray", wraplength=520).pack(anchor="w", pady=(0, UI_PAD))
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
    ctk.CTkLabel(main_content, text=dev_info, font=ctk.CTkFont(family=UI_FONT_FAMILY, size=F.small), text_color="gray").pack(anchor="w", padx=UI_PAD_LG, pady=(0, UI_PAD))

    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_event.set(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
