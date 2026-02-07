"""
Meetings2: Local real-time transcription with system audio + microphone.

GUI: Start / Stop, scrollable live transcript.
Pipeline: WASAPI mic + loopback -> stereo mixer -> 5s chunks -> temp WAV -> Parakeet ASR -> text.
"""
# Force CPU before any torch/nemo import (avoids CUDA-related warnings and code paths)
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from config import SAMPLE_RATE, CHUNK_DURATION_SEC, USE_PYAUDIOWPATCH
from logging_config import setup_logging, get_logger
if USE_PYAUDIOWPATCH:
    from audio_capture_pyaudiowpatch import AudioMixer
else:
    from audio_capture import AudioMixer
from chunk_recorder import ChunkRecorder
from transcriber import ParakeetTranscriber, ensure_parakeet_model

log = get_logger("meetings2.app")


def _excepthook(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions to file and stderr, then show dialog so we can see why the app closed."""
    if exc_type is None:
        return
    lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    msg = "".join(lines)
    log.critical("Uncaught exception:\n%s", msg)
    try:
        root = tk._default_root
        if root and root.winfo_exists():
            messagebox.showerror("Error", msg[:2000] + ("..." if len(msg) > 2000 else ""))
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


class App:
    def __init__(self, initial_parakeet_model=None):
        self.root = tk.Tk()
        self.root.title("Meetings2 — Live transcription (mic + system audio)")
        self.root.minsize(500, 400)
        self.root.geometry("700x500")

        self.mixer = AudioMixer(sample_rate=SAMPLE_RATE)
        self.recorder = ChunkRecorder(
            sample_rate=SAMPLE_RATE,
            chunk_duration_sec=CHUNK_DURATION_SEC,
            on_chunk_ready=self._on_chunk_ready,
        )
        self.transcriber = ParakeetTranscriber(
            result_callback=self._on_transcript,
            initial_model=initial_parakeet_model,
        )

        self._building_ui()
        self._running = False

    def _building_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        self.btn_start = ttk.Button(top, text="Start", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 5))
        self.btn_stop = ttk.Button(top, text="Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)

        ttk.Label(top, text=f"  • {SAMPLE_RATE} Hz, {CHUNK_DURATION_SEC}s chunks").pack(side=tk.LEFT, padx=(15, 0))

        self.text_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            state=tk.NORMAL,
            padx=10,
            pady=10,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.text_area.insert(tk.END, "Click Start to begin capturing mic + system audio and transcribing.\n")
        self.text_area.see(tk.END)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_chunk_ready(self, wav_path: str):
        """Called from recorder thread when a 5s chunk is written."""
        self.transcriber.submit(wav_path)

    def _on_transcript(self, text: str):
        """Called from transcriber thread when a chunk is transcribed."""
        def append():
            if not self.text_area.winfo_exists():
                return
            self.text_area.insert(tk.END, text + " ")
            self.text_area.see(tk.END)
        self.root.after(0, append)

    def _start(self):
        if self._running:
            return
        try:
            log.debug("Start: setting callback and starting transcriber.")
            self.mixer.set_stereo_callback(self.recorder.push_stereo)
            self.transcriber.start()
            log.debug("Start: starting mixer (mic + loopback).")
            self.mixer.start()
            self._running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, "\n[Recording started.]\n")
            self.text_area.see(tk.END)
            log.info("Recording started.")
        except Exception as e:
            msg = traceback.format_exc()
            log.exception("Start failed: %s", e)
            messagebox.showerror("Start error", msg[:2000] + ("..." if len(msg) > 2000 else ""))
            self._running = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def _stop(self):
        if not self._running:
            return
        self._running = False
        self.mixer.stop()
        self.recorder.flush()
        self.transcriber.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.text_area.insert(tk.END, "\n[Recording stopped.]\n")
        self.text_area.see(tk.END)
        log.info("Recording stopped.")

    def _on_close(self):
        if self._running:
            self._stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    setup_logging()
    log.info("Meetings2 starting (capture backend: %s).", "PyAudioWPatch" if USE_PYAUDIOWPATCH else "SoundCard")
    sys.excepthook = _excepthook
    # Show GUI immediately; load Parakeet in background so boot is instant
    app = App(initial_parakeet_model=None)

    def load_model_in_background():
        try:
            model = ensure_parakeet_model()
            app.transcriber.set_initial_model(model)
        except Exception as e:
            log.exception("Background Parakeet model load failed: %s", e)

    t = threading.Thread(target=load_model_in_background, daemon=True)
    t.start()
    app.run()


if __name__ == "__main__":
    main()
