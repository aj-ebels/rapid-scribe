"""
Recording session controller: capture, transcription, and live events without GUI dependencies.
Used by the Electron sidecar via app/ipc.py.
"""
from __future__ import annotations

import multiprocessing
import queue
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from . import dev_config as _dev_cfg
from .audio_leveler import AudioLevelerConfig
from .capture import (
    CAPTURE_SAMPLE_RATE_MEETING,
    FRAMES_PER_READ_MEETING,
    MIXER_GAIN_MEETING,
    SAMPLE_RATE,
    capture_worker,
    capture_worker_loopback,
    meeting_chunk_ready,
)
from .devices import get_default_monitor_device, get_effective_audio_device
from .diagnostic import write as diag
from .settings import AUDIO_MODE_DEFAULT, AUDIO_MODE_LOOPBACK, AUDIO_MODE_MEETING
from .transcript_join import join_transcription_items
from .transcription import (
    STANDARD_TRANSCRIPTION_MODEL,
    list_installed_transcription_models,
    start_transcription_subprocess,
)

if sys.platform == "win32":
    from .audio_mixer import AudioMixer
    from .chunk_recorder import ChunkRecorder

_PARAGRAPH_GAP_SEC = 6.0
_LEVEL_FLOOR = 0.003
_LEVEL_CEILING = 0.08

EmitFn = Callable[[str, dict], None]


def _discard_pending_chunks(chunk_queue: queue.Queue) -> int:
    from pathlib import Path as _Path

    discarded = 0
    while True:
        try:
            item = chunk_queue.get_nowait()
            if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], str) and item[0] != "error":
                try:
                    _Path(item[0]).unlink(missing_ok=True)
                except Exception:
                    pass
            discarded += 1
        except queue.Empty:
            break
    if discarded:
        diag("chunk_queue_drained", discarded=discarded)
    return discarded


def _discard_pending_text(text_queue: queue.Queue) -> int:
    discarded = 0
    while True:
        try:
            text_queue.get_nowait()
            discarded += 1
        except queue.Empty:
            break
    if discarded:
        diag("text_queue_drained", discarded=discarded)
    return discarded


def _level_monitor_worker(device_index, level_queue: queue.Queue, stop_event: threading.Event):
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


def _build_leveler_config(settings: dict) -> AudioLevelerConfig:
    cfg = AudioLevelerConfig()
    cfg.enabled = bool(settings.get("audio_auto_level", True))
    cfg.input_sensitivity = max(0.5, min(3.0, float(settings.get("input_sensitivity", 0.8))))
    cfg.target_rms = max(0.01, min(0.2, float(settings.get("agc_target_rms", 0.035))))
    cfg.max_gain_db = max(6.0, min(30.0, float(settings.get("agc_max_boost_db", 9.0))))
    cfg.expander_enabled = bool(settings.get("audio_expander_enabled", True))
    cfg.hangover_blocks = max(2, min(20, int(settings.get("audio_hangover_blocks", 6))))
    return cfg


def is_transcription_model_installed() -> bool:
    models, err = list_installed_transcription_models()
    if err or not models:
        return False
    return STANDARD_TRANSCRIPTION_MODEL in [m["repo_id"] for m in models]


class RecordingSession:
    """Headless recording + live transcript streaming."""

    def __init__(self, settings: dict, emit: EmitFn):
        self.settings = settings
        self.emit = emit
        self.running = False
        self._stopping = False
        self.stop_event = threading.Event()
        # multiprocessing.Queue/Event — required for transcription subprocess on Windows (spawn/pickle).
        self.chunk_queue = multiprocessing.Queue()
        self.text_queue = multiprocessing.Queue()
        self.level_queue: queue.Queue = queue.Queue(maxsize=200)
        self.transcription_stop_event = multiprocessing.Event()
        self.transcription_process = None
        self.capture_thread = None
        self.capture_threads: list = []
        self.level_thread = None
        self.mixer = None
        self.recorder = None
        self._transcript_tail = "\n"
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

    def model_status(self) -> dict:
        installed = is_transcription_model_installed()
        return {
            "installed": installed,
            "ready": installed,
            "model_id": STANDARD_TRANSCRIPTION_MODEL,
        }

    def start(self) -> tuple[bool, str | None]:
        if self.running or self._stopping:
            return False, "Already recording or stopping"
        if not is_transcription_model_installed():
            return False, "Transcription model not installed"
        self.stop_event.clear()
        _discard_pending_text(self.text_queue)
        mode, mic_idx, loopback_idx = get_effective_audio_device(self.settings)
        diag("start_recording", mode=mode, mic_idx=mic_idx, loopback_idx=loopback_idx)
        self.emit("status", {"message": "starting", "mode": mode})

        if mode == AUDIO_MODE_DEFAULT:
            dev_idx, err = get_default_monitor_device()
            if err or dev_idx is None:
                self.emit("capture_error", {"message": err or "No monitor device found"})
                return False, err or "No input device"
            self.capture_thread = threading.Thread(
                target=capture_worker,
                args=(dev_idx, self.chunk_queue, self.stop_event, self.settings),
                daemon=True,
            )
            self.capture_threads = [self.capture_thread]
            self.level_thread = threading.Thread(
                target=_level_monitor_worker,
                args=(dev_idx, self.level_queue, self.stop_event),
                daemon=True,
            )
            self.level_thread.start()
        elif mode == AUDIO_MODE_LOOPBACK:
            if sys.platform != "win32":
                self.emit("capture_error", {"message": "Loopback only supported on Windows"})
                return False, "Loopback only supported on Windows"
            self.capture_thread = threading.Thread(
                target=capture_worker_loopback,
                args=(loopback_idx, self.chunk_queue, self.stop_event, self.level_queue),
                daemon=True,
            )
            self.capture_threads = [self.capture_thread]
        elif mode == AUDIO_MODE_MEETING:
            if sys.platform != "win32":
                self.emit("capture_error", {"message": "Meeting mode only supported on Windows"})
                return False, "Meeting mode only supported on Windows"
            try:
                use_silence = _dev_cfg.CHUNKING_MODE == "vad"
                if use_silence:
                    chunk_sec = None
                    min_sec = max(0.5, _dev_cfg.VAD_MIN_CHUNK_SEC)
                    max_sec = max(min_sec, _dev_cfg.VAD_MAX_CHUNK_SEC)
                    silence_sec = max(0.1, _dev_cfg.VAD_SILENCE_SEC)
                else:
                    chunk_sec = max(3.0, min(30.0, _dev_cfg.CHUNK_DURATION_SEC))
                    min_sec = max_sec = silence_sec = None
                self.mixer = AudioMixer(
                    sample_rate=CAPTURE_SAMPLE_RATE_MEETING,
                    frames_per_read=FRAMES_PER_READ_MEETING,
                    gain=MIXER_GAIN_MEETING,
                    mic_leveler_config=_build_leveler_config(self.settings),
                )

                def _push_level(rms):
                    try:
                        self.level_queue.put_nowait(rms)
                    except queue.Full:
                        pass

                self.mixer.set_level_callback(_push_level)
                self.mixer.start(loopback_device_index=loopback_idx, mic_device_index=mic_idx)
                _recorder_kwargs = dict(
                    sample_rate=self.mixer.sample_rate,
                    asr_sample_rate=SAMPLE_RATE,
                    on_chunk_ready=lambda wav_path, rms: meeting_chunk_ready(self.chunk_queue, wav_path, rms),
                    use_silence_chunking=use_silence,
                    leveler_config=_build_leveler_config(self.settings),
                    adaptive_silence=bool(self.settings.get("adaptive_audio_gating", True)),
                )
                if use_silence:
                    _recorder_kwargs.update(
                        min_chunk_sec=float(min_sec),
                        max_chunk_sec=float(max_sec),
                        silence_duration_sec=float(silence_sec),
                    )
                else:
                    _recorder_kwargs["chunk_duration_sec"] = float(chunk_sec)
                self.recorder = ChunkRecorder(**_recorder_kwargs)
                self.mixer.set_stereo_callback(self.recorder.push_stereo)
                self.capture_thread = None
                self.capture_threads = []
                diag("meeting_started", mixer_ok=True)
            except Exception as e:
                diag("meeting_start_failed", error=str(e))
                self.emit("capture_error", {"message": str(e)})
                if self.mixer is not None:
                    try:
                        self.mixer.stop()
                    except Exception:
                        pass
                self.mixer = None
                self.recorder = None
                return False, str(e)
        else:
            self.emit("capture_error", {"message": f"Unknown mode: {mode}"})
            return False, f"Unknown mode: {mode}"

        if self.capture_thread:
            self.capture_thread.start()

        proc = self.transcription_process
        if proc is None or not proc.is_alive():
            self.transcription_stop_event.clear()
            self.transcription_process = start_transcription_subprocess(
                self.chunk_queue,
                self.text_queue,
                self.transcription_stop_event,
                STANDARD_TRANSCRIPTION_MODEL,
            )
            # Brief wait so spawn failures surface immediately.
            for _ in range(10):
                if self.transcription_process.is_alive():
                    break
                threading.Event().wait(0.05)
            if not self.transcription_process.is_alive():
                code = self.transcription_process.exitcode
                msg = f"Transcription process failed to start (exit {code})"
                diag("transcription_start_failed", exitcode=code)
                self.emit("capture_error", {"message": msg})
                return False, msg

        self.running = True
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.emit("status", {"message": "recording", "mode": mode})
        return True, None

    def stop(self) -> None:
        if not self.running or self._stopping:
            return
        self.running = False
        self.stop_event.set()
        self._stopping = True
        self._poll_stop.set()
        mixer = self.mixer
        recorder = self.recorder
        capture_threads = list(self.capture_threads or ([self.capture_thread] if self.capture_thread else []))
        self.mixer = None
        self.recorder = None
        self.capture_threads = []
        self.capture_thread = None
        self.emit("status", {"message": "stopping"})

        def _shutdown_worker():
            if mixer is not None:
                try:
                    mixer.stop()
                except Exception as e:
                    diag("mixer_stop_failed", error=str(e))
            if recorder is not None:
                try:
                    recorder.flush()
                except Exception as e:
                    diag("recorder_flush_failed", error=str(e))
            for t in capture_threads:
                try:
                    if t and t.is_alive():
                        t.join(timeout=2.0)
                except Exception:
                    pass
            _discard_pending_chunks(self.chunk_queue)
            self._flush_transcript_backlog()
            self._stopping = False
            self.emit("status", {"message": "stopped"})
            self.emit("audio_level", {"level": 0.0})

        threading.Thread(target=_shutdown_worker, daemon=True).start()

    def shutdown_transcription(self) -> None:
        """Final shutdown when sidecar exits."""
        if self.running:
            self.stop()
        self.transcription_stop_event.set()
        proc = self.transcription_process
        if proc is not None and proc.is_alive():
            try:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass

    def _poll_loop(self):
        while self.running and not self._poll_stop.is_set():
            self._poll_once()
            self._poll_stop.wait(0.05)
        self._poll_once()

    def _drain_text_queue(self) -> list:
        items = []
        while True:
            try:
                items.append(self.text_queue.get_nowait())
            except Exception:
                break
        return items

    def _emit_transcript_items(self, items: list) -> None:
        if not items:
            return
        combined, self._transcript_tail = join_transcription_items(
            items,
            initial_tail=self._transcript_tail,
            paragraph_gap_sec=_PARAGRAPH_GAP_SEC,
        )
        if combined:
            self.emit("transcript_line", {"text": combined})

    def _poll_once(self):
        self._emit_transcript_items(self._drain_text_queue())

        level = 0.0
        try:
            while True:
                level = self.level_queue.get_nowait()
        except queue.Empty:
            pass
        if level or self.running:
            p = (float(level) - _LEVEL_FLOOR) / max(1e-6, (_LEVEL_CEILING - _LEVEL_FLOOR))
            p = min(1.0, max(0.0, p))
            self.emit("audio_level", {"level": p if self.running else 0.0})

    def _flush_transcript_backlog(self):
        self._emit_transcript_items(self._drain_text_queue())
