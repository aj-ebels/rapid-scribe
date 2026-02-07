"""
WASAPI-based capture and real-time mixer.

- Microphone: standard capture.
- System audio: loopback capture (SoundCard exposes loopback as a "microphone" on Windows).
- Stereo buffer: Left = mic, Right = loopback. Loopback gaps (silence) are filled so streams stay in sync.
"""

import threading
import time
import numpy as np
import soundcard as sc

from config import SAMPLE_RATE, FRAMES_PER_READ, MIXER_GAIN
from logging_config import get_logger, flush_log

log = get_logger("meetings2.audio_capture")


def _get_loopback_mic():
    """Return the loopback 'microphone' for the default speaker (Windows WASAPI)."""
    default_speaker = sc.default_speaker()
    speaker_name = default_speaker.name
    mics = sc.all_microphones(include_loopback=True)
    for m in mics:
        if not getattr(m, "isloopback", False):
            continue
        if m.name == speaker_name or speaker_name in m.name or m.name in speaker_name:
            return m
    for m in mics:
        if getattr(m, "isloopback", False):
            return m
    return None


class AudioMixer:
    """
    Captures microphone and system loopback at the same sample rate,
    outputs a stereo stream: Left = mic, Right = loopback.
    Fills loopback gaps with silence so both streams stay time-aligned.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, frames_per_read: int = FRAMES_PER_READ, gain: float = MIXER_GAIN):
        self.sample_rate = sample_rate
        self.frames_per_read = frames_per_read
        self.gain = gain
        self._mic = None
        self._loopback = None
        self._mic_recorder = None
        self._loopback_recorder = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stereo_callback = None  # callable(stereo_frames: np.ndarray) called from capture thread

    def _ensure_mono(self, data: np.ndarray) -> np.ndarray:
        """Convert to mono (mean across channels) if needed."""
        if data.ndim == 1:
            return data.astype(np.float32)
        return np.mean(data, axis=1).astype(np.float32)

    def _next_mic_block(self):
        if self._mic_recorder is None:
            return None
        try:
            block = self._mic_recorder.record(numframes=self.frames_per_read)
            return self._ensure_mono(block)
        except Exception as e:
            log.exception("Mic record failed: %s", e)
            return None

    def _next_loopback_block(self):
        if self._loopback_recorder is None:
            return None
        try:
            block = self._loopback_recorder.record(numframes=None)
            if block is None or block.size == 0:
                return np.zeros(self.frames_per_read, dtype=np.float32)
            return self._ensure_mono(block)
        except Exception as e:
            log.exception("Loopback record failed: %s", e)
            return np.zeros(self.frames_per_read, dtype=np.float32)

    def _run_capture(self):
        """Capture loop: read mic (always) and loopback (may be less when silent); output stereo."""
        while self._running:
            mic_block = self._next_mic_block()
            if mic_block is None:
                time.sleep(0.001)
                continue
            n_mic = len(mic_block)
            if n_mic == 0:
                continue

            loopback_block = self._next_loopback_block()
            if loopback_block is None or len(loopback_block) == 0:
                loopback_block = np.zeros(n_mic, dtype=np.float32)
            else:
                n_lb = len(loopback_block)
                if n_lb < n_mic:
                    pad = np.zeros(n_mic - n_lb, dtype=np.float32)
                    loopback_block = np.concatenate([loopback_block, pad])
                elif n_lb > n_mic:
                    loopback_block = loopback_block[:n_mic]

            stereo = np.column_stack([
                mic_block * self.gain,
                loopback_block * self.gain,
            ]).astype(np.float32)

            with self._lock:
                cb = self._stereo_callback
            if cb is not None:
                try:
                    cb(stereo)
                except Exception as e:
                    log.exception("Stereo callback failed: %s", e)

    def set_stereo_callback(self, callback):
        """Set callable(stereo_frames) invoked from capture thread for each block."""
        with self._lock:
            self._stereo_callback = callback

    def start(self):
        """Start capture. Uses default mic and default speaker loopback."""
        if self._running:
            return
        log.debug("Getting default microphone and loopback device.")
        self._mic = sc.default_microphone()
        self._loopback = _get_loopback_mic()
        if self._loopback is None:
            log.error(
                "No loopback device found. all_microphones(include_loopback=True) did not return a loopback device."
            )
            raise RuntimeError(
                "No loopback device found. On Windows, SoundCard should list speaker loopback when include_loopback=True."
            )
        # Open loopback first, then mic (order can avoid driver crashes on some Windows setups)
        log.debug("Opening loopback recorder first (rate=%s, blocksize=None).", self.sample_rate)
        flush_log()
        try:
            self._loopback_recorder = self._loopback.recorder(
                samplerate=self.sample_rate,
                blocksize=None,
            )
        except Exception as e:
            log.exception("Creating loopback recorder failed: %s", e)
            raise
        log.debug("Loopback recorder created; about to open stream (__enter__).")
        flush_log()
        try:
            self._loopback_recorder.__enter__()
        except Exception as e:
            log.exception("Loopback recorder __enter__ failed: %s", e)
            raise
        log.debug("Loopback stream opened.")
        flush_log()
        log.debug("Opening mic recorder (rate=%s, blocksize=None).", self.sample_rate)
        flush_log()
        try:
            self._mic_recorder = self._mic.recorder(
                samplerate=self.sample_rate,
                blocksize=None,
            )
        except Exception as e:
            log.exception("Creating mic recorder failed: %s", e)
            raise
        log.debug("Mic recorder created; about to open stream (__enter__).")
        flush_log()
        try:
            self._mic_recorder.__enter__()
        except Exception as e:
            log.exception("Mic recorder __enter__ failed: %s", e)
            raise
        log.debug("Mic stream opened.")
        flush_log()
        self._running = True
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()
        log.debug("Capture thread started.")

    def stop(self):
        """Stop capture and release devices."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._mic_recorder is not None:
            try:
                self._mic_recorder.__exit__(None, None, None)
            except Exception as e:
                log.exception("Mic recorder __exit__: %s", e)
            self._mic_recorder = None
        if self._loopback_recorder is not None:
            try:
                self._loopback_recorder.__exit__(None, None, None)
            except Exception as e:
                log.exception("Loopback recorder __exit__: %s", e)
            self._loopback_recorder = None
        self._mic = None
        self._loopback = None
        log.debug("Capture stopped.")

    def is_running(self):
        return self._running
