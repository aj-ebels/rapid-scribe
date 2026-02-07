"""
Sliding-window recorder: accumulates stereo audio from the mixer into 5-second chunks,
saves each chunk to a temp WAV (mono mix for ASR), then deletes the file after transcription.
"""

import os
import threading
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import resample

from config import SAMPLE_RATE, CHUNK_DURATION_SEC, TEMP_DIR, ASR_SAMPLE_RATE
from logging_config import get_logger

log = get_logger("meetings2.chunk_recorder")


class ChunkRecorder:
    """
    Receives stereo blocks (L=mic, R=loopback), accumulates CHUNK_DURATION_SEC seconds,
    writes a mono WAV to temp, and invokes a callback with the file path. Caller transcribes
    and then we delete the file (or caller can delete).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_sec: float = CHUNK_DURATION_SEC,
        temp_dir: str = TEMP_DIR,
        on_chunk_ready=None,
    ):
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_duration_sec)
        self.temp_dir = temp_dir
        self.on_chunk_ready = on_chunk_ready  # callable(wav_path: str)
        self._buffer = []
        self._buffer_frames = 0
        self._lock = threading.Lock()
        os.makedirs(self.temp_dir, exist_ok=True)

    def push_stereo(self, stereo_frames: np.ndarray):
        """Accept a block of stereo (frames x 2). When we have enough for a chunk, write WAV and callback."""
        if stereo_frames is None or stereo_frames.size == 0:
            return
        with self._lock:
            self._buffer.append(stereo_frames.copy())
            self._buffer_frames += len(stereo_frames)
            if self._buffer_frames >= self.chunk_frames:
                self._flush_chunk_locked()

    def _flush_chunk_locked(self):
        """Concatenate buffer, write mono WAV, clear buffer, invoke callback."""
        if self._buffer_frames == 0:
            return
        concatenated = np.concatenate(self._buffer, axis=0)
        self._buffer.clear()
        self._buffer_frames = 0
        # Use only up to chunk_frames so we don't send more than one chunk
        to_write = concatenated[: self.chunk_frames]
        if len(to_write) < self.chunk_frames:
            pad = np.zeros((self.chunk_frames - len(to_write), 2), dtype=np.float32)
            to_write = np.concatenate([to_write, pad], axis=0)
        # Mono for ASR: mix L and R (both already gained in mixer)
        mono = np.mean(to_write, axis=1).astype(np.float32)
        # Resample to ASR rate if capture rate differs (e.g. capture 48k, ASR wants 16k)
        if self.sample_rate != ASR_SAMPLE_RATE:
            num_out = int(len(mono) * ASR_SAMPLE_RATE / self.sample_rate)
            mono = resample(mono, num_out).astype(np.float32)
        wav_path = os.path.join(self.temp_dir, f"chunk_{uuid.uuid4().hex}.wav")
        try:
            sf.write(wav_path, mono, ASR_SAMPLE_RATE)
        except Exception as e:
            log.exception("Failed to write chunk WAV %s: %s", wav_path, e)
            return
        cb = self.on_chunk_ready
        if cb is not None:
            try:
                cb(wav_path)
            except Exception as e:
                log.exception("on_chunk_ready callback failed: %s", e)

    def flush(self):
        """Flush any remaining buffer as a final chunk (may be shorter)."""
        with self._lock:
            if self._buffer_frames > 0:
                self._flush_chunk_locked()
