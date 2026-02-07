"""
Sliding-window recorder: accumulates stereo audio (L=mic, R=loopback) into fixed-duration
chunks, saves each as a mono WAV at ASR sample rate (16 kHz), then invokes callback with path.
The remainder after each chunk is kept in the buffer so no audio is dropped at boundaries.
"""

import os
import threading
import uuid
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class ChunkRecorder:
    """
    Receives stereo blocks (L=mic, R=loopback), accumulates chunk_duration_sec seconds,
    writes a mono WAV (mean of L+R) at asr_sample_rate, invokes on_chunk_ready(path).
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_duration_sec: float,
        asr_sample_rate: int = 16000,
        temp_dir: str = None,
        on_chunk_ready=None,
    ):
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_duration_sec)
        self.asr_sample_rate = asr_sample_rate
        self.temp_dir = temp_dir or os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "MeetingsChunks")
        self.on_chunk_ready = on_chunk_ready
        self._buffer = []
        self._buffer_frames = 0
        self._lock = threading.Lock()
        _ensure_dir(self.temp_dir)

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
        if self._buffer_frames == 0:
            return
        concatenated = np.concatenate(self._buffer, axis=0)
        to_write = concatenated[: self.chunk_frames]
        # Keep remainder in buffer (sliding window) so we never drop audio at chunk boundaries.
        remainder = concatenated[self.chunk_frames:]
        if len(remainder) > 0:
            self._buffer = [remainder]
            self._buffer_frames = len(remainder)
        else:
            self._buffer.clear()
            self._buffer_frames = 0
        if len(to_write) < self.chunk_frames:
            pad = np.zeros((self.chunk_frames - len(to_write), 2), dtype=np.float32)
            to_write = np.concatenate([to_write, pad], axis=0)
        mono = np.mean(to_write, axis=1).astype(np.float32)
        if self.sample_rate != self.asr_sample_rate:
            num_out = int(len(mono) * self.asr_sample_rate / self.sample_rate)
            mono = resample(mono, num_out).astype(np.float32)
        mono_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        wav_path = os.path.join(self.temp_dir, f"chunk_{uuid.uuid4().hex}.wav")
        try:
            wavfile.write(wav_path, self.asr_sample_rate, mono_int16)
        except Exception as e:
            try:
                from diagnostic import write as diag
                diag("chunk_write_failed", path=wav_path, error=str(e))
            except ImportError:
                pass
            import logging
            logging.getLogger(__name__).exception("Failed to write chunk WAV %s: %s", wav_path, e)
            return
        try:
            from diagnostic import write as diag
            diag("chunk_flushed", path=wav_path, written_frames=len(to_write), remainder_frames=len(remainder))
        except ImportError:
            pass
        cb = self.on_chunk_ready
        if cb is not None:
            try:
                cb(wav_path)
            except Exception as e:
                try:
                    from diagnostic import write as diag
                    diag("on_chunk_ready_failed", path=wav_path, error=str(e))
                except ImportError:
                    pass
                import logging
                logging.getLogger(__name__).exception("on_chunk_ready failed: %s", e)

    def flush(self):
        """Flush any remaining buffer as a final chunk."""
        with self._lock:
            if self._buffer_frames > 0:
                self._flush_chunk_locked()
