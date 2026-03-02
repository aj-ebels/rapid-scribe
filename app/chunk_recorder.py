"""
Sliding-window recorder: accumulates stereo audio (L=mic, R=loopback) into chunks,
saves each as a mono WAV at ASR sample rate (16 kHz), then invokes callback with path.

Supports two chunking modes:
- Fixed: emit every chunk_duration_sec seconds (sliding window; remainder kept).
- Silence-based: emit when min chunk length is met and silence_duration_sec of
  consecutive silence is seen, or when max chunk length is reached (no mid-chunk cuts).
"""

import os
import threading
import uuid
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from .audio_leveler import AudioLeveler, AudioLevelerConfig


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _is_silent_block(stereo_frames: np.ndarray, threshold: float = 0.005) -> bool:
    """True if the block has very low energy (mono RMS below threshold). 0.005 is intentional to reduce ASR hallucination on near-silence."""
    mono = np.mean(stereo_frames, axis=1).astype(np.float64)
    rms = np.sqrt(np.mean(mono ** 2))
    return rms < threshold


def _rms32(samples: np.ndarray) -> float:
    """Fast RMS for float32 audio buffers."""
    if samples is None or samples.size == 0:
        return 0.0
    buf = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(buf * buf)))


class ChunkRecorder:
    """
    Receives stereo blocks (L=mic, R=loopback). In fixed mode: accumulates
    chunk_duration_sec seconds per chunk (sliding window). In silence mode: accumulates
    until min_chunk_sec is met and silence_duration_sec of silence is seen, or
    max_chunk_sec is reached. Writes mono WAV at asr_sample_rate and invokes
    on_chunk_ready(path).
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_duration_sec: float = 5.0,
        asr_sample_rate: int = 16000,
        temp_dir: str = None,
        on_chunk_ready=None,
        *,
        use_silence_chunking: bool = False,
        min_chunk_sec: float = 1.5,
        max_chunk_sec: float = 8.0,
        silence_duration_sec: float = 0.5,
        silence_rms_threshold: float = 0.005,
        leveler_config: AudioLevelerConfig | None = None,
        adaptive_silence: bool = True,
        silence_hangover_blocks: int = 4,
    ):
        self.sample_rate = sample_rate
        self.asr_sample_rate = asr_sample_rate
        self.temp_dir = temp_dir or os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "MeetingsChunks")
        self.on_chunk_ready = on_chunk_ready
        self._buffer = []
        self._buffer_frames = 0
        self._lock = threading.Lock()
        _ensure_dir(self.temp_dir)
        self._silence_hangover_blocks = max(0, int(silence_hangover_blocks))
        self._silence_hangover_left = 0
        self._adaptive_silence = bool(adaptive_silence)
        self._noise_floor_rms = silence_rms_threshold
        self._noise_floor_alpha = 0.04
        self._leveler = AudioLeveler(leveler_config or AudioLevelerConfig())

        self._use_silence_chunking = use_silence_chunking
        if use_silence_chunking:
            self._min_chunk_frames = int(sample_rate * min_chunk_sec)
            self._max_chunk_frames = int(sample_rate * max_chunk_sec)
            self._silence_frames = int(sample_rate * silence_duration_sec)
            self._silence_rms_threshold = silence_rms_threshold
            self._consecutive_silent_frames = 0
            self.chunk_frames = self._max_chunk_frames  # for flush() / fixed fallback
        else:
            self.chunk_frames = int(sample_rate * chunk_duration_sec)
            self._min_chunk_frames = self._max_chunk_frames = self._silence_frames = 0

    def push_stereo(self, stereo_frames: np.ndarray):
        """Accept a block of stereo (frames x 2). When chunk condition is met, write WAV and callback."""
        if stereo_frames is None or stereo_frames.size == 0:
            return
        pending = None
        with self._lock:
            self._buffer.append(stereo_frames)
            n = len(stereo_frames)
            self._buffer_frames += n

            if self._use_silence_chunking:
                block = np.asarray(stereo_frames, dtype=np.float32)
                block_mono = np.mean(block, axis=1) if block.ndim == 2 else block.reshape(-1)
                block_rms = _rms32(block_mono)
                if self._adaptive_silence:
                    learn_upper = max(self._silence_rms_threshold * 5.0, self._noise_floor_rms * 1.7)
                    if block_rms <= learn_upper:
                        a = self._noise_floor_alpha
                        self._noise_floor_rms = (1.0 - a) * self._noise_floor_rms + a * block_rms
                    adaptive_threshold = max(self._silence_rms_threshold, self._noise_floor_rms * 2.6)
                else:
                    adaptive_threshold = self._silence_rms_threshold
                is_silent_now = _is_silent_block(stereo_frames, adaptive_threshold)
                if is_silent_now:
                    if self._silence_hangover_left > 0:
                        self._silence_hangover_left -= 1
                        is_silent_now = False
                else:
                    self._silence_hangover_left = self._silence_hangover_blocks
                    self._consecutive_silent_frames = 0
                if is_silent_now:
                    self._consecutive_silent_frames += n

                if self._buffer_frames >= self._max_chunk_frames:
                    pending = self._flush_full_buffer_locked()
                elif (
                    self._buffer_frames >= self._min_chunk_frames
                    and self._consecutive_silent_frames >= self._silence_frames
                ):
                    pending = self._flush_full_buffer_locked()
            else:
                if self._buffer_frames >= self.chunk_frames:
                    pending = self._flush_fixed_chunk_locked()
        if pending is not None:
            to_write, remainder_frames, pad_to_fixed = pending
            self._write_chunk(to_write, remainder_frames, pad_to_fixed=pad_to_fixed)

    def _flush_full_buffer_locked(self):
        """Emit the entire buffer as one chunk (silence-based or max reached)."""
        if self._buffer_frames == 0:
            return None
        concatenated = np.concatenate(self._buffer, axis=0)
        self._buffer.clear()
        self._buffer_frames = 0
        self._consecutive_silent_frames = 0
        return concatenated, 0, False

    def _flush_fixed_chunk_locked(self):
        """Emit exactly chunk_frames; keep remainder (fixed-duration mode)."""
        if self._buffer_frames == 0:
            return None
        concatenated = np.concatenate(self._buffer, axis=0)
        to_write = concatenated[: self.chunk_frames]
        remainder = concatenated[self.chunk_frames:]
        remainder_frames = len(remainder)
        if len(remainder) > 0:
            self._buffer = [remainder]
            self._buffer_frames = remainder_frames
        else:
            self._buffer.clear()
            self._buffer_frames = 0
        return to_write, remainder_frames, True

    def _write_chunk(self, to_write: np.ndarray, remainder_frames: int, pad_to_fixed: bool = False):
        """Convert to_write (stereo) to mono, optionally pad, resample, write WAV, invoke callback."""
        if len(to_write) == 0:
            return
        if pad_to_fixed and len(to_write) < self.chunk_frames:
            pad = np.zeros((self.chunk_frames - len(to_write), 2), dtype=np.float32)
            to_write = np.concatenate([to_write, pad], axis=0)
        # Adaptive downmix: when one side dominates, preserve the stronger source.
        if to_write.ndim == 2 and to_write.shape[1] >= 2:
            mic = to_write[:, 0].astype(np.float32)
            loopback = to_write[:, 1].astype(np.float32)
            mic_rms = _rms32(mic)
            loop_rms = _rms32(loopback)
            eps = 1e-8
            if mic_rms < eps and loop_rms < eps:
                mono = 0.5 * (mic + loopback)
            elif loop_rms < mic_rms * 0.2:
                mono = mic
            elif mic_rms < loop_rms * 0.2:
                mono = loopback
            else:
                s = mic_rms + loop_rms + eps
                mono = (mic * (mic_rms / s) + loopback * (loop_rms / s)).astype(np.float32)
        else:
            mono = np.mean(to_write, axis=1).astype(np.float32)
        mono, level_stats = self._leveler.process(mono)
        if self.sample_rate != self.asr_sample_rate:
            num_out = int(len(mono) * self.asr_sample_rate / self.sample_rate)
            mono = resample(mono, num_out).astype(np.float32)
        chunk_rms = _rms32(mono)
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
            diag(
                "chunk_flushed",
                path=wav_path,
                written_frames=len(to_write),
                remainder_frames=remainder_frames,
                rms_in=round(float(level_stats.get("rms_in", 0.0)), 6),
                rms_out=round(float(level_stats.get("rms_out", 0.0)), 6),
                gain_db=round(float(level_stats.get("gain_db", 0.0)), 2),
                noise_floor=round(float(level_stats.get("noise_floor", 0.0)), 6),
            )
        except ImportError:
            pass
        cb = self.on_chunk_ready
        if cb is not None:
            try:
                cb(wav_path, chunk_rms)
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
        pending = None
        with self._lock:
            if self._buffer_frames > 0:
                if self._use_silence_chunking:
                    pending = self._flush_full_buffer_locked()
                else:
                    pending = self._flush_fixed_chunk_locked()
        if pending is not None:
            to_write, remainder_frames, pad_to_fixed = pending
            self._write_chunk(to_write, remainder_frames, pad_to_fixed=pad_to_fixed)
