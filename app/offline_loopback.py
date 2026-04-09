"""
Offline replay of fixed-duration ASR chunks (same strategy as capture_worker_loopback:
resample to 16 kHz mono, no AudioLeveler, _is_silent gate).

Used by the eval harness and shared constants with capture.capture_worker_loopback /
capture_worker_fixed.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from .capture import CHUNK_DURATION_SEC, SAMPLE_RATE, _is_silent, _rms32


def iter_fixed_duration_chunks_from_wav(
    path: str | Path,
    temp_dir=None,
    *,
    duration_sec: float | None = None,
):
    """
    Yield (wav_path, rms, gap_sec) for each non-silent chunk of length duration_sec
    (at 16 kHz after resampling), non-overlapping windows on the source timeline.

    duration_sec defaults to CHUNK_DURATION_SEC (loopback / app default).
    """
    path = Path(path)
    d = float(duration_sec if duration_sec is not None else CHUNK_DURATION_SEC)
    d = max(3.0, min(30.0, d))
    chunk_samples_16k = int(SAMPLE_RATE * d)

    rate, data = wavfile.read(str(path))
    if data.ndim == 1:
        pcm = data[:, np.newaxis]
        ch = 1
    else:
        pcm = data
        ch = pcm.shape[1]
    if np.issubdtype(pcm.dtype, np.integer):
        pcm = pcm.astype(np.float64) / float(np.iinfo(pcm.dtype).max)
    else:
        pcm = np.clip(pcm.astype(np.float64), -1.0, 1.0)

    n_frames = pcm.shape[0]
    frames_per_chunk = int(rate * d)
    base = temp_dir or os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "MeetingsOfflineFixedChunks")
    os.makedirs(base, exist_ok=True)

    chunk_index = 0
    prev_emit_end_16k = 0
    i = 0
    while i < n_frames:
        end = min(i + frames_per_chunk, n_frames)
        block = pcm[i:end].copy()
        consumed = block.shape[0]
        if consumed < frames_per_chunk:
            pad = np.zeros((frames_per_chunk - consumed, ch), dtype=np.float64)
            block = np.vstack([block, pad])
        i += consumed

        mono = np.mean(block, axis=1).astype(np.float32)
        if rate != SAMPLE_RATE:
            n_out = int(round(len(mono) * SAMPLE_RATE / rate))
            mono = resample(mono, n_out).astype(np.float32)
        mono = mono[:chunk_samples_16k]
        if len(mono) < chunk_samples_16k:
            mono = np.pad(mono, (0, chunk_samples_16k - len(mono)))

        chunk_start_16k = chunk_index * chunk_samples_16k
        chunk_index += 1

        if _is_silent(mono):
            continue

        rms = _rms32(mono)
        gap = max(0.0, (chunk_start_16k - prev_emit_end_16k) / float(SAMPLE_RATE))
        prev_emit_end_16k = chunk_start_16k + chunk_samples_16k

        wav_path = os.path.join(base, f"fixed_{uuid.uuid4().hex}.wav")
        audio_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
        yield (wav_path, rms, gap)


def iter_loopback_chunks_from_wav(path: str | Path, temp_dir=None):
    """Same as iter_fixed_duration_chunks_from_wav with duration_sec=CHUNK_DURATION_SEC."""
    return iter_fixed_duration_chunks_from_wav(
        path, temp_dir, duration_sec=CHUNK_DURATION_SEC
    )
