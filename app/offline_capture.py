"""
Replay a WAV file through the same VAD + leveler chunking as capture_worker (16 kHz mono).

Keep algorithm parameters in sync with capture.capture_worker.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from .audio_leveler import AudioLeveler
from .capture import SAMPLE_RATE, _build_leveler_config, _rms32

# Mirror capture_worker
BLOCK_SIZE = 512
MIN_CHUNK_FRAMES = int(SAMPLE_RATE * 1.0)
MAX_CHUNK_FRAMES = int(SAMPLE_RATE * 3.0)
SILENCE_TRIGGER_FRAMES = int(SAMPLE_RATE * 0.35)
HANGOVER_BLOCKS = 5
SILENCE_RMS_THRESHOLD = 0.005


def _load_mono_float32_16k(path: str | Path) -> np.ndarray:
    """Load WAV, convert to mono float32 in [-1, 1] at SAMPLE_RATE."""
    path = Path(path)
    rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = np.mean(data.astype(np.float64), axis=1)
    else:
        data = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        maxv = np.iinfo(data.dtype).max
        data = data / float(maxv)
    else:
        data = np.clip(data, -1.0, 1.0)
    data = data.astype(np.float32)
    if rate != SAMPLE_RATE:
        n_out = int(round(len(data) * SAMPLE_RATE / rate))
        data = resample(data, n_out).astype(np.float32)
    return data


def iter_chunks_from_wav(path: str | Path, leveler_settings=None, temp_dir=None):
    """
    Yield (wav_path, rms, start_sample, end_sample) for each chunk (sample indices in
    the resampled mono stream, same timeline as capture_worker).

    Caller should unlink each wav_path after use. Temp files live under temp_dir
    (default: system temp / MeetingsOfflineChunks).
    """
    samples = _load_mono_float32_16k(path)
    leveler = AudioLeveler(_build_leveler_config(leveler_settings))
    base = temp_dir or os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "MeetingsOfflineChunks")
    os.makedirs(base, exist_ok=True)

    buf = []
    buf_frames = 0
    buf_start_sample = 0
    consecutive_silent = 0
    noise_floor = SILENCE_RMS_THRESHOLD
    hangover_left = 0

    def flush(buf, buf_frames, start_sample, end_sample):
        if not buf or buf_frames == 0:
            return None
        audio = np.concatenate(buf).astype(np.float32)
        audio, _stats = leveler.process(audio)
        rms = _rms32(audio)
        if rms < SILENCE_RMS_THRESHOLD * 0.5:
            return None
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav_path = os.path.join(base, f"offline_{uuid.uuid4().hex}.wav")
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
        return (wav_path, rms, start_sample, end_sample)

    for i in range(0, len(samples), BLOCK_SIZE):
        block = samples[i : i + BLOCK_SIZE]
        if block.size < BLOCK_SIZE:
            block = np.pad(block, (0, BLOCK_SIZE - block.size))
        if not buf:
            buf_start_sample = i
        block_rms = _rms32(block)

        learn_upper = max(SILENCE_RMS_THRESHOLD * 5.0, noise_floor * 1.7)
        if block_rms <= learn_upper:
            noise_floor = 0.96 * noise_floor + 0.04 * block_rms
        adaptive_threshold = max(SILENCE_RMS_THRESHOLD, noise_floor * 2.5)

        is_silent = block_rms < adaptive_threshold
        if is_silent:
            if hangover_left > 0:
                hangover_left -= 1
                is_silent = False
        else:
            hangover_left = HANGOVER_BLOCKS
            consecutive_silent = 0

        buf.append(block)
        buf_frames += len(block)
        if is_silent:
            consecutive_silent += len(block)

        if buf_frames >= MAX_CHUNK_FRAMES:
            out = flush(buf, buf_frames, buf_start_sample, buf_start_sample + buf_frames)
            if out is not None:
                yield out
            buf, buf_frames, consecutive_silent = [], 0, 0
        elif buf_frames >= MIN_CHUNK_FRAMES and consecutive_silent >= SILENCE_TRIGGER_FRAMES:
            out = flush(buf, buf_frames, buf_start_sample, buf_start_sample + buf_frames)
            if out is not None:
                yield out
            buf, buf_frames, consecutive_silent = [], 0, 0

    out = flush(buf, buf_frames, buf_start_sample, buf_start_sample + buf_frames)
    if out is not None:
        yield out
