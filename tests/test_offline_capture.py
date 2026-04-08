"""Offline WAV chunking (no onnx-asr)."""
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from app.offline_capture import iter_chunks_from_wav


@pytest.fixture()
def speech_like_wav(tmp_path: Path) -> Path:
    """Short 16 kHz mono WAV with non-silent audio (sine + noise)."""
    sr = 16000
    t = np.arange(sr * 2.5) / sr
    sig = 0.15 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sig += np.random.default_rng(0).standard_normal(len(sig), dtype=np.float32) * 0.02
    pcm = (np.clip(sig, -1.0, 1.0) * 32767).astype(np.int16)
    p = tmp_path / "utt.wav"
    wavfile.write(str(p), sr, pcm)
    return p


def test_iter_chunks_from_wav_yields_files(speech_like_wav, tmp_path):
    paths = []
    try:
        for wav_path, _rms, _s0, _s1 in iter_chunks_from_wav(
            speech_like_wav, leveler_settings=None, temp_dir=str(tmp_path / "chunks")
        ):
            paths.append(Path(wav_path))
            assert Path(wav_path).is_file()
    finally:
        for q in paths:
            q.unlink(missing_ok=True)
    assert len(paths) >= 1


def test_empty_wav(tmp_path):
    sr = 16000
    p = tmp_path / "silent.wav"
    wavfile.write(str(p), sr, np.zeros(sr, dtype=np.int16))
    out = list(iter_chunks_from_wav(p, temp_dir=str(tmp_path / "c")))
    assert out == []
