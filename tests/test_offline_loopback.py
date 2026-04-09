"""Offline loopback-style chunking (no onnx-asr)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from app.offline_loopback import iter_fixed_duration_chunks_from_wav, iter_loopback_chunks_from_wav


@pytest.fixture()
def loud_wav_16k(tmp_path: Path) -> Path:
    sr = 16000
    t = np.arange(sr * 11) / sr
    sig = 0.12 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767).astype(np.int16)
    p = tmp_path / "lb.wav"
    wavfile.write(str(p), sr, pcm)
    return p


def test_loopback_yields_about_two_chunks_for_eleven_seconds(loud_wav_16k, tmp_path):
    paths = []
    try:
        chunks = list(
            iter_loopback_chunks_from_wav(loud_wav_16k, temp_dir=str(tmp_path / "lb"))
        )
        for wav_path, _rms, _gap in chunks:
            paths.append(Path(wav_path))
            assert Path(wav_path).is_file()
    finally:
        for q in paths:
            q.unlink(missing_ok=True)
    # 11 s at 5 s windows -> 3 windows; all non-silent -> up to 3 chunks
    assert 2 <= len(chunks) <= 3


def test_fixed_duration_matches_loopback_for_mono_16k(loud_wav_16k, tmp_path):
    d = str(tmp_path / "a")
    e = str(tmp_path / "b")
    a = list(iter_loopback_chunks_from_wav(loud_wav_16k, temp_dir=d))
    b = list(iter_fixed_duration_chunks_from_wav(loud_wav_16k, temp_dir=e, duration_sec=5.0))
    for paths in (a, b):
        for wav_path, _, _gap in paths:
            Path(wav_path).unlink(missing_ok=True)
    assert len(a) == len(b)
