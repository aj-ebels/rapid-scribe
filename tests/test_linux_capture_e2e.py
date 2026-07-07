"""
End-to-end tests for the Linux system-audio capture paths with a fake sounddevice
stream: the loopback worker must produce 16 kHz WAV chunks from monitor audio, and
the sounddevice meeting mixer must emit time-aligned stereo (L=mic, R=monitor) blocks.
"""
import queue
import sys
import threading
import time

import numpy as np
import pytest
from scipy.io import wavfile

from app import capture
from app import audio_mixer_sd
from app.audio_mixer_sd import AudioMixerSD


class FakeInputStream:
    """Stands in for sounddevice.InputStream: a real-time-paced 440 Hz tone.

    Availability is wall-clock based like a real PortAudio stream: read() blocks
    until enough frames have "arrived", and read_available drops as frames are
    consumed — so drain loops terminate the same way they do on real hardware.
    """

    def __init__(self, device=None, channels=1, samplerate=48000, blocksize=1024,
                 dtype="float32", on_read=None, time_scale=20.0):
        self.device = device
        self.channels = max(1, int(channels))
        self.samplerate = int(samplerate)
        self.blocksize = max(1, int(blocksize or 1024))
        self._consumed = 0
        self._t0 = time.monotonic()
        self._on_read = on_read
        # Frames "arrive" time_scale× faster than real time to keep tests quick
        self._time_scale = time_scale

    @property
    def read_available(self):
        arrived = int((time.monotonic() - self._t0) * self.samplerate * self._time_scale)
        # Quantize to whole periods like PortAudio, so drain loops actually reach 0
        arrived = (arrived // self.blocksize) * self.blocksize
        return max(0, arrived - self._consumed)

    def read(self, n):
        while self.read_available < n:
            time.sleep(0.0005)
        t = (np.arange(n) + self._consumed) / self.samplerate
        self._consumed += n
        tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        frames = np.tile(tone[:, None], (1, self.channels))
        if self._on_read is not None:
            self._on_read(self)
        return frames, False

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


MONITOR_DEVICE = {
    "index": 2,
    "name": "Monitor of Built-in Audio Analog Stereo",
    "max_input_channels": 2,
    "default_samplerate": 48000,
}
MIC_DEVICE = {
    "index": 0,
    "name": "HDA Intel PCH: ALC3204 Analog (hw:0,0)",
    "max_input_channels": 1,
    "default_samplerate": 44100,
}


def _fake_query_devices(device=None, kind=None):
    if kind == "input":
        return MIC_DEVICE
    if device == MONITOR_DEVICE["index"]:
        return MONITOR_DEVICE
    if device == MIC_DEVICE["index"]:
        return MIC_DEVICE
    return [MIC_DEVICE, MONITOR_DEVICE]


def test_loopback_worker_produces_16k_wav_chunks(monkeypatch, tmp_path):
    if sys.platform == "win32":
        pytest.skip("non-Windows path")
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr(capture.sd, "query_devices", _fake_query_devices)

    stop_event = threading.Event()
    reads = {"n": 0}

    def _stop_after_3s(stream):
        reads["n"] += 1
        if reads["n"] * 1024 >= stream.samplerate * 3:
            stop_event.set()

    monkeypatch.setattr(
        capture.sd, "InputStream",
        lambda **kw: FakeInputStream(on_read=_stop_after_3s, **kw),
    )

    chunk_queue = queue.Queue()
    capture.capture_worker_loopback(MONITOR_DEVICE["index"], chunk_queue, stop_event)

    items = []
    while not chunk_queue.empty():
        items.append(chunk_queue.get_nowait())
    assert items, "expected at least one chunk from 3 s of tone"
    assert all(kind != "error" for kind, _ in items)

    total_samples = 0
    for wav_path, rms in items:
        assert rms > 0.01
        rate, data = wavfile.read(wav_path)
        assert rate == capture.SAMPLE_RATE
        total_samples += len(data)
    # 3 s of tone resampled from 48 kHz to 16 kHz
    assert total_samples == pytest.approx(3 * capture.SAMPLE_RATE, rel=0.1)


def test_meeting_mixer_emits_stereo_blocks(monkeypatch):
    monkeypatch.setattr(audio_mixer_sd.sd, "query_devices", _fake_query_devices)
    monkeypatch.setattr(audio_mixer_sd.sd, "InputStream", FakeInputStream)

    mixer = AudioMixerSD(sample_rate=48000, frames_per_read=4096)
    blocks = []
    levels = []
    done = threading.Event()

    def _collect(stereo):
        blocks.append(stereo)
        if len(blocks) >= 5:
            done.set()

    mixer.set_stereo_callback(_collect)
    mixer.set_level_callback(levels.append)
    mixer.start(
        loopback_device_index=MONITOR_DEVICE["index"],
        mic_device_index=MIC_DEVICE["index"],
    )
    try:
        assert done.wait(timeout=5.0), "mixer produced fewer than 5 blocks in 5 s"
    finally:
        mixer.stop()

    assert not mixer.is_running()
    # Output rate follows the mic's native rate
    assert mixer.sample_rate == MIC_DEVICE["default_samplerate"]
    stereo = blocks[0]
    assert stereo.ndim == 2 and stereo.shape[1] == 2
    assert stereo.shape[0] == 4096
    # Both channels carry signal: left = mic tone, right = monitor tone
    assert float(np.abs(stereo[:, 0]).max()) > 0.01
    assert float(np.abs(stereo[:, 1]).max()) > 0.01
    assert levels and max(levels) > 0.01
