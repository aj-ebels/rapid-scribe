"""
Linux/portable audio-path tests: monitor-source device resolution, the sounddevice
meeting mixer's time-aligned loopback buffer, and the loopback worker's error path.
These run on any platform — sounddevice is only queried through monkeypatched stubs.
"""
import queue
import sys
import threading

import numpy as np
import pytest

from app import devices
from app.audio_mixer_sd import AudioMixerSD, MAX_LOOPBACK_BUF_SAMPLES


FAKE_DEVICES = [
    {"index": 0, "name": "HDA Intel PCH: ALC3204 Analog (hw:0,0)", "max_input_channels": 2, "default_samplerate": 44100},
    {"index": 1, "name": "Built-in Audio Analog Stereo", "max_input_channels": 0, "default_samplerate": 48000},
    {"index": 2, "name": "Monitor of Built-in Audio Analog Stereo", "max_input_channels": 2, "default_samplerate": 48000},
    {"index": 3, "name": "Monitor of USB Headset Analog Stereo", "max_input_channels": 2, "default_samplerate": 48000},
]


def _patch_query_devices(monkeypatch, default_output_name="USB Headset Analog Stereo"):
    def fake_query_devices(device=None, kind=None):
        if kind == "output":
            return {"index": 1, "name": default_output_name}
        if kind == "input":
            return {"index": 0, "name": FAKE_DEVICES[0]["name"]}
        if device is not None:
            return FAKE_DEVICES[device]
        return FAKE_DEVICES

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)


def test_list_loopback_devices_returns_monitors_on_linux(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("non-Windows path")
    _patch_query_devices(monkeypatch)
    monitors, err = devices.list_loopback_devices()
    assert err is None
    assert [d["index"] for d in monitors] == [2, 3]
    assert all("monitor" in d["name"].lower() for d in monitors)


def test_default_loopback_prefers_default_output_monitor(monkeypatch):
    _patch_query_devices(monkeypatch, default_output_name="USB Headset Analog Stereo")
    idx, err = devices.get_default_loopback_device()
    assert err is None
    assert idx == 3  # monitor matching the default output, not the first monitor


def test_default_loopback_falls_back_to_first_monitor(monkeypatch):
    _patch_query_devices(monkeypatch, default_output_name="HDMI Output")
    idx, err = devices.get_default_loopback_device()
    assert err is None
    assert idx == 2


def test_default_loopback_errors_without_monitor_source(monkeypatch):
    def fake_query_devices(device=None, kind=None):
        return [FAKE_DEVICES[0]]

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)
    idx, err = devices.get_default_loopback_device()
    assert idx is None
    assert "monitor" in err.lower()


def test_mixer_loopback_buffer_is_time_aligned():
    mixer = AudioMixerSD(sample_rate=48000)
    mixer._append_loopback_chunk(np.arange(10, dtype=np.float32))
    mixer._append_loopback_chunk(np.arange(10, 20, dtype=np.float32))
    out = mixer._consume_loopback_chunk(15)
    assert np.array_equal(out, np.arange(15, dtype=np.float32))
    # Underrun pads with silence
    out = mixer._consume_loopback_chunk(10)
    assert np.array_equal(out[:5], np.arange(15, 20, dtype=np.float32))
    assert np.all(out[5:] == 0.0)
    assert mixer._loopback_buffer_samples == 0


def test_mixer_loopback_buffer_caps_growth():
    mixer = AudioMixerSD(sample_rate=48000)
    block = np.ones(48000, dtype=np.float32)
    for _ in range(10):
        mixer._append_loopback_chunk(block)
    assert mixer._loopback_buffer_samples <= MAX_LOOPBACK_BUF_SAMPLES


def test_loopback_worker_reports_error_when_no_monitor(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("non-Windows path")
    from app import capture, pulse_monitor

    monkeypatch.setattr(
        devices, "get_default_loopback_device",
        lambda: (None, devices.NO_MONITOR_SOURCE_MSG),
    )
    # Sound-server fallback also finds nothing
    monkeypatch.setattr(
        pulse_monitor, "get_default_pulse_monitor",
        lambda: (None, "pactl not found"),
    )
    chunk_queue = queue.Queue()
    stop_event = threading.Event()
    capture.capture_worker_loopback(None, chunk_queue, stop_event)
    kind, msg = chunk_queue.get_nowait()
    assert kind == "error"
    assert "monitor" in msg.lower()
