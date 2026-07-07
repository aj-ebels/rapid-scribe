"""
Tests for the Pulse-native system-audio fallback: when PortAudio is built without
the PulseAudio host API (common on Ubuntu), monitor sources are enumerated with
pactl and captured with a parec subprocess. Covers pactl parsing, device/target
resolution with synthetic indices, the PulseMonitorStream, and end-to-end capture
through a fake parec process.
"""
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from scipy.io import wavfile

from app import devices, pulse_monitor
from app.pulse_monitor import PulseMonitorStream


PACTL_SHORT_SOURCES = (
    "1\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tmodule-alsa-card.c\ts32le 2ch 48000Hz\tIDLE\n"
    "2\talsa_input.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\ts32le 2ch 48000Hz\tSUSPENDED\n"
    "7\talsa_output.usb-headset.analog-stereo.monitor\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tRUNNING\n"
)
PACTL_INFO = (
    "Server Name: PulseAudio (on PipeWire 1.0.5)\n"
    "Default Sink: alsa_output.usb-headset.analog-stereo\n"
    "Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo\n"
)


def _patch_pactl(monkeypatch, sources=PACTL_SHORT_SOURCES, info=PACTL_INFO):
    def fake_run(cmd, **kwargs):
        assert cmd[0] == "pactl"
        out = sources if cmd[1:] == ["list", "short", "sources"] else info
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(pulse_monitor.subprocess, "run", fake_run)


def _patch_no_portaudio_monitors(monkeypatch):
    """PortAudio sees only plain ALSA devices — no monitor sources."""
    plain = [
        {"index": 0, "name": "HDA Intel PCH: ALC3204 Analog (hw:0,0)", "max_input_channels": 2, "default_samplerate": 44100},
        {"index": 1, "name": "default", "max_input_channels": 32, "default_samplerate": 48000},
    ]

    def fake_query_devices(device=None, kind=None):
        if kind == "output":
            return {"index": 1, "name": "default"}
        if kind == "input":
            return plain[0]
        if device is not None:
            return plain[device]
        return plain

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)


# -- pactl parsing -------------------------------------------------------------


def test_list_pulse_monitor_sources_parses_pactl(monkeypatch):
    _patch_pactl(monkeypatch)
    monitors, err = pulse_monitor.list_pulse_monitor_sources()
    assert err is None
    assert [(m["pactl_index"], m["name"]) for m in monitors] == [
        (1, "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"),
        (7, "alsa_output.usb-headset.analog-stereo.monitor"),
    ]


def test_default_pulse_monitor_follows_default_sink(monkeypatch):
    _patch_pactl(monkeypatch)
    source, err = pulse_monitor.get_default_pulse_monitor()
    assert err is None
    assert source == "alsa_output.usb-headset.analog-stereo.monitor"


def test_pulse_listing_reports_missing_pactl(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("pactl")

    monkeypatch.setattr(pulse_monitor.subprocess, "run", fake_run)
    monitors, err = pulse_monitor.list_pulse_monitor_sources()
    assert monitors == []
    assert "pulseaudio-utils" in err


# -- device listing and target resolution ---------------------------------------


def test_list_loopback_devices_falls_back_to_pulse(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("non-Windows path")
    _patch_no_portaudio_monitors(monkeypatch)
    _patch_pactl(monkeypatch)
    listed, err = devices.list_loopback_devices()
    assert err is None
    assert [d["index"] for d in listed] == [
        devices._encode_pulse_index(1),
        devices._encode_pulse_index(7),
    ]
    assert all(devices.is_pulse_synthetic_index(d["index"]) for d in listed)


def test_resolve_target_prefers_portaudio_monitor(monkeypatch):
    def fake_query_devices(device=None, kind=None):
        if kind == "output":
            return {"index": 1, "name": "Built-in Audio Analog Stereo"}
        return [
            {"index": 0, "name": "Monitor of Built-in Audio Analog Stereo", "max_input_channels": 2, "default_samplerate": 48000},
        ]

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)
    target, err = devices.resolve_loopback_target(None)
    assert err is None
    assert target == {"backend": "portaudio", "index": 0}


def test_resolve_target_falls_back_to_pulse_default(monkeypatch):
    _patch_no_portaudio_monitors(monkeypatch)
    _patch_pactl(monkeypatch)
    target, err = devices.resolve_loopback_target(None)
    assert err is None
    assert target == {"backend": "pulse", "source": "alsa_output.usb-headset.analog-stereo.monitor"}


def test_resolve_target_decodes_synthetic_index(monkeypatch):
    _patch_pactl(monkeypatch)
    target, err = devices.resolve_loopback_target(devices._encode_pulse_index(1))
    assert err is None
    assert target == {"backend": "pulse", "source": "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"}


def test_resolve_target_stale_synthetic_index_uses_default(monkeypatch):
    _patch_pactl(monkeypatch)
    target, err = devices.resolve_loopback_target(devices._encode_pulse_index(99))
    assert err is None
    assert target == {"backend": "pulse", "source": "alsa_output.usb-headset.analog-stereo.monitor"}


def test_resolve_target_positive_index_is_portaudio():
    target, err = devices.resolve_loopback_target(5)
    assert err is None
    assert target == {"backend": "portaudio", "index": 5}


# -- PulseMonitorStream via a fake parec process --------------------------------


class FakeParecProcess:
    """Writes a 440 Hz float32 tone to a real pipe, like parec's stdout."""

    def __init__(self, rate=48000, channels=2, duration_sec=3.0, paced=False):
        r, w = os.pipe()
        self.stdout = os.fdopen(r, "rb")
        self._w = os.fdopen(w, "wb")
        self._rate = rate
        self._channels = channels
        self._total = int(rate * duration_sec)
        self._paced = paced
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self):
        pos = 0
        block = 1024
        try:
            while pos < self._total and not self._stop.is_set():
                t = (np.arange(block) + pos) / self._rate
                tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
                frames = np.tile(tone[:, None], (1, self._channels))
                self._w.write(frames.tobytes())
                pos += block
                if self._paced:
                    time.sleep(block / self._rate / 20.0)  # 20× real time
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                self._w.close()
            except Exception:
                pass

    def poll(self):
        return None if self._thread.is_alive() else 0

    def terminate(self):
        self._stop.set()

    def wait(self, timeout=None):
        self._thread.join(timeout=timeout)
        return 0

    def kill(self):
        self.terminate()


def _patch_parec(monkeypatch, **proc_kwargs):
    def fake_popen(cmd, **kwargs):
        assert cmd[0] == "parec"
        rate = int(cmd[cmd.index("--rate") + 1])
        channels = int(cmd[cmd.index("--channels") + 1])
        return FakeParecProcess(rate=rate, channels=channels, **proc_kwargs)

    monkeypatch.setattr(pulse_monitor.subprocess, "Popen", fake_popen)


def test_pulse_monitor_stream_reads_frames(monkeypatch):
    _patch_parec(monkeypatch, duration_sec=1.0)
    with PulseMonitorStream("fake.monitor", samplerate=48000, channels=2) as stream:
        frames, overflowed = stream.read(4096)
        assert not overflowed
        assert frames.shape == (4096, 2)
        assert float(np.abs(frames).max()) > 0.1
        # After EOF and drain, read raises so callers can stop cleanly
        with pytest.raises(RuntimeError):
            while True:
                stream.read(48000)


def test_loopback_worker_produces_chunks_via_pulse(monkeypatch, tmp_path):
    if sys.platform == "win32":
        pytest.skip("non-Windows path")
    from app import capture

    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr(
        devices, "resolve_loopback_target",
        lambda idx=None: ({"backend": "pulse", "source": "fake.monitor"}, None),
    )
    _patch_parec(monkeypatch, duration_sec=3.0)

    chunk_queue = queue.Queue()
    stop_event = threading.Event()  # worker exits when the fake parec hits EOF
    capture.capture_worker_loopback(None, chunk_queue, stop_event)

    items = []
    while not chunk_queue.empty():
        items.append(chunk_queue.get_nowait())
    assert items, "expected chunks from 3 s of tone via the pulse backend"
    assert all(kind != "error" for kind, _ in items)
    total_samples = 0
    for wav_path, rms in items:
        assert rms > 0.01
        rate, data = wavfile.read(wav_path)
        assert rate == capture.SAMPLE_RATE
        total_samples += len(data)
    assert total_samples == pytest.approx(3 * capture.SAMPLE_RATE, rel=0.1)


def test_meeting_mixer_works_via_pulse(monkeypatch):
    from app import audio_mixer_sd
    from tests.test_linux_capture_e2e import FakeInputStream, MIC_DEVICE

    def fake_query_devices(device=None, kind=None):
        return MIC_DEVICE

    monkeypatch.setattr(audio_mixer_sd.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(audio_mixer_sd.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(
        audio_mixer_sd, "resolve_loopback_target",
        lambda idx=None: ({"backend": "pulse", "source": "fake.monitor"}, None),
    )
    _patch_parec(monkeypatch, duration_sec=30.0, paced=True)

    mixer = audio_mixer_sd.AudioMixerSD(sample_rate=48000, frames_per_read=4096)
    blocks = []
    done = threading.Event()

    def _collect(stereo):
        blocks.append(stereo)
        if len(blocks) >= 5:
            done.set()

    mixer.set_stereo_callback(_collect)
    mixer.start(loopback_device_index=None, mic_device_index=MIC_DEVICE["index"])
    try:
        assert done.wait(timeout=5.0), "mixer produced fewer than 5 blocks in 5 s"
    finally:
        mixer.stop()

    stereo = np.concatenate(blocks[:5])
    assert stereo.shape[1] == 2
    assert float(np.abs(stereo[:, 0]).max()) > 0.01, "mic (left) carries signal"
    assert float(np.abs(stereo[:, 1]).max()) > 0.01, "system audio (right) carries signal"
