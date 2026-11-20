"""Mic capture must not block, and a stall or a PortAudio error must reopen."""
import sys
import types

from app import capture, chunk_recorder, diagnostic


def test_diagnostic_log_rotates(tmp_path):
    path = tmp_path / "diagnostic.log"
    for _ in range(4):
        path.write_bytes(b"x" * 30)
        diagnostic.rotate_log(path, max_bytes=10, backups=3)
    assert (tmp_path / "diagnostic.log.1").is_file()
    assert (tmp_path / "diagnostic.log.2").is_file()
    assert (tmp_path / "diagnostic.log.3").is_file()
    assert not (tmp_path / "diagnostic.log.4").exists()


def test_capture_diagnostics_use_the_package_import():
    assert chunk_recorder.diag is diagnostic.write
    assert capture.diag is diagnostic.write


def test_run_capture_reopens_on_stall_and_on_stream_error(monkeypatch):
    """0 for >3s of fake time, then PortAudio -9999, then one full 4096-frame block."""
    pa = types.ModuleType("pyaudiowpatch")
    pa.paInt16 = 8
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", pa)
    from app.audio_mixer import MIC_STALL_SEC, AudioMixer
    import app.audio_mixer as audio_mixer

    clock = {"ms": 0}

    def monotonic():
        return clock["ms"] / 1000.0

    def sleep(dt):
        clock["ms"] += int(round(dt * 1000))

    monkeypatch.setattr(audio_mixer.time, "monotonic", monotonic)
    monkeypatch.setattr(audio_mixer.time, "sleep", sleep)
    monkeypatch.setattr(audio_mixer.time, "time", monotonic)

    class Mic:
        def __init__(self):
            self.reads = []
            self.stage = "wait"

        def get_read_available(self):
            if clock["ms"] < int(MIC_STALL_SEC * 1000):
                return 0
            if self.stage == "wait":
                self.stage = "error"
                return 0
            if self.stage == "error":
                self.stage = "data"
                return -9999
            return 4096

        def read(self, n, exception_on_overflow=False):
            self.reads.append(n)
            mixer._running = False
            return b"\x00\x00" * n

    class Loop:
        def get_read_available(self):
            return 0

        def read(self, n, exception_on_overflow=False):
            raise AssertionError("loopback read")

    reasons = []

    def reopen(self, reason="read_error"):
        reasons.append(reason)
        return True

    monkeypatch.setattr(AudioMixer, "_reopen_streams", reopen)
    mic = Mic()
    mixer = AudioMixer(sample_rate=48000, frames_per_read=4096)
    mixer._running = True
    mixer._mic_stream = mic
    mixer._loopback_stream = Loop()
    mixer._mic_sample_rate = 48000
    mixer._loopback_sample_rate = 48000
    mixer._mic_channels = 1
    mixer._mic_device_index = 3
    mixer._run_capture()

    assert mic.reads == [4096]
    assert reasons[0] == "mic_stalled"
    assert "read_error" in reasons
