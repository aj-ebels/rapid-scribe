"""The mic read must not block, and a default-device change must reopen capture."""
import inspect

from app import capture, chunk_recorder, diagnostic
from app.devices import default_devices_changed, frames_to_read


def test_frames_to_read_skips_the_blocking_read_until_samples_exist():
    assert frames_to_read(0, 4096) == 0
    assert frames_to_read(None, 4096) == 0
    assert frames_to_read(100, 4096) == 100
    assert frames_to_read(8000, 4096) == 4096


def test_default_device_change_reopens_only_when_following_the_system_default():
    same = dict(
        follow_mic=True,
        follow_loopback=True,
        current_mic_name="Headset Mic",
        current_loopback_name="Headphones [Loopback]",
        default_mic_name="Headset Mic",
        default_loopback_name="Headphones [Loopback]",
    )
    assert default_devices_changed(**same) is False
    assert default_devices_changed(**dict(same, default_mic_name="Laptop Mic")) is True
    assert default_devices_changed(
        **dict(same, default_loopback_name="Speakers [Loopback]")
    ) is True
    # An explicit device stays put when Windows changes the default.
    assert default_devices_changed(
        **dict(same, follow_mic=False, follow_loopback=False, default_mic_name="Laptop Mic")
    ) is False
    assert default_devices_changed(**dict(same, default_mic_name="")) is False


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
    assert "from diagnostic import" not in inspect.getsource(chunk_recorder)
    assert "from diagnostic import" not in inspect.getsource(capture.meeting_chunk_ready)
