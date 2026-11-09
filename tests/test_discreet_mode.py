"""Unit tests for discreet (incognito) capture-chrome helpers."""
from app.discreet_mode import (
    is_discreet,
    pulse_allowed,
    should_refresh_idle_status,
    status_for_capture_state,
    STATUS_DISCREET_CAPTURING,
    STATUS_DISCREET_READY,
)
from app.settings import load_settings


def test_is_discreet_defaults_false():
    assert is_discreet(None) is False
    assert is_discreet({}) is False
    assert is_discreet({"discreet_mode": False}) is False
    assert is_discreet({"discreet_mode": True}) is True


def test_pulse_allowed_inverse_of_discreet():
    assert pulse_allowed({"discreet_mode": False}) is True
    assert pulse_allowed({"discreet_mode": True}) is False


def test_status_matrix():
    loud_cap = "Recording & transcribing…"
    loud_ready = "Ready to record & transcribe"
    off = {"discreet_mode": False}
    on = {"discreet_mode": True}
    assert status_for_capture_state(off, running=True, mode_off_capturing=loud_cap, mode_off_ready=loud_ready) == loud_cap
    assert status_for_capture_state(off, running=False, mode_off_capturing=loud_cap, mode_off_ready=loud_ready) == loud_ready
    assert status_for_capture_state(on, running=True, mode_off_capturing=loud_cap, mode_off_ready=loud_ready) == STATUS_DISCREET_CAPTURING
    assert status_for_capture_state(on, running=False, mode_off_capturing=loud_cap, mode_off_ready=loud_ready) == STATUS_DISCREET_READY


def test_idle_status_keeps_stopped_and_refreshes_ready_copy():
    assert should_refresh_idle_status("Stopped") is False
    assert should_refresh_idle_status("Stopping...") is False
    assert should_refresh_idle_status("Install the transcription model first (Model tab → Download & install)") is False
    assert should_refresh_idle_status("Checking…") is False
    assert should_refresh_idle_status("Ready") is True
    assert should_refresh_idle_status("Ready to record & transcribe") is True


def test_load_settings_includes_discreet_mode_default(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "settings.json")
    s = load_settings()
    assert s.get("discreet_mode") is False
