"""Unit tests for discreet (incognito) capture-chrome helpers."""
from pathlib import Path

import pytest

from app.discreet_mode import (
    is_discreet,
    pulse_allowed,
    recolor_icon_calm,
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


def test_load_settings_includes_discreet_mode_default(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "settings.json")
    s = load_settings()
    assert s.get("discreet_mode") is False


def test_recolor_icon_calm_changes_pixels():
    pytest.importorskip("PIL")
    from PIL import Image

    # Solid red opaque pixel should change under calm recolor
    img = Image.new("RGBA", (4, 4), (220, 30, 30, 255))
    out = recolor_icon_calm(img)
    assert out.size == img.size
    assert out.getpixel((1, 1)) != img.getpixel((1, 1))
    # Alpha preserved
    assert out.getpixel((1, 1))[3] == 255


def test_recolor_icon_from_assets_if_present():
    pytest.importorskip("PIL")
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    path = root / "assets" / "icons" / "record.png"
    if not path.exists():
        pytest.skip("record.png not present")
    img = Image.open(path).convert("RGBA")
    out = recolor_icon_calm(img)
    assert out.size == img.size
