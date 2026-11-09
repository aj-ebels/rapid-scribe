"""
Discreet (incognito) capture-chrome helpers — pure logic, no Tk.
"""
from __future__ import annotations

from typing import Optional

# Bland copy while discreet (KTD5)
STATUS_DISCREET_CAPTURING = "On"
STATUS_DISCREET_READY = "Ready"


def is_discreet(settings: Optional[dict]) -> bool:
    """True when settings request discreet capture chrome."""
    if not isinstance(settings, dict):
        return False
    return bool(settings.get("discreet_mode", False))


def pulse_allowed(settings: Optional[dict]) -> bool:
    """Recording transcript pulse is allowed only when not discreet."""
    return not is_discreet(settings)


def volume_meter_visible(settings: Optional[dict]) -> bool:
    """Live input level is shown only when incognito is off."""
    return pulse_allowed(settings)


def should_refresh_idle_status(current: Optional[str]) -> bool:
    """Whether an idle status line should follow the current mode's ready copy.

    Stopping and Stopped stay put so mode-off still ends on "Stopped" (R7, KTD5).
    Install prompts stay explicit. Ready-style lines switch between discreet and normal.
    """
    if not isinstance(current, str) or not current:
        return False
    if current in ("Stopping...", "Stopped"):
        return False
    if "Install the transcription model" in current:
        return False
    return current.startswith("Ready")


def status_for_capture_state(
    settings: Optional[dict],
    *,
    running: bool,
    mode_off_capturing: str,
    mode_off_ready: str,
) -> str:
    """
    Pick status bar text for idle/capturing.
    mode_off_* are the existing loud strings used when discreet is off.
    """
    if is_discreet(settings):
        return STATUS_DISCREET_CAPTURING if running else STATUS_DISCREET_READY
    return mode_off_capturing if running else mode_off_ready
