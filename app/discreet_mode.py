"""
Discreet (incognito) capture-chrome helpers — pure logic + optional icon recolor.
GUI applies results; this module stays free of Tk.
"""
from __future__ import annotations

from typing import Optional

# Bland copy while discreet (KTD5)
STATUS_DISCREET_CAPTURING = "On"
STATUS_DISCREET_READY = "Ready"

# Calm tint for recolored record/stop icons (blue-gray, not red)
_DISCREET_TINT_RGB = (90, 120, 150)


def is_discreet(settings: Optional[dict]) -> bool:
    """True when settings request discreet capture chrome."""
    if not isinstance(settings, dict):
        return False
    return bool(settings.get("discreet_mode", False))


def pulse_allowed(settings: Optional[dict]) -> bool:
    """Recording transcript pulse is allowed only when not discreet."""
    return not is_discreet(settings)


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


def recolor_icon_calm(pil_image):
    """
    Return a new PIL RGBA image: desaturated then tinted calm blue-gray.
    Leaves alpha unchanged. Caller owns original.
    """
    from PIL import Image

    img = pil_image.convert("RGBA")
    pixels = img.load()
    w, h = img.size
    tr, tg, tb = _DISCREET_TINT_RGB
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            # Luma desaturate, then multiply by tint / 255
            luma = int(0.299 * r + 0.587 * g + 0.114 * b)
            nr = min(255, (luma * tr) // 255)
            ng = min(255, (luma * tg) // 255)
            nb = min(255, (luma * tb) // 255)
            pixels[x, y] = (nr, ng, nb, a)
    return img
