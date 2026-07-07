"""
Audio device listing and resolution: sounddevice inputs, WASAPI loopback, effective device from settings.
"""
import sys
import sounddevice as sd

from .settings import (
    load_settings,
    AUDIO_MODE_DEFAULT,
    AUDIO_MODE_LOOPBACK,
    AUDIO_MODE_MEETING,
)


def list_audio_devices():
    """List all audio devices with index, name, and input channels."""
    devices = []
    try:
        all_devices = sd.query_devices()
        for dev in all_devices:
            devices.append({
                "index": dev.get("index", len(devices)),
                "name": dev.get("name", "Unknown"),
                "max_input_channels": dev.get("max_input_channels", 0),
                "default_samplerate": dev.get("default_samplerate", 0),
            })
    except Exception as e:
        return [], str(e)
    return devices, None


def _list_monitor_inputs():
    """Inputs that carry system audio on PulseAudio/PipeWire ('Monitor of …' sources)."""
    devices, err = list_audio_devices()
    if err:
        return [], err
    monitors = [
        d for d in devices
        if d.get("max_input_channels", 0) > 0 and "monitor" in (d.get("name") or "").lower()
    ]
    return monitors, None


# Monitor sources that only the sound server knows about (PortAudio built without
# the PulseAudio host API) get synthetic negative indices so they fit everywhere a
# PortAudio device index is stored/displayed: pactl index i <-> -(PULSE_INDEX_BASE + i).
PULSE_INDEX_BASE = 1000


def _encode_pulse_index(pactl_index: int) -> int:
    return -(PULSE_INDEX_BASE + int(pactl_index))


def _decode_pulse_index(synthetic_index: int) -> int:
    return -int(synthetic_index) - PULSE_INDEX_BASE


def is_pulse_synthetic_index(index) -> bool:
    return index is not None and int(index) <= -PULSE_INDEX_BASE


def list_loopback_devices():
    """List system-audio capture devices: WASAPI loopback on Windows, monitor sources elsewhere."""
    if sys.platform != "win32":
        monitors, err = _list_monitor_inputs()
        if monitors or sys.platform == "darwin":
            return monitors, err
        # PortAudio sees no monitor sources (ALSA-only build) — ask the sound server directly.
        from .pulse_monitor import list_pulse_monitor_sources
        pulse_monitors, pulse_err = list_pulse_monitor_sources()
        if pulse_monitors:
            return [
                {
                    "index": _encode_pulse_index(m["pactl_index"]),
                    "name": m["name"],
                    "max_input_channels": 2,
                    "default_samplerate": 48000,
                }
                for m in pulse_monitors
            ], None
        return [], err or pulse_err
    try:
        import pyaudiowpatch as pyaudio
        with pyaudio.PyAudio() as p:
            devices = []
            for loopback in p.get_loopback_device_info_generator():
                devices.append({
                    "index": loopback["index"],
                    "name": loopback["name"],
                    "default_samplerate": loopback.get("defaultSampleRate", 48000),
                    "max_input_channels": loopback.get("maxInputChannels", 2),
                })
            return devices, None
    except ImportError:
        return [], "pyaudiowpatch not installed"
    except Exception as e:
        return [], str(e)


NO_MONITOR_SOURCE_MSG = (
    "No system-audio monitor source found. System-audio capture needs PulseAudio or "
    "PipeWire (with pipewire-pulse) running, plus the pulseaudio-utils tools "
    "(pactl/parec). Check `pactl list short sources` for a '….monitor' source."
)


def get_default_loopback_device():
    """
    Resolve the system-audio capture device on Linux/macOS: prefer the monitor source
    of the default output device, else the first monitor source. PortAudio devices only —
    for the Pulse-native fallback, use resolve_loopback_target().
    Returns (device_index, error_message).
    """
    monitors, err = _list_monitor_inputs()
    if err:
        return None, err
    if not monitors:
        return None, NO_MONITOR_SOURCE_MSG
    try:
        default_out = sd.query_devices(kind="output")
        out_name = (default_out.get("name") or "").lower()
    except Exception:
        out_name = ""
    if out_name:
        for d in monitors:
            if out_name in (d.get("name") or "").lower():
                return d["index"], None
    return monitors[0]["index"], None


def resolve_loopback_target(loopback_device_index=None):
    """
    Resolve where system audio should be captured from on Linux/macOS.
    Returns (target, error_message) where target is
      {"backend": "portaudio", "index": int}   — monitor device visible to PortAudio, or
      {"backend": "pulse", "source": str}      — sound-server source captured via parec.
    """
    # Explicit selection from settings
    if loopback_device_index is not None:
        if is_pulse_synthetic_index(loopback_device_index):
            from .pulse_monitor import list_pulse_monitor_sources
            pactl_index = _decode_pulse_index(loopback_device_index)
            monitors, err = list_pulse_monitor_sources()
            if err:
                return None, err
            for m in monitors:
                if m["pactl_index"] == pactl_index:
                    return {"backend": "pulse", "source": m["name"]}, None
            if monitors:  # saved source is gone (indices shift across reboots) — use default
                from .pulse_monitor import get_default_pulse_monitor
                source, err = get_default_pulse_monitor()
                if source:
                    return {"backend": "pulse", "source": source}, None
            return None, NO_MONITOR_SOURCE_MSG
        return {"backend": "portaudio", "index": int(loopback_device_index)}, None

    # Default: PortAudio monitor of the default output if visible…
    index, _err = get_default_loopback_device()
    if index is not None:
        return {"backend": "portaudio", "index": index}, None
    # …else fall back to the sound server directly.
    from .pulse_monitor import get_default_pulse_monitor
    source, pulse_err = get_default_pulse_monitor()
    if source:
        return {"backend": "pulse", "source": source}, None
    return None, f"{NO_MONITOR_SOURCE_MSG} ({pulse_err})" if pulse_err else NO_MONITOR_SOURCE_MSG


def get_default_monitor_device():
    """
    Select default input device: first input whose name contains 'monitor' or 'RDPSource'.
    Falls back to system default input if none found.
    """
    devices, err = list_audio_devices()
    if err:
        return None, err
    keywords = ("monitor", "RDPSource")
    for d in devices:
        if d["max_input_channels"] <= 0:
            continue
        name = (d.get("name") or "").lower()
        if any(kw.lower() in name for kw in keywords):
            return d["index"], None
    try:
        default = sd.query_devices(kind="input")
        return default["index"], None
    except Exception as e:
        return None, str(e)


def get_effective_audio_device(app):
    """
    Resolve capture mode and device indices from settings.
    Returns (mode, mic_device_index, loopback_device_index).
    """
    settings = load_settings() if app is None else getattr(app, "settings", load_settings())
    mode = settings.get("audio_mode") or AUDIO_MODE_DEFAULT
    if mode == AUDIO_MODE_LOOPBACK:
        return (AUDIO_MODE_LOOPBACK, None, settings.get("loopback_device_index"))
    # "meeting_ffmpeg" = legacy setting value; treat as in-process Meeting
    if mode == AUDIO_MODE_MEETING or mode == "meeting_ffmpeg":
        return (AUDIO_MODE_MEETING, settings.get("meeting_mic_device"), settings.get("loopback_device_index"))
    return (AUDIO_MODE_DEFAULT, None, None)
