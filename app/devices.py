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


def list_loopback_devices():
    """List system-audio capture devices: WASAPI loopback on Windows, monitor sources elsewhere."""
    if sys.platform != "win32":
        return _list_monitor_inputs()
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
    "PipeWire exposing a 'Monitor of …' input device. Check that audio is set up "
    "(e.g. pipewire-pulse or pulseaudio is running) and that PortAudio can see it."
)


def get_default_loopback_device():
    """
    Resolve the system-audio capture device on Linux/macOS: prefer the monitor source
    of the default output device, else the first monitor source.
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
