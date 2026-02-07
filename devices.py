"""
Audio device listing and resolution: sounddevice inputs, WASAPI loopback, effective device from settings.
"""
import sys
import sounddevice as sd

from settings import (
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


def list_loopback_devices():
    """On Windows with pyaudiowpatch: list WASAPI loopback devices. Otherwise empty."""
    if sys.platform != "win32":
        return [], None
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
    if mode == AUDIO_MODE_MEETING or mode == "meeting_ffmpeg":
        return (AUDIO_MODE_MEETING, settings.get("meeting_mic_device"), settings.get("loopback_device_index"))
    return (AUDIO_MODE_DEFAULT, None, None)
