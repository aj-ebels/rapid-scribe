"""
WASAPI capture via PyAudioWPatch (mic + loopback). Use when SoundCard crashes on your system.
Same interface as audio_capture.AudioMixer: set_stereo_callback, start, stop.
"""

import threading
import time
import numpy as np

from config import SAMPLE_RATE, FRAMES_PER_READ, MIXER_GAIN
from logging_config import get_logger, flush_log

log = get_logger("meetings2.audio_capture")

try:
    import pyaudiowpatch as pyaudio
except ImportError as e:
    raise ImportError("PyAudioWPatch is required when USE_PYAUDIOWPATCH is True. Install with: pip install pyaudiowpatch") from e

# PyAudio uses int16; we deliver float32
PA_FORMAT = pyaudio.paInt16
BYTES_PER_SAMPLE = 2


def _bytes_to_float_mono(data: bytes, channels: int) -> np.ndarray:
    """Convert raw int16 bytes to mono float32 (-1..1)."""
    arr = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return (arr.astype(np.float32) / 32768.0)


class AudioMixer:
    """
    Captures microphone and system loopback with PyAudioWPatch (WASAPI).
    Outputs stereo: Left = mic, Right = loopback. Fills loopback gaps with silence.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, frames_per_read: int = FRAMES_PER_READ, gain: float = MIXER_GAIN):
        self.sample_rate = sample_rate
        self.frames_per_read = frames_per_read
        self.gain = gain
        self._p = None
        self._mic_stream = None
        self._loopback_stream = None
        self._mic_channels = 1
        self._loopback_channels = 1
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stereo_callback = None

    def set_stereo_callback(self, callback):
        with self._lock:
            self._stereo_callback = callback

    def _run_capture(self):
        """
        Mic is read blocking (so we get a steady stream). Loopback is read only when
        data is available (get_read_available); when system is silent we use zeros.
        This avoids blocking on loopback when there's no system audio, which would
        otherwise freeze the mic stream.
        """
        while self._running:
            try:
                mic_data = self._mic_stream.read(self.frames_per_read, exception_on_overflow=False)
                if mic_data is None or len(mic_data) == 0:
                    time.sleep(0.001)
                    continue
                mic_mono = _bytes_to_float_mono(mic_data, self._mic_channels)
            except Exception as e:
                log.exception("Mic read failed: %s", e)
                time.sleep(0.001)
                continue
            n_mic = len(mic_mono)

            try:
                # Only read loopback if data is available; otherwise use silence.
                # (When system is silent, loopback read() would block and starve the mic.)
                available = self._loopback_stream.get_read_available()
                if available <= 0:
                    loopback_mono = np.zeros(n_mic, dtype=np.float32)
                else:
                    to_read = min(available, self.frames_per_read)
                    loopback_data = self._loopback_stream.read(to_read, exception_on_overflow=False)
                    if loopback_data is None or len(loopback_data) == 0:
                        loopback_mono = np.zeros(n_mic, dtype=np.float32)
                    else:
                        loopback_mono = _bytes_to_float_mono(loopback_data, self._loopback_channels)
                        if len(loopback_mono) < n_mic:
                            loopback_mono = np.concatenate([
                                loopback_mono,
                                np.zeros(n_mic - len(loopback_mono), dtype=np.float32),
                            ])
                        elif len(loopback_mono) > n_mic:
                            loopback_mono = loopback_mono[:n_mic]
            except Exception as e:
                log.exception("Loopback read failed: %s", e)
                loopback_mono = np.zeros(n_mic, dtype=np.float32)

            stereo = np.column_stack([
                mic_mono * self.gain,
                loopback_mono * self.gain,
            ]).astype(np.float32)
            with self._lock:
                cb = self._stereo_callback
            if cb is not None:
                try:
                    cb(stereo)
                except Exception as e:
                    log.exception("Stereo callback failed: %s", e)

    def start(self):
        if self._running:
            return
        log.debug("PyAudioWPatch: initializing WASAPI.")
        flush_log()
        self._p = pyaudio.PyAudio()
        try:
            wasapi_info = self._p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as e:
            log.exception("WASAPI not available: %s", e)
            self._p.terminate()
            self._p = None
            raise RuntimeError("WASAPI is not available on this system.") from e

        default_input_index = wasapi_info["defaultInputDevice"]
        default_output_index = wasapi_info["defaultOutputDevice"]
        mic_info = self._p.get_device_info_by_index(default_input_index)
        speakers_info = self._p.get_device_info_by_index(default_output_index)

        # Resolve loopback device (same as default output but as input/loopback)
        loopback_info = None
        if speakers_info.get("isLoopbackDevice"):
            loopback_info = speakers_info
        else:
            for dev in self._p.get_loopback_device_info_generator():
                if speakers_info["name"] in dev["name"]:
                    loopback_info = dev
                    break
        if loopback_info is None:
            self._p.terminate()
            self._p = None
            raise RuntimeError("Default loopback device not found. Run: python -m pyaudiowpatch")

        self._loopback_channels = max(1, int(loopback_info.get("maxInputChannels", 1)))
        self._mic_channels = max(1, int(mic_info.get("maxInputChannels", 1)))

        log.debug("Opening loopback stream (device=%s, rate=%s, ch=%s).", loopback_info["name"], self.sample_rate, self._loopback_channels)
        flush_log()
        self._loopback_stream = self._p.open(
            format=PA_FORMAT,
            channels=self._loopback_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.frames_per_read,
            input=True,
            input_device_index=loopback_info["index"],
        )
        log.debug("Loopback stream opened. Opening mic stream (device=%s).", mic_info["name"])
        flush_log()
        self._mic_stream = self._p.open(
            format=PA_FORMAT,
            channels=self._mic_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.frames_per_read,
            input=True,
            input_device_index=default_input_index,
        )
        log.debug("Mic stream opened. Starting capture thread.")
        flush_log()
        self._running = True
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception as e:
                log.exception("Mic stream close: %s", e)
            self._mic_stream = None
        if self._loopback_stream is not None:
            try:
                self._loopback_stream.stop_stream()
                self._loopback_stream.close()
            except Exception as e:
                log.exception("Loopback stream close: %s", e)
            self._loopback_stream = None
        if self._p is not None:
            try:
                self._p.terminate()
            except Exception as e:
                log.exception("PyAudio terminate: %s", e)
            self._p = None
        log.debug("Capture stopped.")

    def is_running(self):
        return self._running
