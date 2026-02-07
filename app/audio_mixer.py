"""
In-process WASAPI mic + loopback capture (PyAudioWPatch).
Single thread: mic read is blocking (steady stream); loopback is read only when
data is available (get_read_available), otherwise silence — so loopback never blocks the mic.
Output: stereo callback with Left=mic, Right=loopback at capture sample rate.
"""

import logging
import threading
import time
import numpy as np

log = logging.getLogger(__name__)

try:
    import pyaudiowpatch as pyaudio
except ImportError as e:
    raise ImportError("PyAudioWPatch is required for in-process mic+loopback. pip install pyaudiowpatch") from e

PA_FORMAT = pyaudio.paInt16


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

    def __init__(self, sample_rate: int, frames_per_read: int = 4096, gain: float = 0.7):
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
        """Set callable(stereo_frames: np.ndarray) invoked from capture thread for each block."""
        with self._lock:
            self._stereo_callback = callback

    def _run_capture(self):
        """
        Mic is read blocking (steady stream). Loopback is read only when
        data is available (get_read_available); when system is silent we use zeros.
        This avoids blocking on loopback when there's no system audio.
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

    def start(self, loopback_device_index=None, mic_device_index=None):
        """Start capture. Uses default devices unless indices are provided."""
        if self._running:
            return
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
        if mic_device_index is not None:
            default_input_index = mic_device_index
        mic_info = self._p.get_device_info_by_index(default_input_index)
        speakers_info = self._p.get_device_info_by_index(default_output_index)

        if loopback_device_index is not None:
            loopback_info = self._p.get_device_info_by_index(loopback_device_index)
        elif speakers_info.get("isLoopbackDevice"):
            loopback_info = speakers_info
        else:
            loopback_info = None
            for dev in self._p.get_loopback_device_info_generator():
                if speakers_info["name"] in dev["name"]:
                    loopback_info = dev
                    break
        if loopback_info is None:
            self._p.terminate()
            self._p = None
            raise RuntimeError("Default loopback device not found.")

        self._loopback_channels = max(1, int(loopback_info.get("maxInputChannels", 1)))
        self._mic_channels = max(1, int(mic_info.get("maxInputChannels", 1)))

        self._loopback_stream = self._p.open(
            format=PA_FORMAT,
            channels=self._loopback_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.frames_per_read,
            input=True,
            input_device_index=loopback_info["index"],
        )
        self._mic_stream = self._p.open(
            format=PA_FORMAT,
            channels=self._mic_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.frames_per_read,
            input=True,
            input_device_index=default_input_index,
        )
        self._running = True
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()
        log.debug("In-process mic+loopback capture started.")

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
