"""
In-process WASAPI mic + loopback capture (PyAudioWPatch).
Single thread: mic read is blocking (steady stream). Loopback is drained into a
time-aligned ring buffer; each block consumes exactly the loopback samples that
correspond to that block's duration, so the loopback timeline is continuous and
transcription quality is preserved.
Output: stereo callback with Left=mic, Right=loopback at capture sample rate.
"""

import logging
import threading
import time
import numpy as np
from scipy.signal import resample

log = logging.getLogger(__name__)

# Max loopback buffer: ~5 sec at 48 kHz so we stay time-aligned without unbounded growth
MAX_LOOPBACK_BUF_SAMPLES = 240000

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
        self._mic_sample_rate = None  # native mic rate (may differ from loopback)
        self._loopback_sample_rate = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stereo_callback = None
        self._level_callback = None
        # Time-aligned loopback: samples at loopback sample rate, consumed per block
        self._loopback_buffer = []

    def set_stereo_callback(self, callback):
        """Set callable(stereo_frames: np.ndarray) invoked from capture thread for each block."""
        with self._lock:
            self._stereo_callback = callback

    def set_level_callback(self, callback):
        """Set callable(rms: float) invoked from capture thread with mic RMS per block."""
        with self._lock:
            self._level_callback = callback

    def _run_capture(self):
        """
        Mic is read blocking (steady stream). Loopback is drained into a buffer;
        each block consumes exactly the loopback samples for that block's duration,
        then resamples to mic length. This keeps the loopback timeline continuous
        so transcription of system audio (e.g. other participants) is usable.
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
            mic_rms = float(np.sqrt(np.mean(mic_mono.astype(np.float64) ** 2)))

            # Drain all available loopback into the time-aligned buffer (non-blocking)
            try:
                while True:
                    available = self._loopback_stream.get_read_available()
                    if available <= 0:
                        break
                    to_read = min(available, self.frames_per_read * 2)
                    loopback_data = self._loopback_stream.read(to_read, exception_on_overflow=False)
                    if loopback_data is None or len(loopback_data) == 0:
                        break
                    chunk = _bytes_to_float_mono(loopback_data, self._loopback_channels)
                    self._loopback_buffer.append(chunk)
                    total = sum(arr.size for arr in self._loopback_buffer)
                    if total > MAX_LOOPBACK_BUF_SAMPLES:
                        flat = np.concatenate(self._loopback_buffer)
                        self._loopback_buffer = [flat[-MAX_LOOPBACK_BUF_SAMPLES:]]
            except Exception as e:
                log.exception("Loopback read failed: %s", e)

            # Consume exactly the loopback samples that match this block's duration (time-aligned)
            n_loopback_needed = int(round(n_mic * self._loopback_sample_rate / self._mic_sample_rate))
            if n_loopback_needed <= 0:
                n_loopback_needed = n_mic
            flat = np.concatenate(self._loopback_buffer) if self._loopback_buffer else np.array([], dtype=np.float32)
            if len(flat) >= n_loopback_needed:
                loopback_chunk = flat[:n_loopback_needed].astype(np.float32)
                remainder = flat[n_loopback_needed:]
                self._loopback_buffer = [remainder] if len(remainder) > 0 else []
            else:
                loopback_chunk = np.zeros(n_loopback_needed, dtype=np.float32)
                if len(flat) > 0:
                    loopback_chunk[: len(flat)] = flat
                self._loopback_buffer = []

            # Resample loopback to output (mic) rate so we get exactly n_mic samples
            if len(loopback_chunk) != n_mic:
                loopback_mono = resample(loopback_chunk, n_mic).astype(np.float32)
            else:
                loopback_mono = loopback_chunk

            with self._lock:
                level_cb = self._level_callback
            if level_cb is not None:
                try:
                    loopback_rms = float(np.sqrt(np.mean(loopback_mono.astype(np.float64) ** 2)))
                    level_cb(max(mic_rms, loopback_rms))
                except Exception as e:
                    log.debug("Level callback failed: %s", e)

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

        # Open each device at its native default sample rate (avoids -9997 when dock/mic differ)
        def _device_rate(info, fallback=48000):
            r = info.get("defaultSampleRate") or fallback
            r = int(r) if r else fallback
            return r if r > 0 else fallback

        self._loopback_sample_rate = _device_rate(loopback_info)
        self._mic_sample_rate = _device_rate(mic_info)
        self.sample_rate = self._mic_sample_rate  # output rate = mic rate; loopback resampled in loop

        try:
            self._loopback_stream = self._p.open(
                format=PA_FORMAT,
                channels=self._loopback_channels,
                rate=self._loopback_sample_rate,
                frames_per_buffer=self.frames_per_read,
                input=True,
                input_device_index=loopback_info["index"],
            )
        except OSError as e:
            self._p.terminate()
            self._p = None
            raise RuntimeError(
                "Loopback device does not support its default rate %d Hz: %s"
                % (self._loopback_sample_rate, e)
            ) from e
        try:
            self._mic_stream = self._p.open(
                format=PA_FORMAT,
                channels=self._mic_channels,
                rate=self._mic_sample_rate,
                frames_per_buffer=self.frames_per_read,
                input=True,
                input_device_index=default_input_index,
            )
        except OSError as e:
            try:
                self._loopback_stream.stop_stream()
                self._loopback_stream.close()
            except Exception:
                pass
            self._p.terminate()
            self._p = None
            raise RuntimeError(
                "Microphone does not support its default rate %d Hz: %s"
                % (self._mic_sample_rate, e)
            ) from e

        if self._mic_sample_rate != self.sample_rate or self._loopback_sample_rate != self.sample_rate:
            log.info(
                "Meeting capture: mic=%d Hz, loopback=%d Hz, output=%d Hz.",
                self._mic_sample_rate, self._loopback_sample_rate, self.sample_rate,
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
