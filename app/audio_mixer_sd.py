"""
In-process mic + system-audio capture via sounddevice/PortAudio (Linux/macOS).
System audio comes from the PulseAudio/PipeWire monitor source of the output
device — PortAudio exposes it as a normal input, so no WASAPI-style loopback
API is needed. Same architecture as the Windows AudioMixer (app/audio_mixer.py):
a single thread blocks on mic reads (steady stream) and drains the monitor
stream into a time-aligned ring buffer; each block consumes exactly the monitor
samples that correspond to that block's duration, so the system-audio timeline
is continuous and transcription quality is preserved.
Output: stereo callback with Left=mic, Right=system audio at the mic sample rate.
"""

import logging
import threading
import time
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from .audio_leveler import AudioLeveler, AudioLevelerConfig
from .devices import resolve_loopback_target
from .pulse_monitor import PulseMonitorStream

log = logging.getLogger(__name__)

# Max loopback buffer: ~5 sec at 48 kHz so we stay time-aligned without unbounded growth
MAX_LOOPBACK_BUF_SAMPLES = 240000
STREAM_REOPEN_COOLDOWN_SEC = 2.0
MIC_ERROR_REOPEN_THRESHOLD = 5


def _to_mono(frames: np.ndarray) -> np.ndarray:
    """Collapse an (n, ch) float32 block to mono float32."""
    if frames.ndim > 1 and frames.shape[1] > 1:
        return frames.mean(axis=1).astype(np.float32)
    return np.asarray(frames, dtype=np.float32).reshape(-1)


def _rms32(samples: np.ndarray) -> float:
    """Fast RMS for float32 buffers."""
    if samples is None or samples.size == 0:
        return 0.0
    buf = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(buf * buf)))


class AudioMixerSD:
    """
    Captures microphone and the system-audio monitor source with sounddevice.
    Outputs stereo: Left = mic, Right = system audio. Fills gaps with silence.
    Public API matches app.audio_mixer.AudioMixer.
    """

    def __init__(
        self,
        sample_rate: int,
        frames_per_read: int = 4096,
        gain: float = 0.7,
        *,
        mic_gain: float | None = None,
        loopback_gain: float | None = None,
        mic_leveler_config: AudioLevelerConfig | None = None,
    ):
        self.sample_rate = sample_rate
        self.frames_per_read = frames_per_read
        self.mic_gain = float(gain if mic_gain is None else mic_gain)
        self.loopback_gain = float(gain if loopback_gain is None else loopback_gain)
        self._mic_leveler = AudioLeveler(mic_leveler_config or AudioLevelerConfig())
        self._mic_stream = None
        self._loopback_stream = None
        self._mic_channels = 1
        self._loopback_channels = 1
        self._mic_sample_rate = None  # native mic rate (may differ from monitor source)
        self._loopback_sample_rate = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stereo_callback = None
        self._level_callback = None
        self._health_callback = None
        # Time-aligned loopback: samples at loopback sample rate, consumed per block
        self._loopback_buffer = deque()
        self._loopback_buffer_samples = 0
        self._mic_device_index = None
        self._loopback_target = None  # {"backend": "portaudio", "index"} or {"backend": "pulse", "source"}
        self._mic_read_errors = 0
        self._last_reopen_attempt = 0.0

    def _close_stream(self, stream, name: str):
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as e:
            log.debug("%s stream stop failed: %s", name, e)
        try:
            stream.close()
        except Exception as e:
            log.debug("%s stream close failed: %s", name, e)

    def _stop_stream_only(self, stream, name: str):
        if stream is None:
            return
        try:
            stream.stop()
        except Exception as e:
            log.debug("%s stream stop failed: %s", name, e)

    def _append_loopback_chunk(self, chunk: np.ndarray):
        if chunk is None or chunk.size == 0:
            return
        arr = np.asarray(chunk, dtype=np.float32)
        self._loopback_buffer.append(arr)
        self._loopback_buffer_samples += arr.size
        if self._loopback_buffer_samples <= MAX_LOOPBACK_BUF_SAMPLES:
            return
        to_drop = self._loopback_buffer_samples - MAX_LOOPBACK_BUF_SAMPLES
        while to_drop > 0 and self._loopback_buffer:
            head = self._loopback_buffer[0]
            if head.size <= to_drop:
                self._loopback_buffer.popleft()
                self._loopback_buffer_samples -= head.size
                to_drop -= head.size
            else:
                self._loopback_buffer[0] = head[to_drop:]
                self._loopback_buffer_samples -= to_drop
                to_drop = 0

    def _consume_loopback_chunk(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            return np.array([], dtype=np.float32)
        out = np.zeros(n_samples, dtype=np.float32)
        written = 0
        while written < n_samples and self._loopback_buffer:
            head = self._loopback_buffer[0]
            take = min(head.size, n_samples - written)
            out[written:written + take] = head[:take]
            written += take
            if take >= head.size:
                self._loopback_buffer.popleft()
            else:
                self._loopback_buffer[0] = head[take:]
            self._loopback_buffer_samples -= take
        if self._loopback_buffer_samples < 0:
            self._loopback_buffer_samples = 0
        return out

    def _open_streams(self):
        self._mic_stream = sd.InputStream(
            device=self._mic_device_index,
            channels=self._mic_channels,
            samplerate=self._mic_sample_rate,
            blocksize=self.frames_per_read,
            dtype="float32",
        )
        if self._loopback_target["backend"] == "portaudio":
            self._loopback_stream = sd.InputStream(
                device=self._loopback_target["index"],
                channels=self._loopback_channels,
                samplerate=self._loopback_sample_rate,
                blocksize=self.frames_per_read,
                dtype="float32",
            )
        else:
            self._loopback_stream = PulseMonitorStream(
                self._loopback_target["source"],
                samplerate=self._loopback_sample_rate,
                channels=self._loopback_channels,
            )
        self._mic_stream.start()
        self._loopback_stream.start()

    def _reopen_streams(self) -> bool:
        if not self._running:
            return False
        now = time.time()
        if now - self._last_reopen_attempt < STREAM_REOPEN_COOLDOWN_SEC:
            return False
        self._last_reopen_attempt = now
        try:
            self._close_stream(self._mic_stream, "Mic")
            self._close_stream(self._loopback_stream, "Loopback")
            self._open_streams()
            self._loopback_buffer.clear()
            self._loopback_buffer_samples = 0
            self._mic_read_errors = 0
            log.warning("Capture streams were reopened after repeated mic read failures.")
            return True
        except Exception as e:
            log.exception("Failed to reopen capture streams: %s", e)
            return False

    def set_stereo_callback(self, callback):
        """Set callable(stereo_frames: np.ndarray) invoked from capture thread for each block."""
        with self._lock:
            self._stereo_callback = callback

    def set_level_callback(self, callback):
        """Set callable(rms: float) invoked from capture thread with mic RMS per block."""
        with self._lock:
            self._level_callback = callback

    def set_health_callback(self, callback):
        """Set callable(status: dict) invoked from capture thread with signal-health info."""
        with self._lock:
            self._health_callback = callback

    def _run_capture(self):
        """
        Mic is read blocking (steady stream). The monitor source is drained into a
        buffer; each block consumes exactly the monitor samples for that block's
        duration, then resamples to mic length. This keeps the system-audio timeline
        continuous so transcription of other participants is usable.
        """
        while self._running:
            if self._mic_stream is None or self._loopback_stream is None:
                break
            try:
                mic_frames, _overflowed = self._mic_stream.read(self.frames_per_read)
                if mic_frames is None or len(mic_frames) == 0:
                    time.sleep(0.005)
                    continue
                mic_mono = _to_mono(mic_frames)
                mic_rms_raw = _rms32(mic_mono)
                mic_mono, mic_stats = self._mic_leveler.process(mic_mono)
                self._mic_read_errors = 0
            except Exception as e:
                if not self._running:
                    break
                self._mic_read_errors += 1
                if self._mic_read_errors == 1 or self._mic_read_errors % MIC_ERROR_REOPEN_THRESHOLD == 0:
                    log.warning("Mic read failed (%d): %s", self._mic_read_errors, e)
                if self._mic_read_errors >= MIC_ERROR_REOPEN_THRESHOLD:
                    self._reopen_streams()
                time.sleep(min(0.05 * self._mic_read_errors, 0.5))
                continue
            n_mic = len(mic_mono)
            mic_rms = _rms32(mic_mono)

            # Drain all available monitor audio into the time-aligned buffer (non-blocking)
            try:
                while self._running:
                    available = self._loopback_stream.read_available
                    if available <= 0:
                        break
                    to_read = min(available, self.frames_per_read * 2)
                    loopback_frames, _overflowed = self._loopback_stream.read(to_read)
                    if loopback_frames is None or len(loopback_frames) == 0:
                        break
                    self._append_loopback_chunk(_to_mono(loopback_frames))
            except Exception as e:
                log.warning("Loopback read failed: %s", e)

            # Consume exactly the loopback samples that match this block's duration (time-aligned)
            n_loopback_needed = int(round(n_mic * self._loopback_sample_rate / self._mic_sample_rate))
            if n_loopback_needed <= 0:
                n_loopback_needed = n_mic
            loopback_chunk = self._consume_loopback_chunk(n_loopback_needed)

            # Resample loopback to output (mic) rate so we get exactly n_mic samples
            if len(loopback_chunk) != n_mic:
                loopback_mono = resample(loopback_chunk, n_mic).astype(np.float32)
            else:
                loopback_mono = loopback_chunk

            with self._lock:
                level_cb = self._level_callback
                health_cb = self._health_callback
            if level_cb is not None:
                try:
                    loopback_rms = _rms32(loopback_mono)
                    # Use raw mic RMS for the UI meter so AGC does not make silence look loud.
                    level_cb(max(mic_rms_raw, loopback_rms))
                except Exception as e:
                    log.debug("Level callback failed: %s", e)
            if health_cb is not None:
                try:
                    health_cb({
                        "source": "meeting_mic",
                        "rms_in": float(mic_stats.get("rms_in", mic_rms)),
                        "rms_out": float(mic_stats.get("rms_out", mic_rms)),
                        "noise_floor": float(mic_stats.get("noise_floor", 0.0)),
                        "threshold": float(mic_stats.get("threshold", 0.0)),
                        "gain_db": float(mic_stats.get("gain_db", 0.0)),
                        "clip_count": int(mic_stats.get("clip_count", 0)),
                        "active": bool(mic_stats.get("active", 0.0) >= 0.5),
                    })
                except Exception as e:
                    log.debug("Health callback failed: %s", e)

            stereo = np.column_stack([
                mic_mono * self.mic_gain,
                loopback_mono * self.loopback_gain,
            ]).astype(np.float32)
            with self._lock:
                cb = self._stereo_callback
            if cb is not None:
                try:
                    cb(stereo)
                except Exception as e:
                    log.exception("Stereo callback failed: %s", e)

    def start(self, loopback_device_index=None, mic_device_index=None):
        """Start capture. Uses default mic and the default output's monitor source unless indices are provided."""
        if self._running:
            return

        if mic_device_index is None:
            try:
                mic_device_index = sd.query_devices(kind="input")["index"]
            except Exception as e:
                raise RuntimeError("No default microphone found: %s" % e) from e
        target, err = resolve_loopback_target(loopback_device_index)
        if err or target is None:
            raise RuntimeError(err or "No system-audio monitor source found.")
        self._loopback_target = target

        try:
            mic_info = sd.query_devices(mic_device_index)
        except Exception as e:
            raise RuntimeError("Could not query audio devices: %s" % e) from e

        self._mic_device_index = int(mic_device_index)
        # Mono is enough for ASR; cap at stereo so odd ALSA devices don't open 32 channels.
        self._mic_channels = max(1, min(2, int(mic_info.get("max_input_channels") or 1)))

        # Open each device at its native default sample rate (avoids errors when rates differ)
        def _device_rate(info, fallback=48000):
            r = info.get("default_samplerate") or fallback
            r = int(r) if r else fallback
            return r if r > 0 else fallback

        if target["backend"] == "portaudio":
            try:
                loopback_info = sd.query_devices(target["index"])
            except Exception as e:
                raise RuntimeError("Could not query audio devices: %s" % e) from e
            loopback_name = loopback_info.get("name")
            self._loopback_channels = max(1, min(2, int(loopback_info.get("max_input_channels") or 1)))
            self._loopback_sample_rate = _device_rate(loopback_info)
        else:
            # Sound-server capture: parec resamples to whatever we ask for.
            loopback_name = target["source"]
            self._loopback_channels = 2
            self._loopback_sample_rate = 48000
        self._mic_sample_rate = _device_rate(mic_info)
        self.sample_rate = self._mic_sample_rate  # output rate = mic rate; loopback resampled in loop

        try:
            self._open_streams()
        except Exception as e:
            self._close_stream(self._mic_stream, "Mic")
            self._close_stream(self._loopback_stream, "Loopback")
            self._mic_stream = None
            self._loopback_stream = None
            raise RuntimeError(
                "Could not open capture streams (mic=%r %d Hz, monitor=%r %d Hz): %s"
                % (
                    mic_info.get("name"), self._mic_sample_rate,
                    loopback_name, self._loopback_sample_rate, e,
                )
            ) from e

        if self._mic_sample_rate != self._loopback_sample_rate:
            log.info(
                "Meeting capture: mic=%d Hz, monitor=%d Hz, output=%d Hz.",
                self._mic_sample_rate, self._loopback_sample_rate, self.sample_rate,
            )

        self._running = True
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()
        log.debug("In-process mic+monitor capture started.")

    def stop(self):
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.5)
            if t.is_alive():
                # Unblock a potentially pending read without closing handles cross-thread.
                self._stop_stream_only(self._mic_stream, "Mic")
                self._stop_stream_only(self._loopback_stream, "Loopback")
                t.join(timeout=2.5)
            if t.is_alive():
                log.warning("Capture thread did not stop cleanly; deferring stream close to avoid backend crash.")
            else:
                self._close_stream(self._mic_stream, "Mic")
                self._mic_stream = None
                self._close_stream(self._loopback_stream, "Loopback")
                self._loopback_stream = None
            self._thread = None
        else:
            self._close_stream(self._mic_stream, "Mic")
            self._mic_stream = None
            self._close_stream(self._loopback_stream, "Loopback")
            self._loopback_stream = None
        self._loopback_buffer.clear()
        self._loopback_buffer_samples = 0
        self._mic_read_errors = 0
        log.debug("Capture stopped.")

    def is_running(self):
        return self._running
