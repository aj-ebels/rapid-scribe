"""
Audio capture: default input and loopback workers, silence detection, meeting-mode callback.
"""
import sys
import tempfile
import queue
from pathlib import Path

import numpy as np

from .diagnostic import write as diag
from .audio_leveler import AudioLeveler, AudioLevelerConfig
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_SEC = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
# In-process Meeting (mic+loopback): capture at 48k, then resample to 16k for ASR
CAPTURE_SAMPLE_RATE_MEETING = 48000
FRAMES_PER_READ_MEETING = 4096
MIXER_GAIN_MEETING = 0.7
CHUNK_PATH = Path(tempfile.gettempdir()) / "meetings_chunk.wav"
SILENCE_RMS_THRESHOLD = 0.005


def _rms32(chunk: np.ndarray) -> float:
    """Fast RMS for float32 buffers."""
    if chunk is None or chunk.size == 0:
        return 0.0
    buf = np.asarray(chunk, dtype=np.float32)
    return float(np.sqrt(np.mean(buf * buf)))


def _is_silent(chunk: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """True if chunk has very low energy (silence or negligible noise)."""
    rms = _rms32(chunk)
    return rms < threshold


def _build_leveler_config(leveler_settings: dict | None) -> AudioLevelerConfig:
    leveler_settings = leveler_settings or {}
    cfg = AudioLevelerConfig()
    cfg.enabled = bool(leveler_settings.get("audio_auto_level", True))
    cfg.input_sensitivity = max(0.5, min(3.0, float(leveler_settings.get("input_sensitivity", 0.8))))
    cfg.target_rms = max(0.01, min(0.2, float(leveler_settings.get("agc_target_rms", 0.035))))
    cfg.max_gain_db = max(6.0, min(30.0, float(leveler_settings.get("agc_max_boost_db", 9.0))))
    cfg.expander_enabled = bool(leveler_settings.get("audio_expander_enabled", True))
    return cfg


def capture_worker(device_index, chunk_queue, stop_event, leveler_settings=None, health_queue=None):
    """Record chunks; save to temp file only when not silent, put path in queue."""
    leveler = AudioLeveler(_build_leveler_config(leveler_settings))
    while not stop_event.is_set():
        try:
            chunk = sd.rec(
                CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                device=device_index,
                blocking=True,
            )
            if stop_event.is_set():
                break
            mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
            mono, stats = leveler.process(mono)
            if health_queue is not None:
                try:
                    health_queue.put_nowait({
                        "source": "default_mic",
                        "rms_in": float(stats.get("rms_in", 0.0)),
                        "rms_out": float(stats.get("rms_out", 0.0)),
                        "noise_floor": float(stats.get("noise_floor", 0.0)),
                        "threshold": float(stats.get("threshold", 0.0)),
                        "gain_db": float(stats.get("gain_db", 0.0)),
                        "clip_count": int(stats.get("clip_count", 0)),
                        "active": bool(stats.get("active", 0.0) >= 0.5),
                    })
                except queue.Full:
                    pass
            if _is_silent(mono, threshold=max(SILENCE_RMS_THRESHOLD, float(stats.get("threshold", SILENCE_RMS_THRESHOLD)) * 0.75)):
                continue
            rms = _rms32(mono)
            chunk_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, chunk_int16)
            try:
                chunk_queue.put((str(CHUNK_PATH), rms), timeout=1.0)
                diag(
                    "chunk_queued",
                    path=str(CHUNK_PATH),
                    worker="default",
                    rms_in=round(float(stats.get("rms_in", 0.0)), 6),
                    rms_out=round(float(stats.get("rms_out", rms)), 6),
                    gain_db=round(float(stats.get("gain_db", 0.0)), 2),
                    noise_floor=round(float(stats.get("noise_floor", 0.0)), 6),
                    clip_count=int(stats.get("clip_count", 0)),
                )
            except queue.Full:
                pass
        except Exception as e:
            diag("capture_error", worker="default", error=str(e))
            if not stop_event.is_set():
                try:
                    chunk_queue.put_nowait(("error", str(e)))
                except queue.Full:
                    pass
            break


def capture_worker_loopback(loopback_device_index, chunk_queue, stop_event, level_queue=None):
    """Record from WASAPI loopback only (PyAudioWPatch). Puts chunk paths into chunk_queue.
    If level_queue is provided, pushes RMS (float) per read block for a level indicator."""
    if sys.platform != "win32":
        try:
            chunk_queue.put_nowait(("error", "Loopback is only supported on Windows with pyaudiowpatch."))
        except queue.Full:
            pass
        return
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        try:
            chunk_queue.put_nowait(("error", "pyaudiowpatch not installed. pip install pyaudiowpatch"))
        except queue.Full:
            pass
        return
    try:
        with pyaudio.PyAudio() as p:
            if loopback_device_index is not None:
                dev = p.get_device_info_by_index(loopback_device_index)
            else:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                if not default_speakers.get("isLoopbackDevice"):
                    for loopback in p.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            default_speakers = loopback
                            break
                dev = default_speakers
            rate = int(dev["defaultSampleRate"])
            ch = int(dev["maxInputChannels"])
            chunk_frames_loopback = int(rate * CHUNK_DURATION_SEC) * ch
            while not stop_event.is_set():
                buf = []
                with p.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    frames_per_buffer=1024,
                    input=True,
                    input_device_index=dev["index"],
                ) as stream:
                    while len(buf) * 1024 < chunk_frames_loopback and not stop_event.is_set():
                        try:
                            data = stream.read(1024, exception_on_overflow=False)
                            block = np.frombuffer(data, dtype=np.int16)
                            buf.append(block)
                            if level_queue is not None:
                                mono = block.reshape(-1, ch).astype(np.float32) / 32768.0
                                if mono.size:
                                    rms = _rms32(mono)
                                    try:
                                        level_queue.put_nowait(rms)
                                    except queue.Full:
                                        pass
                        except Exception:
                            break
                if stop_event.is_set():
                    break
                if not buf:
                    continue
                raw = np.concatenate(buf)[:chunk_frames_loopback]
                raw = raw.reshape(-1, ch).astype(np.float64) / 32768.0
                mono = np.mean(raw, axis=1).astype(np.float32)
                if rate != SAMPLE_RATE:
                    num_out = int(round(len(mono) * SAMPLE_RATE / rate))
                    mono = resample(mono, num_out).astype(np.float32)
                mono = mono[:CHUNK_SAMPLES]
                if len(mono) < CHUNK_SAMPLES:
                    mono = np.pad(mono, (0, CHUNK_SAMPLES - len(mono)))
                if _is_silent(mono):
                    continue
                rms = _rms32(mono)
                chunk_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
                wavfile.write(str(CHUNK_PATH), SAMPLE_RATE, chunk_int16)
                try:
                    chunk_queue.put((str(CHUNK_PATH), rms), timeout=1.0)
                    diag("chunk_queued", path=str(CHUNK_PATH), worker="loopback")
                except queue.Full:
                    pass
    except Exception as e:
        diag("capture_error", worker="loopback", error=str(e))
        if not stop_event.is_set():
            try:
                chunk_queue.put_nowait(("error", str(e)))
            except queue.Full:
                pass


def meeting_chunk_ready(app, wav_path: str, rms: float = None):
    """Callback from ChunkRecorder (in-process Meeting mode): queue WAV path and RMS for transcription."""
    try:
        from diagnostic import write as diag
        diag("chunk_queued", path=wav_path, worker="meeting")
    except ImportError:
        pass
    try:
        app.chunk_queue.put_nowait((wav_path, rms) if rms is not None else wav_path)
    except queue.Full:
        pass
