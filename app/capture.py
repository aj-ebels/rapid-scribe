"""
Audio capture: default input and loopback workers, silence detection, meeting-mode callback.
"""
import os
import sys
import tempfile
import queue
import uuid
from pathlib import Path

import numpy as np

from .diagnostic import write as diag
from .audio_leveler import AudioLeveler, AudioLevelerConfig
from . import dev_config
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
    # dev_config.LEVELING_ENABLED is the authoritative on/off; settings values tune parameters only.
    cfg.enabled = dev_config.LEVELING_ENABLED
    cfg.input_sensitivity = max(0.5, min(3.0, float(leveler_settings.get("input_sensitivity", 0.8))))
    cfg.target_rms = max(0.01, min(0.2, float(leveler_settings.get("agc_target_rms", 0.035))))
    cfg.max_gain_db = max(6.0, min(30.0, float(leveler_settings.get("agc_max_boost_db", 9.0))))
    cfg.expander_enabled = bool(leveler_settings.get("audio_expander_enabled", True))
    return cfg


def _take_front_mono_samples(buf: list, n_take: int) -> tuple[np.ndarray, list]:
    """Pop the first n_take float32 mono samples from a list of 1-D blocks; return (taken, buf)."""
    if n_take <= 0:
        return np.array([], dtype=np.float32), buf
    parts = []
    remaining = n_take
    while remaining > 0 and buf:
        b = buf[0]
        if len(b) <= remaining:
            parts.append(b)
            remaining -= len(b)
            buf.pop(0)
        else:
            parts.append(b[:remaining])
            buf[0] = b[remaining:]
            remaining = 0
    if not parts:
        return np.array([], dtype=np.float32), buf
    return np.concatenate(parts).astype(np.float32), buf


def capture_worker(device_index, chunk_queue, stop_event, leveler_settings=None, health_queue=None):
    """Default microphone capture: routes to fixed or VAD worker based on dev_config.CHUNKING_MODE.

    Pathway is set exclusively in app/dev_config.py — not from user-facing settings.
    """
    if dev_config.CHUNKING_MODE == "vad":
        capture_worker_vad(device_index, chunk_queue, stop_event, leveler_settings, health_queue)
    else:
        capture_worker_fixed(device_index, chunk_queue, stop_event, leveler_settings, health_queue)


def capture_worker_fixed(device_index, chunk_queue, stop_event, leveler_settings=None, health_queue=None):
    """16 kHz mic capture: fixed-duration chunks.  Duration from dev_config.CHUNK_DURATION_SEC."""
    duration_sec = max(3.0, min(30.0, dev_config.CHUNK_DURATION_SEC))
    chunk_samples = int(SAMPLE_RATE * duration_sec)

    block_queue: queue.Queue = queue.Queue(maxsize=200)
    BLOCK_SIZE = 512
    temp_dir = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "MeetingsChunks")
    os.makedirs(temp_dir, exist_ok=True)

    def _audio_callback(indata, frames, time_info, status):
        try:
            block_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    buf: list = []
    buf_frames = 0

    def _emit_chunk(mono: np.ndarray):
        if mono.size < chunk_samples:
            mono = np.pad(mono.astype(np.float32), (0, chunk_samples - mono.size))
        else:
            mono = mono[:chunk_samples].astype(np.float32)
        if _is_silent(mono):
            return
        rms = _rms32(mono)
        audio_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        wav_path = os.path.join(temp_dir, f"chunk_{uuid.uuid4().hex}.wav")
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
        diag(
            "chunk_queued",
            path=wav_path,
            worker="default_fixed",
            rms_out=round(rms, 6),
        )
        try:
            chunk_queue.put((wav_path, rms), timeout=1.0)
        except queue.Full:
            pass

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype=np.float32,
            callback=_audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    block = block_queue.get(timeout=0.3)
                except queue.Empty:
                    continue
                buf.append(block)
                buf_frames += len(block)
                while buf_frames >= chunk_samples:
                    mono, buf = _take_front_mono_samples(buf, chunk_samples)
                    buf_frames = sum(len(x) for x in buf)
                    _emit_chunk(mono)

            while not block_queue.empty():
                try:
                    block = block_queue.get_nowait()
                    buf.append(block)
                    buf_frames += len(block)
                except queue.Empty:
                    break
            while buf_frames >= chunk_samples:
                mono, buf = _take_front_mono_samples(buf, chunk_samples)
                buf_frames = sum(len(x) for x in buf)
                _emit_chunk(mono)
            if buf_frames > 0:
                mono, buf = _take_front_mono_samples(buf, buf_frames)
                _emit_chunk(mono)

    except Exception as e:
        diag("capture_error", worker="default_fixed", error=str(e))
        if not stop_event.is_set():
            try:
                chunk_queue.put_nowait(("error", str(e)))
            except queue.Full:
                pass


def capture_worker_vad(device_index, chunk_queue, stop_event, leveler_settings=None, health_queue=None):
    """16 kHz mic capture: VAD silence-based chunking.  Leveling from dev_config.LEVELING_ENABLED."""
    leveler = AudioLeveler(_build_leveler_config(leveler_settings))
    block_queue: queue.Queue = queue.Queue(maxsize=200)

    # VAD / chunking parameters — all driven by dev_config
    BLOCK_SIZE = 512
    MIN_CHUNK_FRAMES     = int(SAMPLE_RATE * max(0.5, dev_config.VAD_MIN_CHUNK_SEC))
    MAX_CHUNK_FRAMES     = int(SAMPLE_RATE * max(1.0, dev_config.VAD_MAX_CHUNK_SEC))
    SILENCE_TRIGGER_FRAMES = int(SAMPLE_RATE * max(0.1, dev_config.VAD_SILENCE_SEC))
    HANGOVER_BLOCKS = 5

    temp_dir = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "MeetingsChunks")
    os.makedirs(temp_dir, exist_ok=True)

    def _audio_callback(indata, frames, time_info, status):
        """Runs on the audio thread — only enqueue, never block."""
        try:
            block_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop block rather than stalling the audio thread

    def _flush(buf, buf_frames):
        """Levelize, gate, write WAV, and enqueue the accumulated buffer."""
        if not buf or buf_frames == 0:
            return
        audio = np.concatenate(buf).astype(np.float32)
        audio, stats = leveler.process(audio)
        rms = _rms32(audio)
        if rms < SILENCE_RMS_THRESHOLD * 0.5:
            return  # skip effectively silent chunks
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wav_path = os.path.join(temp_dir, f"chunk_{uuid.uuid4().hex}.wav")
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
        diag(
            "chunk_queued",
            path=wav_path,
            worker="default",
            rms_out=round(rms, 6),
            gain_db=round(float(stats.get("gain_db", 0.0)), 2),
            noise_floor=round(float(stats.get("noise_floor", 0.0)), 6),
        )
        try:
            chunk_queue.put((wav_path, rms), timeout=1.0)
        except queue.Full:
            pass

    buf = []
    buf_frames = 0
    consecutive_silent = 0
    noise_floor = SILENCE_RMS_THRESHOLD
    hangover_left = 0

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype=np.float32,
            callback=_audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    block = block_queue.get(timeout=0.3)
                except queue.Empty:
                    continue

                block_rms = _rms32(block)

                # Adaptive noise floor (exponential moving average)
                learn_upper = max(SILENCE_RMS_THRESHOLD * 5.0, noise_floor * 1.7)
                if block_rms <= learn_upper:
                    noise_floor = 0.96 * noise_floor + 0.04 * block_rms
                adaptive_threshold = max(SILENCE_RMS_THRESHOLD, noise_floor * 2.5)

                # Hangover: hold the speech gate open for HANGOVER_BLOCKS after last voiced block
                is_silent = block_rms < adaptive_threshold
                if is_silent:
                    if hangover_left > 0:
                        hangover_left -= 1
                        is_silent = False
                else:
                    hangover_left = HANGOVER_BLOCKS
                    consecutive_silent = 0

                buf.append(block)
                buf_frames += len(block)
                if is_silent:
                    consecutive_silent += len(block)

                # Emit conditions
                if buf_frames >= MAX_CHUNK_FRAMES:
                    _flush(buf, buf_frames)
                    buf, buf_frames, consecutive_silent = [], 0, 0
                elif buf_frames >= MIN_CHUNK_FRAMES and consecutive_silent >= SILENCE_TRIGGER_FRAMES:
                    _flush(buf, buf_frames)
                    buf, buf_frames, consecutive_silent = [], 0, 0

            # Drain remaining audio when recording stops
            while not block_queue.empty():
                try:
                    block = block_queue.get_nowait()
                    buf.append(block)
                    buf_frames += len(block)
                except queue.Empty:
                    break
            _flush(buf, buf_frames)

    except Exception as e:
        diag("capture_error", worker="default", error=str(e))
        if not stop_event.is_set():
            try:
                chunk_queue.put_nowait(("error", str(e)))
            except queue.Full:
                pass


def capture_worker_loopback(loopback_device_index, chunk_queue, stop_event, level_queue=None, settings=None):
    """WASAPI loopback (PyAudioWPatch): resample to 16 kHz mono.

    Pathway is identical to Default input — fixed windows or VAD — both controlled by
    dev_config.CHUNKING_MODE and dev_config.CHUNK_DURATION_SEC.  Loopback is never leveled.
    """
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

    # Fixed-window constants — only used when CHUNKING_MODE == "fixed".
    # CHUNK_DURATION_SEC has no effect on the VAD path.
    if dev_config.CHUNKING_MODE != "vad":
        _fixed_duration_sec = max(3.0, min(30.0, dev_config.CHUNK_DURATION_SEC))
        chunk_samples_16k = int(SAMPLE_RATE * _fixed_duration_sec)
    else:
        chunk_samples_16k = 0  # unused in VAD path

    # VAD tuning — only used when CHUNKING_MODE == "vad".
    VAD_BLOCK_SIZE         = 512
    VAD_MIN_FRAMES         = int(SAMPLE_RATE * max(0.5, dev_config.VAD_MIN_CHUNK_SEC))
    VAD_MAX_FRAMES         = int(SAMPLE_RATE * max(1.0, dev_config.VAD_MAX_CHUNK_SEC))
    VAD_SILENCE_FRAMES     = int(SAMPLE_RATE * max(0.1, dev_config.VAD_SILENCE_SEC))
    VAD_HANGOVER_BLOCKS    = 5

    temp_dir = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "MeetingsChunks")
    os.makedirs(temp_dir, exist_ok=True)

    def _emit(mono: np.ndarray, worker_tag: str):
        mono = np.clip(mono.astype(np.float32), -1.0, 1.0)
        if _is_silent(mono):
            return
        rms = _rms32(mono)
        if rms < SILENCE_RMS_THRESHOLD * 0.5:
            return
        chunk_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        wav_path = os.path.join(temp_dir, f"lb_{uuid.uuid4().hex}.wav")
        wavfile.write(wav_path, SAMPLE_RATE, chunk_int16)
        try:
            chunk_queue.put((wav_path, rms), timeout=1.0)
            diag("chunk_queued", path=wav_path, worker=worker_tag, rms_out=round(rms, 6))
        except queue.Full:
            pass

    # Fixed-window state
    fix_buf: list = []
    fix_frames = 0

    def _feed_fixed(y: np.ndarray):
        nonlocal fix_buf, fix_frames
        fix_buf.append(y)
        fix_frames += len(y)
        while fix_frames >= chunk_samples_16k:
            mono, fix_buf = _take_front_mono_samples(fix_buf, chunk_samples_16k)
            fix_frames = sum(len(x) for x in fix_buf)
            _emit(mono, "loopback_fixed")

    # VAD state (inline — mirrors capture_worker_vad)
    vad_buf: list           = []
    vad_buf_frames: int     = 0
    vad_consecutive_silent  = 0
    vad_noise_floor         = SILENCE_RMS_THRESHOLD
    vad_hangover_left       = 0

    def _feed_vad(y: np.ndarray):
        nonlocal vad_buf, vad_buf_frames, vad_consecutive_silent, vad_noise_floor, vad_hangover_left
        for i in range(0, len(y), VAD_BLOCK_SIZE):
            block = y[i : i + VAD_BLOCK_SIZE]
            block_rms = _rms32(block)
            learn_upper = max(SILENCE_RMS_THRESHOLD * 5.0, vad_noise_floor * 1.7)
            if block_rms <= learn_upper:
                vad_noise_floor = 0.96 * vad_noise_floor + 0.04 * block_rms
            adaptive_threshold = max(SILENCE_RMS_THRESHOLD, vad_noise_floor * 2.5)
            is_silent = block_rms < adaptive_threshold
            if is_silent:
                if vad_hangover_left > 0:
                    vad_hangover_left -= 1
                    is_silent = False
            else:
                vad_hangover_left = VAD_HANGOVER_BLOCKS
                vad_consecutive_silent = 0
            vad_buf.append(block)
            vad_buf_frames += len(block)
            if is_silent:
                vad_consecutive_silent += len(block)
            if vad_buf_frames >= VAD_MAX_FRAMES:
                audio = np.concatenate(vad_buf).astype(np.float32)
                vad_buf, vad_buf_frames, vad_consecutive_silent = [], 0, 0
                _emit(audio, "loopback_vad")
            elif vad_buf_frames >= VAD_MIN_FRAMES and vad_consecutive_silent >= VAD_SILENCE_FRAMES:
                audio = np.concatenate(vad_buf).astype(np.float32)
                vad_buf, vad_buf_frames, vad_consecutive_silent = [], 0, 0
                _emit(audio, "loopback_vad")

    def _feed_16k(y: np.ndarray):
        if dev_config.CHUNKING_MODE == "vad":
            _feed_vad(y)
        else:
            _feed_fixed(y)

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
            native_batch = max(int(rate * 0.05), 512)
            pending = np.array([], dtype=np.float32)

            with p.open(
                format=pyaudio.paInt16,
                channels=ch,
                rate=rate,
                frames_per_buffer=1024,
                input=True,
                input_device_index=dev["index"],
            ) as stream:
                while not stop_event.is_set():
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                    except Exception:
                        break
                    block = np.frombuffer(data, dtype=np.int16)
                    if block.size == 0:
                        continue
                    raw = block.reshape(-1, ch).astype(np.float64) / 32768.0
                    mono_native = np.mean(raw, axis=1).astype(np.float32)
                    pending = np.concatenate([pending, mono_native])
                    if level_queue is not None and mono_native.size:
                        try:
                            level_queue.put_nowait(_rms32(mono_native))
                        except queue.Full:
                            pass
                    while len(pending) >= native_batch and not stop_event.is_set():
                        piece = pending[:native_batch]
                        pending = pending[native_batch:]
                        n_out = int(round(len(piece) * SAMPLE_RATE / rate))
                        _feed_16k(resample(piece, n_out).astype(np.float32))

                if len(pending) > 0:
                    n_out = int(round(len(pending) * SAMPLE_RATE / rate))
                    _feed_16k(resample(pending, n_out).astype(np.float32))

                # Flush remainder
                if dev_config.CHUNKING_MODE == "vad":
                    if vad_buf_frames > 0:
                        _emit(np.concatenate(vad_buf).astype(np.float32), "loopback_vad")
                else:
                    while fix_frames >= chunk_samples_16k:
                        mono, fix_buf = _take_front_mono_samples(fix_buf, chunk_samples_16k)
                        fix_frames = sum(len(x) for x in fix_buf)
                        _emit(mono, "loopback_fixed")
                    if fix_frames > 0:
                        mono, fix_buf = _take_front_mono_samples(fix_buf, fix_frames)
                        _emit(mono, "loopback_fixed")

    except Exception as e:
        diag("capture_error", worker="loopback", error=str(e))
        if not stop_event.is_set():
            try:
                chunk_queue.put_nowait(("error", str(e)))
            except queue.Full:
                pass


def meeting_chunk_ready(chunk_queue, wav_path: str, rms: float = None):
    """Callback from ChunkRecorder (in-process Meeting mode): queue WAV path and RMS for transcription."""
    diag("chunk_queued", path=wav_path, worker="meeting")
    try:
        chunk_queue.put_nowait((wav_path, rms) if rms is not None else wav_path)
    except queue.Full:
        pass
