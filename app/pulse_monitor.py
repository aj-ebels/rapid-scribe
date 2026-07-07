"""
PulseAudio/PipeWire monitor-source capture without PortAudio.

Many distro builds of PortAudio only include the ALSA host API, so PulseAudio/
PipeWire 'Monitor of …' sources never show up in sounddevice's device list even
though the sound server exposes them. This module talks to the sound server
directly: `pactl` for enumeration and a `parec` subprocess for capture (both
ship in pulseaudio-utils; PipeWire serves them via pipewire-pulse).

PulseMonitorStream mirrors the small slice of the sounddevice.InputStream API
the capture code uses: start/stop/close, read(n) -> (frames, overflowed),
read_available, and context-manager protocol.
"""

import shutil
import subprocess
import threading
from collections import deque

import numpy as np

PACTL_TIMEOUT_SEC = 3.0
PAREC_MISSING_MSG = (
    "parec not found. Install pulseaudio-utils (Ubuntu/Debian: "
    "sudo apt-get install pulseaudio-utils) for system-audio capture."
)
PACTL_MISSING_MSG = (
    "pactl not found. Install pulseaudio-utils (Ubuntu/Debian: "
    "sudo apt-get install pulseaudio-utils)."
)


def pulse_tools_available() -> bool:
    """True if the PulseAudio CLI tools needed for the fallback are on PATH."""
    return shutil.which("pactl") is not None and shutil.which("parec") is not None


def _run_pactl(args):
    """Run pactl and return (stdout, error_message)."""
    try:
        proc = subprocess.run(
            ["pactl", *args],
            capture_output=True,
            text=True,
            timeout=PACTL_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        return None, PACTL_MISSING_MSG
    except Exception as e:
        return None, str(e)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"pactl {' '.join(args)} failed"
        return None, err
    return proc.stdout, None


def list_pulse_monitor_sources():
    """
    Monitor sources known to the sound server, regardless of PortAudio.
    Returns ([{"pactl_index": int, "name": str}], error_message).
    """
    out, err = _run_pactl(["list", "short", "sources"])
    if err:
        return [], err
    monitors = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()
        if ".monitor" in name.lower():
            monitors.append({"pactl_index": idx, "name": name})
    return monitors, None


def get_default_pulse_monitor():
    """
    Monitor source of the default sink (what the speakers are playing).
    Returns (source_name, error_message).
    """
    monitors, err = list_pulse_monitor_sources()
    if err:
        return None, err
    if not monitors:
        return None, "The sound server reports no monitor sources."
    out, err = _run_pactl(["info"])
    if not err and out:
        for line in out.splitlines():
            if line.startswith("Default Sink:"):
                default_monitor = line.split(":", 1)[1].strip() + ".monitor"
                for m in monitors:
                    if m["name"] == default_monitor:
                        return m["name"], None
                break
    return monitors[0]["name"], None


class PulseMonitorStream:
    """
    Capture a PulseAudio/PipeWire source via a parec subprocess.
    The sound server resamples to the requested rate/channels for us.
    """

    def __init__(self, source_name: str, samplerate: int = 48000, channels: int = 2):
        self.source_name = source_name
        self.samplerate = int(samplerate)
        self.channels = max(1, int(channels))
        self._proc = None
        self._reader = None
        self._buffer = deque()
        self._buffered_frames = 0
        self._cond = threading.Condition()
        self._eof = False
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        if self._proc is not None:
            return
        cmd = [
            "parec",
            "--device", self.source_name,
            "--format=float32le",
            "--rate", str(self.samplerate),
            "--channels", str(self.channels),
            "--latency-msec=50",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError:
            raise RuntimeError(PAREC_MISSING_MSG) from None
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self):
        bytes_per_frame = 4 * self.channels
        read_size = 1024 * bytes_per_frame
        stdout = self._proc.stdout
        leftover = b""
        while not self._closed:
            data = stdout.read(read_size)
            if not data:
                break
            data = leftover + data
            usable = (len(data) // bytes_per_frame) * bytes_per_frame
            leftover = data[usable:]
            if usable == 0:
                continue
            frames = np.frombuffer(data[:usable], dtype=np.float32).reshape(-1, self.channels)
            with self._cond:
                self._buffer.append(frames)
                self._buffered_frames += len(frames)
                self._cond.notify_all()
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def stop(self):
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def close(self):
        self._closed = True
        self.stop()
        proc = self._proc
        if proc is not None:
            try:
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        reader = self._reader
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- reading -------------------------------------------------------------

    @property
    def read_available(self) -> int:
        with self._cond:
            return self._buffered_frames

    def read(self, n: int):
        """Blocking read of n frames; returns (frames (n, channels) float32, overflowed=False).

        Raises RuntimeError if the parec process ends before n frames are available.
        """
        n = int(n)
        with self._cond:
            while self._buffered_frames < n and not self._eof:
                self._cond.wait(timeout=0.2)
            if self._buffered_frames < n:
                raise RuntimeError(
                    "System-audio capture (parec, source %r) ended unexpectedly." % self.source_name
                )
            out = np.empty((n, self.channels), dtype=np.float32)
            written = 0
            while written < n:
                head = self._buffer[0]
                take = min(len(head), n - written)
                out[written:written + take] = head[:take]
                written += take
                if take >= len(head):
                    self._buffer.popleft()
                else:
                    self._buffer[0] = head[take:]
                self._buffered_frames -= take
            return out, False
