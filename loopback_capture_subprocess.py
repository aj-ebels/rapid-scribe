#!/usr/bin/env python3
"""
Loopback capture subprocess for Meeting mode (dual-subprocess).
Uses ONLY pyaudiowpatch (no sounddevice). Writes raw float32 mono 16 kHz PCM to stdout.
Run by main.py so the main process never opens any audio device.
"""
import sys

# Only pyaudiowpatch + numpy - no sounddevice in this process
import numpy as np

def main():
    if sys.platform != "win32":
        sys.stderr.write("Loopback is Windows-only.\n")
        sys.exit(1)
    if len(sys.argv) < 2:
        # device_index can be "default" or integer
        sys.stderr.write("Usage: loopback_capture_subprocess.py <device_index|default> [sample_rate] [chunk_samples]\n")
        sys.exit(1)
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        sys.stderr.write("pyaudiowpatch not installed.\n")
        sys.exit(1)

    device_arg = sys.argv[1]
    device_index = None if device_arg.lower() == "default" else int(device_arg)
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
    chunk_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
    chunk_duration_sec = chunk_samples / sample_rate

    out = sys.stdout.buffer
    try:
        with pyaudio.PyAudio() as p:
            if device_index is not None:
                dev = p.get_device_info_by_index(device_index)
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
            chunk_frames = int(rate * chunk_duration_sec) * ch
            while True:
                buf = []
                with p.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    frames_per_buffer=1024,
                    input=True,
                    input_device_index=dev["index"],
                ) as stream:
                    while len(buf) * 1024 < chunk_frames:
                        try:
                            data = stream.read(1024, exception_on_overflow=False)
                            buf.append(np.frombuffer(data, dtype=np.int16))
                        except Exception:
                            break
                if not buf:
                    continue
                raw = np.concatenate(buf)[:chunk_frames]
                raw = raw.reshape(-1, ch).astype(np.float64) / 32768.0
                mono = np.mean(raw, axis=1).astype(np.float32)
                if rate != sample_rate:
                    num_out = int(round(len(mono) * sample_rate / rate))
                    mono = np.array(np.interp(
                        np.linspace(0, len(mono) - 1, num_out),
                        np.arange(len(mono)),
                        mono
                    ), dtype=np.float32)
                mono = mono[:chunk_samples]
                if len(mono) < chunk_samples:
                    mono = np.pad(mono, (0, chunk_samples - len(mono)))
                out.write(mono.tobytes())
                out.flush()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
    except Exception as e:
        sys.stderr.write(f"loopback_capture error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
