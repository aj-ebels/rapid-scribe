#!/usr/bin/env python3
"""
Microphone capture subprocess for Meeting mode.
Uses ONLY sounddevice (no pyaudiowpatch). Writes raw float32 PCM to stdout.
Run by main.py so mic and loopback use completely separate processes and no shared audio state.
"""
import sys

# Only sounddevice + numpy - no pyaudiowpatch in this process
import numpy as np
import sounddevice as sd

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: mic_capture_subprocess.py <device_index> [sample_rate] [chunk_samples]\n")
        sys.exit(1)
    device_index = int(sys.argv[1])
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
    chunk_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 80000  # 5 sec at 16 kHz

    out = sys.stdout.buffer
    try:
        while True:
            chunk = sd.rec(
                chunk_samples,
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                device=device_index,
                blocking=True,
            )
            if chunk is None or chunk.size == 0:
                continue
            flat = np.asarray(chunk, dtype=np.float32).flatten()
            out.write(flat.tobytes())
            out.flush()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
    except Exception as e:
        sys.stderr.write(f"mic_capture error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
