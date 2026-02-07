# Research: Capturing Microphone + Loopback Simultaneously on Windows

## Problem

In "Meeting mode" we need both:
- **Microphone** — what you're saying
- **Loopback** (WASAPI) — what you're hearing (system/call audio)

On some Windows setups, when both streams are opened in the same process (e.g. both via PyAudioWPatch/PortAudio), the **mic stream reports no signal (RMS ~0)** once loopback is active, even though the mic works when used alone.

---

## What the APIs Allow

### WASAPI (Microsoft)

- **Two separate `IAudioClient` instances** in the same process are supported:
  1. **Mic:** capture stream with data flow `eCapture` on a capture endpoint.
  2. **Loopback:** capture stream with data flow `eRender` and flag `AUDCLNT_STREAMFLAGS_LOOPBACK` on a render endpoint.

- **Shared mode** is required for loopback; exclusive mode can block or conflict.
- Official docs: [Loopback Recording](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording).

So in theory, one process can open mic + loopback with two WASAPI clients. In practice, when both are opened through **the same library** (e.g. PortAudio/PyAudioWPatch), some drivers or the library’s use of the API can cause the mic to go silent.

---

## Root Cause Hypotheses

1. **Single PortAudio host** — Using one PyAudio instance (and one PortAudio DLL) for both streams may lead to internal state or device contention (e.g. default device or shared resources).
2. **Driver behavior** — Some Realtek/other drivers are known to have quirks with multiple active streams (see PortAudio issue #935: loopback not progressing without playback).
3. **Order / timing** — Opening loopback first vs mic first can change which stream “wins” on some systems.

---

## Recommended Solutions (Best First)

### 1. **Split libraries: sounddevice (mic) + PyAudioWPatch (loopback only)** — **Recommended**

- **Mic:** capture with **sounddevice** (standard PortAudio; on Windows often uses MME or default host API, not necessarily WASAPI).
- **Loopback:** capture with **PyAudioWPatch** only for the loopback device (WASAPI loopback).

**Why it works:** Two different runtimes:
- `sounddevice` uses the standard PortAudio build (no WASAPI loopback).
- `pyaudiowpatch` uses a patched PortAudio with WASAPI loopback.

So the mic is not going through the same WASAPI/PortAudio path as the loopback, which avoids the “second stream kills the first” behavior on affected machines.

**Implementation:** In meeting mode, start two workers: one that uses `sd.rec()` (or `sd.InputStream`) for the selected mic device, and one that uses `pyaudiowpatch` to open only the default (or selected) loopback device. Mix the two streams (e.g. 50/50 or configurable) and push mixed chunks to the existing transcription pipeline.

---

### 2. **Subprocess for mic**

- **Main process:** open only **loopback** with PyAudioWPatch; read mixed audio from a pipe.
- **Child process:** run a small script that uses **sounddevice** to capture the mic and write raw PCM to stdout (or a named pipe).

Main process reads mic PCM from the subprocess and mixes with loopback, then sends to transcription.

**Pros:** Complete isolation between mic and loopback stacks; no shared PortAudio.  
**Cons:** More complex (process lifecycle, pipe buffering, sample alignment), and slight latency/sync concerns.

---

### 3. **Single virtual device (VoiceMeeter / VB-Audio)**

- User installs **VoiceMeeter** (or similar) and creates one virtual input that mixes:
  - hardware mic, and  
  - system/call audio (e.g. WDM loopback or similar).

- In the app we capture **one** device (the virtual input) with sounddevice or PyAudioWPatch.

**Pros:** No dual-stream logic; works everywhere the virtual driver works.  
**Cons:** Requires user to install and configure external software; not a pure in-app solution.

---

### 4. **Explicit shared mode and ordering (single library)**

If we keep using **only** PyAudioWPatch for both streams:

- Ensure both streams use **shared mode** (no exclusive). PyAudioWPatch may not expose this; worth checking.
- Try **open order**: e.g. open **loopback first**, then **mic** (or the reverse), and keep that order consistent.
- Use **two separate `PyAudio()` instances** (if the library allows) so each stream has its own context.

This is the least invasive change but may not fix driver-specific issues; still worth trying if we want to avoid adding sounddevice for mic.

---

### 5. **Native WASAPI (ctypes / C extension)**

- Bypass PortAudio and call WASAPI from Python (ctypes or a small C extension).
- Create two `IAudioClient` instances explicitly (shared mode): one for mic, one for loopback.

**Pros:** Full control; matches Microsoft’s recommended “two IAudioClient” approach.  
**Cons:** More code and Windows-only; maintenance burden.

---

## Recommendation (updated)

When the above still leave the mic silent, the app offers two further options:

1. **Dual subprocess (Meeting mode)**  
   The **main process opens no audio device**. It spawns:
   - **Mic subprocess** (`mic_capture_subprocess.py` or `--mic-capture`): only sounddevice, writes float32 PCM to stdout.
   - **Loopback subprocess** (`loopback_capture_subprocess.py` or `--loopback-capture`): only pyaudiowpatch, writes float32 PCM to stdout.  
   Main process only reads both pipes and mixes. No process has two capture streams open.

2. **Meeting (FFmpeg)**  
   Use **FFmpeg** to capture both sources (DirectShow: mic + Stereo Mix or virtual device), mix with `amix`, output s16le 16 kHz mono to a pipe. The app reads the pipe and transcribes. No Python audio APIs are used for capture. Requires FFmpeg on PATH and correct dshow device names (`ffmpeg -list_devices true -f dshow -i dummy`).

3. **VoiceMeeter workaround**  
   Install VoiceMeeter, create a virtual output that mixes mic + system audio, and select that as the app’s input (Default input or Meeting microphone). Single capture device, no dual-stream logic.

- _(Obsolete)_ **Meeting mode** = sounddevice for **mic** (selected device index from settings) + PyAudioWPatch for **loopback only** (default or selected output’s loopback).
- Run two capture workers; mix fixed-size chunks (same sample rate, e.g. 16 kHz mono) and feed the mixed stream into the existing chunk → transcription queue.
- Keep **Default** and **Loopback-only** modes as they are (single stream: sounddevice default input, or PyAudioWPatch loopback only).

This gives the best chance of reliable mic + loopback on the problematic setups without requiring external software or a subprocess.

---

## References

- [Loopback Recording (Microsoft)](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording)
- [PyAudioWPatch loopback example](https://github.com/s0d3s/PyAudioWPatch/blob/master/examples/pawp_record_wasapi_loopback.py)
- [PortAudio WASAPI loopback issue #935](https://github.com/PortAudio/portaudio/issues/935)
- sounddevice: `WasapiSettings(exclusive=False)` for shared mode when using WASAPI
- NAudio (C#): uses different APIs for mic vs loopback (WaveIn vs WasapiLoopbackCapture), analogous to using two different stacks in Python
