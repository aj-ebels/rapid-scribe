# Rapid Scribe 🎙️

Real-time meeting capture and transcription for Windows and Linux (and WSL2). Record system audio and your mic, see live speech-to-text, and — if you bring your own OpenAI key — generates AI summaries and Q&A.

**License:** [MIT](LICENSE) — use, modify, and distribute under the terms in that file.

## 🚀 Quick start (Windows installer)

1. Download the latest **`Rapid Scribe vX.X-Setup.exe`** from **[Releases](https://github.com/aj-ebels/rapid-scribe/releases)** and run it.
2. If Windows shows **“Windows protected your PC”**, choose **More info** → **Run anyway** (unsigned installer; only proceed if you trust this project).
3. After install, pin or add a shortcut as you like. Uninstall anytime via **Settings → Apps**.

**Requires 64-bit Windows 10 or 11 on Intel or AMD (x64).** Snapdragon / ARM Copilot+ PCs are not supported. If the installer reports an unsupported Windows version, see [PACKAGING.md — Troubleshooting](PACKAGING.md#troubleshooting).

### One-time setup

1. **Models** tab — download the transcription model (once, ~650 MB).
2. **Settings** tab — paste your **OpenAI API key** and **Save** (optional but needed for AI Summary and Ask AI). Optionally pick OpenAI models for Summary, Ask AI, and export naming. Calls use your key and incur OpenAI usage; the app rate-limits requests to reduce accidental spam.
3. For summaries, **Call Summary** is the default AI prompt (Notion-style output). Add your own prompts under **AI Prompts** if you like.

### Using the app

1. **Record** (the red button, top right) — activate live transcription from your mic and system audio.
2. **Manual Notes** — jot things down; prompts can incorporate these into the summary.
3. **AI Summary → Generate** — summary built from transcript + notes. It'll even suggest a filename for you.
4. **Ask AI** — chat with the meeting content.
5. **Export** — pick a filename, export Markdown (summary, notes, full transcript).

The app checks for updates when a new release ships.

## 🛠️ Run from source (for development)

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Linux (Ubuntu)** — install PortAudio and Tk first, then the same steps:

```bash
sudo apt-get install libportaudio2 python3-tk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Meeting and loopback modes capture system audio from the PulseAudio/PipeWire **monitor source** of your output device (the input named "Monitor of …"). Modern Ubuntu (PipeWire with `pipewire-pulse`) and PulseAudio desktops expose this out of the box. If no monitor device is found, the app falls back with a message — check `pactl list sources short` for a `.monitor` source.

**WSL2** — same as Linux; plain mic input works via WSLg. System-audio capture depends on what WSLg's PulseAudio exposes (often only `RDPSource`).

On first run: download the model under **Models**, then drop an OpenAI key under **Settings** if you want the AI summarization and Q&A to function.

## ✨ Features

- **Live transcription** — Parakeet (ONNX) speech-to-text, scrolling in real time
- **Meeting mode** — Mic + system audio (calls, browser) in a single stream
- **AI summary** — Optional OpenAI summaries and export filenames
- **Custom prompts** — Define your own AI prompts under **AI Prompts** to tailor summaries
- **Ask AI** — Chat over your transcript and notes
- **In-app updates** — You'll be told when a new version exists
- **Problem reporting** — Uncaught errors offer an email focused on **that** exception (traceback + short log tail). **Settings → Report a problem…** sends a broader manual report. Set `RAPID_SCRIBE_REPORT_EMAIL` or `_DEFAULT_RECIPIENT_EMAIL` in `app/error_report.py`.


See **[PACKAGING.md](PACKAGING.md)** for the full ritual. TL;DR:

```powershell
# Windows
pip install -r requirements.txt -r requirements-build.txt
pyinstaller meetings.spec
# Compile installer.iss in Inno Setup → installer_output\Rapid Scribe vX.X-Setup.exe
```

```bash
# Linux
pip install -r requirements.txt -r requirements-build.txt
./scripts/build_linux.sh
# Output: dist/rapid-scribe/ — see PACKAGING.md for .desktop install
```

## 📚 Docs

| Doc | Description |
|-----|-------------|
| [PACKAGING.md](PACKAGING.md) | Build, installer, updates, troubleshooting |
| [dev.md](dev.md) | Development notes, performance |

## 📋 Requirements

- **Python** 3.10 or 3.11 (64-bit) — source install only
- **64-bit Windows 10 or 11 on Intel/AMD (x64)** — Snapdragon / ARM Windows is not supported — or **Linux** (Ubuntu 22.04+ recommended) with PulseAudio or PipeWire
- Meeting/loopback capture uses WASAPI loopback on Windows and the PulseAudio/PipeWire monitor source on Linux; on WSL2, system-audio capture depends on WSLg
- **OpenAI API key** (optional) for AI summary and Q&A
