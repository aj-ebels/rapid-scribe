# Rapid Scribe

Real-time meeting capture and transcription for Windows (and WSL2). Record system audio and your mic, see live speech-to-text, and — if you bring your own OpenAI key — generates AI summaries and Q&A.

**License:** [MIT](LICENSE) — use, modify, and distribute under the terms in that file.

## Quick start (Windows installer)

1. Download the latest **`Rapid Scribe vX.X-Setup.exe`** from **[Releases](https://github.com/aj-ebels/rapid-scribe/releases)** and run it.
2. If Windows shows **“Windows protected your PC”**, choose **More info** → **Run anyway** (unsigned installer; only proceed if you trust this project).
3. After install, pin or add a shortcut as you like. Uninstall anytime via **Settings → Apps**.

### One-time setup

1. **Models** tab — download the transcription model (once, ~650 MB).
2. **Settings** tab — paste your **OpenAI API key** and **Save** (optional but needed for AI Summary and Ask AI). Calls use your key and incur OpenAI usage; the app rate-limits requests to reduce accidental spam.
3. For summaries, **Call Summary** is the default AI prompt (Notion-style output). Add your own prompts under **AI Prompts** if you like.

### Using the app

1. **Record** (the red button, top right) — activate live transcription from your mic and system audio.
2. **Manual Notes** — jot things down; prompts can incorporate these into the summary.
3. **AI Summary → Generate** — summary built from transcript + notes. It'll even suggest a filename for you.
4. **Ask AI** — chat with the meeting content.
5. **Export** — pick a filename, export Markdown (summary, notes, full transcript).

The app checks for updates when a new release ships.

## Run from source (for development)

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**WSL2 / Linux** — install PortAudio first: `sudo apt-get install libportaudio2 portaudio19-dev`, then the same steps. Meeting/loopback capture is Windows-only.

On first run: download the model under **Models**, then drop an OpenAI key under **Settings** if you want the AI summarization and Q&A to function.

## Features

- **Live transcription** — Parakeet (ONNX) speech-to-text, scrolling in real time
- **Meeting mode** — Mic + system audio (calls, browser) in a single stream
- **AI summary** — Optional OpenAI summaries and export filenames
- **Ask AI** — Chat over your transcript and notes
- **In-app updates** — You'll be told when a new version exists

## Build the installer

See **[PACKAGING.md](PACKAGING.md)** for the full ritual. TL;DR:

```powershell
pip install -r requirements.txt -r requirements-build.txt
pyinstaller meetings.spec
# Compile installer.iss in Inno Setup → installer_output\Rapid Scribe vX.X-Setup.exe
```

## Docs

| Doc | Description |
|-----|-------------|
| [PACKAGING.md](PACKAGING.md) | Build, installer, updates, troubleshooting |
| [dev.md](dev.md) | Development notes, performance |

## Requirements

- **Python** 3.10 or 3.11 (64-bit) — source install only
- **Windows** for meeting/loopback capture; WSL2 works fine for plain mic input
- **OpenAI API key** (optional) for AI summary and Q&A
