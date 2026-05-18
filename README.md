# Rapid Scribe

Real-time meeting capture and transcription for Windows (and WSL2). Record system audio and microphone, get live speech-to-text, and generate AI summaries and notes in one app.

**License:** [MIT](LICENSE) — use, modify, and distribute under the terms in that file.

## Quick start (Windows installer)

1. Download the latest **`Rapid Scribe vX.X-Setup.exe`** from **[Releases](https://github.com/aj-ebels/rapid-scribe/releases)** and run it.
2. If Windows shows **“Windows protected your PC”**, choose **More info** → **Run anyway** (unsigned installer; only proceed if you trust this project).
3. After install, pin or add a shortcut as you like. Uninstall anytime via **Settings → Apps**.

### One-time setup

1. **Models** tab — download the transcription model (once, ~650 MB).
2. **Settings** tab — paste your **OpenAI API key** and **Save** (optional; needed for AI Summary and Ask AI). Calls use your key and incur OpenAI usage; the app rate-limits requests to reduce accidental spam.
3. For summaries, **Call Summary** is the default AI prompt (Notion-style output). Add your own prompts under **AI Prompts** if you like.

### Using the app

1. **Record** (red button, top right) during a meeting — live transcription from your mic and meeting/system audio.
2. **Manual Notes** — your notes can feed into the AI summary (depends on the selected prompt).
3. **AI Summary → Generate** — summary from transcript + notes; can suggest an export filename.
4. **Ask AI** — chat over the meeting content.
5. **Export** — set a filename if needed, then export Markdown (summary, notes, and full transcript).

The app checks for updates when a new release is published.

## Run from source (for development)

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**WSL2 / Linux** — install PortAudio first: `sudo apt-get install libportaudio2 portaudio19-dev`, then the same steps. Meeting/loopback capture is Windows-only.

On first run: download the model under **Models**, then add an OpenAI key under **Settings** if you want AI features.

## Features

- **Live transcription** — Parakeet (ONNX) speech-to-text while you record
- **Meeting mode** — Mic + system audio (calls, browser) in one stream
- **AI summary** — Optional OpenAI summaries and export names
- **Ask AI** — Chat over transcript and notes
- **In-app updates** — Notified when a new version is available

## Build the installer

See **[PACKAGING.md](PACKAGING.md)**. Summary:

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
- **Windows** for meeting/loopback capture; WSL2 supported for default mic input
- **OpenAI API key** (optional) for AI summary and Ask AI; stored only in your local app data
