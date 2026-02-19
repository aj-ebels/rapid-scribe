# Rapid Scribe

Real-time meeting capture and transcription for Windows (and WSL2). Record system audio and microphone, get live speech-to-text, and generate AI summaries and notes, all in one app.

## Features

- **Live transcription** — Parakeet (ONNX) speech-to-text as you record
- **Meeting mode** — Mic + system audio (e.g. calls, browser) in one stream
- **AI summary** — Optional OpenAI-powered summaries and export names
- **Ask AI** — Chat over your meeting transcript and notes
- **In-app updates** — Notified when a new version is available (when repo is public)

## Quick start (from source)

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**WSL2 / Linux** — Install PortAudio first: `sudo apt-get install libportaudio2 portaudio19-dev`, then the same steps. Meeting/loopback mode is Windows-only.

On first run, open **Models**, download the default speech model (~650 MB one-time), then add your **OpenAI API key** in **Settings** if you want AI summary and Ask AI.

## Download (Windows)

Pre-built installer and release notes: **[Releases](https://github.com/aj-ebels/rapid-scribe/releases)**. The app checks for updates automatically when the repo is public.

## Building the installer

See **[PACKAGING.md](PACKAGING.md)** for PyInstaller build steps and Inno Setup installer. Summary:

```powershell
pip install -r requirements.txt -r requirements-build.txt
pyinstaller meetings.spec
# Then compile installer.iss in Inno Setup → installer_output\Rapid Scribe v3.0-Setup.exe
```

## Docs

| Doc | Description |
|-----|-------------|
| [PACKAGING.md](PACKAGING.md) | Build, installer, update check, troubleshooting |
| [dev.md](dev.md) | Development notes, run from source, performance |

## Requirements

- **Python** 3.10 or 3.11 (64-bit)
- **Windows** for meeting/loopback capture; WSL2 supported for default input
- **OpenAI API key** (optional) for AI summary and Ask AI; stored only in your app data

## License

See [LICENSE](LICENSE) in this repo.
