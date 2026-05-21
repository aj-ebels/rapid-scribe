# Rapid Scribe 🎙️

Real-time meeting capture and transcription for Windows (and WSL2). Record system audio and your mic, see live speech-to-text, and — if you bring your own OpenAI key — generate AI summaries and Q&A.

**License:** [MIT](LICENSE) — use, modify, and distribute under the terms in that file.

## Quick start (Windows installer)

1. Download the latest **`Rapid Scribe Setup X.X.X.exe`** from **[Releases](https://github.com/aj-ebels/rapid-scribe/releases)** and run it.
2. If Windows shows **“Windows protected your PC”**, choose **More info** → **Run anyway** (unsigned installer; only proceed if you trust this project).
3. After install, pin or add a shortcut as you like. Uninstall anytime via **Settings → Apps**.

### One-time setup

1. **Models** tab — download the transcription model (once, ~650 MB).
2. **Settings** tab — paste your **OpenAI API key** and **Save** (optional but needed for AI Summary and Ask AI).
3. For summaries, use the default AI prompts or add your own under **AI Prompts**.

### Using the app

1. **Record** — live transcription from your mic and/or system audio.
2. **Manual Notes** — jot things down; prompts can incorporate these into the summary.
3. **AI Summary → Generate** — summary from transcript + notes.
4. **Ask AI** — chat with the meeting content.
5. **Export** — save Markdown (summary, notes, full transcript).

Packaged builds check for updates on startup via `electron-updater`.

## Run from source (development)

**Prerequisites:** Node.js 20+, Python 3.10 or 3.11 (64-bit).

```powershell
cd c:\Users\aebels\Documents\apps\Meetings
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

npm install
npm run dev
```

This starts the Vite dev server and Electron, spawning the Python sidecar via `python -m sidecar.sidecar`.

**Sidecar only** (headless IPC test):

```powershell
python -m sidecar.sidecar
# Pipe JSON lines to stdin, e.g. {"id":"1","type":"ping","params":{}}
```

## Architecture

- **Electron** — window, single-instance lock, file dialogs, auto-update.
- **React (Vite)** — UI in `renderer/`.
- **Python sidecar** — audio capture, ONNX transcription, OpenAI, storage (`sidecar/sidecar.py` + `app/*`).

Communication: newline-delimited JSON on the sidecar’s stdin/stdout. See [ELECTRON_MIGRATION_PLAN.md](ELECTRON_MIGRATION_PLAN.md).

## Features

- **Live transcription** — Parakeet (ONNX) speech-to-text
- **Meeting mode** — Mic + system audio (calls, browser) on Windows
- **AI summary** — Optional OpenAI summaries
- **Custom prompts** — Tailor summaries under **AI Prompts**
- **Ask AI** — Chat over transcript and notes
- **Auto-update** — GitHub releases (packaged app)

## Build the installer

See **[PACKAGING.md](PACKAGING.md)**. Summary:

```powershell
pip install -r requirements.txt -r requirements-build.txt
pyinstaller meetings.spec
npm install
npm run dist
```

Output: `installer_output/Rapid Scribe Setup X.X.X.exe` (includes the PyInstaller sidecar under `resources/sidecar/`).

## Docs

| Doc | Description |
|-----|-------------|
| [PACKAGING.md](PACKAGING.md) | Build, installer, updates |
| [dev.md](dev.md) | Development notes |
| [ELECTRON_MIGRATION_PLAN.md](ELECTRON_MIGRATION_PLAN.md) | Migration design |

## Requirements

- **Windows** for meeting/loopback capture; WSL2 works for mic-only via the sidecar
- **OpenAI API key** (optional) for AI summary and Q&A
