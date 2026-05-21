# Development notes

## Architecture

- **Electron main** (`electron/main.ts`) — spawns the sidecar, forwards IPC events to the renderer.
- **Renderer** (`renderer/`) — React + Vite UI.
- **Python sidecar** (`sidecar/sidecar.py`, `app/ipc.py`, `app/session.py`) — recording, transcription, OpenAI, storage.

## Transcription subprocess

Transcription runs in a separate process so ONNX inference does not hold the sidecar GIL. See `app/transcription.py` (`start_transcription_subprocess`) and `app/session.py` (queues/process).

## Local run

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
npm install
npm run dev
```

## Sidecar-only smoke test

```powershell
python -m sidecar.sidecar
```

Example commands (one JSON object per line on stdin):

```json
{"id":"1","type":"ping","params":{}}
{"id":"2","type":"model_status","params":{}}
{"id":"3","type":"list_devices","params":{}}
```

## Build for testers

```powershell
pyinstaller meetings.spec
npm run dist
```

Share `installer_output\Rapid Scribe Setup *.exe`.

## Performance

- Sidecar is spawned on Electron `ready` before the window loads (splash until `ping` succeeds).
- Transcription subprocess stays alive between recordings to avoid cold model load.
