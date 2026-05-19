# Electron Migration Plan — Rapid Scribe

> Goal: the **simplest** path to ship Rapid Scribe as an Electron application,
> minimizing rewrite of the audio + transcription pipelines (the hard parts)
> while replacing the CustomTkinter UI with a modern web UI.

## TL;DR

Keep all Python code (audio capture, ONNX/Parakeet transcription, OpenAI calls,
storage, update check). Replace **only** the `app/gui.py` (CustomTkinter) layer
with an **Electron shell**, and have Electron spawn the existing Python code as
a **headless sidecar child process**. Communicate over **JSON lines on
stdin/stdout** (no network sockets required).

This is the smallest viable surface: the only Python file that effectively
disappears is `app/gui.py`. Everything else is reused unchanged or with very
small refactors.

```
┌─────────────────────────────┐        stdin (commands)         ┌──────────────────────────────┐
│ Electron main (Node/TS)     │  ───────────────────────────▶   │ Python sidecar (PyInstaller) │
│  • spawns sidecar           │                                 │  • app/capture.py            │
│  • forwards IPC ↔ renderer  │  ◀───────────────────────────   │  • app/transcription.py      │
│  • single-instance lock     │       stdout (events, JSON)     │  • app/audio_mixer.py        │
│  • auto-update              │                                 │  • app/ai_summary.py         │
│  • file dialogs / exports   │                                 │  • app/settings.py, etc.     │
└──────────────┬──────────────┘                                 └──────────────────────────────┘
               │ contextBridge
               ▼
   ┌──────────────────────┐
   │ Renderer (React/Vite)│  ← new UI, replaces app/gui.py
   └──────────────────────┘
```

---

## Why this is the simplest option

We considered three approaches:

| Option | Effort | Install size | Notes |
|---|---|---|---|
| **A. Electron shell + Python sidecar (this plan)** | Low | Large (Electron + Python + ONNX) | Reuses ~5,300 of ~5,480 LOC. Only the Tk UI is replaced. |
| B. Hybrid — port settings/OpenAI/updates to Node, keep audio+ASR Python | Medium | Same | Trims sidecar surface but adds duplicate code paths and re-tests. |
| C. Full port to Node (onnxruntime-node, naudiodon/Web Audio, openai-node) | High | Smaller | Reimplements the entire WASAPI loopback + Parakeet preprocessing pipeline. High risk of audio/ASR regressions. |

Option **A** is recommended as the "simplest". The install size already includes
~650 MB of ONNX model in the current build, so adding Electron’s ~120 MB on top
is not a regression that justifies the risk of Option C.

---

## Architecture

### Processes

1. **Electron main process** (`electron/main.ts`) — Owns the BrowserWindow,
   single-instance lock, native menus, file save dialogs (for Markdown export),
   `electron-updater`, and the Python sidecar lifecycle.
2. **Renderer process** (`renderer/`) — React + Vite (or vanilla TS) UI that
   reproduces the current tabs: Record / Transcript / AI Summary / Ask AI /
   Manual Notes / AI Prompts / Models / Settings.
3. **Python sidecar** (`sidecar/sidecar.py`) — A new ~150-line entrypoint that
   imports the existing `app/*` modules and exposes them via a JSON-line
   protocol on stdio. Built with PyInstaller (reuses `meetings.spec` with a new
   entry script and `console=True`).

### IPC

- **Renderer ↔ Main**: standard `ipcRenderer`/`ipcMain` via a `contextBridge`
  preload script. Typed channels (`record:start`, `record:stop`,
  `transcript:line`, `summary:chunk`, `settings:get`, etc.).
- **Main ↔ Sidecar**: newline-delimited JSON on the sidecar’s stdin/stdout.
  - Commands (Electron → Python):
    `{"id": "...", "type": "start_recording", "params": {"mode": "meeting"}}`
  - Events / responses (Python → Electron):
    `{"id": "...", "type": "transcript_line", "data": {"ts": ..., "text": "..."}}`
  - `stderr` is captured into `%APPDATA%\Rapid Scribe\sidecar.log` (replaces
    the current `diagnostic.log`).
- Backpressure is not a concern: transcript lines are small and infrequent;
  AI summary streaming is line-oriented; audio bytes never cross the IPC
  boundary (capture stays inside Python).

### IPC command surface (minimum viable set)

Derived directly from the public methods used by `app/gui.py` today:

| Command | Maps to | Streams events? |
|---|---|---|
| `start_recording` / `stop_recording` | `capture.py` + `audio_mixer.py` + `chunk_recorder.py` | `transcript_line`, `audio_level`, `chunk_queued`, `capture_error` |
| `list_devices` | `app/devices.py` | no |
| `download_model` / `model_status` | `transcription.py` (model load path) + `huggingface_hub` | `model_progress` |
| `get_settings` / `set_settings` | `app/settings.py` | no |
| `get_api_key` / `set_api_key` | `app/api_key_storage.py` | no |
| `list_prompts` / `save_prompt` / `delete_prompt` | `app/prompts.py` | no |
| `generate_summary` | `app/ai_summary.py` | `summary_chunk`, `summary_done` |
| `ask_ai` | `app/ai_summary.py` (chat path) | `ask_chunk`, `ask_done` |
| `list_meetings` / `load_meeting` / `save_meeting` / `delete_meeting` | `app/meetings_storage.py` | no |
| `check_for_update` | `app/update_check.py` (or replaced by `electron-updater`) | no |

That is the **entire** Python surface area the Electron app needs.

### Data / config

- Keep `%APPDATA%\Meetings\` (or rename to `Rapid Scribe` — one-shot migration
  on first launch) so existing users keep their settings, prompts, API key, and
  meetings history.
- Move `single_instance.lock` ownership to Electron (`app.requestSingleInstanceLock()`).
  The sidecar no longer needs `app/single_instance.py`.

---

## Repository layout after the change

```
/  (repo root)
├── app/                       # Python — unchanged except gui.py removed
│   ├── capture.py
│   ├── transcription.py
│   ├── audio_mixer.py
│   ├── ai_summary.py
│   ├── settings.py
│   ├── ...
│   └── ipc.py                 # NEW: JSON-line dispatcher (~150 LOC)
├── sidecar/
│   └── sidecar.py             # NEW: PyInstaller entry, wires commands to app/*
├── electron/
│   ├── main.ts                # window, lifecycle, sidecar spawn
│   ├── preload.ts             # contextBridge
│   └── sidecar-client.ts      # stdio JSON-line client + typed API
├── renderer/                  # Vite + React (or vanilla TS)
│   ├── index.html
│   ├── src/
│   │   ├── App.tsx
│   │   └── tabs/{Record,Transcript,Summary,AskAI,Models,Settings,Prompts}.tsx
│   └── vite.config.ts
├── build/                     # electron-builder config + icons
├── package.json
├── tsconfig.json
├── meetings.spec              # updated: entry = sidecar/sidecar.py, console=True
├── installer.iss              # REMOVED (electron-builder makes the installer)
└── main.py                    # REMOVED (replaced by sidecar entry)
```

---

## Phased execution

Each phase is independently shippable behind a feature flag, so you can keep
the Tk app working until the Electron one reaches parity.

### Phase 1 — Headless sidecar (Python only, no Electron yet)

Goal: drive the existing pipeline from stdio so it can be tested without any UI.

1. Add `app/ipc.py` — a small dispatcher that reads JSON lines from stdin and
   writes JSON lines to stdout. Use one thread per long-running stream (transcript,
   summary), and a `queue.Queue` to serialize writes.
2. Add `sidecar/sidecar.py` — registers handlers that call the existing modules
   (`capture.start_*`, `transcription.start_transcription_subprocess`,
   `ai_summary.generate_summary`, `settings.load/save`, etc.).
3. Refactor (small): in modules that today call into `gui.py` for callbacks
   (e.g. status updates), swap to passing a callback object so the sidecar can
   inject "emit event" instead of "update Tk widget".
4. Smoke test from a terminal:
   `python -m sidecar.sidecar` and pipe in newline-delimited JSON commands.
5. Update `meetings.spec` to build the sidecar as a console exe
   (`console=True`, entry = `sidecar/sidecar.py`, drop `customtkinter` and
   `tkinter`-related hidden imports — saves ~30 MB).

Done criteria: `dist/Rapid Scribe Sidecar/rapid-scribe-sidecar.exe` records a
meeting, streams transcript JSON to stdout, generates a summary, all driven by
piped JSON. **Old Tk app still runs from `main.py` for fallback.**

### Phase 2 — Electron shell + sidecar bring-up

1. `npm init` + add `electron`, `electron-builder`, `vite`, `react`,
   `typescript`, `@vitejs/plugin-react`.
2. `electron/main.ts`:
   - `app.requestSingleInstanceLock()` (replaces `app/single_instance.py`).
   - Resolve sidecar path: `path.join(process.resourcesPath, 'sidecar', 'rapid-scribe-sidecar.exe')`
     in production, `dist/Rapid Scribe Sidecar/rapid-scribe-sidecar.exe` in dev.
   - `child_process.spawn` it with `stdio: ['pipe','pipe','pipe']`.
   - On exit (non-zero), surface in the UI and tail the last lines of
     `sidecar.log`.
3. `electron/sidecar-client.ts`: typed wrapper with `send(cmd, params)` →
   `Promise`, plus an `EventEmitter` for streamed events keyed by request id.
4. `electron/preload.ts`: expose `window.api = { startRecording, onTranscript, ... }`
   via `contextBridge.exposeInMainWorld`.
5. Show a minimal Renderer (one button + a log pane) that proves end-to-end
   IPC: click Record → see transcript lines appear.

### Phase 3 — Renderer parity with the Tk UI

Recreate the existing tabs in React. The Tk UI in `app/gui.py` is the
specification; map each tab 1:1. None of these views require new backend
features — they all map onto the IPC surface listed above.

Order of implementation (highest user value first):

1. **Record** tab — start/stop button, level meter (`audio_level` events),
   mode selector (mic / meeting).
2. **Transcript** view — virtualized list bound to `transcript_line` events.
3. **Settings** tab — devices, theme, API key (proxied to `set_api_key`).
4. **Models** tab — download progress UI bound to `model_progress`.
5. **AI Summary** tab — prompt picker + streamed `summary_chunk` rendering.
6. **Ask AI** tab — chat UI streamed via `ask_chunk`.
7. **AI Prompts** tab — CRUD on `prompts.json`.
8. **Manual Notes** + **Export** (use Electron `dialog.showSaveDialog`).

### Phase 4 — Packaging and updates

1. `electron-builder` config:
   - `extraResources`: ship the PyInstaller `dist/Rapid Scribe Sidecar/` folder
     under `resources/sidecar/`.
   - `nsis` target (replaces Inno Setup). Per-user install by default.
   - `appId: "com.rapidscribe.app"`, icon = existing `icon.ico`.
2. Replace `app/update_check.py` with **`electron-updater`** pointed at the
   same GitHub releases. (You can also keep the existing Python implementation
   and just call it through IPC — but `electron-updater` is the natural fit and
   removes ~155 lines of Python.)
3. Two-step CI build:
   - `pyinstaller meetings.spec` → produces the sidecar folder.
   - `npm run build && npm run dist` → electron-builder picks up the sidecar
     via `extraResources` and produces `Rapid Scribe Setup vX.Y.exe`.
4. Delete `installer.iss` and the old `dist/Rapid Scribe/` build path once the
   Electron installer ships.

### Phase 5 — Cleanup

- Delete `app/gui.py`, `app/single_instance.py`, `main.py`, `themes/`,
  `customtkinter` from `requirements.txt`, the `_splash` block, and the
  Tk-specific assets.
- Update `README.md`, `dev.md`, `PACKAGING.md` for the new flow.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sidecar startup time on cold launch (PyInstaller unpack + ONNX import). | Spawn the sidecar immediately on Electron `ready`, before the window is shown. Show a splash in the renderer that flips to ready on the first `pong` from the sidecar. This mirrors today’s Tk splash. |
| ONNX/Parakeet model already cached in `%USERPROFILE%\.cache\huggingface\hub` — must keep working. | No code change needed; `huggingface_hub` resolves the same cache regardless of the host process. |
| Existing users’ settings live in `%APPDATA%\Meetings\`. | Read from the old path on first run, migrate-copy to the new path (or just keep the old path — recommended for the "simplest" plan). |
| WASAPI loopback (`pyaudiowpatch`) is Windows-only and process-bound. | Unchanged. Stays inside the sidecar exactly as today. |
| Multi-process transcription (Python `multiprocessing.spawn`) inside a frozen sidecar. | Already working in the current PyInstaller build (`freeze_support()` is in `main.py`). Move that block verbatim into `sidecar/sidecar.py`. |
| Antivirus flags on a new unsigned exe. | Same situation as today; no regression. Consider code signing later. |
| Bundle size grows by ~120 MB (Electron). | Acceptable — the model is already the dominant cost. Document in `PACKAGING.md`. |

---

## Out of scope for the "simplest" plan

These are explicitly **not** part of this plan, to keep the migration small:

- Porting audio capture, the audio mixer, or the ONNX transcription pipeline to
  Node/TypeScript.
- Replacing OpenAI calls with `openai-node` (the Python SDK stays).
- Replacing the meetings storage / prompts files with a database.
- macOS or Linux builds (current app is Windows-first; the sidecar would need
  `sounddevice`-only paths, no WASAPI loopback).
- Code signing and notarization.

These can each be tackled later as independent follow-ups once the Electron
shell is shipping.

---

## Estimated change surface

- **New code**: ~150 LOC Python (`app/ipc.py` + `sidecar/sidecar.py`),
  ~300 LOC TypeScript (Electron main + preload + sidecar client),
  ~1,500–2,500 LOC TypeScript/TSX for the renderer (depends on how faithfully
  the Tk UI is reproduced).
- **Deleted code**: `app/gui.py` (2,460 LOC), `app/single_instance.py` (70),
  `main.py` (76), `installer.iss`, plus the `customtkinter` and Tk splash bits.
- **Touched but mostly unchanged**: `capture.py`, `transcription.py`,
  `audio_mixer.py`, `ai_summary.py`, `settings.py`, `meetings_storage.py`,
  `prompts.py`, `update_check.py` — refactor any GUI-callback parameters to
  accept a generic `emit(event, data)` function. No business-logic changes.

The net is a smaller Python codebase and a new TS codebase of comparable size
to the deleted Tk UI, with the audio and ASR pipelines untouched.
