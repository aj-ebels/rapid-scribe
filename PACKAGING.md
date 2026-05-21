# Packaging for Windows (Electron + Python sidecar)

Rapid Scribe ships as an **Electron** app with a **PyInstaller** Python sidecar for audio and transcription.

## What you get

- **Electron installer:** `installer_output/Rapid Scribe Setup X.Y.Z.exe` (NSIS, per-user install).
- **Sidecar folder:** `dist/Rapid Scribe Sidecar/` — bundled under `resources/sidecar/` in the installer.
- **User data:** `%APPDATA%\Meetings\` for settings, prompts, API key, meetings, and `sidecar.log`.

## Prerequisites

1. **Windows 10/11** (64-bit).
2. **Python 3.10 or 3.11** (64-bit) with venv:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt -r requirements-build.txt
   ```

3. **Node.js 20+** for Electron:

   ```powershell
   npm install
   ```

4. **Optional:** Download the transcription model once in dev (`npm run dev` → Models tab) so Hugging Face cache is warm.

## Build steps

### 1. Python sidecar

```powershell
pyinstaller meetings.spec
```

Output: `dist\Rapid Scribe Sidecar\rapid-scribe-sidecar.exe`

Test:

```powershell
dist\"Rapid Scribe Sidecar"\rapid-scribe-sidecar.exe
```

(Expect a `ready` JSON line on stdout; send `{"id":"1","type":"ping","params":{}}`.)

### 2. Electron app + installer

```powershell
npm run dist
```

This runs `vite build`, compiles `electron/*.ts`, and runs `electron-builder`, copying the sidecar via `extraResources` in `package.json`.

### 3. Share

Upload `installer_output\Rapid Scribe Setup *.exe` to GitHub Releases. Configure `publish` in `package.json` for `electron-updater`.

## Development

```powershell
npm run dev
```

Uses `python -m sidecar.sidecar` from the venv when the PyInstaller exe is not built yet.

## Bundle size notes

- ONNX model (~650 MB) is downloaded on first use (Hugging Face cache), not embedded in the installer.
- Electron adds ~120 MB; the sidecar folder is similar in size to the old Tk PyInstaller bundle (without CustomTkinter).

## Updates

- **Packaged app:** `electron-updater` checks GitHub releases on startup.
- **Settings → Check for updates** still calls the Python `check_for_updates` via IPC as a fallback.

## Troubleshooting

- **Sidecar failed to start:** Tail `%APPDATA%\Meetings\sidecar.log` and the error dialog in Electron.
- **No loopback on Windows:** Ensure `pyaudiowpatch` is in the venv before `pyinstaller`.
- **Transcription model missing:** Models tab → Download & install (requires network).

## Legacy

The previous **CustomTkinter + Inno Setup** flow (`main.py`, `installer.iss`, `dist/Rapid Scribe/`) has been replaced by this Electron layout. See [ELECTRON_MIGRATION_PLAN.md](ELECTRON_MIGRATION_PLAN.md).
