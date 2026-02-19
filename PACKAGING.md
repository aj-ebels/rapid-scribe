# Packaging for Windows

This document explains how to build **Rapid Scribe** into a Windows executable and folder that you can share so others can install and run it without Python.

## What you get

- **Output:** A folder `dist/Rapid Scribe/` containing:
  - `Rapid Scribe.exe` — double-click to run
  - Python runtime and dependencies (no Python install needed on the user’s PC)
  - `themes/`, `icon.ico`, and default `prompts.json`
- **User data:** On first run, the app creates `%APPDATA%\Meetings\` for settings, prompts, and API key. The exe can live in Program Files; settings remain writable.

## Prerequisites

1. **Windows 10/11** (same OS you’re building on is recommended).
2. **Python 3.10 or 3.11** (64-bit). Create and use a venv:

   ```powershell
   cd c:\Users\aebels\Documents\apps\Meetings
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install app and build dependencies:**

   ```powershell
   pip install -r requirements.txt -r requirements-build.txt
   ```

4. **Optional but recommended:** Install the default transcription model once so the build environment matches:

   ```powershell
   python main.py
   ```

   Then in the app: open the **Models** tab, ensure the default model is selected, click **Download & install**. Close the app.

5. **Required for transcription in the built app:** The spec bundles the `onnx_asr` package’s `preprocessors` folder (e.g. `nemo128.onnx`) so that the frozen exe can load the model. Ensure `onnx_asr` is installed in the venv when you run `pyinstaller` (it is included via `requirements.txt`).

## Build steps

1. **From the project root (where `meetings.spec` is):**

   ```powershell
   pyinstaller meetings.spec
   ```

2. **Output location:**  
   `dist\Rapid Scribe\`  
   - Run `dist\Rapid Scribe\Rapid Scribe.exe` to test.

3. **Share with others:**  
   - **Option A — Installer (recommended):** Build the installer (see [Installer (Inno Setup)](#installer-inno-setup) below), then share the generated setup exe. Users run it to install like any Windows app; they can uninstall via **Settings → Apps → Installed apps**.
   - **Option B — Zip:** Zip the entire `dist\Rapid Scribe` folder and send it. Users unzip to a folder (e.g. `C:\Program Files\Rapid Scribe` or their preferred location) and run `Rapid Scribe.exe`. No Python or pip needed.

## Installer (Inno Setup)

The project includes an **Inno Setup** script so you can build a normal Windows installer. Users get a single setup exe, Start Menu and Desktop shortcuts, and the app appears in **Settings → Apps** so they can uninstall it like any other program. User data in `%APPDATA%\Meetings\` is not removed on uninstall.

1. **Build the app first** (see [Build steps](#build-steps) above):
   ```powershell
   pyinstaller meetings.spec
   ```

2. **Install [Inno Setup](https://jrsoftware.org/isinfo.php)** (free) if you don't have it.

3. **Build the installer:**
   - Open `installer.iss` in Inno Setup and click **Build → Compile**, or
   - From the project root: `"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss`
   The setup exe is created in `installer_output\` (e.g. `Rapid-Scribe-Setup-1.0.exe`).

4. **Share** the setup exe. Users double-click to install, then run the app from the Start Menu or Desktop. To uninstall: **Settings → Apps → Installed apps** → **Rapid Scribe** → **Uninstall**.

## In-app update check

The app can prompt users when a newer version is available (Settings tab → **Check for updates**, and once automatically shortly after startup). To enable it, configure a source.

### Public GitHub repo (simplest)

If your repo is **public**, the app can read the latest release with no password:

1. Publish releases (tag e.g. `v3.0`) and attach the setup exe as an asset.
2. In `app/update_check.py`, set **GITHUB_REPO** to your repo:
   ```python
   GITHUB_REPO = "your-username/Meetings"   # use your actual GitHub username and repo name
   ```
   Or set the environment variable **UPDATE_CHECK_GITHUB_REPO** to `your-username/Meetings` when building or running the app (so you don’t edit code).

No token or password is needed for a public repo.

### Private GitHub repo

GitHub’s API does **not** allow unauthenticated access to private repos. You should **not** put a Personal Access Token in the app (it would be shipped to every user and would grant access to your repo). Use one of these instead:

- **Make the repo public** and use the steps above (no token).
- **Use a public version URL:** Host a small JSON file that is publicly readable (e.g. a [public GitHub Gist](https://gist.github.com)) with:
  ```json
  {"version": "3.0", "url": "https://github.com/your-username/Meetings/releases/latest"}
  ```
  Set **UPDATE_CHECK_JSON_URL** (env var or in `app/update_check.py`) to that URL (for a Gist use the raw URL, e.g. `https://gist.githubusercontent.com/.../raw/.../version.json`). When you publish a new release, update the Gist’s `version` (and optionally `url`). When users click “Download”, they’re taken to the release page; for a private repo they must be logged in to GitHub with access to see and download the asset.

If no source is configured, the **Check for updates** button still runs but reports that no update source is configured.

## Diagnosing "no audio transcribed" in the built app

When the built exe runs but nothing gets transcribed, the app writes a **diagnostic log** so you can see where the pipeline stops (no console window = failures are otherwise silent).

1. **Run the built app**, choose your audio mode, click **Start**, speak or play audio for 10–20 seconds, then **Stop**.
2. **Open the log file:**  
   `%APPDATA%\Meetings\diagnostic.log`  
   (e.g. `C:\Users\<You>\AppData\Roaming\Meetings\diagnostic.log`)
3. **Read the last run.** You’ll see lines like:
   - `start_recording mode=...` — confirms which mode and devices were used.
   - `meeting_started` or `meeting_start_failed` — in Meeting mode, whether the mixer started.
   - `chunk_queued` — audio chunks are being captured and queued.
   - `transcription_model_loaded` or `transcription_model_load_failed` — model loaded or failed (e.g. not downloaded).
   - `transcription_got_path` — worker received a WAV path (check `exists=True`).
   - `transcription_ok` / `transcription_recognize_failed` — recognize step succeeded or threw.

**Typical causes when nothing is transcribed:**

- **No `chunk_queued`** → Capture is failing (device, permissions, or silence threshold). Check for `capture_error` or `meeting_start_failed`.
- **`transcription_model_load_failed`** → Model not installed. Have the user open the **Models** tab and click **Download & install**.
- **`transcription_got_path exists=False`** → WAV path not valid for the worker (e.g. temp dir difference). Check `chunk_write_failed` in the log.
- **`transcription_recognize_failed`** → ONNX/model error; the log will show the exception.

You can also run with a console to see tracebacks: in `meetings.spec` set `console=True`, rebuild, and run the exe from a terminal.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| **“Module not found” when running the exe** | Add the missing module to `hiddenimports` in `meetings.spec`, then rebuild. |
| **Theme or icon missing** | Confirm `themes/` and `icon.ico` are next to `meetings.spec` and that the spec’s `datas` includes them (they are in the default spec). |
| **Transcription model not found** | Models are stored in the user’s Hugging Face cache (e.g. `%USERPROFILE%\.cache\huggingface\hub`). Users must open the **Models** tab and click **Download & install** once. |
| **Antivirus blocks the exe** | PyInstaller executables are sometimes flagged. You can: (1) Sign the exe (code signing cert), or (2) Distribute via an installer and/or zip and ask users to add an exclusion. |
| **Large folder size** | Expected (Python + numpy/scipy/ONNX). To shrink: exclude unneeded packages in the spec `excludes`, or use UPX (set `upx=True` in the spec if you have UPX installed). |

## Single-file build (optional)

The default spec builds a **directory** (`dist/Rapid Scribe/`) so that `icon.ico` and `prompts.json` sit next to the exe (as your code expects).  
If you want a **single .exe** instead:

- In `meetings.spec`, replace the `EXE` + `COLLECT` block with a single `EXE(..., onefile=True)` and include all binaries/datas in that `EXE`. Then theme and data paths will use the extracted `_MEIPASS` path (already handled in the GUI). You would still ship `prompts.json` and `icon.ico` next to the exe for first-run defaults, or copy them from the bundle at runtime.

Keeping the **directory** build is usually simpler and avoids long startup (no unpack step).
