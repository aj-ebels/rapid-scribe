# Packaging for Windows

This document explains how to build the **Blue Bridge Meeting Companion** into a Windows executable and folder that you can share so others can install and run it without Python.

## What you get

- **Output:** A folder `dist/Meetings/` containing:
  - `Meetings.exe` — double-click to run
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

## Build steps

1. **From the project root (where `meetings.spec` is):**

   ```powershell
   pyinstaller meetings.spec
   ```

2. **Output location:**  
   `dist\Meetings\`  
   - Run `dist\Meetings\Meetings.exe` to test.

3. **Share with others:**  
   Zip the entire `dist\Meetings` folder (or use an installer — see below) and send it. Users should:
   - Unzip to a folder (e.g. `C:\Program Files\Meetings` or `C:\Users\<name>\Meetings`).
   - Run `Meetings.exe`.  
   No Python or pip install needed.

## Optional: Installer (e.g. Inno Setup)

To give users a proper “Install” experience (shortcut, uninstaller):

1. **Download and install [Inno Setup](https://jrsoftware.org/isinfo.php)** (free).
2. **Create a script** (e.g. `installer.iss`) that:
   - Points to `dist\Meetings` as the source.
   - Installs `Meetings.exe` and the whole folder.
   - Creates a Start Menu shortcut and optional Desktop shortcut.
   - Sets the app data directory the same way (no need to change your code).

Example minimal `installer.iss` (adjust paths and app name as needed):

```iss
[Setup]
AppName=Blue Bridge Meeting Companion
AppVersion=1.0
DefaultDirName={autopf}\Meetings
DefaultGroupName=Meetings
OutputDir=installer_output
OutputBaseFilename=Meetings-Setup
Compression=lzma2
SolidCompression=yes

[Files]
Source: "dist\Meetings\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\Blue Bridge Meeting Companion"; Filename: "{app}\Meetings.exe"
Name: "{autodesktop}\Blue Bridge Meeting Companion"; Filename: "{app}\Meetings.exe"
```

Build the installer from Inno Setup (Compile), then share the generated `.exe` installer.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| **“Module not found” when running the exe** | Add the missing module to `hiddenimports` in `meetings.spec`, then rebuild. |
| **Theme or icon missing** | Confirm `themes/` and `icon.ico` are next to `meetings.spec` and that the spec’s `datas` includes them (they are in the default spec). |
| **Transcription model not found** | Models are stored in the user’s Hugging Face cache (e.g. `%USERPROFILE%\.cache\huggingface\hub`). Users must open the **Models** tab and click **Download & install** once. |
| **Antivirus blocks the exe** | PyInstaller executables are sometimes flagged. You can: (1) Sign the exe (code signing cert), or (2) Distribute via an installer and/or zip and ask users to add an exclusion. |
| **Large folder size** | Expected (Python + numpy/scipy/ONNX). To shrink: exclude unneeded packages in the spec `excludes`, or use UPX (set `upx=True` in the spec if you have UPX installed). |

## Single-file build (optional)

The default spec builds a **directory** (`dist/Meetings/`) so that `icon.ico` and `prompts.json` sit next to the exe (as your code expects).  
If you want a **single .exe** instead:

- In `meetings.spec`, replace the `EXE` + `COLLECT` block with a single `EXE(..., onefile=True)` and include all binaries/datas in that `EXE`. Then theme and data paths will use the extracted `_MEIPASS` path (already handled in the GUI). You would still ship `prompts.json` and `icon.ico` next to the exe for first-run defaults, or copy them from the bundle at runtime.

Keeping the **directory** build is usually simpler and avoids long startup (no unpack step).
