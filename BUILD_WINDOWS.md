# Building the Windows app (Option B)

Build this **on a Windows machine** so colleagues can run the app without installing Python.

## 1. On your Windows PC (native Windows, not WSL)

You must run the build **in native Windows** — open **Command Prompt** or **PowerShell** from Windows itself (or from the Start menu), **not** inside WSL or a Linux terminal. PyInstaller creates an executable for the OS it runs on; building inside WSL would produce a Linux binary, not a Windows `.exe`.

**Install Python 3.10 or 3.11** (if needed): [python.org/downloads](https://www.python.org/downloads/). During setup, check **“Add Python to PATH”**.

Open **Command Prompt** or **PowerShell** (in Windows) and run:

```cmd
cd path\to\meetings
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller meetings.spec
```

## 2. What you get

- Folder: **`dist\meetings`**
- Inside: **`meetings.exe`** and **`Run Meetings Transcriber.bat`** (and the DLLs/libraries the app needs).

## 3. Share with colleagues

1. **Zip the folder**: Right‑click `dist\meetings` → **Send to** → **Compressed (zipped) folder**. Name it e.g. `MeetingsTranscriber-Windows.zip`.
2. Send the zip (email, Teams, shared drive, etc.).
3. Tell colleagues:
   - Unzip the folder anywhere (e.g. Desktop or Documents).
   - Double‑click **`Run Meetings Transcriber.bat`** (or **`meetings.exe`**) to start the app.
   - The first time they run it, the app will download the speech model (~150 MB); after that it starts quickly.

No Python or other install needed on their side.

## Troubleshooting

- **“Python is not recognized”** — Reinstall Python and tick “Add Python to PATH”, or use the full path to `python.exe`.
- **Build errors about missing modules** — Run `pip install -r requirements.txt` again in the same venv, then `pyinstaller meetings.spec` again.
- **Antivirus warns on the .exe** — Common with PyInstaller apps. They can add an exception or run the .bat; the app is safe if they got the zip from you.
