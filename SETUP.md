# Setup & distribution

## Easiest for colleagues: two options

### Option A — They have Python (recommended to start)

Share the project folder (or Git repo). Colleagues install once per machine, then run one command.

**1. Install system dependencies (once per machine)**

- **Linux / WSL2 (Ubuntu):**
  ```bash
  sudo apt-get update
  sudo apt-get install -y libportaudio2 portaudio19-dev
  ```
- **macOS:**  
  `brew install portaudio`
- **Windows:**  
  No extra system install. Ensure a working microphone or “Stereo Mix” / loopback device for system audio.

**2. Install and run**

```bash
cd meetings
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On **Windows**, after setup you can:
- Double‑click **`run_from_source.bat`** (Command Prompt), or
- In PowerShell, run **`.\run_from_source.ps1`** or **`.venv\Scripts\python.exe main.py`** (avoids execution policy).

If **PowerShell** blocks `.venv\Scripts\activate` with "running scripts is disabled", you can do everything without activating:
- Install deps: **`.venv\Scripts\python.exe -m pip install -r requirements.txt`**
- Run app: **`.venv\Scripts\python.exe main.py`**
- Or use **cmd** and `activate.bat` instead.

First run will download the Parakeet model (from Hugging Face); after that it starts quickly. For NVIDIA GPU, use `onnx-asr[gpu,hub]` in `requirements.txt` instead of `onnx-asr[cpu,hub]` (requires CUDA).

---

### Option B — Windows: one zip, no Python (easiest for colleagues)

You build the app **on a Windows machine** once, then share a single zip. Colleagues unzip and double‑click to run.

**You — build on Windows (see [BUILD_WINDOWS.md](BUILD_WINDOWS.md) for full steps):**

1. On Windows: open Command Prompt or PowerShell in the project folder.
2. Run:
```powershell
cd path\to\meetings
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m pyinstaller meetings.spec
```
3. Zip the folder **`dist\meetings`** (e.g. `MeetingsTranscriber-Windows.zip`) and share it.

**Colleagues — install and open:**

1. **Download** the zip you sent them.
2. **Unzip** the folder anywhere (e.g. Desktop or Documents).
3. **Double‑click** **`Run Meetings Transcriber.bat`** (or **`meetings.exe`**) to start the app.
4. First run will download the Parakeet speech model; after that it starts quickly.

No Python or other software to install. For build details and troubleshooting, see **[BUILD_WINDOWS.md](BUILD_WINDOWS.md)**.

*(Linux/macOS: you can still use Option A with Python, or build a standalone binary per platform if needed.)*

---

## Quick reference

| Who          | Action |
|-------------|--------|
| **You**     | Share repo link or a zip of the project (or of `dist/meetings` if you used Option B). |
| **Colleagues (Python)** | Install system deps above → `pip install -r requirements.txt` → `python main.py`. |
| **Colleagues (Windows, no Python)** | Unzip the folder you shared → double‑click **Run Meetings Transcriber.bat** or **meetings.exe**. |

## Audio input (system / “what you hear”)

- **WSL2 / Linux:** Choose the PulseAudio **Monitor** (or similar) input so the app records system audio.
- **Windows:** Use “Stereo Mix” or a virtual loopback (e.g. VB-Audio Cable, or “Listen” to playback device) as the default recording device.
- **macOS:** Use BlackHole or similar virtual device, or a “Monitor”/loopback source, as input.

## Troubleshooting

- **PortAudio / sounddevice errors:** Install the system library for your OS (see step 1 under Option A).
- **No input device:** In the app, the footer shows the selected input. On Windows, set the desired recording device as default in System Settings → Sound.
- **First run slow:** The app downloads the Parakeet model on first use (in the terminal, before the window opens); later runs are faster.
- **"X connection broken" during first run:** The model now loads before the GUI starts. If the download was interrupted, run `python main.py` again; Hugging Face usually resumes the download.
