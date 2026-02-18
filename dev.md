# Development Notes

## Performance (incl. running alongside Teams)

- **Transcription in subprocess**: Transcription runs in a separate process so ONNX inference does not hold the main process GIL; this avoids UI/audio freezes and reduces contention with other apps (e.g. Teams). See `app/transcription.py` (`start_transcription_subprocess`) and `app/gui.py` (multiprocessing queues/process).
- **Batched transcript updates**: The GUI drains all ready transcript lines each poll tick and does a single text insert/scroll to limit widget updates when several chunks complete at once.

## Run the app
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Package for Windows (share with others)
See **[PACKAGING.md](PACKAGING.md)** for full steps. Quick build:

```powershell
pip install -r requirements.txt -r requirements-build.txt
pyinstaller meetings.spec
```

Then share the `dist\Rapid Scribe` folder (or build the installer: open `installer.iss` in Inno Setup and Compile; see PACKAGING.md).