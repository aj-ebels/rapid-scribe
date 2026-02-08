# Development Notes

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

Then share the `dist\Blue Bridge Meeting Companion` folder (or build the installer: open `installer.iss` in Inno Setup and Compile; see PACKAGING.md).