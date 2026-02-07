# PyInstaller spec — Windows one-folder build (easy to zip and share).
# Build on Windows: pip install pyinstaller && pip install -r requirements.txt && pyinstaller meetings.spec
# Output: dist/meetings/ (folder with meetings.exe + launcher .bat). Zip that folder and share.

import sys
from pathlib import Path

block_cipher = None

# Include the Windows launcher and app icon (for window + taskbar when running from exe)
added_datas = []
if sys.platform == 'win32':
    added_datas = [('Run Meetings Transcriber.bat', '.')]
    # Bundle icon.ico so the app window and taskbar show it; add icon.ico to project root first
    if (Path(__file__).resolve().parent / 'icon.ico').exists():
        added_datas.append(('icon.ico', '.'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_datas,
    hiddenimports=[
        'customtkinter',
        'onnx_asr',
        'onnxruntime',
        'sounddevice',
        'numpy',
        'scipy',
        'scipy.io.wavfile',
        'huggingface_hub',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder: exe is small, rest is in the same folder (faster startup, reliable with heavy deps)
# Use icon.ico for the .exe file (Explorer, taskbar pin). Create icon.ico in project root.
exe_icon = 'icon.ico' if (Path(__file__).resolve().parent / 'icon.ico').exists() else None
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='meetings',
    debug=False,
    icon=exe_icon,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='meetings',
)
