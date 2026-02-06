# PyInstaller spec — Windows one-folder build (easy to zip and share).
# Build on Windows: pip install pyinstaller && pip install -r requirements.txt && pyinstaller meetings.spec
# Output: dist/meetings/ (folder with meetings.exe + launcher .bat). Zip that folder and share.

import sys

block_cipher = None

# Include the Windows launcher so it's inside the built folder
added_datas = []
if sys.platform == 'win32':
    added_datas = [('Run Meetings Transcriber.bat', '.')]

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
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='meetings',
    debug=False,
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
