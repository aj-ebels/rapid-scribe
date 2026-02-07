# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Blue Bridge Meeting Companion (Windows).
# Build: pyinstaller meetings.spec
# Output: dist/Meetings/Meetings.exe (+ folder with dependencies and data)

import os

# Data files to bundle (relative to spec directory)
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
datas = [
    (os.path.join(SPEC_DIR, "themes"), "themes"),
    (os.path.join(SPEC_DIR, "icon.ico"), "."),
]
# Bundle prompts.json if it exists (default prompts for first run)
prompts_src = os.path.join(SPEC_DIR, "prompts.json")
if os.path.isfile(prompts_src):
    datas.append((prompts_src, "."))

# Hidden imports for dynamic / optional modules
hiddenimports = [
    "customtkinter",
    "sounddevice",
    "numpy",
    "scipy",
    "openai",
    "huggingface_hub",
    "onnx",
    "onnxruntime",
]
# Windows-only
if __import__("sys").platform == "win32":
    hiddenimports.append("pyaudiowpatch")

a = Analysis(
    [os.path.join(SPEC_DIR, "main.py")],
    pathex=[SPEC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Meetings",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPEC_DIR, "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Meetings",
)
