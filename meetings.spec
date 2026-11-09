# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Rapid Scribe (Windows and Linux).
# Build: pyinstaller meetings.spec
# Output (Windows): dist/Rapid Scribe/Rapid Scribe.exe (+ folder with dependencies and data)
# Output (Linux):   dist/rapid-scribe/rapid-scribe (+ folder with dependencies and data)

import os
import sys

APP_NAME = "Rapid Scribe" if sys.platform == "win32" else "rapid-scribe"

# Data files to bundle (relative to spec directory)
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
datas = [
    (os.path.join(SPEC_DIR, "themes"), "themes"),
    (os.path.join(SPEC_DIR, "assets", "icons"), "assets/icons"),
    (os.path.join(SPEC_DIR, "icon.ico"), "."),
]
# Bundle prompts.json if it exists (default prompts for first run)
prompts_src = os.path.join(SPEC_DIR, "prompts.json")
if os.path.isfile(prompts_src):
    datas.append((prompts_src, "."))

# Bundle onnx_asr preprocessors so model load finds them when frozen.
# The directory copy is recursive: onnx-asr 0.12+ keeps nemo128.onnx in preprocessors/data.
try:
    import onnx_asr
    _onnx_asr_root = os.path.dirname(os.path.abspath(onnx_asr.__file__))
    _preprocessors = os.path.join(_onnx_asr_root, "preprocessors")
    if os.path.isdir(_preprocessors):
        datas.append((_preprocessors, "onnx_asr/preprocessors"))
except Exception:
    pass

# dist-info. onnx-asr 0.11+ calls importlib.metadata.version("onnx-asr") on import.
# Without this, Start dies with "No package metadata was found for onnx-asr".
import importlib.util

from PyInstaller.utils.hooks import copy_metadata

_bm_file = os.path.join(SPEC_DIR, "app", "bundle_metadata.py")
_bm_spec = importlib.util.spec_from_file_location("rapid_scribe_bundle_metadata", _bm_file)
_bm = importlib.util.module_from_spec(_bm_spec)
_bm_spec.loader.exec_module(_bm)
for _dist in _bm.METADATA_DISTS:
    _entries = copy_metadata(_dist)
    if not _entries:
        raise SystemExit(f"copy_metadata({_dist!r}) returned no files")
    datas += _entries

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
    "onnx_asr",
]
# Windows-only
if sys.platform == "win32":
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
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPEC_DIR, "icon.ico") if sys.platform == "win32" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
