# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Rapid Scribe sidecar (headless, Electron host).
# Build: pyinstaller meetings.spec
# Output: dist/Rapid Scribe Sidecar/rapid-scribe-sidecar.exe

import os

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
datas = [
    (os.path.join(SPEC_DIR, "assets", "icons"), "assets/icons"),
    (os.path.join(SPEC_DIR, "icon.ico"), "."),
]
prompts_src = os.path.join(SPEC_DIR, "prompts.json")
if os.path.isfile(prompts_src):
    datas.append((prompts_src, "."))

try:
    import onnx_asr
    _onnx_asr_root = os.path.dirname(os.path.abspath(onnx_asr.__file__))
    _preprocessors = os.path.join(_onnx_asr_root, "preprocessors")
    if os.path.isdir(_preprocessors):
        datas.append((_preprocessors, "onnx_asr/preprocessors"))
except Exception:
    pass

hiddenimports = [
    "sounddevice",
    "numpy",
    "scipy",
    "openai",
    "huggingface_hub",
    "onnx",
    "onnxruntime",
    "onnx_asr",
]
if __import__("sys").platform == "win32":
    hiddenimports.append("pyaudiowpatch")

a = Analysis(
    [os.path.join(SPEC_DIR, "sidecar", "sidecar.py")],
    pathex=[SPEC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["customtkinter", "tkinter", "PIL"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rapid-scribe-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    console=True,
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
    name="Rapid Scribe Sidecar",
)
