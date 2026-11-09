"""Copy PyInstaller warn files and fail if a required top-level module is missing."""
from __future__ import annotations

import pathlib
import re
import shutil
import sys

REQUIRED = (
    "customtkinter",
    "sounddevice",
    "numpy",
    "scipy",
    "openai",
    "huggingface_hub",
    "onnx",
    "onnxruntime",
    "onnx_asr",
    "pyaudiowpatch",
)
EXACT_MISSING = re.compile(r"^missing module named (" + "|".join(REQUIRED) + r")(\s|$)")
INTERESTING = re.compile(r"onnx_asr|pyaudiowpatch|customtkinter|hidden import", re.I)


def missing_required(lines: list[str]) -> list[str]:
    return [line for line in lines if EXACT_MISSING.match(line)]


def main() -> None:
    warns = list(pathlib.Path("build").rglob("warn-*.txt"))
    if not warns:
        sys.exit("PyInstaller did not write build/**/warn-*.txt")
    out_dir = pathlib.Path("ci-logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    bad: list[str] = []
    for path in warns:
        shutil.copyfile(path, out_dir / path.name)
        print(f"==== {path} ====")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            if INTERESTING.search(line):
                print(line)
        bad.extend(f"{path}: {line}" for line in missing_required(lines))
    if bad:
        print("Required modules reported missing:")
        print("\n".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
