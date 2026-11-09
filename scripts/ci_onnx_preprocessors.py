"""Fail unless onnx_asr preprocessors exist, either installed or inside the PyInstaller dist."""
from __future__ import annotations

import os
import sys


def _onnx_files(directory: str) -> list[str]:
    """Onnx files under onnx_asr/preprocessors, including preprocessors/data.

    onnx-asr 0.12 moved nemo128.onnx into preprocessors/data. The spec copies
    the preprocessors directory recursively, so both layouts count.
    """
    found = []
    for dirpath, _, files in os.walk(directory):
        normalized = "/" + dirpath.replace("\\", "/").rstrip("/")
        if not normalized.endswith("/onnx_asr/preprocessors") and "/onnx_asr/preprocessors/" not in normalized:
            continue
        for name in files:
            if name.endswith(".onnx"):
                found.append(os.path.join(dirpath, name))
    return found


def check_installed() -> None:
    import onnx_asr

    root = os.path.dirname(os.path.abspath(onnx_asr.__file__))
    pre = os.path.join(root, "preprocessors")
    print("onnx_asr:", root)
    if not os.path.isdir(pre):
        sys.exit(f"missing preprocessors directory: {pre}")
    found = _onnx_files(pre)
    print("preprocessors:", sorted(os.path.relpath(path, pre) for path in found))
    if not any(os.path.basename(path) == "nemo128.onnx" for path in found):
        listing = []
        for dirpath, _, files in os.walk(pre):
            listing.append(f"{dirpath}: {files}")
        sys.exit("nemo128.onnx not found under onnx_asr/preprocessors\n" + "\n".join(listing))


def check_dist() -> None:
    root = os.path.join("dist", "Rapid Scribe")
    if not os.path.isdir(root):
        sys.exit(f"PyInstaller dist folder missing: {root}")
    found = _onnx_files(root)
    print("bundled preprocessor onnx files:")
    for path in found:
        print(" ", path, os.path.getsize(path))
    if not any(os.path.basename(path) == "nemo128.onnx" for path in found):
        sys.exit("PyInstaller output is missing onnx_asr/preprocessors/**/nemo128.onnx")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--installed"
    if mode == "--installed":
        check_installed()
    elif mode == "--dist":
        check_dist()
    else:
        sys.exit(f"usage: {sys.argv[0]} --installed|--dist")


if __name__ == "__main__":
    main()
