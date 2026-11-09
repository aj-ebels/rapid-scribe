"""Headless check for the frozen app: package metadata and preprocessor data.

Start fails with "No package metadata was found for onnx-asr" when dist-info
was left out of the bundle. Importing onnx_asr hits that immediately.
The preprocessor file is read the same way onnx_asr loads nemo128.onnx.
"""
from __future__ import annotations

import os
import sys
import traceback

from .bundle_metadata import METADATA_DISTS


def main(argv: list[str]) -> int:
    out_path = _out_path(argv)
    try:
        text, ok = _check()
    except Exception:
        text = traceback.format_exc()
        ok = False
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    stream = sys.stdout or sys.stderr
    if stream is not None:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
    return 0 if ok else 1


def _out_path(argv: list[str]) -> str | None:
    if "--frozen-smoke" in argv:
        idx = argv.index("--frozen-smoke")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return os.environ.get("RAPID_SCRIBE_SMOKE_LOG") or None


def _check() -> tuple[str, bool]:
    lines: list[str] = []
    ok = True
    frozen = bool(getattr(sys, "frozen", False))
    lines.append(f"frozen {int(frozen)}")
    if os.environ.get("RAPID_SCRIBE_REQUIRE_FROZEN") == "1" and not frozen:
        lines.append("ERROR not frozen")
        ok = False

    import importlib.metadata

    for dist in METADATA_DISTS:
        try:
            lines.append(f"{dist} {importlib.metadata.version(dist)}")
        except Exception as exc:
            lines.append(f"ERROR {dist}: {exc}")
            ok = False

    try:
        import onnx_asr

        lines.append(f"import onnx_asr {onnx_asr.__version__}")
    except Exception as exc:
        lines.append(f"ERROR import onnx_asr: {exc}")
        ok = False
        lines.append("FAIL" if not ok else "OK")
        return "\n".join(lines) + "\n", ok

    try:
        from importlib.resources import files

        import onnx_asr.preprocessors

        blob = files(onnx_asr.preprocessors).joinpath("data").joinpath("nemo128.onnx").read_bytes()
        if not blob:
            raise RuntimeError("nemo128.onnx is empty")
        lines.append(f"nemo128.onnx {len(blob)}")
    except Exception as exc:
        lines.append(f"ERROR nemo128.onnx: {exc}")
        ok = False

    lines.append("OK" if ok else "FAIL")
    return "\n".join(lines) + "\n", ok
