"""Confirm the requested release version matches app/__init__.py and installer.iss."""
from __future__ import annotations

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def requested_version() -> str:
    raw = os.environ.get("REQUESTED_VERSION", "").strip()
    if raw[:1] in ("v", "V"):
        raw = raw[1:]
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", raw):
        sys.exit(f"Version must look like 3.20 (got {raw!r})")
    return raw


def app_version() -> str:
    text = (ROOT / "app" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        sys.exit("app/__init__.py has no __version__")
    return match.group(1)


def main() -> None:
    requested = requested_version()
    current = app_version()
    if current != requested:
        sys.exit(f"Requested {requested} but app/__init__.py is {current}")
    iss = (ROOT / "installer.iss").read_text(encoding="utf-8")
    if f"AppVersion={current}" not in iss:
        sys.exit("installer.iss AppVersion does not match __version__")
    needle = f"OutputBaseFilename=Rapid Scribe v{current}-Setup"
    if needle not in iss:
        sys.exit(f"installer.iss is missing {needle}")
    print(current)


if __name__ == "__main__":
    main()
