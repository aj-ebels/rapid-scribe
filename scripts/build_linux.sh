#!/usr/bin/env bash
# Build Rapid Scribe for Linux with PyInstaller.
#
# Prereqs (Ubuntu):
#   sudo apt-get install libportaudio2 python3-tk
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt -r requirements-build.txt
#
# Output: dist/rapid-scribe/ (self-contained folder; run dist/rapid-scribe/rapid-scribe)
#
# To install system-wide for the current machine:
#   sudo cp -r dist/rapid-scribe /opt/rapid-scribe
#   sudo cp packaging/linux/rapid-scribe.desktop /usr/share/applications/
#   sudo cp assets/icons/app.png /usr/share/icons/hicolor/256x256/apps/rapid-scribe.png
set -euo pipefail
cd "$(dirname "$0")/.."

pyinstaller meetings.spec --noconfirm

echo
echo "Build complete: dist/rapid-scribe/"
echo "Run with: ./dist/rapid-scribe/rapid-scribe"
