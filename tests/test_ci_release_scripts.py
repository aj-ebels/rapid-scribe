"""Checks used by the Windows installer workflow."""
import os
import subprocess
import sys
from pathlib import Path

from scripts.ci_pyinstaller_warn import missing_required
from scripts.ci_release_version import app_version
from scripts.ci_setup_size import MAX_BYTES, MIN_BYTES, V3_19_BYTES, size_error


def test_app_version_matches_installer():
    version = app_version()
    text = (Path(__file__).resolve().parents[1] / "installer.iss").read_text(encoding="utf-8")
    assert f"AppVersion={version}" in text
    assert f"OutputBaseFilename=Rapid Scribe v{version}-Setup" in text


def test_release_version_script_accepts_matching_version():
    version = app_version()
    env = os.environ.copy()
    env["REQUESTED_VERSION"] = version
    root = os.path.join(os.path.dirname(__file__), "..")
    proc = subprocess.run(
        [sys.executable, "scripts/ci_release_version.py"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == version


def test_release_version_script_rejects_mismatch():
    env = os.environ.copy()
    env["REQUESTED_VERSION"] = "9.9"
    root = os.path.join(os.path.dirname(__file__), "..")
    proc = subprocess.run(
        [sys.executable, "scripts/ci_release_version.py"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_warn_file_flags_exact_required_modules_only():
    lines = [
        "missing module named numpy - imported by something",
        "missing module named numpy.core - imported by something",
        "missing module named onnx_asr.preprocessors - imported by onnx_asr",
        "missing module named pyaudiowpatch ",
        "missing module named unrelated",
    ]
    missing = missing_required(lines)
    assert missing == [
        "missing module named numpy - imported by something",
        "missing module named pyaudiowpatch ",
    ]


def test_setup_size_window_accepts_v3_19_and_rejects_extremes():
    assert MIN_BYTES < V3_19_BYTES < MAX_BYTES
    assert size_error(V3_19_BYTES) is None
    assert size_error(MIN_BYTES - 1) is not None
    assert size_error(MAX_BYTES + 1) is not None


def test_onnx_preprocessor_dist_walk(tmp_path, monkeypatch):
    from scripts.ci_onnx_preprocessors import _onnx_files

    bundled = tmp_path / "Rapid Scribe" / "_internal" / "onnx_asr" / "preprocessors"
    data = bundled / "data"
    data.mkdir(parents=True)
    (bundled / "legacy.onnx").write_bytes(b"old")
    (data / "nemo128.onnx").write_bytes(b"ok")
    (tmp_path / "Rapid Scribe" / "other.onnx").write_bytes(b"no")
    monkeypatch.chdir(tmp_path)
    found = _onnx_files(str(tmp_path / "Rapid Scribe"))
    names = sorted(os.path.basename(path) for path in found)
    assert names == ["legacy.onnx", "nemo128.onnx"]
