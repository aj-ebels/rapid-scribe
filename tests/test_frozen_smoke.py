"""Unfrozen run of the check the Windows installer later runs inside the exe."""
from pathlib import Path

from app.bundle_metadata import METADATA_DISTS
from app.frozen_smoke import main


def test_metadata_dists_include_the_import_crash():
    assert "onnx-asr" in METADATA_DISTS
    assert "onnxruntime" in METADATA_DISTS
    assert "huggingface_hub" in METADATA_DISTS


def test_spec_copies_those_dists():
    spec = (Path(__file__).resolve().parents[1] / "meetings.spec").read_text(encoding="utf-8")
    assert "copy_metadata" in spec
    assert "bundle_metadata" in spec


def test_frozen_smoke_reads_metadata_and_preprocessor(tmp_path, monkeypatch):
    monkeypatch.delenv("RAPID_SCRIBE_REQUIRE_FROZEN", raising=False)
    log = tmp_path / "out.txt"
    code = main(["--frozen-smoke", str(log)])
    text = log.read_text(encoding="utf-8")
    assert code == 0, text
    assert "onnx-asr 0.12.0" in text
    assert "nemo128.onnx" in text
    assert text.strip().endswith("OK")


def test_frozen_smoke_rejects_unfrozen_process_when_ci_requires_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_SCRIBE_REQUIRE_FROZEN", "1")
    log = tmp_path / "out.txt"
    code = main(["--frozen-smoke", str(log)])
    text = log.read_text(encoding="utf-8")
    assert code == 1
    assert "ERROR not frozen" in text
    assert text.strip().endswith("FAIL")
