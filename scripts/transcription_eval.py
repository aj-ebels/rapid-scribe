#!/usr/bin/env python3
"""
CLI: run transcription accuracy evaluation against a JSON manifest of WAV + reference transcripts.

Usage (from repository root):
  pip install -r requirements.txt -r requirements-dev.txt
  python scripts/transcription_eval.py fixtures/transcription_eval/cases.json

Options:
  --model HF_REPO   Override transcription model for all cases
  --mode MODE       mic | loopback | both | triple | mic_vad | mic_vad_nolevel
                    (default: both). triple = mic + loopback + mic_vad_nolevel (three-way compare).
                    mic_vad_nolevel = VAD chunks without AGC; mic_vad = VAD + leveler (legacy).
  --json            Print machine-readable JSON summary to stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/transcription_eval.py` without PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.transcription_harness import (  # noqa: E402
    format_report,
    run_manifest,
    summarize_results,
)


def main():
    p = argparse.ArgumentParser(description="Transcription evaluation harness")
    p.add_argument(
        "manifest",
        type=Path,
        help="JSON manifest (list of cases with audio_file, reference text)",
    )
    p.add_argument("--model", type=str, default=None, help="Hugging Face repo id for onnx-asr")
    p.add_argument(
        "--mode",
        choices=("mic", "loopback", "both", "triple", "mic_vad", "mic_vad_nolevel"),
        default="both",
        help="Simulated capture path. triple = mic + loopback + mic_vad_nolevel.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text report")
    args = p.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        sys.exit(2)
    if args.mode == "both":
        capture_modes = ("mic", "loopback")
    elif args.mode == "triple":
        capture_modes = ("mic", "loopback", "mic_vad_nolevel")
    elif args.mode == "mic_vad":
        capture_modes = ("mic_vad",)
    else:
        capture_modes = (args.mode,)
    results = run_manifest(manifest, model_id=args.model, capture_modes=capture_modes)
    if args.json:
        out = {
            "summary": summarize_results(results),
            "results": [
                {
                    "id": r.case_id,
                    "capture_mode": r.capture_mode,
                    "audio_path": r.audio_path,
                    "wer": r.wer,
                    "cer": r.cer,
                    "chunk_count": r.chunk_count,
                    "error": r.error,
                    "hypothesis": r.hypothesis,
                }
                for r in results
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(format_report(results))


if __name__ == "__main__":
    main()
