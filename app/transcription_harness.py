"""
Run offline transcription evaluation: WAV → capture-style chunks → ASR → GUI-style join → metrics.

This does not start the GUI or sounddevice. It uses the same ONNX model stack as production
(get_transcription_model / recognize). Transcription-worker RMS gating is not applied here.

Eval paths (same WAV, same reference):
  - mic: fixed-duration windows, no leveler (default mic / loopback-aligned).
  - loopback: fixed windows (CHUNK_DURATION_SEC).
  - mic_vad_nolevel: VAD chunking, no leveler (isolates leveler vs chunking).
  - mic_vad: VAD + leveler (legacy capture_worker_vad).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .offline_capture import iter_chunks_from_wav
from .offline_loopback import iter_fixed_duration_chunks_from_wav, iter_loopback_chunks_from_wav
from .transcription import clear_transcription_model_cache, get_transcription_model
from .transcription_metrics import char_error_rate, normalize_for_compare, word_error_rate
from .transcript_join import join_transcription_items


@dataclass
class EvalCaseResult:
    case_id: str
    audio_path: str
    reference_excerpt: str
    hypothesis: str
    wer: float
    cer: float
    chunk_count: int
    error: str | None = None
    capture_mode: str = "mic"  # "mic" | "loopback" | "mic_vad" | "mic_vad_nolevel"


def load_cases_manifest(manifest_path: str | Path) -> list[dict]:
    """Load evaluation cases from JSON. See fixtures/transcription_eval/cases.example.json."""
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON array of case objects")
    return data


def _resolve_case_asset(manifest_dir: Path, rel: str) -> Path:
    """
    Resolve paths like samples/foo.wav relative to the manifest folder, or if missing,
    relative to the project root (parent of fixtures/).
    """
    primary = (manifest_dir / rel).resolve()
    if primary.is_file():
        return primary
    repo_root = manifest_dir.parent.parent
    alt = (repo_root / rel).resolve()
    if alt.is_file():
        return alt
    return primary


def _reference_path(manifest_dir: Path, case: dict) -> Path | None:
    ref = case.get("reference_text_file")
    if ref:
        return _resolve_case_asset(manifest_dir, ref)
    return None


def _load_reference_text(manifest_dir: Path, case: dict) -> str:
    if "reference_text" in case and case["reference_text"] is not None:
        return str(case["reference_text"])
    rp = _reference_path(manifest_dir, case)
    if rp is None or not rp.is_file():
        raise ValueError(f"Case {case.get('id')!r} needs reference_text or reference_text_file")
    return rp.read_text(encoding="utf-8")


def run_single_case(
    *,
    case_id: str,
    audio_path: Path,
    reference_text: str,
    model_id: str | None,
    leveler_settings: dict | None,
    temp_dir: str | None,
    capture_mode: str = "mic",
    chunk_duration_sec: float | None = None,
) -> EvalCaseResult:
    """Transcribe one WAV; return metrics vs reference."""
    if capture_mode not in ("mic", "loopback", "mic_vad", "mic_vad_nolevel"):
        return EvalCaseResult(
            case_id=case_id,
            audio_path=str(audio_path),
            reference_excerpt="",
            hypothesis="",
            wer=1.0,
            cer=1.0,
            chunk_count=0,
            error=f"unknown capture_mode {capture_mode!r}",
            capture_mode=capture_mode,
        )
    ref_excerpt = reference_text[:120] + ("…" if len(reference_text) > 120 else "")
    paths_to_delete: list[Path] = []
    try:
        model = get_transcription_model(model_id)
        segments: list[tuple[str, float]] = []
        chunk_count = 0
        prev_end_sample = 0
        sample_rate = 16000

        if capture_mode == "mic_vad":
            chunk_iter = iter_chunks_from_wav(
                audio_path, leveler_settings=leveler_settings, temp_dir=temp_dir, use_leveler=True
            )
        elif capture_mode == "mic_vad_nolevel":
            chunk_iter = iter_chunks_from_wav(
                audio_path, leveler_settings=leveler_settings, temp_dir=temp_dir, use_leveler=False
            )
        elif capture_mode == "mic":
            chunk_iter = iter_fixed_duration_chunks_from_wav(
                audio_path, temp_dir=temp_dir, duration_sec=chunk_duration_sec
            )
        else:
            chunk_iter = iter_loopback_chunks_from_wav(audio_path, temp_dir=temp_dir)

        for batch in chunk_iter:
            if capture_mode in ("mic_vad", "mic_vad_nolevel"):
                wav_path, _rms, start_sample, end_sample = batch
                gap = max(0.0, (start_sample - prev_end_sample) / float(sample_rate))
                prev_end_sample = end_sample
            else:
                wav_path, _rms, gap = batch
            chunk_count += 1
            p = Path(wav_path)
            paths_to_delete.append(p)
            try:
                result = model.recognize(str(p))
                text = result if isinstance(result, str) else getattr(result, "text", str(result))
            except Exception as e:
                return EvalCaseResult(
                    case_id=case_id,
                    audio_path=str(audio_path),
                    reference_excerpt=ref_excerpt,
                    hypothesis="",
                    wer=1.0,
                    cer=1.0,
                    chunk_count=chunk_count,
                    error=f"recognize failed: {e}",
                    capture_mode=capture_mode,
                )
            text = (text or "").strip()
            if not text:
                segments.append(("", gap))
                continue
            segments.append((text, gap))

        hypothesis, _tail = join_transcription_items(segments, initial_tail="\n")
        wer = word_error_rate(reference_text, hypothesis)
        cer = char_error_rate(reference_text, hypothesis)
        return EvalCaseResult(
            case_id=case_id,
            audio_path=str(audio_path),
            reference_excerpt=ref_excerpt,
            hypothesis=hypothesis,
            wer=wer,
            cer=cer,
            chunk_count=chunk_count,
            error=None,
            capture_mode=capture_mode,
        )
    finally:
        for p in paths_to_delete:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def run_manifest(
    manifest_path: str | Path,
    *,
    model_id: str | None = None,
    leveler_settings: dict | None = None,
    temp_dir: str | None = None,
    capture_modes: tuple[str, ...] | None = None,
) -> list[EvalCaseResult]:
    """Run all cases in a manifest; model_id overrides per-case model when provided.

    capture_modes: e.g. ("mic", "loopback") to compare default mic vs loopback file replay.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent
    cases = load_cases_manifest(manifest_path)
    modes = capture_modes or ("mic", "loopback")
    results: list[EvalCaseResult] = []
    clear_transcription_model_cache()
    last_model = None
    for case in cases:
        cid = str(case.get("id", "")).strip() or f"case_{len(results)}"
        audio_rel = case.get("audio_file")
        if not audio_rel:
            for mode in modes:
                results.append(
                    EvalCaseResult(
                        case_id=cid,
                        audio_path="",
                        reference_excerpt="",
                        hypothesis="",
                        wer=1.0,
                        cer=1.0,
                        chunk_count=0,
                        error="missing audio_file",
                        capture_mode=mode,
                    )
                )
            continue
        audio_path = _resolve_case_asset(manifest_dir, audio_rel)
        if not audio_path.is_file():
            for mode in modes:
                results.append(
                    EvalCaseResult(
                        case_id=cid,
                        audio_path=str(audio_path),
                        reference_excerpt="",
                        hypothesis="",
                        wer=1.0,
                        cer=1.0,
                        chunk_count=0,
                        error="audio_file not found",
                        capture_mode=mode,
                    )
                )
            continue
        try:
            ref = _load_reference_text(manifest_dir, case)
        except ValueError as e:
            for mode in modes:
                results.append(
                    EvalCaseResult(
                        case_id=cid,
                        audio_path=str(audio_path),
                        reference_excerpt="",
                        hypothesis="",
                        wer=1.0,
                        cer=1.0,
                        chunk_count=0,
                        error=str(e),
                        capture_mode=mode,
                    )
                )
            continue
        mid = model_id or case.get("transcription_model")
        if last_model is not None and mid != last_model:
            clear_transcription_model_cache()
        last_model = mid
        cd = case.get("chunk_duration_sec")
        if isinstance(cd, (int, float)):
            cd = max(3.0, min(30.0, float(cd)))
        else:
            cd = None
        for mode in modes:
            results.append(
                run_single_case(
                    case_id=cid,
                    audio_path=audio_path,
                    reference_text=ref,
                    model_id=mid,
                    leveler_settings=leveler_settings or case.get("leveler_settings"),
                    temp_dir=temp_dir,
                    capture_mode=mode,
                    chunk_duration_sec=cd,
                )
            )
    return results


def summarize_results(results: list[EvalCaseResult]) -> dict:
    """Aggregate mean WER/CER over cases that succeeded, split by capture_mode."""
    ok = [r for r in results if r.error is None and r.chunk_count > 0]
    out: dict = {"cases": len(results), "ok": len(ok), "mean_wer": None, "mean_cer": None}
    if ok:
        out["mean_wer"] = sum(r.wer for r in ok) / len(ok)
        out["mean_cer"] = sum(r.cer for r in ok) / len(ok)
    by_mode: dict[str, list[EvalCaseResult]] = {}
    for r in ok:
        by_mode.setdefault(r.capture_mode, []).append(r)
    for mode, rs in sorted(by_mode.items()):
        out[f"mean_wer_{mode}"] = sum(x.wer for x in rs) / len(rs)
        out[f"mean_cer_{mode}"] = sum(x.cer for x in rs) / len(rs)
        out[f"ok_{mode}"] = len(rs)
    return out


def format_report(results: list[EvalCaseResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"=== {r.case_id} [{r.capture_mode}] ===")
        if r.error:
            lines.append(f"  ERROR: {r.error}")
            continue
        lines.append(f"  audio: {r.audio_path}")
        lines.append(f"  chunks: {r.chunk_count}")
        lines.append(f"  WER: {r.wer:.4f}  CER: {r.cer:.4f}")
        lines.append(f"  ref (excerpt): {r.reference_excerpt!r}")
        hyp_excerpt = normalize_for_compare(r.hypothesis)[:200]
        lines.append(f"  hyp (excerpt): {hyp_excerpt!r}")
        lines.append("")
    lines.append(str(summarize_results(results)))
    return "\n".join(lines)
