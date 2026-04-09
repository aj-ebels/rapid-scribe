#!/usr/bin/env python3
"""
Automated VAD parameter sweep for transcription accuracy.

Feeds a benchmark WAV file through the same inline VAD code used by the live app's
loopback capture path, testing many parameter combinations without manual re-recording.
Results are saved to fixtures/manual_eval/results/ and the comparison table is printed.

The VAD logic here is a verbatim copy of the inline VAD in capture_worker_loopback
(app/capture.py). If you change that code, update the _run_vad function here too.

Usage
-----
    python scripts/run_vad_scenarios.py                     # run all scenarios
    python scripts/run_vad_scenarios.py --skip-existing     # skip already-saved results
    python scripts/run_vad_scenarios.py --wav path/to.wav   # use a different WAV
    python scripts/run_vad_scenarios.py --dry-run           # print what would run, exit

Scenario format (see SCENARIOS below)
--------------------------------------
Each entry is a dict with:
    label   str   Filename stem for results/  (no spaces; use underscores)
    mode    str   "vad" or "fixed"
    min     float (vad only)  VAD_MIN_CHUNK_SEC
    max     float (vad only)  VAD_MAX_CHUNK_SEC
    sil     float (vad only)  VAD_SILENCE_SEC
    dur     float (fixed only) CHUNK_DURATION_SEC
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample as scipy_resample

# ---------------------------------------------------------------------------
# Scenarios — edit this list to define which combinations to sweep
# ---------------------------------------------------------------------------

_min_set = [2, 3, 4, 5, 6, 7]
_max_set = [10, 15, 20, 25, 30]
_sil_set = [0.25, 0.50, 0.75, 1.00]

SCENARIOS: list[dict] = [
    # ── Fixed window baselines ───────────────────────────────────────────
    dict(label="fixed_5s",  mode="fixed", dur=5.0),
    dict(label="fixed_7s",  mode="fixed", dur=7.0),
    dict(label="fixed_10s", mode="fixed", dur=10.0),

    # ── Full VAD grid ────────────────────────────────────────────────────
    *[
        dict(label=f"vad_{mn}_{mx}_{sl}", mode="vad", min=float(mn), max=float(mx), sil=sl)
        for mn in _min_set
        for mx in _max_set
        for sl in _sil_set
    ],
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WAV = _ROOT / "samples" / "benchmark_sample.wav"
_RESULTS_DIR = _ROOT / "fixtures" / "manual_eval" / "results"

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
SILENCE_RMS_THRESHOLD = 0.005
_VAD_BLOCK_SIZE = 512
_VAD_HANGOVER_BLOCKS = 5


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _load_wav_16k_mono(path: Path) -> np.ndarray:
    """Load any WAV to 16 kHz mono float32."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / 32768.0
    if sr != SAMPLE_RATE:
        n_out = int(round(len(data) * SAMPLE_RATE / sr))
        data = scipy_resample(data, n_out).astype(np.float32)
    return np.clip(data, -1.0, 1.0)


def _run_vad(audio: np.ndarray, min_sec: float, max_sec: float, sil_sec: float) -> list[np.ndarray]:
    """
    Process audio through the same inline VAD as capture_worker_loopback (VAD mode).
    Returns list of float32 chunks ready to be written to WAV and transcribed.

    *** Keep this in sync with the inline VAD in app/capture.py ***
    """
    min_frames = int(SAMPLE_RATE * max(0.5, min_sec))
    max_frames = int(SAMPLE_RATE * max(1.0, max_sec))
    sil_frames = int(SAMPLE_RATE * max(0.1, sil_sec))

    buf: list[np.ndarray] = []
    buf_frames = 0
    consecutive_silent = 0
    noise_floor = SILENCE_RMS_THRESHOLD
    hangover_left = 0
    chunks: list[np.ndarray] = []

    for i in range(0, len(audio), _VAD_BLOCK_SIZE):
        block = audio[i : i + _VAD_BLOCK_SIZE]
        if block.size == 0:
            continue
        block_rms = _rms(block)

        learn_upper = max(SILENCE_RMS_THRESHOLD * 5.0, noise_floor * 1.7)
        if block_rms <= learn_upper:
            noise_floor = 0.96 * noise_floor + 0.04 * block_rms
        adaptive_threshold = max(SILENCE_RMS_THRESHOLD, noise_floor * 2.5)

        is_silent = block_rms < adaptive_threshold
        if is_silent:
            if hangover_left > 0:
                hangover_left -= 1
                is_silent = False
        else:
            hangover_left = _VAD_HANGOVER_BLOCKS
            consecutive_silent = 0

        buf.append(block)
        buf_frames += len(block)
        if is_silent:
            consecutive_silent += len(block)

        if buf_frames >= max_frames:
            chunks.append(np.concatenate(buf).astype(np.float32))
            buf, buf_frames, consecutive_silent = [], 0, 0
        elif buf_frames >= min_frames and consecutive_silent >= sil_frames:
            chunks.append(np.concatenate(buf).astype(np.float32))
            buf, buf_frames, consecutive_silent = [], 0, 0

    if buf_frames > 0:
        chunks.append(np.concatenate(buf).astype(np.float32))

    return [c for c in chunks if _rms(c) >= SILENCE_RMS_THRESHOLD * 0.5]


def _run_fixed(audio: np.ndarray, duration_sec: float) -> list[np.ndarray]:
    """Slice audio into fixed-duration windows (same logic as capture_worker_fixed)."""
    chunk_samples = int(SAMPLE_RATE * max(3.0, duration_sec))
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        if _rms(chunk) >= SILENCE_RMS_THRESHOLD:
            chunks.append(chunk.astype(np.float32))
    return chunks


def _chunks_to_transcript(chunks: list[np.ndarray], model) -> str:
    """Write each chunk to a temp WAV, transcribe, join."""
    temp_dir = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "MeetingsSweep")
    os.makedirs(temp_dir, exist_ok=True)
    texts: list[str] = []
    for chunk in chunks:
        wav_path = os.path.join(temp_dir, f"chunk_{uuid.uuid4().hex}.wav")
        try:
            audio_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
            result = model.recognize(wav_path)
            text = result if isinstance(result, str) else getattr(result, "text", str(result))
            if text and text.strip():
                texts.append(text.strip())
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def _scenario_header(s: dict) -> str:
    if s["mode"] == "vad":
        return (
            f"# VAD_MIN_CHUNK_SEC: float = {s['min']}\n"
            f"# VAD_MAX_CHUNK_SEC: float = {s['max']}\n"
            f"# VAD_SILENCE_SEC: float = {s['sil']}\n"
            f"# mode: automated (vad inline replay)\n"
        )
    else:
        return (
            f"# CHUNK_DURATION_SEC: float = {s['dur']}\n"
            f"# mode: automated (fixed window replay)\n"
        )


def run_scenario(s: dict, audio: np.ndarray, model) -> str:
    if s["mode"] == "vad":
        chunks = _run_vad(audio, s["min"], s["max"], s["sil"])
    else:
        chunks = _run_fixed(audio, s["dur"])
    transcript = _chunks_to_transcript(chunks, model)
    return transcript, len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Automated VAD parameter sweep.")
    parser.add_argument("--wav", default=str(_DEFAULT_WAV), help="WAV file to replay.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenarios whose result file already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scenario list and exit without running.")
    args = parser.parse_args(argv)

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        print(f"ERROR: WAV file not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"WAV: {wav_path}")
        print(f"Results dir: {_RESULTS_DIR}\n")
        print(f"{'#':>3}  {'Label':<22}  {'Mode':<6}  Params")
        print("  " + "-" * 60)
        for i, s in enumerate(SCENARIOS, 1):
            if s["mode"] == "vad":
                params = f"min={s['min']}  max={s['max']}  sil={s['sil']}"
            else:
                params = f"dur={s['dur']}"
            result_path = _RESULTS_DIR / f"{s['label']}.txt"
            exists = " (exists)" if result_path.exists() else ""
            print(f"  {i:>3}  {s['label']:<22}  {s['mode']:<6}  {params}{exists}")
        sys.exit(0)

    # ── Load audio ──────────────────────────────────────────────────────
    print(f"\nLoading WAV: {wav_path}")
    audio = _load_wav_16k_mono(wav_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"  {duration:.1f}s at {SAMPLE_RATE} Hz mono")

    # ── Load transcription model (once) ─────────────────────────────────
    sys.path.insert(0, str(_ROOT))
    print("Loading transcription model (first time may download)...")
    from app.transcription import get_transcription_model
    model = get_transcription_model()
    print("  Model ready.\n")

    # ── Run scenarios ────────────────────────────────────────────────────
    total = len(SCENARIOS)
    skipped = 0
    ran = 0

    for i, s in enumerate(SCENARIOS, 1):
        label = s["label"]
        result_path = _RESULTS_DIR / f"{label}.txt"

        if args.skip_existing and result_path.is_file():
            print(f"  [{i:>2}/{total}] SKIP  {label} (file exists)")
            skipped += 1
            continue

        if s["mode"] == "vad":
            params_str = f"min={s['min']}s  max={s['max']}s  sil={s['sil']}s"
        else:
            params_str = f"dur={s['dur']}s"
        print(f"  [{i:>2}/{total}] {label:<24} {params_str} ...", end="", flush=True)

        transcript, n_chunks = run_scenario(s, audio, model)

        header = _scenario_header(s)
        header += f"# chunks: {n_chunks}\n"
        result_path.write_text(header + "\n" + transcript + "\n", encoding="utf-8")

        print(f" {n_chunks} chunks, {len(transcript.split())} words")
        ran += 1

    print(f"\nDone. {ran} ran, {skipped} skipped.")

    # ── Print comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    try:
        from scripts.compare_transcripts import main as compare_main
        compare_main([])
    except Exception as e:
        print(f"Could not run comparison: {e}")
        print(f"Run manually: python scripts/compare_transcripts.py")


if __name__ == "__main__":
    main()
