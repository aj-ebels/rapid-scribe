#!/usr/bin/env python3
"""
Manual transcription accuracy comparison.

Workflow
--------
1. Edit app/dev_config.py for the combination you want to test:
       CHUNKING_MODE  = "fixed"   # or "vad"
       LEVELING_ENABLED = True    # or False

2. Open the app, speak the script in fixtures/manual_eval/reference.txt, stop recording.

3. Copy the full transcript from the app (Transcript tab → select all → copy).

4. Create a new .txt file in fixtures/manual_eval/results/ and paste.
   Name it to describe the config, e.g.:
       fixed_leveled.txt
       fixed_noleveler.txt
       vad_leveled.txt
       vad_noleveler.txt

   Optional metadata header (lines starting with #, must be at the top):
       # config: CHUNKING_MODE=fixed LEVELING_ENABLED=True
       # date: 2026-04-09
       # notes: quiet room, Surface mic

5. Run this script:
       python scripts/compare_transcripts.py

   Optional flags:
       --ref  path/to/other_reference.txt   (default: fixtures/manual_eval/reference.txt)
       --dir  path/to/results/              (default: fixtures/manual_eval/results)
       --json                               emit JSON to stdout instead of text table
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Spoken-form equivalences
#
# Each group is a tuple: (canonical_form, alt1, alt2, ...).
# After basic normalization both reference and hypothesis have every alternate
# form replaced by the canonical form before WER/CER is computed.
# That means "50%" and "fifty percent" score as identical.
#
# Rules:
#   - All forms are lowercase with no punctuation (matches post-normalize state).
#   - Longer phrases must appear before shorter ones that share words — the list
#     is automatically sorted longest-first so you can add in any order.
#   - Add new groups here whenever you see a test result where the model wrote a
#     number/symbol differently from the spoken reference and you consider both
#     acceptable.
# ---------------------------------------------------------------------------

SPOKEN_EQUIVALENTS: list[tuple[str, ...]] = [
    # Percentages
    ("fifty percent",       "50 percent", "50"),     # "50%" → "50" after punct strip

    # Years
    ("twenty twenty six",   "2026"),

    # Ordinals used in dates
    ("seventh",             "7th"),                  # covers "april seventh" / "april 7th"

    # Versions / decimals
    ("three point oh",      "three point zero", "3 0", "3"),  # "3.0" → "3 0" after strip

    # Room designations
    ("twelve b",            "12 b", "12b"),

    # Spoken file names / formats
    ("dot wav",             "dot wave"),              # model often says "wave"
]

def _build_equiv_subs(
    groups: list[tuple[str, ...]],
) -> list[tuple[re.Pattern, str]]:
    """Compile one regex per alternate form, longest match first."""
    subs: list[tuple[re.Pattern, str]] = []
    for group in groups:
        canonical = group[0]
        for alt in group[1:]:
            # Word-boundary aware: match only complete tokens.
            pattern = re.compile(r"(?<!\w)" + re.escape(alt) + r"(?!\w)")
            subs.append((pattern, canonical))
    # Sort: longer patterns first to prevent partial substitution.
    subs.sort(key=lambda x: -len(x[0].pattern))
    return subs

_EQUIV_SUBS = _build_equiv_subs(SPOKEN_EQUIVALENTS)


def _spoken_normalize(text: str) -> str:
    """Basic normalization then apply spoken-form equivalences."""
    text = _normalize(text)
    for pattern, canonical in _EQUIV_SUBS:
        text = pattern.sub(canonical, text)
    return text


# ---------------------------------------------------------------------------
# WER / CER helpers (no external dependencies)
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein distance on arbitrary sequences."""
    m, n = len(a), len(b)
    # Use two-row rolling array for O(n) space.
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _spoken_normalize(reference).split()
    hyp_words = _spoken_normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(_spoken_normalize(reference))
    hyp_chars = list(_spoken_normalize(hypothesis))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# Result file parsing
# ---------------------------------------------------------------------------

def _parse_result_file(path: Path) -> tuple[dict, str]:
    """Return (metadata_dict, transcript_text)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    meta: dict = {}
    body_lines: list[str] = []
    in_header = True
    for line in lines:
        if in_header and line.startswith("#"):
            # Try to parse   # key: value
            m = re.match(r"#\s*(\w+)\s*:\s*(.+)", line)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
        else:
            in_header = False
            body_lines.append(line)
    return meta, "\n".join(body_lines).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare manually captured transcripts against a reference script."
    )
    parser.add_argument(
        "--ref",
        default=str(_root() / "fixtures" / "manual_eval" / "reference.txt"),
        help="Reference (ground-truth) script file.",
    )
    parser.add_argument(
        "--dir",
        default=str(_root() / "fixtures" / "manual_eval" / "results"),
        help="Folder containing captured transcript .txt files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable table.",
    )
    args = parser.parse_args(argv)

    ref_path = Path(args.ref)
    results_dir = Path(args.dir)

    if not ref_path.is_file():
        print(f"ERROR: reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    reference = ref_path.read_text(encoding="utf-8").strip()

    result_files = sorted(
        p for p in results_dir.glob("*.txt") if p.name != ".gitkeep"
    )
    if not result_files:
        print(
            f"No result files found in {results_dir}\n"
            "Paste a transcript into a .txt file there and re-run.",
            file=sys.stderr,
        )
        sys.exit(0)

    rows = []
    for path in result_files:
        meta, hypothesis = _parse_result_file(path)
        if not hypothesis:
            print(f"  Skipping {path.name} — empty transcript.", file=sys.stderr)
            continue
        wer = word_error_rate(reference, hypothesis)
        cer = char_error_rate(reference, hypothesis)
        rows.append(
            {
                "label": path.stem,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "meta": meta,
                "hypothesis": hypothesis,
            }
        )

    rows.sort(key=lambda r: (r["wer"], r["cer"], r["label"]))

    if args.json:
        out = {
            "reference_file": str(ref_path),
            "reference_excerpt": reference[:200],
            "ranked": [
                {
                    "rank": i + 1,
                    "label": r["label"],
                    "wer": r["wer"],
                    "cer": r["cer"],
                    "meta": r["meta"],
                }
                for i, r in enumerate(rows)
            ],
            "results": rows,
        }
        print(json.dumps(out, indent=2))
        return

    # ── Text table ──────────────────────────────────────────────────────────
    width = 72
    bar = "-" * width

    print()
    print(f"Reference:  {ref_path}")
    ref_words = _normalize(reference).split()
    print(f"            {len(ref_words)} words · {len(reference)} chars")
    print(bar)
    print()

    if not rows:
        print("No valid results to compare.")
        return

    def _num(meta: dict, key: str) -> str:
        """Extract numeric portion from a meta value like 'float = 4.0' or '4.0'."""
        raw = meta.get(key, "")
        if not raw:
            return ""
        m = re.search(r"[\d.]+", raw)
        return m.group() if m else raw

    def _mode(meta: dict) -> str:
        raw = meta.get("mode", "")
        if "vad" in raw:
            return "vad"
        if "fixed" in raw:
            return "fixed"
        config = meta.get("config", "")
        if "vad" in config.lower():
            return "vad"
        if "fixed" in config.lower():
            return "fixed"
        return raw[:6] if raw else ""

    print("Ranked by WER  (lower = better)\n")
    col_label = max(len(r["label"]) for r in rows)
    header = (
        f"  {'#':>3}  {'Label':<{col_label}}"
        f"  {'WER':>6}  {'CER':>6}  {'Ch':>3}"
        f"  {'Mode':<5}  {'Min':>5}  {'Max':>5}  {'Sil':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(rows, start=1):
        m = r["meta"]
        chunks_str  = f"{m.get('chunks', '?'):>3}"
        mode_str    = f"{_mode(m):<5}"
        min_str     = f"{_num(m, 'VAD_MIN_CHUNK_SEC'):>5}"
        max_str     = f"{_num(m, 'VAD_MAX_CHUNK_SEC'):>5}"
        sil_str     = f"{_num(m, 'VAD_SILENCE_SEC'):>5}"
        print(
            f"  {i:>3}  {r['label']:<{col_label}}"
            f"  {r['wer']:>6.4f}  {r['cer']:>6.4f}  {chunks_str}"
            f"  {mode_str}  {min_str}  {max_str}  {sil_str}"
        )

    print()
    print(bar)

    # ── VAD parameter summary tables ─────────────────────────────────────────
    vad_rows = [r for r in rows if _mode(r["meta"]) == "vad"]
    if vad_rows:
        def _avg_table(title: str, key: str) -> None:
            from collections import defaultdict
            groups: dict[str, list] = defaultdict(list)
            for r in vad_rows:
                val = _num(r["meta"], key)
                if val:
                    groups[val].append(r)
            if not groups:
                return
            print(f"\n  By {key}")
            print(f"  {'Value':>6}  {'Avg WER':>8}  {'Avg CER':>8}  {'N':>4}  {'Best WER':>9}")
            print("  " + "-" * 44)
            for val in sorted(groups, key=lambda x: float(x)):
                bucket = groups[val]
                avg_wer = sum(r["wer"] for r in bucket) / len(bucket)
                avg_cer = sum(r["cer"] for r in bucket) / len(bucket)
                best_wer = min(r["wer"] for r in bucket)
                print(f"  {val:>6}  {avg_wer:>8.4f}  {avg_cer:>8.4f}  {len(bucket):>4}  {best_wer:>9.4f}")

        print()
        print("VAD Parameter Averages  (vad-mode results only)\n")
        _avg_table("VAD_MIN_CHUNK_SEC", "VAD_MIN_CHUNK_SEC")
        _avg_table("VAD_MAX_CHUNK_SEC", "VAD_MAX_CHUNK_SEC")
        _avg_table("VAD_SILENCE_SEC",   "VAD_SILENCE_SEC")
        print()
        print(bar)


if __name__ == "__main__":
    main()
