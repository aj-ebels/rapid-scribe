"""
Developer-controlled audio capture pathway toggles.

These are the ONLY authoritative switches for chunking strategy and noise leveling.
They are not exposed in the UI; flip a value here, save the file, and restart the app.
Both Default input (mic) and Loopback always follow the same pathway — they stay in lockstep.

QUICK REFERENCE
---------------
  CHUNKING_MODE      "fixed"  slice audio into equal-length windows (best accuracy in eval)
                     "vad"    slice at detected pauses (lower latency, may clip words)

  CHUNK_DURATION_SEC  window length in seconds for "fixed" mode only — no effect on VAD

  VAD_MIN_CHUNK_SEC   (vad only) don't emit until at least this many seconds have built up
  VAD_MAX_CHUNK_SEC   (vad only) force-emit after this many seconds even with no pause
  VAD_SILENCE_SEC     (vad only) how long a quiet gap must be to trigger end of chunk

  LEVELING_ENABLED    True  = apply AGC + limiter to mic signal before transcription
                      False = raw mic (same processing as loopback, good for A/B)
"""
from __future__ import annotations

# ── Chunking strategy ─────────────────────────────────────────────────────────
#
#  "fixed"  Emit a new chunk every CHUNK_DURATION_SEC seconds (sliding window).
#           Produces consistent windows with no word clipping at silence boundaries.
#           Best accuracy in manual eval.
#
#  "vad"    Voice Activity Detection — listen for silence and use it as a natural
#           sentence boundary before sending audio to the transcription model.
#           Lower latency but aggressive short windows can clip words mid-phrase.
#           Tuned by VAD_MIN_CHUNK_SEC, VAD_MAX_CHUNK_SEC, VAD_SILENCE_SEC below.
#
CHUNKING_MODE: str = "vad"  # "fixed" | "vad"

# ── Fixed-window duration ─────────────────────────────────────────────────────
# How long each audio slice is when CHUNKING_MODE == "fixed".
# Has NO effect on VAD mode — use VAD_MAX_CHUNK_SEC for VAD's hard cap.
# Longer  → fewer cross-boundary cuts, but more lag before you see the transcript.
# Shorter → transcript appears sooner, but more chance a word gets cut in half.
# Valid range: 3.0 – 30.0 seconds.
CHUNK_DURATION_SEC: float = 7.0

# ── VAD (Voice Activity Detection) tuning ─────────────────────────────────────
# Only used when CHUNKING_MODE == "vad".
#
# VAD_MIN_CHUNK_SEC
#   Minimum audio to accumulate before a chunk can be emitted.
#   Prevents micro-chunks from very short pauses mid-sentence.
#   Example: 1.5 means "wait at least 1.5 s of speech before considering a cut."
#
# VAD_MAX_CHUNK_SEC
#   Hard cap: force-emit even if no silence has been detected yet.
#   Keeps the transcript from falling too far behind during continuous speech.
#   Example: 10.0 means "always send every 10 seconds at most."
#
# VAD_SILENCE_SEC
#   How many consecutive seconds of quiet audio trigger the end of a chunk.
#   Shorter = more responsive cuts; longer = waits longer for a "real" pause.
#   Example: 0.45 means a 450 ms gap in speech triggers a cut.
#
VAD_MIN_CHUNK_SEC: float = 5.0
VAD_MAX_CHUNK_SEC: float = 25.0
VAD_SILENCE_SEC:   float = 0.5

# ── Noise leveling ────────────────────────────────────────────────────────────
#
#  True   Apply AudioLeveler (automatic gain control + expander/limiter) to the
#         microphone signal before transcription. Helps quiet laptop mics and
#         very loud or reverberant rooms.
#         Used for: Default input (both fixed and VAD modes), Meeting mic channel.
#         Loopback is never leveled regardless of this setting.
#
#  False  Raw mic signal — same pipeline as the Loopback path.
#         Use this for direct A/B comparison with loopback, or when leveling
#         is making things worse (e.g., boosting background noise too aggressively).
#
LEVELING_ENABLED: bool = False
