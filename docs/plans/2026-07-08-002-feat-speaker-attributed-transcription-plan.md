# feat: Speaker-attributed transcription ("who said what")

**Created:** 2026-07-08
**Origin:** Research + brainstorm dialogue (2026-07-08). User goal: meeting transcript should show *who* is saying each part. Chosen direction: **local, offline** — Phase A (channel split) first, then Phase B (offline diarization). No cloud APIs.
**Depth:** Standard

---

## Summary

Add speaker labels to the meeting transcript in two stages, both fully offline:

- **Phase A — Channel attribution (live).** Capture already produces a stereo stream where **Left = your mic, Right = system/loopback audio** (`app/audio_mixer.py:7`). Today `ChunkRecorder` throws that separation away by downmixing to mono (`app/chunk_recorder.py:166-190`). Phase A preserves it: each chunk is tagged with its dominant source and rendered as **"Me"** (mic) vs **"Others"** (everyone on the call). Near-zero cost, no new models, works in real time.

- **Phase B — Offline diarization (at stop).** After recording ends, run a **sherpa-onnx** offline diarization pass over the retained session audio to split the loopback side into distinct remote speakers ("Speaker 1", "Speaker 2", …). Align diarized time segments against **word-level ASR timestamps** (`onnx-asr` `.with_timestamps()`), anchored by the Phase A channel split so the local user is never confused with remote voices. Optional, gated behind a setting and a one-time model download.

Phase A ships independently and delivers most of the value for 1-on-1 / "me + the call" meetings. Phase B layers multi-participant breakdown on top.

---

## Problem Frame

The transcript is a single undifferentiated blob. `transcription_worker` calls `model.recognize(path)` which returns plain text with no speaker or timing metadata (`app/transcription.py:539`), and `poll_text_queue` → `join_transcription_items` concatenates it into one scrolling log (`app/gui.py:242`, `app/transcript_join.py`). Users cannot tell who said what.

The pipeline downmixes stereo→mono *before* ASR (`_write_chunk`, `app/chunk_recorder.py:166-190`), discarding the one signal that trivially separates the local user from the remote call. And ASR output carries no timestamps, so there is nothing to align real diarization against.

**Success means:**
- Meeting-mode transcripts visibly attribute speech to a speaker (at minimum "Me" vs "Others"), live and in exports.
- With diarization enabled, remote participants are further split into stable per-speaker labels for the finished transcript.
- No regression to the existing mono ASR path; default/loopback modes and non-meeting flows are unaffected.
- Stays 100% offline; no new required network calls beyond optional model downloads through the existing Models tab.

---

## Requirements

| ID | Requirement |
|---|---|
| **Phase A** | |
| R1 | In Meeting mode, tag each emitted chunk with a source: `me` (mic dominant), `others` (loopback dominant), or `mixed` (both active). Reuse the mic/loopback RMS already computed in `_write_chunk` (`chunk_recorder.py:181-188`). |
| R2 | Carry the source tag end-to-end: `ChunkRecorder.on_chunk_ready` → `meeting_chunk_ready` → `chunk_queue` → `transcription_worker` → `text_queue`, without breaking the existing `(path, rms)` / bare-path item shapes used by default & loopback workers. |
| R3 | Render attribution in the transcript as speaker-prefixed blocks; start a new labeled block only when the speaker changes (avoid a label on every chunk). |
| R4 | Configurable display names: `local_speaker_name` (default "Me"), `remote_speaker_name` (default "Others"). Setting `speaker_labels_enabled` (default **on** for meeting mode) toggles the whole feature. |
| R5 | Speaker attribution appears in the on-disk transcript and in Markdown export. Backward-compatible: old meetings with a plain `transcript` still load & render. |
| **Phase B** | |
| R6 | Retain full-session 16 kHz mono audio during recording so a post-hoc diarization pass has something to run on. Written incrementally; deleted with the meeting unless the user keeps it. |
| R7 | On **Stop**, when `diarization_enabled` is on, run sherpa-onnx offline diarization (pyannote segmentation ONNX + speaker-embedding ONNX + clustering) in a subprocess/thread; show an "Identifying speakers…" status without freezing the GUI. |
| R8 | Obtain **word-level timestamps** by re-transcribing the retained session WAV once with `onnx_asr … .with_timestamps()`, producing the authoritative timed word list for alignment. |
| R9 | Align: assign each ASR word to the diarized segment covering its midpoint. Anchor with Phase A — mic-sourced audio is always the local user; only the loopback side is clustered into `Speaker N`. |
| R10 | Diarization config: `diarization_num_speakers` (`-1` = auto) and `diarization_cluster_threshold`, surfaced in Settings with sensible defaults. |
| R11 | Diarization models install/uninstall through the existing Models tab infrastructure (mirror `INT8_ONLY_REPOS` handling in `app/transcription.py`). Feature stays disabled & clearly labeled until models are present. |
| R12 | Store the diarized, speaker-attributed transcript in the meeting record as a new field; never overwrite the raw live transcript. |
| **Both** | |
| R13 | Unit tests (no GUI/Tk): channel-tag decision, speaker-block join/rendering, word→segment alignment, and the item-shape compatibility of the transcription queue. |

---

## Key Technical Decisions

| ID | Decision | Rationale |
|---|---|---|
| KTD1 | **Phase A = per-chunk dominant-source tag**, not dual-channel re-transcription | Chunks are short (1.5–8 s) and silence-based chunking tends to break at speaker pauses, so a chunk almost always belongs to one side. Reuses the existing `mic_rms`/`loop_rms` computation — no extra ASR cost, no new model. Dual-channel transcription (transcribe L and R separately) is the documented fallback if fidelity proves insufficient. |
| KTD2 | **Extend queue items to an optional 3rd element `(path, rms, source)`** | `transcription_worker` already parses items positionally (`app/transcription.py:491-492`); adding an optional index is backward-compatible with the bare-path and `(path, rms)` shapes from default/loopback workers. Avoids a schema/IPC redesign. |
| KTD3 | **Emit `(text, gap, speaker)` from the worker; extend `join_transcription_items` with a speaker param** | Keeps the join logic (space vs paragraph) in one place; a speaker change forces a new labeled block. `speaker=None` reproduces today's exact output, so non-meeting modes are untouched. |
| KTD4 | **sherpa-onnx for real diarization**, not pyannote/PyTorch or NeMo | Same ONNX runtime the app already ships (`onnx-asr`), CPU-capable, fully offline, small models. pyannote pulls in PyTorch + gated HF models — contrary to the app's lightweight ONNX ethos. |
| KTD5 | **Diarization is offline/post-stop, not streaming** | sherpa-onnx diarization is batch-only and clustering must see the whole recording to keep speaker IDs stable. Per-chunk diarization would relabel speakers every few seconds. Live view keeps Phase A labels; Phase B finalizes at stop. |
| KTD6 | **Re-transcribe the full session WAV with timestamps at stop** rather than accumulating per-chunk word timings live | One authoritative global timeline; avoids threading chunk offsets and per-chunk timestamp bookkeeping through the live path. Cost is a single extra ASR pass at stop, off the UI thread. |
| KTD7 | **Anchor diarization with the channel split (hybrid)** — only cluster the loopback channel | The local user is already cleanly isolated on L, which is the hardest case for blind diarization. Clustering only remote voices is easier and more accurate, and "Me" stays "Me". |
| KTD8 | **New meeting field `diarized_transcript` (+ `speaker_map`), keep raw `transcript`** | Non-destructive; old meetings still load. Mirrors how summary/notes are separate fields in `app/meetings_storage.py`. |
| KTD9 | **New settings as flat keys** (`speaker_labels_enabled`, `local_speaker_name`, `remote_speaker_name`, `diarization_enabled`, `diarization_num_speakers`, `diarization_cluster_threshold`) | Matches existing `app/settings.py` style; no nested-schema migration. |

---

## Prior Art & Lessons (2026-07-08 research)

Comparable open-source local meeting notetakers, and what each teaches us:

| Project | Stack | Lesson for Rapid Scribe |
|---|---|---|
| **[Meetily](https://github.com/Zackriya-Solutions/meetily)** — our acknowledged ancestor (`main.py` docstring cites it) | Parakeet/Whisper via whisper.cpp, **dual mic + system capture with "intelligent ducking & clipping prevention"**, Ollama/cloud summaries, Rust | Their ducking/clipping mix strategy is directly relevant to `audio_mixer.py` / `audio_leveler.py`. Their **diarization is still "coming soon"**, so shipping Phase B puts us *ahead* of the upstream — worth watching their PRs, as they'll hit the same whisper.cpp/Parakeet timestamp-alignment problems. |
| **[OpenWhispr](https://openwhispr.com/blog/local-speaker-diarization)** — shipped 100% local diarization (Apr 2026) | Local ONNX diarization + **speaker fingerprints / voice enrollment in a local SQLite file** | Most instructive for us. Their voiceprint store makes speaker identity **stable across meetings** ("Alice" stays "Alice"), not just per-session `Speaker N`. Motivates **Phase C** below. |
| **[Hyprnote](https://github.com/fastrepl/hyprnote)** (YC S25) | Tauri/Rust, local-first, own local Whisper server ("owhisper") | Broader UX/architecture reference for a polished local-first notetaker; internals largely undisclosed. |
| **[WhisperX](https://github.com/m-bain/whisperX)** — canonical diarization blueprint | faster-whisper → wav2vec2 forced alignment → pyannote → temporal word→speaker assignment | **Validates our Phase B design exactly.** Its explicit lesson: get accurate **word timestamps first**, *then* diarize, *then* assign each word to the segment its timestamp falls inside. That is precisely KTD6/R8/R9. |

**Library landscape considered:** **sherpa-onnx** (our pick — ONNX runtime we already ship, offline, CPU, proven in OpenWhispr-class apps); **pyannote.audio** (accuracy standard used by WhisperX, but PyTorch + gated HF models — keep only as a quality *benchmark* to measure our gap, per Risks); **diart** (streaming diarization — only relevant if we ever want *live* speaker labels, which sherpa-onnx can't do); **NeMo / 3D-Speaker / WeSpeaker** (sources of the embedding backbones sherpa-onnx consumes).

**Our differentiator:** nearly all of these mix mic + system audio to mono and rely on *blind* diarization to recover the local-vs-remote split. Rapid Scribe already captures **L=mic / R=loopback separately**, so Phase A's "Me vs Others" is free *and more reliable* than diarization for that distinction. Keeping the channel anchor (KTD7) while layering sherpa-onnx only on the remote side is a genuinely stronger design than a pure-diarization pipeline — none of the comparable apps do this.

---

## Implementation — Phase A (channel attribution, live)

**A1. `app/chunk_recorder.py` — tag the dominant source.**
In `_write_chunk`, the mic/loopback RMS is already computed for the adaptive downmix (`chunk_recorder.py:181-188`). Derive a `source` from those values with a hysteresis ratio (e.g. loopback ≪ mic → `me`; mic ≪ loopback → `others`; comparable → `mixed`), and pass it to the callback: `cb(wav_path, chunk_rms, source)`. Keep the mono downmix exactly as-is (ASR still runs on the mix, so `mixed` chunks transcribe fine).
- Make the callback signature tolerant: try the 3-arg call, fall back to 2-arg, so existing callers/tests don't break.

**A2. `app/capture.py` — `meeting_chunk_ready` forwards the source.**
Accept the optional `source` and enqueue `(wav_path, rms, source)` (falling back to the current shapes when absent). `capture_worker` / `capture_worker_loopback` are single-source and unchanged.

**A3. `app/transcription.py` — carry source through the worker.**
Parse an optional 3rd item element (`app/transcription.py:491`). On successful recognize, emit `(text, gap, source)` instead of `(text, gap)`. Non-meeting modes pass `source=None`.

**A4. `app/transcript_join.py` — speaker-aware joining.**
Extend `join_transcription_items` to accept `(text, gap, speaker)` tuples. When the speaker differs from the previous block, prefix a labeled header (e.g. `\n\n**Me:** `), resolving `me`/`others` through the configured display names; otherwise keep the current space/paragraph logic. `speaker=None` → byte-for-byte current behavior.

**A5. `app/gui.py` — thread speaker through display & save.**
`poll_text_queue` / `_process_transcript_text_backlog` already funnel items into `join_transcription_items` (`app/gui.py:242`, `200`); pass the configured names. Ensure `_transcript_tail` bookkeeping still holds with injected headers. Confirm the saved transcript (schedule-save path around `app/gui.py:1887`) captures the labeled text.

**A6. `app/settings.py` — Phase A settings + Settings-tab controls.**
Add `speaker_labels_enabled`, `local_speaker_name`, `remote_speaker_name` with defaults and load/validate. Add a small Settings section (mirroring existing toggles like `adaptive_audio_gating`).

**A7. Tests.** `tests/` — channel-tag thresholds (given mic/loop RMS pairs), speaker-block rendering & label-change boundaries, and `(path, rms, source)` round-trip compatibility in the worker item parser.

---

## Implementation — Phase B (offline diarization, at stop)

**B1. Retain session audio (`app/chunk_recorder.py`).**
Add an optional session-WAV writer that appends each chunk's 16 kHz mono samples to one growing file for the recording (guarded by `diarization_enabled`). Reuses samples already produced — no second capture path. Store path on the app for the stop handler.
- *Refinement (optional):* also retain the loopback-only channel for KTD7 anchoring; otherwise diarize the mono mix and reconcile with per-chunk `source` tags.

**B2. New module `app/diarization.py`.**
Wrap sherpa-onnx `OfflineSpeakerDiarization`: load segmentation + embedding ONNX models, run on the session WAV (16 kHz mono, which we already produce), return `[(start, end, speaker_id)]`. Config from KTD10 settings (`num_speakers`, `cluster_threshold`). Runs in a subprocess/thread (mirror `start_transcription_subprocess`).

**B3. Word timestamps (`app/transcription.py`).**
Add a helper that loads the model with `.with_timestamps()` and transcribes the full session WAV once, returning `[(word, start, end)]`. Used only in the stop-time pass, not the live loop.

**B4. Alignment (`app/diarization.py` or new `app/speaker_align.py`).**
Assign each word to the diarized segment containing its midpoint. Merge with Phase A: words from `me`-tagged spans → local name; remaining words → `Speaker N` from clustering. Emit the same `(text, gap, speaker)` stream shape so the existing renderer/join logic is reused.

**B5. Stop-time orchestration (`app/gui.py`).**
In the stop path (`start_stop` → `_shutdown_worker`, `app/gui.py:361-405`), after mixer/recorder shutdown and if `diarization_enabled`: set status "Identifying speakers…", run B2–B4 off the UI thread, then render the diarized transcript and persist it.

**B6. Storage (`app/meetings_storage.py`).**
Add `diarized_transcript` and `speaker_map` fields; keep `transcript`. Load path renders `diarized_transcript` when present, else falls back to `transcript` (`app/gui.py:1965`).

**B7. Model management (`app/transcription.py` + Models tab in `app/gui.py`).**
Add the two diarization repos to a diarization-model registry mirroring `INT8_ONLY_REPOS`/`INT8_ONLY_FILES` (`app/transcription.py:32-44`), with download/list/uninstall reusing `_download_url_to_file` and the Models-tab UI. Feature disabled with a clear hint until installed.

**B8. Settings (`app/settings.py`).**
Add `diarization_enabled` (default off until models present), `diarization_num_speakers` (`-1`), `diarization_cluster_threshold`; Settings-tab controls + an explanatory note that diarization runs after recording stops.

**B9. Tests.** Word→segment alignment (incl. boundary/overlap and midpoint rule), channel-anchored merge (mic word never labeled a remote speaker), and graceful degradation when models are missing (fall back to Phase A labels).

---

## Implementation — Phase C (voiceprint enrollment, cross-meeting identity) — *stretch*

Inspired by OpenWhispr: turn ephemeral per-session `Speaker N` labels into **persistent named identities**.

**C1. Local voiceprint store.** A local SQLite (or JSON) DB in the app-data dir (`_get_app_data_dir()`, `app/meetings_storage.py:16`) mapping speaker names → averaged embedding vectors. Nothing leaves the device.

**C2. Enrollment.** After a diarized meeting, let the user rename a `Speaker N` → a real name; persist that cluster's mean embedding under the name. Also allow explicit enrollment from a short sample.

**C3. Auto-labeling.** In the Phase B alignment step, before assigning `Speaker N`, compare each remote cluster's embedding against enrolled voiceprints (cosine similarity ≥ threshold) and use the matched name instead. Unmatched clusters fall back to `Speaker N`.

**C4. Settings/UI.** Manage enrolled speakers (list/rename/delete), similarity threshold, and an on/off toggle. Keep it optional — Phase B works fully without any enrollment.

Deferred until Phase B is validated; listed so the storage/embedding shapes in Phase B are designed with this in mind (retain per-cluster mean embeddings, not just labels).

---

## Testing & Verification

- `pytest` unit coverage per R13/B9 (logic only, no Tk).
- Manual: a 1-on-1 recording shows correct **Me/Others** live (Phase A). A ≥3-party call, after stop with diarization on, shows stable **Speaker N** splits on the remote side with "Me" preserved (Phase B).
- Regression: default and loopback modes, and playback of pre-existing meetings, are unchanged.
- Use the `verify` skill to drive the meeting flow end-to-end before committing Phase A.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Overlapping/cross-talk chunks mis-tagged in Phase A | `mixed` label + hysteresis; dual-channel transcription is the documented escalation (KTD1). |
| sherpa-onnx adds dependency weight / packaging complexity (PyInstaller) | ONNX-only, CPU wheels; validate against `meetings.spec` early; keep Phase B optional so Phase A ships regardless. |
| `onnx-asr` timestamp behavior varies by model/version | Verify `.with_timestamps()` on the shipped Parakeet v2/v3 int8 build before wiring B3; fall back to per-chunk offsets if unavailable. |
| Diarization CPU cost on long meetings | Off-thread at stop with progress UI; document expected wait; allow disabling. |
| Speaker labels pollute AI summary prompts | Confirm `{{transcript}}` substitution handles labeled text; likely a benefit (summaries can attribute quotes). |
| sherpa-onnx accuracy gap vs pyannote unknown | **Benchmark sherpa-onnx against pyannote on 2–3 real recordings before committing Phase B**, to quantify the accuracy we trade for the offline/lightweight win (WhisperX uses pyannote as its standard). |

---

## Open Questions

1. Default display names — "Me"/"Others", or prompt for the user's name once? *(Phase C voiceprint enrollment is the longer-term answer for remote names.)*
2. Should the retained session WAV (R6) be kept after diarization (re-run later / debugging / enrollment) or always deleted?
3. Auto-run diarization on every stop when enabled, or a manual **"Identify speakers"** button per meeting?
4. Rename `Speaker N` → real names post-hoc — one-off in `speaker_map` (Phase B), or persisted as voiceprints across meetings (Phase C)?

---

## Rollout

Ship **Phase A** first as a self-contained PR (small, offline, high value). Land **Phase B** as a follow-up once diarization models and packaging are validated — and after the sherpa-onnx-vs-pyannote accuracy benchmark. **Phase C** (voiceprint enrollment, OpenWhispr-style) is a later stretch; design Phase B's embedding storage so it can plug in. All on branch `claude/meeting-transcription-speaker-6vfz7w`.
