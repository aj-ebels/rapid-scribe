"""
Where live transcript text and the recording pulse belong.

The transcript box is one widget. Recording is bound to the meeting that was
on screen when Start was pressed. Text and the red pulse stay with that
meeting when the user selects another one.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from .transcript_join import join_transcription_items

DEST_WIDGET = "widget"
DEST_STORE = "store"
DEST_DROP = "drop"

_PARAGRAPH_GAP_SEC = 6.0
_STATUS_NAME_MAX = 40


class RoutedSegment(NamedTuple):
    destination: str
    combined: str
    tail: str
    recording_transcript: str
    widget_text: str


def start_recording_meeting(shown_meeting_id: Optional[str]) -> Optional[str]:
    """Start records into the meeting currently on screen."""
    if not shown_meeting_id:
        return None
    return shown_meeting_id


def transcript_destination(
    *,
    recording_meeting_id: Optional[str],
    shown_meeting_id: Optional[str],
    recording_meeting_exists: bool,
) -> str:
    """Where the next transcript characters go.

    widget: the on-screen box is the recording meeting (or nothing is recording).
    store: another meeting is on screen; append to the recording meeting's transcript.
    drop: the recording meeting is gone; do not write into the meeting on screen.
    """
    if not recording_meeting_id:
        return DEST_WIDGET
    if not recording_meeting_exists:
        return DEST_DROP
    if recording_meeting_id == shown_meeting_id:
        return DEST_WIDGET
    return DEST_STORE


def route_transcript_segment(
    *,
    items: Sequence,
    tail: str,
    recording_meeting_id: Optional[str],
    shown_meeting_id: Optional[str],
    recording_transcript: str,
    shown_widget_text: str,
    recording_meeting_exists: bool,
    paragraph_gap_sec: float = _PARAGRAPH_GAP_SEC,
) -> RoutedSegment:
    """Join one batch of ASR items and assign the text to the recording meeting.

    When the recording meeting is on screen, the box is the source of truth
    (so an edit in the box is kept). Otherwise the stored transcript is.
    """
    stored = recording_transcript or ""
    widget = shown_widget_text or ""
    destination = transcript_destination(
        recording_meeting_id=recording_meeting_id,
        shown_meeting_id=shown_meeting_id,
        recording_meeting_exists=recording_meeting_exists,
    )
    combined, new_tail = join_transcription_items(
        items,
        initial_tail=tail if tail is not None else "",
        paragraph_gap_sec=paragraph_gap_sec,
    )
    if destination == DEST_DROP or not combined:
        kept_tail = (tail or "") if destination == DEST_DROP else (new_tail or "")
        return RoutedSegment(destination, "", kept_tail, stored, widget)
    if destination == DEST_STORE:
        return RoutedSegment(
            DEST_STORE,
            combined,
            new_tail or "",
            stored + combined,
            widget,
        )
    updated = widget + combined
    saved = updated if recording_meeting_id else stored
    return RoutedSegment(DEST_WIDGET, combined, new_tail or "", saved, updated)


def pulse_on_shown_transcript(
    *,
    running: bool,
    discreet: bool,
    recording_meeting_id: Optional[str],
    shown_meeting_id: Optional[str],
) -> bool:
    """Red pulse is drawn on the transcript box only while it shows the recording meeting."""
    if not running or discreet:
        return False
    return bool(recording_meeting_id) and recording_meeting_id == shown_meeting_id


def tail_after_show(
    *,
    recording_meeting_id: Optional[str],
    shown_transcript: str,
    recording_tail: str,
) -> str:
    """Join cursor after the transcript box switches meetings.

    During a recording the cursor belongs to that recording, including when
    its meeting comes back on screen. Otherwise it is the last character of
    the transcript now shown.
    """
    if recording_meeting_id:
        return recording_tail or ""
    text = shown_transcript or ""
    return text[-1] if text else ""


def tail_after_clear(
    *,
    shown_meeting_id: Optional[str],
    recording_meeting_id: Optional[str],
    recording_tail: str,
) -> str:
    """Clear wipes the box on screen. The recording cursor resets only for that box."""
    if recording_meeting_id and shown_meeting_id != recording_meeting_id:
        return recording_tail or ""
    return ""


def must_flush_widget_before_leave(
    *,
    recording_meeting_id: Optional[str],
    leaving_meeting_id: Optional[str],
) -> bool:
    """The box still holds uncommitted recording text and is about to be swapped."""
    return bool(recording_meeting_id) and recording_meeting_id == leaving_meeting_id


def deletion_stops_recording(
    recording_meeting_id: Optional[str],
    deleted_meeting_id: Optional[str],
) -> bool:
    """Deleting the meeting that is recording stops capture so later text is not reused."""
    return must_flush_widget_before_leave(
        recording_meeting_id=recording_meeting_id,
        leaving_meeting_id=deleted_meeting_id,
    )


def action_transcript(
    *,
    subject_meeting_id: Optional[str],
    shown_meeting_id: Optional[str],
    widget_text: str,
    stored_text: str,
) -> str:
    """Summary, export, copy, and Ask AI read the meeting they target.

    The meeting on screen is read from the box (it may have edits not yet
    stored). Any other meeting, including a recording in the background, is
    read from storage.
    """
    if subject_meeting_id and subject_meeting_id == shown_meeting_id:
        return widget_text or ""
    return stored_text or ""


def named_capture_status(loud: str, meeting_name: Optional[str]) -> str:
    """Non-incognito capture status names the meeting Start bound to."""
    name = (meeting_name or "").strip() or "New Meeting"
    if len(name) > _STATUS_NAME_MAX:
        name = name[: _STATUS_NAME_MAX - 3] + "..."
    return f"{loud} ({name})"
