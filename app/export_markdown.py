"""Markdown assembled for meeting export and clipboard copy.

File export and Copy to Clipboard share one body (AI summary, manual notes,
full transcript — each section only when it has text). Clipboard copy adds a
title from the meeting name, with the meeting date in front when the same
prepend option used for the export filename is on.
"""


def _clean(value) -> str:
    return (value or "").strip()


def build_export_body(summary: str = "", manual_notes: str = "", transcript: str = "") -> str:
    """Return the markdown written to an export file, or "" when nothing is present."""
    parts = []
    summary = _clean(summary)
    manual_notes = _clean(manual_notes)
    transcript = _clean(transcript)
    if summary:
        parts.append("# AI Summary\n\n" + summary)
    if manual_notes:
        parts.append("# Manual Notes\n\n" + manual_notes)
    if transcript:
        parts.append("# Full Transcript\n\n" + transcript)
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts) + "\n"


def build_clipboard_markdown(
    meeting_name: str = "",
    meeting_date: str = "",
    prepend_date: bool = False,
    summary: str = "",
    manual_notes: str = "",
    transcript: str = "",
) -> str:
    """Return export markdown plus a meeting-name title, or "" when there is no body.

    The date is placed in front of the name only when ``prepend_date`` is true
    and both the name and date are non-empty — the same condition as the
    export filename.
    """
    body = build_export_body(summary=summary, manual_notes=manual_notes, transcript=transcript)
    if not body:
        return ""
    name = _clean(meeting_name)
    if not name:
        return body
    date = _clean(meeting_date)
    if prepend_date and date:
        title = f"# {date} {name}"
    else:
        title = f"# {name}"
    return title + "\n\n" + body
