"""Clipboard and file export share one markdown body."""
from app.export_markdown import build_clipboard_markdown, build_export_body


def test_export_body_orders_present_sections_and_skips_blanks():
    text = build_export_body(summary="Sum", manual_notes="  ", transcript="Talk")
    assert text == "# AI Summary\n\nSum\n\n---\n\n# Full Transcript\n\nTalk\n"


def test_export_body_includes_every_present_section():
    text = build_export_body(summary="Sum", manual_notes="Notes", transcript="Talk")
    assert text == (
        "# AI Summary\n\nSum\n\n---\n\n"
        "# Manual Notes\n\nNotes\n\n---\n\n"
        "# Full Transcript\n\nTalk\n"
    )


def test_export_body_empty_when_nothing_present():
    assert build_export_body(summary="", manual_notes=" \n", transcript=None) == ""


def test_clipboard_prepends_date_to_meeting_name():
    text = build_clipboard_markdown(
        meeting_name="Weekly sync",
        meeting_date="2026-10-14",
        prepend_date=True,
        summary="Sum",
        manual_notes="Notes",
        transcript="",
    )
    assert text == (
        "# 2026-10-14 Weekly sync\n\n"
        "# AI Summary\n\nSum\n\n---\n\n"
        "# Manual Notes\n\nNotes\n"
    )


def test_clipboard_uses_name_only_when_date_box_is_off():
    text = build_clipboard_markdown(
        meeting_name="Weekly sync",
        meeting_date="2026-10-14",
        prepend_date=False,
        summary="",
        manual_notes="",
        transcript="Talk",
    )
    assert text == "# Weekly sync\n\n# Full Transcript\n\nTalk\n"


def test_clipboard_omits_title_when_name_is_blank():
    text = build_clipboard_markdown(
        meeting_name="   ",
        meeting_date="2026-10-14",
        prepend_date=True,
        summary="Sum",
    )
    assert text == "# AI Summary\n\nSum\n"


def test_clipboard_keeps_name_when_date_is_blank_even_if_prepend_is_on():
    text = build_clipboard_markdown(
        meeting_name="Standup",
        meeting_date="",
        prepend_date=True,
        transcript="Talk",
    )
    assert text.startswith("# Standup\n\n")


def test_clipboard_empty_when_no_sections():
    assert build_clipboard_markdown(meeting_name="Standup", meeting_date="2026-10-14", prepend_date=True) == ""
