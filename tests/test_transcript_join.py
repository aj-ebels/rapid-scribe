"""Unit tests for transcript joining logic (production code in app/transcript_join.py)."""
from app.transcript_join import join_transcription_items


def test_join_paragraph_gap():
    text, tail = join_transcription_items(
        [("hello", 0.0), ("world", 7.0)],
        initial_tail="\n",
        paragraph_gap_sec=6.0,
    )
    assert "hello" in text and "world" in text
    assert "\n\n" in text
    assert tail == "d"


def test_join_space_within_paragraph():
    text, _ = join_transcription_items(
        [("hello", 0.0), ("world", 1.0)],
        initial_tail="\n",
        paragraph_gap_sec=6.0,
    )
    assert text == "hello world"
