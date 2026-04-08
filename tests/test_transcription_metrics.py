"""Fast unit tests for transcript joining and WER/CER (no ASR model)."""
import pytest

from app.transcript_join import join_transcription_items
from app.transcription_metrics import char_error_rate, normalize_for_compare, word_error_rate


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


def test_wer_perfect():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_one_substitution():
    r = word_error_rate("hello world", "hallo world")
    assert 0 < r <= 1.0


def test_cer():
    assert char_error_rate("ab", "ab") == 0.0
    assert char_error_rate("ab", "ac") == pytest.approx(0.5)


def test_normalize():
    assert normalize_for_compare("  Hello  ") == "hello"
