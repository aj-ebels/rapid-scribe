"""Live transcript text and the red pulse stay on the meeting that is recording."""
from app.recording_route import (
    DEST_DROP,
    DEST_STORE,
    DEST_WIDGET,
    action_transcript,
    deletion_stops_recording,
    must_flush_widget_before_leave,
    named_capture_status,
    pulse_on_shown_transcript,
    route_transcript_segment,
    start_recording_meeting,
    tail_after_clear,
    tail_after_show,
)


def _route(*, items, tail, recording, shown, stored, widget, exists=True):
    return route_transcript_segment(
        items=items,
        tail=tail,
        recording_meeting_id=recording,
        shown_meeting_id=shown,
        recording_transcript=stored,
        shown_widget_text=widget,
        recording_meeting_exists=exists,
    )


def test_transcript_and_pulse_stay_with_the_recording_meeting_across_switches():
    stored_a = "Hello from A."
    widget_b = "Notes for B"
    recording = start_recording_meeting("A")
    assert recording == "A"

    on_a = _route(
        items=[("More A.", 0.2)],
        tail=stored_a[-1],
        recording=recording,
        shown="A",
        stored=stored_a,
        widget=stored_a,
    )
    assert on_a.destination == DEST_WIDGET
    assert on_a.widget_text == "Hello from A. More A."
    assert on_a.recording_transcript == "Hello from A. More A."
    assert pulse_on_shown_transcript(
        running=True, discreet=False, recording_meeting_id=recording, shown_meeting_id="A"
    )
    assert must_flush_widget_before_leave(recording_meeting_id=recording, leaving_meeting_id="A")

    tail_on_b = tail_after_show(
        recording_meeting_id=recording,
        shown_transcript=widget_b,
        recording_tail=on_a.tail,
    )
    assert tail_on_b == on_a.tail
    assert tail_on_b != widget_b[-1]
    assert not pulse_on_shown_transcript(
        running=True, discreet=False, recording_meeting_id=recording, shown_meeting_id="B"
    )

    away = _route(
        items=[("Still A.", 0.2)],
        tail=tail_on_b,
        recording=recording,
        shown="B",
        stored=on_a.recording_transcript,
        widget=widget_b,
    )
    assert away.destination == DEST_STORE
    assert away.widget_text == "Notes for B"
    assert "Still A." not in away.widget_text
    assert away.recording_transcript == "Hello from A. More A. Still A."

    tail_back = tail_after_show(
        recording_meeting_id=recording,
        shown_transcript=away.recording_transcript,
        recording_tail=away.tail,
    )
    assert "Still A." in away.recording_transcript
    assert pulse_on_shown_transcript(
        running=True, discreet=False, recording_meeting_id=recording, shown_meeting_id="A"
    )

    cont = _route(
        items=[("And more.", 0.2)],
        tail=tail_back,
        recording=recording,
        shown="A",
        stored=away.recording_transcript,
        widget=away.recording_transcript,
    )
    assert cont.destination == DEST_WIDGET
    assert cont.widget_text == "Hello from A. More A. Still A. And more."
    assert cont.recording_transcript == cont.widget_text


def test_join_while_away_uses_the_recording_tail_not_the_meeting_on_screen():
    away = _route(
        items=[("there", 0.2)],
        tail="o",
        recording="A",
        shown="B",
        stored="hello",
        widget="xxx",
    )
    assert away.recording_transcript == "hello there"
    assert away.widget_text == "xxx"

    paragraph = _route(
        items=[("later", 7.0)],
        tail="o",
        recording="A",
        shown="B",
        stored="hello",
        widget="xxx",
    )
    assert paragraph.recording_transcript == "hello\n\nlater"
    assert paragraph.widget_text == "xxx"


def test_stop_while_viewing_another_meeting_keeps_text_and_actions_on_the_owner():
    away = _route(
        items=[("captured while away", 0.2)],
        tail=".",
        recording="A",
        shown="B",
        stored="Meeting A.",
        widget="Meeting B only",
    )
    assert away.recording_transcript == "Meeting A. captured while away"
    assert away.widget_text == "Meeting B only"
    assert not pulse_on_shown_transcript(
        running=False, discreet=False, recording_meeting_id="A", shown_meeting_id="A"
    )
    # Auto-summary of the recording, and export/copy of the meeting on screen.
    assert (
        action_transcript(
            subject_meeting_id="A",
            shown_meeting_id="B",
            widget_text=away.widget_text,
            stored_text=away.recording_transcript,
        )
        == "Meeting A. captured while away"
    )
    assert (
        action_transcript(
            subject_meeting_id="B",
            shown_meeting_id="B",
            widget_text=away.widget_text,
            stored_text=away.recording_transcript,
        )
        == "Meeting B only"
    )


def test_start_records_the_meeting_on_screen_and_names_it():
    assert start_recording_meeting("B") == "B"
    assert start_recording_meeting(None) is None
    assert start_recording_meeting("") is None
    status = named_capture_status("Recording & transcribing…", "Meeting B")
    assert status == "Recording & transcribing… (Meeting B)"
    on_b = _route(
        items=[("from B", 0.2)],
        tail="",
        recording="B",
        shown="B",
        stored="",
        widget="",
    )
    assert on_b.destination == DEST_WIDGET
    assert on_b.widget_text == "from B"
    switched = _route(
        items=[("still B", 0.2)],
        tail=on_b.tail,
        recording="B",
        shown="A",
        stored=on_b.recording_transcript,
        widget="A is visible",
    )
    assert switched.widget_text == "A is visible"
    assert switched.recording_transcript == "from B still B"


def test_new_meeting_mid_recording_does_not_receive_transcript_or_pulse():
    created = _route(
        items=[("keep going", 0.2)],
        tail=".",
        recording="A",
        shown="C",
        stored="Already.",
        widget="",
    )
    assert created.destination == DEST_STORE
    assert created.widget_text == ""
    assert created.recording_transcript == "Already. keep going"
    assert not pulse_on_shown_transcript(
        running=True, discreet=False, recording_meeting_id="A", shown_meeting_id="C"
    )


def test_rename_does_not_move_the_recording():
    assert named_capture_status("Meeting (mic + loopback) — recording…", "Renamed standup") == (
        "Meeting (mic + loopback) — recording… (Renamed standup)"
    )
    long_name = "N" * 50
    named = named_capture_status("Recording & transcribing…", long_name)
    assert named.endswith("...)")
    assert "N" * 50 not in named
    routed = _route(
        items=[("same id", 0.2)],
        tail="",
        recording="A",
        shown="B",
        stored="",
        widget="other",
    )
    assert routed.destination == DEST_STORE
    assert routed.recording_transcript == "same id"


def test_deleting_the_recording_meeting_stops_and_drops_later_text():
    assert deletion_stops_recording("A", "A") is True
    assert deletion_stops_recording("A", "B") is False
    assert deletion_stops_recording(None, "A") is False
    dropped = _route(
        items=[("orphan", 0.2)],
        tail="x",
        recording="A",
        shown="B",
        stored="gone",
        widget="stay",
        exists=False,
    )
    assert dropped.destination == DEST_DROP
    assert dropped.widget_text == "stay"
    assert dropped.recording_transcript == "gone"
    assert "orphan" not in dropped.widget_text


def test_clearing_another_meeting_does_not_reset_the_recording_tail():
    assert tail_after_clear(shown_meeting_id="B", recording_meeting_id="A", recording_tail="o") == "o"
    assert tail_after_clear(shown_meeting_id="A", recording_meeting_id="A", recording_tail="o") == ""
    assert tail_after_clear(shown_meeting_id="A", recording_meeting_id=None, recording_tail="o") == ""


def test_pulse_hides_in_incognito_even_on_the_recording_meeting():
    assert not pulse_on_shown_transcript(
        running=True, discreet=True, recording_meeting_id="A", shown_meeting_id="A"
    )
    assert pulse_on_shown_transcript(
        running=True, discreet=False, recording_meeting_id="A", shown_meeting_id="A"
    )


def test_edited_visible_transcript_wins_when_that_meeting_is_recording():
    routed = _route(
        items=[("tail", 0.2)],
        tail="o",
        recording="A",
        shown="A",
        stored="hello",
        widget="hello edited",
    )
    assert routed.widget_text == "hello edited tail"
    assert routed.recording_transcript == "hello edited tail"
