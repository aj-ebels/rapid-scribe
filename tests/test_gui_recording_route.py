"""GUI delivery uses the recording-meeting route, not whichever box is on screen."""
import queue

from app import gui


class _Log:
    def __init__(self, text):
        self.text = text

    def get(self, _start, end):
        if end == "end-1c":
            return self.text
        return self.text + "\n"

    def insert(self, _where, text):
        self.text += text

    def see(self, _where):
        pass

    def delete(self, _start, _end):
        self.text = ""


def _meeting(mid, transcript, name):
    return {
        "id": mid,
        "meeting_name": name,
        "transcript": transcript,
        "manual_notes": "",
        "ai_summary": "",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def _app():
    app = type("App", (), {})()
    app.meetings = [_meeting("A", "Hello", "Meeting A"), _meeting("B", "Notes", "Meeting B")]
    app.recording_meeting_id = "A"
    app.current_meeting_id = "B"
    app.log = _Log("Notes")
    app._transcript_tail = "o"
    app._transcript_text_backlog = []
    app._transcript_typewriter_after_id = None
    app._transcript_typewriter_buffer = ""
    app._transcript_typewriter_index = 0
    app._transcript_typewriter_segment_tail = "\n"
    app._loading_meeting = False
    app._save_meetings_queue = queue.Queue()
    app.running = True
    app.settings = {"discreet_mode": False}
    return app


def test_incoming_text_is_stored_on_the_recording_meeting_while_another_is_shown():
    app = _app()
    app._transcript_text_backlog.append(("away", 0.2))
    gui._process_transcript_text_backlog(app)
    assert app.meetings[0]["transcript"] == "Hello away"
    assert app.log.text == "Notes"
    assert "away" not in app.log.text
    saved = app._save_meetings_queue.get_nowait()
    by_id = {m["id"]: m["transcript"] for m in saved}
    assert by_id["A"] == "Hello away"
    assert by_id["B"] == "Notes"
    assert gui.pulse_on_shown_transcript(
        running=True,
        discreet=False,
        recording_meeting_id=app.recording_meeting_id,
        shown_meeting_id=app.current_meeting_id,
    ) is False


def test_flush_while_viewing_another_meeting_does_not_write_the_visible_box():
    app = _app()
    app._transcript_typewriter_buffer = "REST"
    app._transcript_typewriter_segment_tail = "T"
    gui.flush_transcript_typewriter_immediate(app)
    assert app.log.text == "Notes"
    assert app.meetings[0]["transcript"] == "HelloREST"
    assert app._transcript_tail == "T"


def test_summary_copy_and_export_helper_reads_the_meeting_it_is_asked_for():
    app = _app()
    app.meetings[0]["transcript"] = "Hello away"
    # Shown meeting B: copy/export/summary of B stay on the box.
    assert gui._text_for_meeting_action(app, "B") == "Notes"
    # Auto-summary of the recording meeting reads what was saved, not the box.
    assert gui._text_for_meeting_action(app, "A") == "Hello away"


def test_showing_the_recording_meeting_again_animates_into_its_box():
    app = _app()
    app.current_meeting_id = "A"
    app.log = _Log("Hello")
    app._transcript_tail = "o"
    scheduled = []

    class _Root:
        def after(self, _ms, fn):
            scheduled.append(fn)
            return "timer"

        def after_cancel(self, _id):
            pass

    app.root = _Root()
    app._transcript_text_backlog.append(("there", 0.2))
    gui._process_transcript_text_backlog(app)
    # Join inserts a leading space; the first typewriter tick reveals that space
    # on Meeting A's box. The rest of the word stays on A's timer.
    assert app.log.text == "Hello "
    assert scheduled
    by_id = {m["id"]: m["transcript"] for m in app.meetings}
    assert by_id["B"] == "Notes"
    assert "there" not in by_id["B"]
