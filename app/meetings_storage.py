"""
Meeting instances: local JSON storage for meeting name, notes, transcript, summary, timestamps.
Uses same app-data directory as settings/prompts when frozen; project root in dev.
"""
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent


def _get_app_data_dir():
    """Per-user app data directory (same as prompts/settings)."""
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings"


def _get_meetings_path():
    """Path to meetings.json."""
    if getattr(sys, "frozen", False):
        return _get_app_data_dir() / "meetings.json"
    return _BASE / "meetings.json"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _today_iso_date():
    return date.today().isoformat()


def _normalize_meeting_date(value, fallback=None):
    """Return YYYY-MM-DD when value looks like a date/timestamp; else fallback/today."""
    if isinstance(value, str):
        v = value.strip()
        if v:
            try:
                # Preferred format: YYYY-MM-DD
                return date.fromisoformat(v).isoformat()
            except Exception:
                pass
            try:
                # Legacy/alternate format: full ISO datetime
                return datetime.fromisoformat(v.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                pass
    if isinstance(fallback, str) and fallback.strip():
        try:
            return datetime.fromisoformat(fallback.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            pass
    return _today_iso_date()


def load_meetings():
    """Load meetings from JSON. Returns list of meeting dicts sorted by updated_at descending."""
    path = _get_meetings_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    meetings = data.get("meetings", data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if not isinstance(meetings, list):
        return []
    # Sort by updated_at desc (most recent first)
    meetings = [m for m in meetings if isinstance(m, dict) and m.get("id")]
    for m in meetings:
        # Backfill meeting_date for older records; prefer created_at's calendar date.
        m["meeting_date"] = _normalize_meeting_date(m.get("meeting_date"), fallback=m.get("created_at"))
    meetings.sort(key=lambda m: m.get("updated_at") or m.get("created_at") or "", reverse=True)
    return meetings


def save_meetings(meetings):
    """Persist meetings list to JSON."""
    path = _get_meetings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meetings": meetings}, f, indent=2, ensure_ascii=False)


def create_meeting(meeting_name="New Meeting"):
    """Create a new meeting dict (does not persist). Call save_meetings after appending."""
    now = _now_iso()
    return {
        "id": str(uuid.uuid4()),
        "meeting_name": meeting_name or "New Meeting",
        "meeting_date": _today_iso_date(),
        "manual_notes": "",
        "transcript": "",
        "ai_summary": "",
        "ai_chat_messages": [],
        "created_at": now,
        "updated_at": now,
    }


def ensure_meeting_has_ai_chat_messages(meeting):
    """Ensure meeting has ai_chat_messages list. Modifies in place."""
    if "ai_chat_messages" not in meeting or not isinstance(meeting["ai_chat_messages"], list):
        meeting["ai_chat_messages"] = []


def append_ai_chat_message(meeting, role, content):
    """Append a message to the meeting's single AI chat. role is 'user' or 'assistant'. Does not persist."""
    ensure_meeting_has_ai_chat_messages(meeting)
    meeting["ai_chat_messages"].append({"role": role, "content": (content or "").strip()})
    meeting["updated_at"] = _now_iso()
    return True


def clear_ai_chat_messages(meeting):
    """Clear all messages in the meeting's AI chat. Does not persist."""
    ensure_meeting_has_ai_chat_messages(meeting)
    meeting["ai_chat_messages"].clear()
    meeting["updated_at"] = _now_iso()
    return True


def get_meeting_by_id(meetings, meeting_id):
    """Return meeting dict by id or None."""
    for m in meetings:
        if m.get("id") == meeting_id:
            return m
    return None


def update_meeting_fields(meeting, **fields):
    """Update meeting dict in place; set updated_at. Does not persist."""
    for k, v in fields.items():
        if k in ("id", "created_at"):
            continue
        meeting[k] = v
    meeting["updated_at"] = _now_iso()


def delete_meeting_by_id(meetings, meeting_id):
    """Remove meeting by id from list. Returns True if removed. Call save_meetings after."""
    for i, m in enumerate(meetings):
        if m.get("id") == meeting_id:
            meetings.pop(i)
            return True
    return False


def ensure_at_least_one_meeting(meetings):
    """If list is empty, append one 'New Meeting' and return it. Otherwise return None."""
    if not meetings:
        m = create_meeting("New Meeting")
        meetings.append(m)
        return m
    return None
