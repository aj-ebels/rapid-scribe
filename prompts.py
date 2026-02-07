"""
AI prompt templates: JSON storage, CRUD, placeholder for transcript insertion.
- Dev: read/write project prompts.json so edits show in the repo file.
- Frozen: read/write app-data prompts.json; bundled prompts.json (under _MEIPASS) is copied on first run.
"""
import json
import os
import sys
import uuid
from pathlib import Path

_BASE = Path(__file__).resolve().parent
TRANSCRIPT_PLACEHOLDER = "{{transcript}}"


def _get_app_data_dir():
    """Same as api_key_storage: per-user app data directory."""
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings"


def _get_user_prompts_path():
    """Path to the writable prompts file. In dev = project prompts.json; when frozen = app data."""
    if getattr(sys, "frozen", False):
        return _get_app_data_dir() / "prompts.json"
    return _BASE / "prompts.json"


def _get_bundled_prompts_path():
    """Path to the shipped default prompts. When frozen, bundle is under _MEIPASS (_internal)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "prompts.json"
    return _BASE / "prompts.json"


def load_prompts():
    """Load prompt templates. Uses app-data file; if missing, copies bundled default then loads."""
    user_path = _get_user_prompts_path()
    default_path = _get_bundled_prompts_path()
    if user_path.exists():
        pass  # load from user path
    elif default_path.exists():
        try:
            user_path.parent.mkdir(parents=True, exist_ok=True)
            user_path.write_bytes(default_path.read_bytes())
        except Exception:
            return _load_prompts_from_path(default_path)
    else:
        return []
    return _load_prompts_from_path(user_path)


def _load_prompts_from_path(path):
    """Load prompt list from a JSON file. Returns list of dicts with id, name, prompt."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_prompts(prompts):
    """Save prompt templates to the user app-data file."""
    path = _get_user_prompts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def add_prompt(name, prompt_text):
    """Add a new prompt template. Returns the new prompt dict."""
    prompts = load_prompts()
    p = {"id": str(uuid.uuid4()), "name": name, "prompt": prompt_text}
    prompts.append(p)
    save_prompts(prompts)
    return p


def update_prompt(prompt_id, name, prompt_text):
    """Update an existing prompt. Returns True if found."""
    prompts = load_prompts()
    for p in prompts:
        if p.get("id") == prompt_id:
            p["name"] = name
            p["prompt"] = prompt_text
            save_prompts(prompts)
            return True
    return False


def delete_prompt(prompt_id):
    """Delete a prompt by id. Returns True if found and deleted."""
    prompts = load_prompts()
    new_list = [p for p in prompts if p.get("id") != prompt_id]
    if len(new_list) == len(prompts):
        return False
    save_prompts(new_list)
    return True


def get_prompt_by_id(prompt_id):
    """Return prompt dict by id or None."""
    for p in load_prompts():
        if p.get("id") == prompt_id:
            return p
    return None
