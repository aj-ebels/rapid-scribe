"""
AI prompt templates: JSON storage, CRUD, placeholder for transcript insertion.
"""
import json
import uuid
from pathlib import Path

_BASE = Path(__file__).resolve().parent
PROMPTS_FILE = _BASE / "prompts.json"
TRANSCRIPT_PLACEHOLDER = "{{transcript}}"


def load_prompts():
    """Load prompt templates from JSON file. Returns list of dicts with id, name, prompt."""
    if not PROMPTS_FILE.exists():
        return []
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_prompts(prompts):
    """Save prompt templates to JSON file."""
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
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
