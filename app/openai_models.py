"""
OpenAI model list: fetch, filter, cache, recommended list, and per-feature resolution.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

FALLBACK_MODEL = "gpt-5.2"
RECOMMENDED_COUNT = 5
CACHE_TTL_SEC = 86400
CUSTOM_OPTION = "Custom…"

_CHAT_BLACKLIST = (
    "embedding",
    "whisper",
    "tts",
    "transcribe",
    "realtime",
    "audio",
    "dall-e",
    "image",
    "moderation",
    "search-preview",
    "instruct",
    "codex",
    "base",
    "davinci",
    "babbage",
    "curie",
    "ada",
    "legacy",
    "unsup",
)

_REASONING_BLACKLIST = ("-chat-latest", "embedding", "tts", "whisper", "realtime")
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

_GPT_VERSION = re.compile(r"^gpt-(\d+(?:\.\d+)?)")
_SNAPSHOT_DATE = re.compile(r"-\d{4}-\d{2}-\d{2}")
# gpt-5.5-pro, gpt-5.4-pro, etc. are Responses API only (OpenAI model docs).
_PRO_MODEL = re.compile(r"-pro(?:-|$)")

_FEATURE_KEYS = {
    "summary": ("openai_model_summary", "openai_model_summary_explicit"),
    "ask": ("openai_model_ask", "openai_model_ask_explicit"),
    "export": ("openai_model_export", "openai_model_export_explicit"),
}


def _get_app_data_dir():
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings"


def get_cache_path():
    return _get_app_data_dir() / "openai_models_cache.json"


def is_chat_model(model_id):
    """True for model IDs that work with chat.completions (messages API)."""
    mid = (model_id or "").lower().strip()
    if not mid.startswith("gpt-"):
        return False
    if any(token in mid for token in _CHAT_BLACKLIST):
        return False
    if _PRO_MODEL.search(mid):
        return False
    # Mainline numbered GPT chat models (gpt-5.2, gpt-5.5, gpt-5.4-mini, …)
    if re.match(r"^gpt-\d+\.\d", mid):
        return True
    # Common chat families
    if mid.startswith("gpt-4o") or mid.startswith("gpt-4.1"):
        return True
    if mid.startswith("gpt-4-turbo"):
        return True
    if mid.startswith("gpt-3.5-turbo"):
        return True
    return False


def is_responses_api_only_model(model_id):
    """Pro-tier models listed in models.list but not valid for chat.completions."""
    mid = (model_id or "").lower().strip()
    return bool(mid.startswith("gpt-") and _PRO_MODEL.search(mid))


def supports_reasoning_effort(model_id):
    mid = (model_id or "").lower()
    if not mid or mid.startswith("ft:"):
        return False
    if any(token in mid for token in _REASONING_BLACKLIST):
        return False
    return any(mid.startswith(prefix) for prefix in _REASONING_PREFIXES)


def _gpt_sort_key(model_id):
    match = _GPT_VERSION.match(model_id or "")
    version = float(match.group(1)) if match else 0.0
    is_snapshot = bool(_SNAPSHOT_DATE.search(model_id or ""))
    return (version, 0 if is_snapshot else 1, model_id or "")


def sort_gpt_models(model_ids):
    gpt_ids = [m for m in model_ids if (m or "").startswith("gpt-") and is_chat_model(m)]
    return sorted(gpt_ids, key=_gpt_sort_key, reverse=True)


def _model_family_key(model_id):
    """Collapse dated snapshots to one family (gpt-5.5-pro-2026-04-23 → gpt-5.5-pro)."""
    return _SNAPSHOT_DATE.sub("", model_id or "")


def recommended_models(model_ids, n=RECOMMENDED_COUNT):
    out = []
    seen = set()
    for mid in sort_gpt_models(model_ids):
        key = _model_family_key(mid)
        if key in seen:
            continue
        seen.add(key)
        out.append(mid)
        if len(out) >= n:
            break
    return out


def recommended_default(model_ids):
    rec = recommended_models(model_ids, n=1)
    return rec[0] if rec else FALLBACK_MODEL


def load_cache():
    path = get_cache_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        model_ids = data.get("model_ids")
        if not isinstance(model_ids, list):
            return None
        fetched_at = data.get("fetched_at")
        if fetched_at is not None and not isinstance(fetched_at, (int, float)):
            fetched_at = None
        return {"fetched_at": fetched_at, "model_ids": [str(m) for m in model_ids if m]}
    except Exception:
        return None


def save_cache(model_ids, fetched_at=None):
    fetched_at = fetched_at if fetched_at is not None else time.time()
    path = get_cache_path()
    payload = {"fetched_at": fetched_at, "model_ids": list(model_ids)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(path)
        return True
    except Exception:
        return False


def fetch_models(api_key):
    try:
        from openai import OpenAI
    except ImportError:
        return [], "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        model_ids = [m.id for m in client.models.list() if getattr(m, "id", None)]
        return model_ids, None
    except Exception as e:
        return [], str(e)


def refresh_models(api_key, *, force=False):
    """
    Refresh cached model list. Returns (model_ids, recommended, error, from_cache, fetched_at).
    """
    if not (api_key or "").strip():
        return [FALLBACK_MODEL], [FALLBACK_MODEL], "No API key.", True, None

    cache = load_cache()
    now = time.time()
    cache_fresh = (
        cache
        and cache.get("fetched_at") is not None
        and (now - float(cache["fetched_at"])) < CACHE_TTL_SEC
    )
    need_fetch = force or not cache or not cache_fresh

    error = None
    from_cache = False
    fetched_at = cache.get("fetched_at") if cache else None
    model_ids = [m for m in (cache["model_ids"] if cache else []) if is_chat_model(m)]

    if need_fetch:
        fetched_ids, fetch_error = fetch_models(api_key)
        if fetch_error:
            error = fetch_error
            if cache:
                model_ids = [m for m in cache["model_ids"] if is_chat_model(m)]
                from_cache = True
                fetched_at = cache.get("fetched_at")
            else:
                model_ids = [FALLBACK_MODEL]
                from_cache = True
        else:
            chat_ids = [m for m in fetched_ids if is_chat_model(m)]
            model_ids = chat_ids if chat_ids else [FALLBACK_MODEL]
            fetched_at = now
            save_cache(model_ids, fetched_at)
            from_cache = False
    else:
        from_cache = True

    recommended = recommended_models(model_ids)
    if not recommended:
        recommended = [FALLBACK_MODEL]
    if FALLBACK_MODEL not in model_ids and FALLBACK_MODEL not in recommended:
        # Keep fallback selectable when API list omits it.
        pass
    return model_ids, recommended, error, from_cache, fetched_at


def apply_auto_upgrade(settings, new_default):
    if not (new_default or "").strip():
        return False
    changed = False
    for _feature, (model_key, explicit_key) in _FEATURE_KEYS.items():
        if settings.get(explicit_key):
            continue
        if settings.get(model_key) != new_default:
            settings[model_key] = new_default
            changed = True
    return changed


def resolve_model(settings, feature):
    model_key, _explicit_key = _FEATURE_KEYS.get(feature, (None, None))
    if not model_key:
        return FALLBACK_MODEL
    value = (settings or {}).get(model_key) or ""
    resolved = value.strip() or FALLBACK_MODEL
    if resolved.startswith("gpt-") and not is_chat_model(resolved):
        return FALLBACK_MODEL
    return resolved


def completion_kwargs(model, messages):
    kwargs = {"model": model, "messages": messages}
    if supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = "low"
    return kwargs


def is_unsupported_reasoning_error(exc):
    msg = (getattr(exc, "message", None) or str(exc)).lower()
    return "reasoning" in msg and ("unsupported" in msg or "not support" in msg)


def menu_values_for_model(recommended, current_model):
    values = list(recommended or [])
    current = (current_model or "").strip()
    if current and current not in values and current != CUSTOM_OPTION:
        values.append(current)
    if CUSTOM_OPTION not in values:
        values.append(CUSTOM_OPTION)
    return values
