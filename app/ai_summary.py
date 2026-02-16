"""
AI summary: OpenAI API call with prompt template and transcript.
Rate limited to 1 call/5 seconds and 5 calls/minute.
"""
import re
import threading
import time

# Regex patterns to replace {{transcript}} and {{manual_notes}} (optional spaces inside braces)
_PLACEHOLDER_TRANSCRIPT = re.compile(r"\{\{\s*transcript\s*\}\}")
_PLACEHOLDER_MANUAL_NOTES = re.compile(r"\{\{\s*manual_notes\s*\}\}")

_recent_calls = []
_rate_limit_lock = threading.Lock()


def generate_ai_summary(api_key, prompt_template, transcript, manual_notes=""):
    """
    Call OpenAI API to generate summary.
    Returns (success, result_text_or_error).
    Rate limited to 1 call/5 seconds, 5 calls/minute.
    Replaces {{transcript}} and {{manual_notes}} in the prompt template.
    """
    if not (prompt_template or "").strip():
        return False, "Prompt template is empty."
    if not (transcript or "").strip():
        return False, "Transcript is empty."

    ok, err = _check_rate_limit()
    if not ok:
        return False, err

    # Replace {{transcript}} and {{manual_notes}} (regex allows optional spaces inside braces)
    # Use repl=lambda to avoid interpreting transcript/notes as regex backreferences
    text = _PLACEHOLDER_TRANSCRIPT.sub(lambda m: transcript, prompt_template)
    text = _PLACEHOLDER_MANUAL_NOTES.sub(lambda m: manual_notes or "", text)
    try:
        from openai import OpenAI
    except ImportError:
        return False, "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": text}],
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        return True, (content or "").strip()
    except Exception as e:
        return False, str(e)


def _check_rate_limit():
    """Shared rate limit: 1 call/5 seconds, 5 calls/minute. Returns (True, None) or (False, error_message)."""
    now = time.time()
    with _rate_limit_lock:
        global _recent_calls
        _recent_calls = [t for t in _recent_calls if now - t < 60]
        if len(_recent_calls) >= 5:
            return False, "Rate limit: maximum 5 AI calls per minute. Please try again later."
        if _recent_calls and (now - _recent_calls[-1]) < 5.0:
            wait_s = 5.0 - (now - _recent_calls[-1])
            return False, f"Rate limit: please wait {max(5, int(wait_s))} second(s) between calls."
        _recent_calls.append(now)
    return True, None


def _is_context_length_error(exc):
    """Return True if the exception is OpenAI context length exceeded."""
    msg = (getattr(exc, "message", None) or str(exc)).lower()
    return "context" in msg and ("length" in msg or "limit" in msg or "maximum" in msg)


def ask_meeting_ai(api_key, manual_notes, transcript, ai_summary, chat_messages, new_user_message):
    """
    Call OpenAI API to answer a question about the meeting.
    chat_messages: list of {"role": "user"|"assistant", "content": "..."} in order.
    new_user_message: the latest user query.
    Returns (success, result_text_or_error).
    Uses same rate limit as generate_ai_summary.
    Handles context_length_exceeded with a friendly error message.
    """
    if not (new_user_message or "").strip():
        return False, "Message is empty."

    ok, err = _check_rate_limit()
    if not ok:
        return False, err

    parts = []
    if (manual_notes or "").strip():
        parts.append("## Manual notes\n\n" + (manual_notes or "").strip())
    if (transcript or "").strip():
        parts.append("## Transcript\n\n" + (transcript or "").strip())
    if (ai_summary or "").strip():
        parts.append("## AI summary\n\n" + (ai_summary or "").strip())
    if not parts:
        return False, "Meeting has no content (manual notes, transcript, and AI summary are all empty)."

    context = "\n\n".join(parts)
    system_content = (
        "You are a helpful assistant answering questions about the following meeting. "
        "Use only the meeting content below to answer. If the answer is not in the content, say so.\n\n"
        + context
    )

    messages = [{"role": "system", "content": system_content}]
    for m in (chat_messages or []):
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": (new_user_message or "").strip()})

    try:
        from openai import OpenAI
    except ImportError:
        return False, "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        return True, (content or "").strip()
    except Exception as e:
        if _is_context_length_error(e):
            return False, "Meeting content is too long for the model. Try shortening the transcript or summary."
        return False, str(e)


def generate_export_name(api_key, summary_excerpt):
    """
    Call OpenAI API to generate a very concise export file name from a summary excerpt.
    summary_excerpt: typically the first 250 characters of the summary.
    Returns (success, name_or_error). On success, name is a short string like "ABC Industries call".
    Does not use the same rate limit as generate_ai_summary so it can be called immediately after.
    """
    excerpt = (summary_excerpt or "").strip()
    if not excerpt:
        return False, "Summary excerpt is empty."

    prompt = (
        "Generate a very concise (a few words maximum) file name (no extension) for this meeting summary. Examples: \"ABC Industries call\", \"conference debrief meeting\", \"product launch meeting\". Reply with only the suggested name, nothing else.\n\nSummary excerpt:\n" + excerpt[:250]
    )
    try:
        from openai import OpenAI
    except ImportError:
        return False, "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.2",
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return False, "No name returned."
        return True, content
    except Exception as e:
        return False, str(e)
