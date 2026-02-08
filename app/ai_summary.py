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

    now = time.time()
    with _rate_limit_lock:
        global _recent_calls
        _recent_calls = [t for t in _recent_calls if now - t < 60]
        if len(_recent_calls) >= 5:
            return False, "Rate limit: maximum 5 AI summary calls per minute. Please try again later."
        if _recent_calls and (now - _recent_calls[-1]) < 5.0:
            wait_s = 5.0 - (now - _recent_calls[-1])
            return False, f"Rate limit: please wait {max(5, int(wait_s))} second(s) between calls."
        _recent_calls.append(now)

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
