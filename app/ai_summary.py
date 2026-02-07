"""
AI summary: OpenAI API call with prompt template and transcript.
"""
from prompts import TRANSCRIPT_PLACEHOLDER


def generate_ai_summary(api_key, prompt_template, transcript):
    """
    Call OpenAI API to generate summary.
    Returns (success, result_text_or_error).
    """
    if not (prompt_template or "").strip():
        return False, "Prompt template is empty."
    if not (transcript or "").strip():
        return False, "Transcript is empty."
    text = prompt_template.replace(TRANSCRIPT_PLACEHOLDER, transcript)
    try:
        from openai import OpenAI
    except ImportError:
        return False, "The 'openai' package is not installed. Run: pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
        )
        content = response.choices[0].message.content
        return True, (content or "").strip()
    except Exception as e:
        return False, str(e)
