"""User-facing OpenAI error text for AI Summary, Ask AI, and export naming."""
import sys
import types

from app.ai_summary import format_openai_api_error, generate_ai_summary


_SCREENSHOT_ERROR = (
    "Error code: 429 - {'error': {'message': 'You have no credits remaining. "
    "Add credits to continue using the API at "
    "https://platform.openai.com/settings/organization/billing/', "
    "'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}"
)


def test_format_credit_balance_from_sdk_string():
    text = format_openai_api_error(RuntimeError(_SCREENSHOT_ERROR))
    assert "credit_balance_exhausted" in text
    assert "https://platform.openai.com/settings/organization/billing/" in text
    assert "same organization" in text
    assert "ChatGPT subscription" in text
    assert "Error code: 429" not in text


def test_format_credit_balance_from_exception_code():
    class _ApiError(Exception):
        code = "credit_balance_exhausted"

        def __str__(self):
            return "Error code: 429"

    text = format_openai_api_error(_ApiError())
    assert "prepaid API credits" in text


def test_format_credit_balance_from_body_dict():
    class _ApiError(Exception):
        body = {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}}

        def __str__(self):
            return "Error code: 429"

    text = format_openai_api_error(_ApiError())
    assert "prepaid API credits" in text


def test_format_leaves_other_errors_unchanged():
    raw = "Error code: 404 - The model 'gpt-6.1-sol' does not exist"
    assert format_openai_api_error(RuntimeError(raw)) == raw


def test_generate_ai_summary_replaces_credit_error(monkeypatch):
    import app.ai_summary as ai_summary

    fake = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError(_SCREENSHOT_ERROR)

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    fake.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    ai_summary._recent_calls.clear()

    ok, message = generate_ai_summary("sk-test", "Summarize:\n{{transcript}}", "hello from the meeting")

    assert ok is False
    assert "prepaid API credits" in message
    assert "Error code: 429" not in message
