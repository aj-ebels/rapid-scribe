"""Unit tests for OpenAI model list helpers."""
import json
import time

from app import openai_models


def test_is_chat_model_excludes_non_chat():
    assert openai_models.is_chat_model("gpt-5.2") is True
    assert openai_models.is_chat_model("gpt-5.5") is True
    assert openai_models.is_chat_model("gpt-5.4-mini") is True
    assert openai_models.is_chat_model("gpt-5.5-pro") is False
    assert openai_models.is_chat_model("gpt-5.4-pro") is False
    assert openai_models.is_chat_model("gpt-5.5-pro-2026-04-23") is False
    assert openai_models.is_chat_model("gpt-4o") is True
    assert openai_models.is_chat_model("gpt-5.2-codex") is False
    assert openai_models.is_chat_model("gpt-3.5-turbo-instruct") is False
    assert openai_models.is_chat_model("text-embedding-3-small") is False
    assert openai_models.is_chat_model("whisper-1") is False
    assert openai_models.is_chat_model("dall-e-3") is False


def test_is_responses_api_only_model():
    assert openai_models.is_responses_api_only_model("gpt-5.5-pro") is True
    assert openai_models.is_responses_api_only_model("gpt-5.5") is False


def test_sort_gpt_models_orders_by_version():
    ordered = openai_models.sort_gpt_models(["gpt-4o", "gpt-5.2", "gpt-5.5"])
    assert ordered == ["gpt-5.5", "gpt-5.2", "gpt-4o"]


def test_recommended_models_caps_at_five():
    ids = [f"gpt-5.{i}" for i in range(10)]
    rec = openai_models.recommended_models(ids, n=5)
    assert len(rec) == 5


def test_recommended_models_dedupes_dated_snapshots():
    ids = [
        "gpt-5.5-pro",
        "gpt-5.5-pro-2026-04-23",
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.2",
    ]
    rec = openai_models.recommended_models(ids, n=5)
    assert "gpt-5.5-pro" not in rec
    assert "gpt-5.4-pro" not in rec
    assert "gpt-5.5-2026-04-23" not in rec
    assert "gpt-5.5" in rec
    assert "gpt-5.4-mini" in rec


def test_resolve_model_rejects_pro_models():
    settings = {"openai_model_summary": "gpt-5.5-pro"}
    assert openai_models.resolve_model(settings, "summary") == openai_models.FALLBACK_MODEL


def test_apply_auto_upgrade_updates_implicit_only():
    settings = {
        "openai_model_summary": "gpt-5.2",
        "openai_model_summary_explicit": True,
        "openai_model_ask": "gpt-5.2",
        "openai_model_ask_explicit": False,
        "openai_model_export": "gpt-5.2",
        "openai_model_export_explicit": False,
    }
    changed = openai_models.apply_auto_upgrade(settings, "gpt-5.5")
    assert changed is True
    assert settings["openai_model_summary"] == "gpt-5.2"
    assert settings["openai_model_ask"] == "gpt-5.5"
    assert settings["openai_model_export"] == "gpt-5.5"


def test_supports_reasoning_effort():
    assert openai_models.supports_reasoning_effort("gpt-5.2") is True
    assert openai_models.supports_reasoning_effort("gpt-4o") is False
    assert openai_models.supports_reasoning_effort("gpt-5-chat-latest") is False


def test_refresh_models_uses_cache_on_fetch_failure(tmp_path, monkeypatch):
    cache_path = tmp_path / "openai_models_cache.json"
    cache_path.write_text(
        json.dumps({"fetched_at": time.time(), "model_ids": ["gpt-5.2", "gpt-5.5"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(openai_models, "get_cache_path", lambda: cache_path)
    monkeypatch.setattr(openai_models, "fetch_models", lambda _key: ([], "network error"))

    model_ids, recommended, error, from_cache, _fetched_at = openai_models.refresh_models(
        "sk-test", force=True
    )

    assert error == "network error"
    assert from_cache is True
    assert "gpt-5.5" in model_ids
    assert recommended[0] == "gpt-5.5"


def test_refresh_models_fallback_without_cache(monkeypatch):
    monkeypatch.setattr(openai_models, "load_cache", lambda: None)
    monkeypatch.setattr(openai_models, "fetch_models", lambda _key: ([], "offline"))

    model_ids, recommended, error, from_cache, _fetched_at = openai_models.refresh_models(
        "sk-test", force=True
    )

    assert error == "offline"
    assert from_cache is True
    assert openai_models.FALLBACK_MODEL in model_ids
    assert recommended == [openai_models.FALLBACK_MODEL]


def test_resolve_model():
    settings = {"openai_model_summary": "gpt-4o", "openai_model_ask": ""}
    assert openai_models.resolve_model(settings, "summary") == "gpt-4o"
    assert openai_models.resolve_model(settings, "ask") == openai_models.FALLBACK_MODEL


def test_resolve_model_rejects_non_chat_gpt():
    settings = {"openai_model_summary": "gpt-5.2-codex"}
    assert openai_models.resolve_model(settings, "summary") == openai_models.FALLBACK_MODEL
