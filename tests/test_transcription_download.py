"""Unit tests for transcription model download behavior."""
from app import transcription


def test_download_retries_closed_huggingface_client(monkeypatch):
    calls = []
    resets = []

    def fake_reset():
        resets.append(True)

    def fake_download(_model_id):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("Cannot send a request, as the client has been closed.")

    monkeypatch.setattr(transcription, "_reset_huggingface_hub_http_client", fake_reset)
    monkeypatch.setattr(transcription, "_download_transcription_model_once", fake_download)
    monkeypatch.setattr(transcription, "clear_transcription_model_cache", lambda: None)
    monkeypatch.setattr(transcription, "_invalidate_model_scan_cache", lambda: None)
    monkeypatch.setattr(transcription, "diag", lambda *args, **kwargs: None)

    ok, err = transcription.download_transcription_model("example/model")

    assert ok is True
    assert err is None
    assert len(calls) == 2
    assert len(resets) == 2


def test_download_formats_closed_client_after_retry(monkeypatch):
    def fake_download(_model_id):
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    monkeypatch.setattr(transcription, "_reset_huggingface_hub_http_client", lambda: None)
    monkeypatch.setattr(transcription, "_download_transcription_model_once", fake_download)
    monkeypatch.setattr(transcription, "diag", lambda *args, **kwargs: None)

    ok, err = transcription.download_transcription_model("example/model")

    assert ok is False
    assert "download connection was closed unexpectedly" in err
    assert "Cannot send a request" not in err


def test_download_retries_hub_lookup_error(monkeypatch):
    calls = []

    def fake_download(_model_id):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError(
                "An error happened while trying to locate the files on the Hub and we cannot find "
                "the appropriate snapshot folder for the specified revision on the local disk."
            )

    monkeypatch.setattr(transcription, "_reset_huggingface_hub_http_client", lambda: None)
    monkeypatch.setattr(transcription, "_download_transcription_model_once", fake_download)
    monkeypatch.setattr(transcription, "clear_transcription_model_cache", lambda: None)
    monkeypatch.setattr(transcription, "_invalidate_model_scan_cache", lambda: None)
    monkeypatch.setattr(transcription, "diag", lambda *args, **kwargs: None)

    ok, err = transcription.download_transcription_model("example/model")

    assert ok is True
    assert err is None
    assert len(calls) == 2


def test_download_formats_hub_lookup_error_after_retry(monkeypatch):
    def fake_download(_model_id):
        raise RuntimeError(
            "An error happened while trying to locate the files on the Hub and we cannot find "
            "the appropriate snapshot folder for the specified revision on the local disk."
        )

    monkeypatch.setattr(transcription, "_reset_huggingface_hub_http_client", lambda: None)
    monkeypatch.setattr(transcription, "_download_transcription_model_once", fake_download)
    monkeypatch.setattr(transcription, "diag", lambda *args, **kwargs: None)

    ok, err = transcription.download_transcription_model("example/model")

    assert ok is False
    assert "could not reach Hugging Face" in err
    assert "appropriate snapshot folder" not in err
