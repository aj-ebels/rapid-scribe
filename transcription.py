"""
Transcription: Parakeet/ONNX ASR model loading, cache, installed-models list, transcription worker.
"""
import queue
from pathlib import Path

from settings import DEFAULT_TRANSCRIPTION_MODEL

# Single default: used for dropdown default and for "Install model" (recommended model).
PARAKEET_MODEL = DEFAULT_TRANSCRIPTION_MODEL
STANDARD_TRANSCRIPTION_MODEL = DEFAULT_TRANSCRIPTION_MODEL
_parakeet_model = None
_parakeet_model_id = None

# Substrings in repo_id that we treat as ASR/transcription models
ASR_REPO_PATTERNS = (
    "parakeet", "whisper", "asr", "speech", "stt", "vosk", "gigaam", "canary",
    "conformer", "transcribe",
)


def _format_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_transcription_model(model_id=None):
    """Load and return the ONNX ASR model. Cached per model_id."""
    global _parakeet_model, _parakeet_model_id
    model_id = model_id or PARAKEET_MODEL
    if _parakeet_model is None or _parakeet_model_id != model_id:
        if _parakeet_model is not None:
            _parakeet_model = None
            _parakeet_model_id = None
        import onnx_asr
        _parakeet_model = onnx_asr.load_model(
            model_id,
            quantization="int8",
        )
        _parakeet_model_id = model_id
    return _parakeet_model


def clear_transcription_model_cache():
    """Clear the cached model so the next transcription uses the currently selected model."""
    global _parakeet_model, _parakeet_model_id
    _parakeet_model = None
    _parakeet_model_id = None


def download_transcription_model(model_id):
    """
    Download (and briefly load) a transcription model from Hugging Face so it is cached for use.
    Does not keep the model in memory. Returns (success: bool, error_message: str | None).
    """
    try:
        import onnx_asr
        onnx_asr.load_model(model_id, quantization="int8")
        # Don't cache in our global; we only needed to trigger the download. Next get_transcription_model() will load from cache.
        clear_transcription_model_cache()
        return True, None
    except Exception as e:
        return False, str(e)


def list_installed_transcription_models():
    """List cached Hugging Face models that look like ASR/transcription. Returns (list, error)."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return [], "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
    except Exception as e:
        return [], str(e)
    out = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        rid = (repo.repo_id or "").lower()
        if not any(p in rid for p in ASR_REPO_PATTERNS):
            continue
        out.append({
            "repo_id": repo.repo_id,
            "size_on_disk": repo.size_on_disk,
            "size_str": _format_size(repo.size_on_disk),
            "revision_hashes": [r.commit_hash for r in repo.revisions],
        })
    out.sort(key=lambda x: x["repo_id"])
    return out, None


def uninstall_transcription_model(repo_id, revision_hashes):
    """Remove a cached model from the Hugging Face cache. Returns (success, error_message)."""
    if not revision_hashes:
        return False, "No revisions to delete"
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False, "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
        strategy = cache.delete_revisions(*revision_hashes)
        strategy.execute()
        return True, None
    except Exception as e:
        return False, str(e)


def transcription_worker(chunk_queue, text_queue, stop_event, model_id=None):
    """Take WAV paths from chunk_queue, transcribe with selected model, push text to text_queue, delete file."""
    model = get_transcription_model(model_id or PARAKEET_MODEL)
    while not stop_event.is_set():
        try:
            item = chunk_queue.get(timeout=0.5)
            if isinstance(item, tuple) and item[0] == "error":
                text_queue.put_nowait(("[Error] " + item[1] + "\n"))
                continue
            path = item
            if not Path(path).exists():
                continue
            try:
                result = model.recognize(path)
                text = result if isinstance(result, str) else getattr(result, "text", str(result))
                if text and isinstance(text, str) and text.strip():
                    text_queue.put_nowait(text.strip() + "\n")
            finally:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        except queue.Empty:
            continue
        except Exception as e:
            if not stop_event.is_set():
                text_queue.put_nowait(f"[Transcribe error] {e}\n")
            break
