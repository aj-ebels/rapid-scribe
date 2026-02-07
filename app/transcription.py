"""
Transcription: Parakeet/ONNX ASR model loading, cache, installed-models list, transcription worker.
"""
import queue
from pathlib import Path

from .settings import DEFAULT_TRANSCRIPTION_MODEL
from .diagnostic import write as diag

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


def download_transcription_model(model_id, progress_queue=None):
    """
    Download (and briefly load) a transcription model from Hugging Face so it is cached for use.
    Does not keep the model in memory. Returns (success: bool, error_message: str | None).
    If progress_queue is provided, uses snapshot_download with a progress-reporting tqdm and
    puts (current_bytes, total_bytes) tuples on the queue (total may be 0 until known).
    """
    if progress_queue is not None:
        try:
            from tqdm.auto import tqdm as base_tqdm

            def make_progress_tqdm_class(q):
                """Return a tqdm subclass that pushes (current_bytes, total_bytes) to q. Hub expects a class, not a function."""

                class ProgressTqdm(base_tqdm):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("name", None)  # hub-specific; standard tqdm doesn't accept it
                        self._progress_queue = q
                        super().__init__(*args, **kwargs)

                    def update(self, n=1):
                        super().update(n)
                        if self._progress_queue is not None:
                            try:
                                self._progress_queue.put((self.n, self.total), block=False)
                            except Exception:
                                pass

                return ProgressTqdm

            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_id, tqdm_class=make_progress_tqdm_class(progress_queue))
            clear_transcription_model_cache()
            return True, None
        except Exception as e:
            return False, str(e)

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
    try:
        model = get_transcription_model(model_id or PARAKEET_MODEL)
        diag("transcription_model_loaded", model_id=model_id or PARAKEET_MODEL)
    except Exception as e:
        diag("transcription_model_load_failed", error=str(e))
        if not stop_event.is_set():
            text_queue.put_nowait(f"[Transcribe error] Model load failed: {e}\n")
        return
    while not stop_event.is_set():
        try:
            item = chunk_queue.get(timeout=0.5)
            if isinstance(item, tuple) and item[0] == "error":
                diag("transcription_received_error", msg=item[1])
                text_queue.put_nowait(("[Error] " + item[1] + "\n"))
                continue
            path = item
            diag("transcription_got_path", path=path, exists=Path(path).exists())
            if not Path(path).exists():
                continue
            try:
                result = model.recognize(path)
                text = result if isinstance(result, str) else getattr(result, "text", str(result))
                if text and isinstance(text, str) and text.strip():
                    text_queue.put_nowait(text.strip() + "\n")
                    diag("transcription_ok", path=path, text_len=len(text.strip()))
                else:
                    diag("transcription_empty", path=path)
            except Exception as e:
                diag("transcription_recognize_failed", path=path, error=str(e))
                if not stop_event.is_set():
                    text_queue.put_nowait(f"[Transcribe error] {e}\n")
            finally:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
        except queue.Empty:
            continue
        except Exception as e:
            diag("transcription_worker_error", error=str(e))
            if not stop_event.is_set():
                text_queue.put_nowait(f"[Transcribe error] {e}\n")
            break
