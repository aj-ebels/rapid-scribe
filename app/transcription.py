"""
Transcription: Parakeet/ONNX ASR model loading, cache, installed-models list, transcription worker.
"""
import contextlib
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .settings import DEFAULT_TRANSCRIPTION_MODEL, load_settings
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

# Repos that host both int8 and full-precision files; we only download int8 + metadata (~660 MB vs ~3 GB).
INT8_ONLY_REPOS = frozenset({
    "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "istupakov/parakeet-tdt-0.6b-v3-onnx",
})
# Required files for int8-only Parakeet repos. Keep this explicit so install does not
# depend on Hugging Face's repo snapshot listing when we only need a small subset.
INT8_ONLY_FILES = [
    "config.json",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
    "nemo128.onnx",
]

_CLOSED_HTTP_CLIENT_ERROR = "client has been closed"
_HUB_CACHE_MISS_ERROR = "cannot find the appropriate snapshot folder"
_HUB_LOCATE_FILES_ERROR = "trying to locate the files on the hub"


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
    local_model_path = _get_app_model_dir(model_id) if _is_app_model_installed(model_id) else None
    cache_key = f"{model_id}|{local_model_path}" if local_model_path is not None else model_id
    if _parakeet_model is None or _parakeet_model_id != cache_key:
        if _parakeet_model is not None:
            _parakeet_model = None
            _parakeet_model_id = None
        import onnx_asr
        with _safe_stdout_stderr():
            _parakeet_model = onnx_asr.load_model(
                model_id,
                path=local_model_path,
                quantization="int8",
            )
        _parakeet_model_id = cache_key
    return _parakeet_model


def clear_transcription_model_cache():
    """Clear the cached model so the next transcription uses the currently selected model."""
    global _parakeet_model, _parakeet_model_id
    _parakeet_model = None
    _parakeet_model_id = None


def is_transcription_model_loaded(model_id=None):
    """Return True if the given model is currently loaded in memory (ready to transcribe)."""
    model_id = model_id or PARAKEET_MODEL
    return _parakeet_model is not None and _parakeet_model_id == model_id


@contextlib.contextmanager
def _safe_stdout_stderr():
    """
    Context manager that ensures sys.stdout and sys.stderr have a .write() method.
    In a frozen Windows GUI app they can be None, which breaks tqdm/huggingface_hub.
    """
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def _reset_huggingface_hub_http_client():
    """Drop Hugging Face Hub's shared HTTP client so the next request gets a fresh one."""
    try:
        from huggingface_hub.utils import close_session
        close_session()
    except Exception:
        pass


def _is_closed_http_client_error(exc):
    return _CLOSED_HTTP_CLIENT_ERROR in str(exc).lower()


def _is_hub_lookup_error(exc):
    msg = str(exc).lower()
    return _HUB_CACHE_MISS_ERROR in msg or _HUB_LOCATE_FILES_ERROR in msg


def _is_retryable_download_error(exc):
    return _is_closed_http_client_error(exc) or _is_hub_lookup_error(exc)


def _format_download_error(exc):
    msg = str(exc).strip() or exc.__class__.__name__
    if _is_closed_http_client_error(exc):
        return (
            "The download connection was closed unexpectedly. "
            "Please try Download & install again. If it keeps happening, restart Rapid Scribe and check the network connection."
        )
    if _is_hub_lookup_error(exc):
        return (
            "Rapid Scribe could not reach Hugging Face to download the transcription model, "
            "and the model is not already cached on this computer. Please check the internet connection, VPN/proxy, "
            "or firewall, then try Download & install again."
        )
    return msg


def _get_app_model_root():
    """App-owned model cache used when the Hugging Face Python client is blocked."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base) / "Meetings" / "models"


def _get_app_model_dir(model_id):
    return _get_app_model_root() / model_id.replace("/", "--")


def _is_app_model_installed(model_id):
    if model_id not in INT8_ONLY_REPOS:
        return False
    model_dir = _get_app_model_dir(model_id)
    return model_dir.is_dir() and all((model_dir / filename).is_file() for filename in INT8_ONLY_FILES)


def _get_dir_size(path):
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except Exception:
            pass
    return total


def _direct_model_file_url(model_id, filename):
    return f"https://huggingface.co/{model_id}/resolve/main/{filename}"


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _download_url_with_powershell(url, dest):
    script = (
        "$ProgressPreference='SilentlyContinue'; "
        f"Invoke-WebRequest -Uri {_ps_quote(url)} -OutFile {_ps_quote(dest)} -UseBasicParsing"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )


def _download_url_with_urllib(url, dest):
    with urllib.request.urlopen(url, timeout=60) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)


def _download_url_to_file(url, dest):
    tmp = Path(str(dest) + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        if os.name == "nt":
            try:
                _download_url_with_powershell(url, tmp)
            except Exception as e:
                diag("powershell_download_failed", url=url, error=str(e))
                _download_url_with_urllib(url, tmp)
        else:
            _download_url_with_urllib(url, tmp)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _download_int8_repo_files_to_app_dir(model_id):
    model_dir = _get_app_model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in INT8_ONLY_FILES:
        dest = model_dir / filename
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        diag("direct_model_file_download_start", model_id=model_id, filename=filename)
        _download_url_to_file(_direct_model_file_url(model_id, filename), dest)
        diag("direct_model_file_download_ok", model_id=model_id, filename=filename, size=dest.stat().st_size)
    if not _is_app_model_installed(model_id):
        raise RuntimeError("Direct download finished, but the local model folder is incomplete.")


def _download_int8_repo_files(model_id):
    from huggingface_hub import hf_hub_download

    for filename in INT8_ONLY_FILES:
        hf_hub_download(model_id, filename)


def _download_transcription_model_once(model_id):
    with _safe_stdout_stderr():
        if model_id in INT8_ONLY_REPOS:
            _download_int8_repo_files(model_id)
            return

        import onnx_asr
        onnx_asr.load_model(model_id, quantization="int8")


def download_transcription_model(model_id):
    """
    Download (and briefly load) a transcription model from Hugging Face so it is cached for use.
    Does not keep the model in memory. Returns (success: bool, error_message: str | None).
    """
    last_error = None
    for attempt in range(2):
        try:
            _reset_huggingface_hub_http_client()
            _download_transcription_model_once(model_id)
            # Don't cache in our global; we only needed to trigger the download. Next get_transcription_model() will load from cache.
            clear_transcription_model_cache()
            _invalidate_model_scan_cache()
            diag("transcription_model_download_ok", model_id=model_id, attempt=attempt + 1)
            return True, None
        except Exception as e:
            last_error = e
            diag("transcription_model_download_failed", model_id=model_id, attempt=attempt + 1, error=str(e))
            if attempt == 0 and _is_retryable_download_error(e):
                continue
            break
    if model_id in INT8_ONLY_REPOS:
        try:
            diag("transcription_model_direct_download_fallback", model_id=model_id, hub_error=str(last_error))
            _download_int8_repo_files_to_app_dir(model_id)
            clear_transcription_model_cache()
            _invalidate_model_scan_cache()
            diag("transcription_model_download_ok", model_id=model_id, source="direct")
            return True, None
        except Exception as e:
            diag("transcription_model_direct_download_failed", model_id=model_id, error=str(e))
            return False, (
                "Rapid Scribe could not download the transcription model with either the Hugging Face client "
                "or the Windows downloader. This computer may be blocking app-based downloads from Hugging Face. "
                f"Last error: {_format_download_error(e)}"
            )
    return False, _format_download_error(last_error)


def run_startup_check_worker(result_queue):
    """Run in a subprocess to avoid GIL contention with the GUI. Puts (models, err) on result_queue."""
    try:
        result = list_installed_transcription_models()
        result_queue.put(result)
    except Exception as e:
        result_queue.put(([], str(e)))


def _get_model_scan_cache_path():
    """Path to the model scan result cache file (avoids expensive scan_cache_dir on repeat startups)."""
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "Meetings" / "model_scan_cache.json"


def _invalidate_model_scan_cache():
    """Delete the cached scan result so the next startup re-scans HF cache."""
    try:
        _get_model_scan_cache_path().unlink(missing_ok=True)
    except Exception:
        pass


def _local_model_entries():
    out = []
    for repo_id in INT8_ONLY_REPOS:
        if not _is_app_model_installed(repo_id):
            continue
        model_dir = _get_app_model_dir(repo_id)
        size = _get_dir_size(model_dir)
        out.append({
            "repo_id": repo_id,
            "size_on_disk": size,
            "size_str": _format_size(size),
            "revision_hashes": [],
            "local_path": str(model_dir),
        })
    return out


def _merge_model_entries(hub_models, local_models):
    out = list(hub_models or [])
    existing = {m.get("repo_id") for m in out}
    for model in local_models:
        if model["repo_id"] not in existing:
            out.append(model)
    out.sort(key=lambda x: x["repo_id"])
    return out


def list_installed_transcription_models():
    """List cached Hugging Face models that look like ASR/transcription. Returns (list, error)."""
    local_models = _local_model_entries()
    # Fast path: return cached result if the HF hub directory hasn't changed since last scan.
    cache_path = _get_model_scan_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            hf_dir = Path(cached.get("hf_cache_dir", ""))
            cached_mtime = cached.get("hf_dir_mtime")
            if hf_dir.is_dir() and cached_mtime is not None:
                if abs(hf_dir.stat().st_mtime - cached_mtime) < 1.0:
                    return _merge_model_entries(cached["models"], local_models), None
        except Exception:
            pass

    # Slow path: run the full Hugging Face cache scan.
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return local_models, None if local_models else "huggingface_hub not installed"
    try:
        cache = scan_cache_dir()
    except Exception as e:
        return local_models, None if local_models else str(e)
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
    out = _merge_model_entries(out, local_models)

    # Write result to cache so the next startup can skip the scan.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        hf_dir = Path(HF_HUB_CACHE)
    except Exception:
        hf_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_dir.is_dir():
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "hf_cache_dir": str(hf_dir),
                    "hf_dir_mtime": hf_dir.stat().st_mtime,
                    "models": out,
                }, f, indent=2)
        except Exception:
            pass

    return out, None


def uninstall_transcription_model(repo_id, revision_hashes):
    """Remove a cached model from the Hugging Face cache. Returns (success, error_message)."""
    local_model_dir = _get_app_model_dir(repo_id)
    if local_model_dir.is_dir() and not revision_hashes:
        try:
            shutil.rmtree(local_model_dir)
            _invalidate_model_scan_cache()
            return True, None
        except Exception as e:
            return False, str(e)
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
        _invalidate_model_scan_cache()
        return True, None
    except Exception as e:
        return False, str(e)


# Default: chunks with RMS below this are not transcribed (reduces ASR hallucination on near-silence).
# Can be overridden by settings["min_rms_transcribe"] (e.g. raise to 0.005–0.006 for noisier mics).
MIN_RMS_TRANSCRIBE_DEFAULT = 0.005


def start_transcription_subprocess(chunk_queue, text_queue, stop_event, model_id=None):
    """
    Start the transcription worker in a separate process to avoid GIL contention with
    the GUI and audio capture. Use multiprocessing.Queue for chunk_queue/text_queue
    and multiprocessing.Event for stop_event. Returns the started Process.
    """
    import multiprocessing
    proc = multiprocessing.Process(
        target=transcription_worker,
        args=(chunk_queue, text_queue, stop_event, model_id),
        daemon=True,
    )
    proc.start()
    return proc


def transcription_worker(chunk_queue, text_queue, stop_event, model_id=None):
    """Take WAV paths (or (path, rms)) from chunk_queue, transcribe with selected model, push text to text_queue, delete file.
    Intended to run in a subprocess (via start_transcription_subprocess) to avoid holding the GIL in the main process."""
    try:
        model = get_transcription_model(model_id or PARAKEET_MODEL)
        diag("transcription_model_loaded", model_id=model_id or PARAKEET_MODEL)
    except Exception as e:
        diag("transcription_model_load_failed", error=str(e))
        if not stop_event.is_set():
            text_queue.put_nowait(f"[Transcribe error] Model load failed: {e}\n")
        return
    min_rms = MIN_RMS_TRANSCRIBE_DEFAULT
    min_rms_refresh_every_sec = 3.0
    last_min_rms_refresh = 0.0
    adaptive_gate = True
    noise_floor = max(0.0025, min_rms)
    gate_hangover_chunks = 0
    hangover_left = 0
    last_emit_time = 0.0  # monotonic time of the last successful text emission
    while not stop_event.is_set():
        try:
            item = chunk_queue.get(timeout=1.0)
            if isinstance(item, tuple) and item[0] == "error":
                diag("transcription_received_error", msg=item[1])
                text_queue.put_nowait(("[Error] " + item[1] + "\n"))
                continue
            path = item[0] if isinstance(item, tuple) and len(item) >= 1 else item
            rms = item[1] if isinstance(item, tuple) and len(item) >= 2 else None
            now = time.monotonic()
            if now - last_min_rms_refresh >= min_rms_refresh_every_sec:
                cfg = load_settings()
                cfg_min_rms = cfg.get("min_rms_transcribe", MIN_RMS_TRANSCRIBE_DEFAULT)
                if not isinstance(cfg_min_rms, (int, float)) or cfg_min_rms <= 0:
                    cfg_min_rms = MIN_RMS_TRANSCRIBE_DEFAULT
                min_rms = max(0.001, min(0.05, float(cfg_min_rms)))
                adaptive_gate = bool(cfg.get("adaptive_audio_gating", True))
                gate_hangover_chunks = max(0, min(8, int(cfg.get("audio_gate_hangover_chunks", 0))))
                last_min_rms_refresh = now
            effective_min_rms = min_rms
            if rms is not None and adaptive_gate:
                learn_upper = max(min_rms * 5.0, noise_floor * 1.8)
                if rms <= learn_upper:
                    noise_floor = 0.95 * noise_floor + 0.05 * float(rms)
                effective_min_rms = max(min_rms, noise_floor * 4.5)
            if rms is not None and rms < effective_min_rms:
                if hangover_left > 0:
                    hangover_left -= 1
                else:
                    diag(
                        "transcription_skipped_low_rms",
                        path=path,
                        rms=rms,
                        threshold=effective_min_rms,
                        base_threshold=min_rms,
                        noise_floor=round(noise_floor, 6),
                    )
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue
            else:
                hangover_left = gate_hangover_chunks
            if rms is not None and rms < effective_min_rms and hangover_left >= 0:
                try:
                    diag("transcription_hangover_keep", path=path, rms=rms, threshold=effective_min_rms)
                except Exception:
                    pass
            path_obj = Path(path)
            path_exists = path_obj.exists()
            diag("transcription_got_path", path=path, exists=path_exists)
            if not path_exists:
                continue
            try:
                result = model.recognize(path)
                text = result if isinstance(result, str) else getattr(result, "text", str(result))
                if text and isinstance(text, str) and text.strip():
                    now = time.monotonic()
                    gap = (now - last_emit_time) if last_emit_time > 0 else 0.0
                    last_emit_time = now
                    # Emit a (text, gap_seconds) tuple so the UI can join chunks intelligently.
                    text_queue.put_nowait((text.strip(), gap))
                    diag("transcription_ok", path=path, text_len=len(text.strip()), gap=round(gap, 2))
                else:
                    diag("transcription_empty", path=path)
            except Exception as e:
                diag("transcription_recognize_failed", path=path, error=str(e))
                if not stop_event.is_set():
                    text_queue.put_nowait(f"[Transcribe error] {e}\n")
            finally:
                try:
                    path_obj.unlink(missing_ok=True)
                except Exception:
                    pass
        except queue.Empty:
            continue
        except Exception as e:
            diag("transcription_worker_error", error=str(e))
            if not stop_event.is_set():
                text_queue.put_nowait(f"[Transcribe error] {e}\n")
            break
