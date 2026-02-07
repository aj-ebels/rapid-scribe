"""
Local transcription using NVIDIA NeMo Parakeet ASR.
Loads model once; transcribes WAV files (mono) and returns text.
"""

import os
import threading
from queue import Queue, Empty

from config import PARAKEET_MODEL
from logging_config import get_logger

log = get_logger("meetings2.transcriber")


def _suppress_nemo_and_torch_logging():
    """Reduce NeMo/torch log noise (CUDA, training config, telemetry). Run before importing nemo."""
    import logging
    # Silence known noisy loggers
    for name in (
        "nemo",
        "nemo_logging",
        "nemo.collections.asr",
        "nemo.utils",
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Silence any logger whose name starts with nemo or torch (catches submodules)
    for logger_name in logging.root.manager.loggerDict:
        if "nemo" in logger_name.lower() or "torch" in logger_name.lower():
            logging.getLogger(logger_name).setLevel(logging.ERROR)


def ensure_parakeet_model(model_name: str = PARAKEET_MODEL):
    """
    Download and load the Parakeet model if not already cached. Call before opening the app GUI.
    Returns the loaded model (pass to ParakeetTranscriber(initial_model=...) to avoid loading twice).
    Runs on CPU only; CUDA-related logging is suppressed.
    Raises on failure.
    """
    _suppress_nemo_and_torch_logging()
    log.info("Ensuring Parakeet model is installed: %s (downloads if needed).", model_name)
    import nemo.collections.asr as nemo_asr
    import torch
    _suppress_nemo_and_torch_logging()  # again after import (NeMo creates loggers on import)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(torch.device("cpu"))
    log.info("Parakeet model ready (CPU).")
    return model


class ParakeetTranscriber:
    """NeMo Parakeet ASR: load model once, transcribe file paths from a queue."""

    def __init__(self, model_name: str = PARAKEET_MODEL, result_callback=None, initial_model=None):
        self.model_name = model_name
        self.result_callback = result_callback  # callable(text: str)
        self._model = initial_model  # reuse model loaded at startup if provided
        self._queue = Queue()
        self._running = False
        self._thread = None
        self._model_ready = threading.Event()  # set when _model is available (by us or background loader)

    def set_initial_model(self, model):
        """Inject a pre-loaded model (e.g. from background loader). Idempotent."""
        if model is not None and self._model is None:
            self._model = model
            self._model_ready.set()
            log.debug("Initial model set from background loader.")

    def _load_model(self):
        if self._model is not None:
            return
        # Wait for background loader to set the model (app starts GUI first, loads model in another thread)
        log.info("Waiting for model (loading in background, up to 2 min)...")
        self._model_ready.wait(timeout=120)
        if self._model is not None:
            return
        _suppress_nemo_and_torch_logging()
        log.info("Loading NeMo Parakeet model: %s (background load did not finish in time).", self.model_name)
        import nemo.collections.asr as nemo_asr
        import torch
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self._model = self._model.to(torch.device("cpu"))
        self._model_ready.set()
        log.info("Model loaded (CPU).")

    def _run(self):
        try:
            self._load_model()
        except Exception as e:
            log.exception("Failed to load NeMo Parakeet model: %s", e)
            raise
        while self._running:
            try:
                wav_path = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if wav_path is None:
                break
            text = ""
            try:
                hypotheses = self._model.transcribe([wav_path])
                if hypotheses and len(hypotheses) > 0:
                    h = hypotheses[0]
                    raw = getattr(h, "text", None)
                    text = (raw.strip() if isinstance(raw, str) else "")
            except Exception as e:
                log.exception("Transcribe failed for %s: %s", wav_path, e)
                text = ""
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                log.warning("Could not remove temp file %s: %s", wav_path, e)
            if self.result_callback and text:
                try:
                    self.result_callback(text.strip())
                except Exception as e:
                    log.exception("Result callback failed: %s", e)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.debug("Transcriber thread started.")

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None
        log.debug("Transcriber stopped.")

    def submit(self, wav_path: str):
        """Queue a WAV file for transcription. File will be deleted after transcription."""
        self._queue.put(wav_path)
