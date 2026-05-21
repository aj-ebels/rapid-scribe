"""
JSON-line IPC dispatcher for the Electron sidecar (stdin commands, stdout events).
"""
from __future__ import annotations

import json
import queue
import sys
import threading
from typing import Any, Callable

from . import __version__ as app_version
from .ai_summary import ask_meeting_ai, generate_ai_summary, generate_export_name
from .api_key_storage import clear_openai_api_key, get_openai_api_key, set_openai_api_key
from .devices import list_audio_devices, list_loopback_devices
from .diagnostic import init as init_diagnostic, write as diag
from .meetings_storage import (
    create_meeting,
    delete_meeting_by_id,
    ensure_at_least_one_meeting,
    ensure_meeting_has_ai_chat_messages,
    get_meeting_by_id,
    load_meetings,
    save_meetings,
    update_meeting_fields,
)
from .prompts import add_prompt, delete_prompt, get_prompt_by_id, load_prompts, update_prompt
from .session import RecordingSession
from .settings import load_settings, save_settings
from .transcription import (
    STANDARD_TRANSCRIPTION_MODEL,
    download_transcription_model,
    list_installed_transcription_models,
    uninstall_transcription_model,
)
from .update_check import check_for_updates


class IpcServer:
    """Read JSON lines from stdin; write JSON lines to stdout."""

    def __init__(self, out_stream=None):
        self._out = out_stream or sys.stdout
        self._write_lock = threading.Lock()
        self._session: RecordingSession | None = None
        self._settings = load_settings()
        self._stream_handlers: dict[str, threading.Event] = {}

    def emit(self, event_type: str, data: dict | None = None, request_id: str | None = None):
        msg = {"type": event_type, "data": data or {}}
        if request_id:
            msg["id"] = request_id
        self._write(msg)

    def _write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._write_lock:
            self._out.write(line)
            self._out.flush()

    def _respond(self, request_id: str, ok: bool, result: Any = None, error: str | None = None):
        self._write({
            "id": request_id,
            "type": "response",
            "ok": ok,
            "result": result,
            "error": error,
        })

    def _session_emit(self, event_type: str, data: dict):
        self.emit(event_type, data)

    def _get_session(self) -> RecordingSession:
        if self._session is None:
            self._session = RecordingSession(self._settings, self._session_emit)
        return self._session

    def handle(self, msg: dict):
        request_id = msg.get("id", "")
        cmd = msg.get("type", "")
        params = msg.get("params") or {}

        handlers: dict[str, Callable] = {
            "ping": self._cmd_ping,
            "get_version": self._cmd_get_version,
            "list_devices": self._cmd_list_devices,
            "get_settings": self._cmd_get_settings,
            "set_settings": self._cmd_set_settings,
            "get_api_key": self._cmd_get_api_key,
            "set_api_key": self._cmd_set_api_key,
            "list_prompts": self._cmd_list_prompts,
            "save_prompt": self._cmd_save_prompt,
            "delete_prompt": self._cmd_delete_prompt,
            "model_status": self._cmd_model_status,
            "download_model": self._cmd_download_model,
            "uninstall_model": self._cmd_uninstall_model,
            "start_recording": self._cmd_start_recording,
            "stop_recording": self._cmd_stop_recording,
            "generate_summary": self._cmd_generate_summary,
            "ask_ai": self._cmd_ask_ai,
            "generate_export_name": self._cmd_generate_export_name,
            "list_meetings": self._cmd_list_meetings,
            "load_meeting": self._cmd_load_meeting,
            "save_meeting": self._cmd_save_meeting,
            "delete_meeting": self._cmd_delete_meeting,
            "create_meeting": self._cmd_create_meeting,
            "check_for_update": self._cmd_check_for_update,
            "shutdown": self._cmd_shutdown,
        }

        handler = handlers.get(cmd)
        if not handler:
            self._respond(request_id, False, error=f"Unknown command: {cmd}")
            return
        try:
            handler(request_id, params)
        except Exception as e:
            diag("ipc_error", cmd=cmd, error=str(e))
            self._respond(request_id, False, error=str(e))

    def _cmd_ping(self, request_id: str, _params: dict):
        self._respond(request_id, True, {"pong": True, "version": app_version})

    def _cmd_get_version(self, request_id: str, _params: dict):
        self._respond(request_id, True, {"version": app_version})

    def _cmd_list_devices(self, request_id: str, _params: dict):
        inputs, err_in = list_audio_devices()
        loopbacks, err_lb = list_loopback_devices()
        self._respond(request_id, True, {
            "inputs": inputs,
            "loopbacks": loopbacks,
            "errors": {"inputs": err_in, "loopbacks": err_lb},
        })

    def _cmd_get_settings(self, request_id: str, _params: dict):
        self._settings = load_settings()
        self._respond(request_id, True, self._settings)

    def _cmd_set_settings(self, request_id: str, params: dict):
        self._settings = load_settings()
        if isinstance(params.get("settings"), dict):
            self._settings.update(params["settings"])
        else:
            self._settings.update({k: v for k, v in params.items() if k != "settings"})
        save_settings(self._settings)
        if self._session is not None:
            self._session.settings = self._settings
        self._respond(request_id, True, self._settings)

    def _cmd_get_api_key(self, request_id: str, _params: dict):
        key = get_openai_api_key()
        self._respond(request_id, True, {"has_key": bool(key), "masked": ("*" * 8 + key[-4:]) if len(key) > 4 else ""})

    def _cmd_set_api_key(self, request_id: str, params: dict):
        key = (params.get("api_key") or "").strip()
        if not key:
            clear_openai_api_key()
        else:
            set_openai_api_key(key)
        self._respond(request_id, True, {"has_key": bool(get_openai_api_key())})

    def _cmd_list_prompts(self, request_id: str, _params: dict):
        self._respond(request_id, True, load_prompts())

    def _cmd_save_prompt(self, request_id: str, params: dict):
        prompt_id = params.get("id")
        name = (params.get("name") or "").strip()
        text = (params.get("prompt") or "").strip()
        if not name or not text:
            self._respond(request_id, False, error="Name and prompt text required")
            return
        if prompt_id:
            update_prompt(prompt_id, name, text)
        else:
            add_prompt(name, text)
        self._respond(request_id, True, load_prompts())

    def _cmd_delete_prompt(self, request_id: str, params: dict):
        pid = params.get("id")
        if not pid:
            self._respond(request_id, False, error="id required")
            return
        delete_prompt(pid)
        self._respond(request_id, True, load_prompts())

    def _cmd_model_status(self, request_id: str, _params: dict):
        session = self._get_session()
        models, err = list_installed_transcription_models()
        status = session.model_status()
        status["models"] = models
        status["error"] = err
        self._respond(request_id, True, status)

    def _cmd_download_model(self, request_id: str, params: dict):
        model_id = params.get("model_id") or STANDARD_TRANSCRIPTION_MODEL
        self.emit("model_progress", {"status": "downloading", "model_id": model_id}, request_id)

        def worker():
            ok, err = download_transcription_model(model_id)
            if ok:
                self._settings["transcription_model"] = model_id
                save_settings(self._settings)
                self.emit("model_progress", {"status": "done", "model_id": model_id}, request_id)
                self._respond(request_id, True, {"installed": True})
            else:
                self.emit("model_progress", {"status": "error", "error": err}, request_id)
                self._respond(request_id, False, error=err)

        threading.Thread(target=worker, daemon=True).start()

    def _cmd_uninstall_model(self, request_id: str, params: dict):
        model_id = params.get("model_id") or STANDARD_TRANSCRIPTION_MODEL
        models, _ = list_installed_transcription_models()
        m = next((x for x in (models or []) if x["repo_id"] == model_id), None)
        if not m:
            self._respond(request_id, False, error="Model not found")
            return
        ok, err = uninstall_transcription_model(model_id, m["revision_hashes"])
        self._respond(request_id, ok, {"uninstalled": ok}, error=err)

    def _cmd_start_recording(self, request_id: str, _params: dict):
        session = self._get_session()
        ok, err = session.start()
        self._respond(request_id, ok, {"recording": ok}, error=err)

    def _cmd_stop_recording(self, request_id: str, _params: dict):
        session = self._get_session()
        session.stop()
        self._respond(request_id, True, {"recording": False})

    def _cmd_generate_summary(self, request_id: str, params: dict):
        api_key = get_openai_api_key()
        if not api_key:
            self._respond(request_id, False, error="API key required")
            return
        prompt_id = params.get("prompt_id")
        prompt_text = params.get("prompt")
        if prompt_id and not prompt_text:
            p = get_prompt_by_id(prompt_id)
            prompt_text = (p or {}).get("prompt", "")
        transcript = params.get("transcript", "")
        manual_notes = params.get("manual_notes", "")

        def worker():
            ok, out = generate_ai_summary(api_key, prompt_text, transcript, manual_notes=manual_notes)
            if ok:
                for i, line in enumerate(out.splitlines(keepends=True)):
                    self.emit("summary_chunk", {"text": line, "index": i}, request_id)
                self.emit("summary_done", {"text": out}, request_id)
                self._respond(request_id, True, {"text": out})
            else:
                self._respond(request_id, False, error=out)

        threading.Thread(target=worker, daemon=True).start()

    def _cmd_ask_ai(self, request_id: str, params: dict):
        api_key = get_openai_api_key()
        if not api_key:
            self._respond(request_id, False, error="API key required")
            return

        def worker():
            ok, out = ask_meeting_ai(
                api_key,
                params.get("manual_notes", ""),
                params.get("transcript", ""),
                params.get("ai_summary", ""),
                params.get("chat_messages", []),
                params.get("message", ""),
            )
            if ok:
                self.emit("ask_chunk", {"text": out}, request_id)
                self.emit("ask_done", {"text": out}, request_id)
                self._respond(request_id, True, {"text": out})
            else:
                self._respond(request_id, False, error=out)

        threading.Thread(target=worker, daemon=True).start()

    def _cmd_generate_export_name(self, request_id: str, params: dict):
        api_key = get_openai_api_key()
        if not api_key:
            self._respond(request_id, False, error="API key required")
            return
        summary = params.get("summary", "")[:250]

        def worker():
            ok, out = generate_export_name(api_key, summary)
            self._respond(request_id, ok, {"name": out} if ok else None, error=None if ok else out)

        threading.Thread(target=worker, daemon=True).start()

    def _cmd_list_meetings(self, request_id: str, _params: dict):
        meetings = load_meetings()
        if ensure_at_least_one_meeting(meetings) is not None:
            save_meetings(meetings)
        self._respond(request_id, True, meetings)

    def _cmd_load_meeting(self, request_id: str, params: dict):
        meetings = load_meetings()
        mid = params.get("id")
        m = get_meeting_by_id(meetings, mid) if mid else None
        if not m:
            self._respond(request_id, False, error="Meeting not found")
            return
        ensure_meeting_has_ai_chat_messages(m)
        self._respond(request_id, True, m)

    def _cmd_save_meeting(self, request_id: str, params: dict):
        meetings = load_meetings()
        data = params.get("meeting")
        if not isinstance(data, dict) or not data.get("id"):
            self._respond(request_id, False, error="meeting.id required")
            return
        existing = get_meeting_by_id(meetings, data["id"])
        if existing:
            update_meeting_fields(existing, **{k: v for k, v in data.items() if k != "id"})
        else:
            meetings.append(data)
        save_meetings(meetings)
        self._respond(request_id, True, data)

    def _cmd_delete_meeting(self, request_id: str, params: dict):
        meetings = load_meetings()
        mid = params.get("id")
        if not mid:
            self._respond(request_id, False, error="id required")
            return
        delete_meeting_by_id(meetings, mid)
        if ensure_at_least_one_meeting(meetings) is not None:
            pass  # appended default meeting in place
        save_meetings(meetings)
        self._respond(request_id, True, meetings)

    def _cmd_create_meeting(self, request_id: str, params: dict):
        meetings = load_meetings()
        m = create_meeting(params.get("meeting_name") or "New Meeting")
        meetings.insert(0, m)
        save_meetings(meetings)
        self._respond(request_id, True, m)

    def _cmd_check_for_update(self, request_id: str, _params: dict):
        def worker():
            has_update, latest, url, err = check_for_updates(app_version)
            self._respond(request_id, True, {
                "update_available": has_update,
                "latest_version": latest,
                "download_url": url,
                "error": err,
            })

        threading.Thread(target=worker, daemon=True).start()

    def _cmd_shutdown(self, request_id: str, _params: dict):
        if self._session:
            self._session.shutdown_transcription()
        self._respond(request_id, True, {"shutdown": True})

    def run(self):
        init_diagnostic()
        self.emit("ready", {"version": app_version})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                diag("ipc_json_error", error=str(e))
                continue
            if isinstance(msg, dict):
                self.handle(msg)
