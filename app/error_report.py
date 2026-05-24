"""
User-initiated and uncaught-error flows to report problems via mailto: (default mail client).

Crash reports lead with the exception (type, message, full traceback), then compact
environment and a short diagnostic.log tail. Manual reports emphasize a larger log excerpt.

The body is percent-encoded; URL length is capped on Windows, so we fit the largest prefix
and copy the full report to the clipboard when the draft is truncated.

Recipient: set RAPID_SCRIBE_REPORT_EMAIL or MEETINGS_REPORT_EMAIL, or edit _DEFAULT_RECIPIENT_EMAIL below.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import traceback
import webbrowser
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from . import __version__ as app_version
from .diagnostic import log_file_path

# If you ship builds without env injection, set your Gmail (or support address) here.
_DEFAULT_RECIPIENT_EMAIL = "ajebels+rapidscribe@gmail.com"

# Total mailto: URL length (scheme + to + subject + body). Windows is strict; others are looser.
_MAX_MAILTO_URL_CHARS = 2047 if sys.platform == "win32" else 60_000

_TRUNCATION_TRAILER = (
    "\n\n[… truncated to fit your email app’s URL limit; the complete report was copied to your clipboard …]"
)

_ORIG_SYS_EXCEPTHOOK = sys.__excepthook__
_ORIG_THREADING_EXCEPTHOOK = getattr(threading, "excepthook", None)
_ui_root = None


def set_report_ui_root(root) -> None:
    """Register the main window for scheduling thread-safe dialogs."""
    global _ui_root
    _ui_root = root


def get_report_recipient_email() -> str:
    return (
        os.environ.get("RAPID_SCRIBE_REPORT_EMAIL", "").strip()
        or os.environ.get("MEETINGS_REPORT_EMAIL", "").strip()
        or _DEFAULT_RECIPIENT_EMAIL.strip()
    )


_REDACT_PATTERNS = (
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "Bearer ***"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "sk-***"),
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"), "AIza***"),
    (re.compile(r"(?i)(api[_-]?key|password|token|secret)\s*[:=]\s*\S+"), r"\1:***"),
)


def redact_sensitive_text(text: str) -> str:
    if not text:
        return text
    out = text
    for rx, repl in _REDACT_PATTERNS:
        out = rx.sub(repl, out)
    return out


def _platform_summary() -> str:
    try:
        import platform

        return f"{platform.system()} {platform.release()} ({platform.machine()}) Python {platform.python_version()}"
    except Exception:
        return f"{sys.platform} Python {sys.version.split()[0]}"


def _read_diagnostic_tail(max_bytes: int = 24_000) -> str:
    path = log_file_path()
    try:
        if not path.is_file():
            return "(No diagnostic.log file yet.)\n"
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
            return data.decode("utf-8", errors="replace") + "\n… (truncated) …\n"
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"(Could not read diagnostic log: {e})\n"


# Manual “Report a problem” keeps a larger log tail; automatic crash reports prioritize the exception.
_MANUAL_REPORT_LOG_TAIL_BYTES = 24_000
_ERROR_REPORT_LOG_TAIL_BYTES = 8_192


def _build_error_report_text(
    exc_type,
    exc_value: BaseException,
    exc_tb,
    *,
    user_note: str = "",
    thread_name: Optional[str] = None,
) -> str:
    """Body centered on the failing exception; log tail is secondary context only."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
    exc_name = getattr(exc_type, "__name__", "Exception")
    exc_msg = str(exc_value) if exc_value is not None else ""

    lines = [
        "Rapid Scribe — automatic error report",
        "This email is about the specific error below (not a generic status dump).",
        "",
        "========== THE ERROR ==========",
        f"Exception type: {exc_name}",
        f"Message: {exc_msg or '(none)'}",
        "",
        "---------- Full traceback ----------",
        tb_text,
        "",
        "========== Environment (compact) ==========",
        f"App version: {app_version}",
        f"UTC time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Platform: {_platform_summary()}",
        f"Thread: {thread_name or threading.current_thread().name}",
        f"Frozen build: {getattr(sys, 'frozen', False)}",
        f"Executable: {getattr(sys, 'executable', '')}",
        f"Diagnostic log file: {log_file_path()}",
        "",
    ]
    if user_note.strip():
        lines.extend(["User note:", user_note.strip(), ""])
    lines.extend(
        [
            "========== Recent diagnostic.log (may help correlate; error is above) ==========",
            _read_diagnostic_tail(max_bytes=_ERROR_REPORT_LOG_TAIL_BYTES).rstrip(),
        ]
    )
    return redact_sensitive_text("\n".join(lines))


def _build_manual_report_text(*, user_note: str = "") -> str:
    """User-initiated report without a captured exception — include more log context."""
    lines = [
        "Rapid Scribe problem report (manual)",
        f"App version: {app_version}",
        f"UTC time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Platform: {_platform_summary()}",
        f"Frozen build: {getattr(sys, 'frozen', False)}",
        f"Executable: {getattr(sys, 'executable', '')}",
        f"Diagnostic log: {log_file_path()}",
        "",
    ]
    if user_note.strip():
        lines.extend(["User note:", user_note.strip(), ""])
    lines.extend(
        [
            "--- Last diagnostic.log content (may be empty in dev without MEETINGS_DEBUG=1) ---",
            _read_diagnostic_tail(max_bytes=_MANUAL_REPORT_LOG_TAIL_BYTES).rstrip(),
        ]
    )
    return redact_sensitive_text("\n".join(lines))


def build_report_text(
    *,
    exc_type=None,
    exc_value: Optional[BaseException] = None,
    exc_tb=None,
    user_note: str = "",
    thread_name: Optional[str] = None,
) -> str:
    """
    Build the email body. If *exc_type* / *exc_value* are set (crash path), the report leads
    with that exception and traceback; otherwise (Settings → Report a problem) it leads with
    general context and a larger diagnostic tail.
    """
    if exc_type is not None and exc_value is not None:
        return _build_error_report_text(
            exc_type, exc_value, exc_tb, user_note=user_note, thread_name=thread_name
        )
    return _build_manual_report_text(user_note=user_note)


def _mailto_prefix(to_addr: str, subject: str) -> str:
    return f"mailto:{quote(to_addr, safe='@')}?subject={quote(subject, safe='')}&body="


def _mailto_url_len(prefix: str, body: str) -> int:
    return len(prefix) + len(quote(body, safe=""))


def _fit_body_for_mailto(to_addr: str, subject: str, full_body: str, max_total: int) -> tuple[str, bool]:
    """
    Return (body_for_mailto, truncated). Keeps a prefix of full_body so the full mailto: URL
    stays within max_total characters after percent-encoding.
    """
    prefix = _mailto_prefix(to_addr, subject)
    if _mailto_url_len(prefix, full_body) <= max_total:
        return full_body, False

    lo, hi = 0, len(full_body)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _mailto_url_len(prefix, full_body[:mid]) <= max_total:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    trailer = _TRUNCATION_TRAILER
    while best > 0 and _mailto_url_len(prefix, full_body[:best] + trailer) > max_total:
        best -= 1
    if best <= 0:
        minimal = "[Report too long for mailto URL; full text copied to clipboard.]"
        if _mailto_url_len(prefix, minimal) > max_total:
            return "", True
        return minimal, True
    return full_body[:best] + trailer, True


def copy_report_to_clipboard(root, text: str) -> bool:
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update_idletasks()
        return True
    except Exception:
        return False


def open_problem_report(
    root,
    *,
    exc_type=None,
    exc_value: Optional[BaseException] = None,
    exc_tb=None,
    user_note: str = "",
    thread_name: Optional[str] = None,
) -> bool:
    """
    Open the default mail client with a mailto: draft. Uses as much of the report as fits
    in a safe URL length; copies the full report to the clipboard when the draft is truncated.
    """
    from tkinter import messagebox

    full = build_report_text(
        exc_type=exc_type,
        exc_value=exc_value,
        exc_tb=exc_tb,
        user_note=user_note,
        thread_name=thread_name,
    )
    to_addr = get_report_recipient_email()
    if exc_type is not None and exc_value is not None:
        exc_label = getattr(exc_type, "__name__", "Error")
        subject = f"[Rapid Scribe v{app_version}] Crash: {exc_label}"
    else:
        subject = f"[Rapid Scribe v{app_version}] Problem report"

    if not to_addr:
        copy_report_to_clipboard(root, full)
        messagebox.showinfo(
            "Report copied",
            "The full problem report was copied to your clipboard.\n\n"
            "This build does not have a developer email pre-configured. "
            "Paste the report into an email to the maintainer, or set the environment variable "
            "RAPID_SCRIBE_REPORT_EMAIL before launching the app.",
            parent=root,
        )
        return False

    body, truncated = _fit_body_for_mailto(to_addr, subject, full, _MAX_MAILTO_URL_CHARS)
    if truncated:
        copy_report_to_clipboard(root, full)

    url = _mailto_prefix(to_addr, subject) + quote(body, safe="")

    try:
        webbrowser.open(url)
        if truncated:
            messagebox.showinfo(
                "Problem report",
                "The email draft may be shortened to fit your mail app’s limits.\n\n"
                "The complete report was copied to your clipboard — paste it into the message if needed.",
                parent=root,
            )
        return True
    except Exception as e:
        copy_report_to_clipboard(root, full)
        messagebox.showerror(
            "Could not open email",
            f"Your default mail program could not be opened ({e}).\n\n"
            f"The full report was copied to your clipboard. Send it manually to:\n{to_addr}",
            parent=root,
        )
        return False


def _log_uncaught(exc_type, exc, tb, *, thread_name: str) -> None:
    try:
        from .diagnostic import write as diag

        msg = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        diag("uncaught_exception", thread=thread_name, detail=msg[:2000])
    except Exception:
        pass


def _offer_report_dialog_on_main(exc_type, exc, tb, *, thread_name: Optional[str] = None) -> None:
    from tkinter import messagebox

    if _ui_root is None:
        return
    try:
        if not _ui_root.winfo_exists():
            return
    except Exception:
        return
    try:
        yes = messagebox.askyesno(
            "Unexpected error",
            "Rapid Scribe hit an unexpected error.\n\n"
            "Open a draft email to the developer? The message will focus on this error "
            "(type, message, and traceback), with a short recent log excerpt for context.",
            parent=_ui_root,
        )
    except Exception:
        return
    if yes:
        try:
            open_problem_report(
                _ui_root,
                exc_type=exc_type,
                exc_value=exc,
                exc_tb=tb,
                thread_name=thread_name or threading.current_thread().name,
            )
        except Exception:
            pass


def _sys_excepthook(exc_type, exc, tb) -> None:
    _log_uncaught(exc_type, exc, tb, thread_name=threading.current_thread().name)
    try:
        if threading.current_thread() is threading.main_thread() and _ui_root is not None:
            try:
                if _ui_root.winfo_exists():
                    _offer_report_dialog_on_main(exc_type, exc, tb)
            except Exception:
                pass
    except Exception:
        pass
    try:
        _ORIG_SYS_EXCEPTHOOK(exc_type, exc, tb)
    except Exception:
        try:
            traceback.print_exception(exc_type, exc, tb)
        except Exception:
            pass


def _thread_excepthook(args) -> None:
    try:
        thread_name = getattr(args.thread, "name", "?")
    except Exception:
        thread_name = "?"
    _log_uncaught(args.exc_type, args.exc_value, args.exc_traceback, thread_name=thread_name)
    if _ui_root is not None:
        try:

            def _deferred():
                if _ui_root.winfo_exists():
                    tn = getattr(args.thread, "name", None)
                    _offer_report_dialog_on_main(
                        args.exc_type,
                        args.exc_value,
                        args.exc_traceback,
                        thread_name=tn,
                    )

            _ui_root.after(0, _deferred)
        except Exception:
            pass
    if _ORIG_THREADING_EXCEPTHOOK is not None:
        try:
            _ORIG_THREADING_EXCEPTHOOK(args)
        except Exception:
            pass


def install_error_reporting(root) -> None:
    """Install global hooks and register *root* for dialogs. Call once after the main CTk() exists."""
    set_report_ui_root(root)
    sys.excepthook = _sys_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
