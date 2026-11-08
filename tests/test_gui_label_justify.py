"""Left-anchored wrapped CTkLabels must left-justify paragraphs.

Tkinter Label (and thus CustomTkinter CTkLabel) defaults justify to "center".
With wraplength set, multi-line Settings / Model / Prompts helper text appears
centered even when pack/anchor is west. Call sites that opt into left layout
must pass justify="left".
"""
from __future__ import annotations

import ast
from pathlib import Path

GUI_PATH = Path(__file__).resolve().parents[1] / "app" / "gui.py"


def _kw_str(call: ast.Call, name: str) -> str | None:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _kw_present(call: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in call.keywords)


def _is_ctk_label_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "CTkLabel":
        return True
    return isinstance(func, ast.Name) and func.id == "CTkLabel"


def test_left_anchored_wrapped_labels_set_justify_left():
    tree = ast.parse(GUI_PATH.read_text(encoding="utf-8"), filename=str(GUI_PATH))
    failures: list[str] = []
    for node in ast.walk(tree):
        if not _is_ctk_label_call(node):
            continue
        if not _kw_present(node, "wraplength"):
            continue
        anchor = _kw_str(node, "anchor")
        if anchor is not None and anchor != "w":
            continue  # intentionally centered (or other) layout
        # Default CTkLabel anchor is center; only enforce when call opts into west.
        if anchor != "w":
            continue
        justify = _kw_str(node, "justify")
        if justify != "left":
            failures.append(
                f"line {node.lineno}: left-anchored wrapped CTkLabel missing justify='left' "
                f"(got {justify!r})"
            )
    assert not failures, "\n".join(failures)


def test_ctk_label_defaults_justify_to_center_when_unset():
    """Guard the assumption the gui contract relies on (needs a display)."""
    import os

    import pytest

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no display for CustomTkinter")

    import customtkinter as ctk

    root = ctk.CTk()
    try:
        lbl = ctk.CTkLabel(root, text="a b c d e f g h i j", wraplength=40, anchor="w")
        assert lbl._label.cget("justify") == "center"
        left = ctk.CTkLabel(
            root, text="a b c d e f g h i j", wraplength=40, anchor="w", justify="left"
        )
        assert left._label.cget("justify") == "left"
    finally:
        root.destroy()
