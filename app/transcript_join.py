"""
Join streaming ASR segments the same way the GUI does (space vs paragraph breaks).
"""


def join_transcription_items(
    items,
    *,
    initial_tail="\n",
    paragraph_gap_sec=6.0,
):
    """
    Combine items from the transcription worker into one transcript string.

    items: iterable of either (text, gap_seconds) tuples or plain str (errors/status).
    Returns (combined_text, final_tail_char) where final_tail is the last character
    for continuing a session (matches gui._transcript_tail).
    """
    tail = initial_tail
    parts = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            text, gap = item
            if not text:
                continue
            if not tail or tail in ("\n", " "):
                prefix = ""
            elif gap >= paragraph_gap_sec:
                prefix = "\n\n"
            else:
                prefix = " "
            segment = prefix + text
        else:
            segment = str(item)

        parts.append(segment)
        if segment:
            tail = segment[-1]

    combined = "".join(parts) if parts else ""
    return combined, tail
