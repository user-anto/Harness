"""Readable memory cards; JSON is reserved for explicit export."""

from datetime import datetime, timezone
import math
from pathlib import Path
import re
import textwrap


LABELS = {"explicit": ("SAVED BY YOU", "32"), "user_stated": ("STATED FACT", "36"),
          "inferred": ("TENTATIVE INSIGHT", "33"), "answer_note": ("ANSWER NOTE", "35")}


def safe_text(value):
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "�", str(value))


def age(value):
    if not value:
        return "unknown"
    try:
        seconds = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(value)).total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        return f"{int(seconds // 86400)}d ago"
    except (ValueError, TypeError):
        return "unknown"


def render_memories(records, *, color=False, width=88, detail=False):
    width = max(30, min(100, width))
    inner = width - 4

    def paint(text, code):
        return f"\033[{code}m{text}\033[0m" if color else text

    records = sorted(records, key=lambda record: record.get("updated_at", ""), reverse=True)
    if not records:
        return (paint("MEMORY  ·  No saved memories yet", "1;36") +
                "\n\nTry: /remember I prefer concise answers\nAutomatic facts will appear after background curation finishes.")
    counts = {kind: sum(record.get("kind") == kind for record in records) for kind in LABELS}
    lines = [paint(f"MEMORY  ·  {len(records)} saved", "1;36"),
             paint(f"{counts['explicit']} saved by you  ·  {counts['user_stated']} auto-saved facts  ·  {counts['inferred']} tentative  ·  {counts['answer_note']} answer notes", "90"), ""]
    for record in records[:20]:
        kind = record.get("kind", "user_stated")
        label, code = LABELS.get(kind, LABELS["user_stated"])
        ident = record["id"][:8]
        title = f" {label} · {ident} "
        lines.append(paint("╭─" + title + "─" * max(0, width - len(title) - 3) + "╮", code))

        def row(text, style=None):
            for part in textwrap.wrap(safe_text(text), inner, replace_whitespace=True) or [""]:
                body = part.ljust(inner)
                lines.append(paint("│ ", code) + (paint(body, style) if style else body) + paint(" │", code))

        row(record["statement"], "1")
        row("")
        turns = len({item.get("turn_id") for item in record.get("evidence", [])})
        confidence = record.get("confidence", 0)
        score = ("User-confirmed" if kind == "explicit" else
                 f"Note selection score {confidence:.2f}" if kind == "answer_note" else f"Extractor score {confidence:.2f}")
        if record.get("edited_by") == "user_requested_agent":
            score = "Edited at your request"
        row(f"{score} · {turns} supporting turn{'s' if turns != 1 else ''}", "90")
        workspace = record.get("workspace", "")
        source = Path(workspace).name or workspace or "unknown workspace"
        row(f"From {source} · Updated {age(record.get('updated_at'))}", "90")
        if kind == "answer_note":
            row("Historical assistant answer · verify before reuse", "35")
            row(f"Recorded: {record.get('observed_at', record.get('created_at', 'unknown'))}", "90")
            count = len(record.get("related_sources", []))
            row(f"{count} related source URLs captured" if count else "Source URLs unavailable · unverified note", "90")
        if record.get("expires_at"):
            try:
                remaining = math.ceil((datetime.fromisoformat(record["expires_at"]) - datetime.now(timezone.utc)).total_seconds() / 86400)
                label = "Note retention" if kind == "answer_note" else "Tentative"
                row(f"{label} · expires in {max(0, remaining)} days", "33")
            except (ValueError, TypeError):
                pass
        if detail:
            row("")
            row(f"Full ID: {record['id']}", "90")
            row(f"Fact key: {record.get('fact_key', 'unknown')}", "90")
            row(f"Source: {workspace}", "90")
            if kind == "answer_note":
                row(f"User asked: {record.get('user_prompt', '')}")
                for source in record.get("related_sources", []):
                    row(f"Related URL: {source.get('url', '')}")
            for evidence in record.get("evidence", []):
                role = "Assistant quote" if evidence.get("role") == "assistant" else "Evidence"
                row(f"{role}: “{evidence.get('quote', '')}”")
        lines.append(paint("╰" + "─" * (width - 2) + "╯", code))
        lines.append("")
    if len(records) > 20:
        lines.append(f"Showing the 20 most recently updated of {len(records)}. Use /memories facts, saved, inferred, or notes to filter.")
    lines.append(paint("Details: /memories ID   ·   Delete: harness memory forget ID   ·   JSON: harness memory export", "90"))
    return "\n".join(lines)
