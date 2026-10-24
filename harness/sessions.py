"""Saved conversation discovery and prompt-based session titles."""

from datetime import datetime, timezone
import uuid
import re
import shlex
import sqlite3
from contextlib import closing

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .memory_view import safe_text
from .security import redact


def new_session_id():
    return "chat-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def session_name(prompt):
    """A short first-prompt title, without another model request or secret values."""
    text = safe_text(redact(prompt)).strip()
    text = re.sub(r"^/plan\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:please\s+)?(?:can|could|would) you\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:please\s+|tell me\s+|help me\s+)", "", text, flags=re.I)
    text = re.split(r"[\n?!]", text, maxsplit=1)[0].strip()
    text = " ".join(text.split())
    if len(text) > 72:
        text = text[:69].rsplit(" ", 1)[0] + "…"
    return text[:1].upper() + text[1:] if text else "Untitled conversation"


def list_sessions(state_dir):
    path = state_dir / "sessions.sqlite"
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "checkpoints" not in tables:
            return []
        activity = (dict(connection.execute("SELECT id, touched FROM session_activity"))
                    if "session_activity" in tables else {})
        names = (dict(connection.execute("SELECT id, name FROM session_metadata"))
                 if "session_metadata" in tables else {})
        saver = SqliteSaver(connection, serde=JsonPlusSerializer(
            pickle_fallback=False, allowed_msgpack_modules=[], allowed_json_modules=[]))
        records = []
        for ident, in connection.execute("SELECT DISTINCT thread_id FROM checkpoints"):
            latest = saver.get_tuple({"configurable": {"thread_id": ident}})
            state = latest.checkpoint.get("channel_values", {}) if latest else {}
            if not state.get("workspace"):
                continue
            records.append({"id": ident, "name": names.get(ident, "UNKNOWN"), "workspace": state["workspace"],
                            "updated_at": activity.get(ident, state.get("updated_at", "")),
                            "messages": len(state.get("messages", []))})
        return sorted(records, key=lambda item: item["updated_at"], reverse=True)


def render_sessions(records, *, current=None, color=False):
    def heading(text):
        return f"\033[1;36m{text}\033[0m" if color else text

    lines = [heading("SAVED SESSIONS")]
    if current:
        lines.append(f"Current session: {safe_text(current)}")
    if not records:
        lines.append("No saved conversations yet. A conversation is saved after its first turn.")
    for record in records:
        ident, workspace = record["id"], record["workspace"]
        lines.extend(["", heading(safe_text(record.get("name", "UNKNOWN")) + " | " + safe_text(ident)),
                      f"  Updated: {safe_text(record['updated_at'] or 'unknown')} · {record['messages']} retained messages",
                      f"  Workspace: {safe_text(workspace)}",
                      "  Resume: " + safe_text(f"harness --workspace {shlex.quote(workspace)} --session {shlex.quote(ident)}")])
    lines.extend(["", "Use the same --state-dir when listing or resuming sessions in custom storage."])
    return "\n".join(lines)
