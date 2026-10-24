"""Private, redacted JSONL metadata logs. These files are not immutable."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
import uuid

from .config import private_dir
from .security import redact


class AuditLogger:
    def __init__(self, directory: Path, retention_days=30):
        self.directory = private_dir(directory)
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        for old in self.directory.glob("*.jsonl"):
            if datetime.fromtimestamp(old.stat().st_mtime, timezone.utc) < cutoff:
                old.unlink()
        self.path = self.directory / f"{uuid.uuid4()}.jsonl"
        self._lock = threading.Lock()

    def log(self, event_type: str, content: dict) -> str:
        event_id = str(uuid.uuid4())
        event = {"id": event_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "event_type": event_type, "content": redact(content)}
        with self._lock:
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=True) + "\n")
        return event_id
