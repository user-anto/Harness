"""Read the web-search credential without importing repository runtime settings."""

import os
from pathlib import Path

from dotenv import dotenv_values


def ollama_api_key(settings):
    exported = os.getenv("OLLAMA_API_KEY", "").strip()
    if exported:
        return exported
    path = Path(settings.env_file).expanduser() if settings.env_file else settings.workspace / ".env"
    if not path.is_file():
        return ""
    if path.stat().st_size > 65536:
        raise ValueError("Credential .env file exceeds the 64 KiB limit")
    # No shell evaluation, variable expansion, or changes to process environment.
    values = dotenv_values(path, interpolate=False, encoding="utf-8-sig")
    return (values.get("OLLAMA_API_KEY") or "").strip()
