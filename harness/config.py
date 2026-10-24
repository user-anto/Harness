"""Explicit configuration and private runtime directories."""

from dataclasses import dataclass, fields
import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def default_config_path() -> Path:
    return Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "harness/config.yaml"


@dataclass(frozen=True)
class Settings:
    workspace: Path
    state_dir: Path
    model: str = ""
    embedding_model: str = "mxbai-embed-large:latest"
    env_file: str = ""
    server_url: str = "http://127.0.0.1:8000"
    model_context: int = 131072
    input_cap: int = 32768
    output_tokens: int = 4096
    headroom: float = 0.10
    memory_enabled: bool = True
    memory_background: bool = True
    memory_model: str = "gpt-oss:20b-cloud"
    memory_fallback_model: str = "llama3.2:3b"
    memory_cloud_retry_interval: float = 60.0
    memory_keep_alive: str = "10m"
    memory_server_url: str = "http://127.0.0.1:11434"
    memory_context: int = 8192
    memory_threads: int = 2
    memory_timeout: float = 120.0
    show_reasoning: bool = True
    retention_days: int = 30
    command_timeout: float = 30.0
    output_limit: int = 16000
    server_timeout: float = 120.0
    server_library_path: str = ""
    gpu_layers: int | str = "auto"

    def __post_init__(self):
        if not ((type(self.gpu_layers) is int and self.gpu_layers >= 0)
                or self.gpu_layers in ("auto", "all")):
            raise ValueError("gpu_layers must be a nonnegative integer, auto, or all")
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {self.workspace}")
        url = urlsplit(self.server_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.path not in {"", "/"}
                or url.query or url.fragment):
            raise ValueError("server_url must be an HTTP loopback origin without credentials")
        memory_url = urlsplit(self.memory_server_url)
        if (memory_url.scheme != "http" or memory_url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or memory_url.username or memory_url.password or memory_url.path not in {"", "/"}
                or memory_url.query or memory_url.fragment):
            raise ValueError("memory_server_url must be an HTTP loopback origin without credentials")
        if (not isinstance(self.memory_fallback_model, str)
                or self.memory_fallback_model.endswith(("-cloud", ":cloud"))):
            raise ValueError("memory_fallback_model must name a local model, or be empty to disable fallback")
        if self.memory_cloud_retry_interval <= 0:
            raise ValueError("memory_cloud_retry_interval must be positive")
        if self.memory_context < 4096 or self.memory_threads < 1 or self.memory_timeout <= 0:
            raise ValueError("memory_context must be at least 4096; memory_threads and memory_timeout must be positive")
        import re
        if not isinstance(self.memory_keep_alive, str) or not re.fullmatch(r"[1-9][0-9]*(?:s|m|h)", self.memory_keep_alive):
            raise ValueError("memory_keep_alive must be a positive duration such as 10m")
        if not 0 <= self.headroom < 0.5:
            raise ValueError("headroom must be between 0 and 0.5")
        if min(self.input_cap, self.output_tokens, self.model_context,
               self.retention_days, self.output_limit, self.command_timeout,
               self.server_timeout) <= 0 or self.input_budget < 1:
            raise ValueError("Token budgets, retention, timeouts and output limits must be positive")

    @property
    def input_budget(self) -> int:
        return min(self.input_cap, int(self.model_context * (1 - self.headroom)) - self.output_tokens)


def load_settings(workspace=None, config_path=None, state_dir=None, no_memory=False) -> Settings:
    selected = Path(workspace or Path.cwd()).expanduser().resolve(strict=True)
    default_config = default_config_path()
    config = Path(config_path).expanduser() if config_path else default_config
    raw = {}
    if config.exists():
        raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping")
    elif config_path:
        raise ValueError(f"Configuration does not exist: {config}")
    # Read the old model keys, but never silently read a checkout's config.
    aliases = {"default_orch_model": "model", "ctx-size": "model_context"}
    data = {aliases.get(k, k): (v * 1024 if k == "ctx-size" else v) for k, v in raw.items()}
    allowed = {f.name for f in fields(Settings)} - {"workspace", "state_dir"}
    data = {k: v for k, v in data.items() if k in allowed}
    if data.get("env_file"):
        credential_path = Path(data["env_file"]).expanduser()
        data["env_file"] = str(credential_path if credential_path.is_absolute() else config.parent / credential_path)
    data["server_url"] = os.getenv("HARNESS_LLAMA_SERVER_URL", data.get("server_url", "http://127.0.0.1:8000"))
    if no_memory:
        data["memory_enabled"] = False
    state = Path(state_dir or os.getenv("HARNESS_STATE_DIR") or
                 Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state")) / "harness").expanduser().resolve()
    return Settings(workspace=selected, state_dir=state, **data)


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path
