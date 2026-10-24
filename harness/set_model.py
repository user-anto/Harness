"""Persist the user's default generation model without starting any services."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import yaml

from .config import default_config_path, private_dir


def save_model(model_path):
    supplied = Path(model_path).expanduser()
    model = supplied.resolve(strict=True)
    if not model.is_file() or supplied.suffix.lower() != ".gguf":
        raise ValueError("Model must be an existing .gguf file")
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("Model does not have a GGUF file header")
    config = default_config_path()
    data = {}
    if config.exists():
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {config}")
    data.pop("default_orch_model", None)
    data["model"] = str(model)
    private_dir(config.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".config-", dir=config.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return model, config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harness-set-model", description="Save or replace the default GGUF model for future Harness sessions.")
    parser.add_argument("model", type=Path, help="Path to an existing GGUF model file")
    args = parser.parse_args(argv)
    try:
        model, config = save_model(args.model)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"harness-set-model: {exc}", file=sys.stderr)
        return 1
    print(f"Default model: {model}\nSaved to: {config}\nRun harness to use it. Explicit --config files override this default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
