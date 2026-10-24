import pytest
import yaml

from harness.config import load_settings
from harness.set_model import main, save_model


def test_model_persists_and_replacement_preserves_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    first = tmp_path / "first model.gguf"
    second = tmp_path / "second.gguf"
    for model in (first, second):
        model.write_bytes(b"GGUFtest")
    _, config = save_model(first)
    config.write_text(yaml.safe_dump({"model": str(first), "input_cap": 1234,
                                      "default_orch_model": "obsolete"}))
    assert main([str(second)]) == 0
    data = yaml.safe_load(config.read_text())
    assert data == {"model": str(second), "input_cap": 1234}
    settings = load_settings(workspace=tmp_path, state_dir=tmp_path / "state")
    assert settings.model == str(second)
    assert settings.input_cap == 1234
    assert config.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("name,content", [("bad.gguf", b"not a model"), ("bad.txt", b"GGUF")])
def test_invalid_model_leaves_config_unchanged(tmp_path, monkeypatch, name, content):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = tmp_path / "harness/config.yaml"
    config.parent.mkdir()
    config.write_text("model: previous\n")
    model = tmp_path / name
    model.write_bytes(content)
    assert main([str(model)]) == 1
    assert config.read_text() == "model: previous\n"


def test_missing_model_does_not_create_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert main([str(tmp_path / "missing.gguf")]) == 1
    assert not (tmp_path / "harness").exists()


def test_gguf_symlink_to_extensionless_blob(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    blob = tmp_path / "sha256-model-blob"
    blob.write_bytes(b"GGUFtest")
    link = tmp_path / "model.gguf"
    link.symlink_to(blob)
    model, config = save_model(link)
    assert model == blob
    assert yaml.safe_load(config.read_text())["model"] == str(blob)


def test_invalid_config_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = tmp_path / "harness/config.yaml"
    config.parent.mkdir()
    config.write_text("- list\n")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    assert main([str(model)]) == 1
    assert config.read_text() == "- list\n"
