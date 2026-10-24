from dataclasses import replace
import json
import shlex
import sys
import time

import pytest

from harness.security import arithmetic, contains_secret, redact, workspace_path
from harness.tools import ToolRegistry, terminal_approval


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", ".git/config", ".env", ".aws/credentials"])
def test_paths_reject_escape_and_credentials(settings, path):
    with pytest.raises(PermissionError):
        workspace_path(settings.workspace, path)


def test_symlink_and_new_child_escape(settings, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (settings.workspace / "link").symlink_to(outside, target_is_directory=True)
    tools = ToolRegistry(settings, approve=lambda *args: True)
    assert tools.execute("write_file", {"path": "link/new.txt", "content": "bad"})["status"] == "error"
    assert not (outside / "new.txt").exists()


@pytest.mark.parametrize("response", ["", "y", "no", "yes please", "n"])
def test_approval_requires_exact_yes(monkeypatch, response):
    monkeypatch.setattr("builtins.input", lambda _: response)
    assert terminal_approval("printf hello", "/tmp", "run_cmd") is False


def test_eof_denies(monkeypatch):
    def eof(_):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert not terminal_approval("anything", "/tmp", "run_cmd")


def test_denied_actions_have_no_effect(settings):
    tools = ToolRegistry(settings, approve=lambda *args: False)
    assert tools.execute("write_file", {"path": "x", "content": "bad"})["status"] == "denied"
    assert tools.execute("run_cmd", {"cmd": "touch x"})["status"] == "denied"
    assert not (settings.workspace / "x").exists()


def test_approved_exact_command_cwd_and_nonzero(settings):
    approvals = []
    def approve(*args):
        approvals.append(args)
        return True
    tools = ToolRegistry(settings, approve=approve)
    command = "printf 'hello'; exit 3"
    result = tools.execute("run_cmd", {"cmd": command})
    assert approvals == [(command, str(settings.workspace), "run_cmd")]
    assert result["content"] == "hello" and result["exit_code"] == 3 and result["status"] == "error"


def test_command_timeout_and_flood(settings):
    tools = ToolRegistry(replace(settings, command_timeout=.2, output_limit=100), approve=lambda *a: True)
    start = time.monotonic()
    result = tools.execute("run_cmd", {"cmd": f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(10)'"})
    assert result["timed_out"] and time.monotonic() - start < 3
    result = tools.execute("run_cmd", {"cmd": f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 10000)'"})
    assert result["truncated"] and len(result["content"]) <= 100


def test_approval_path_rechecked(settings, tmp_path):
    target = settings.workspace / "sub"
    target.mkdir()
    def approve(*args):
        target.rmdir()
        target.symlink_to(tmp_path / "outside", target_is_directory=True)
        return True
    tools = ToolRegistry(settings, approve=approve)
    result = tools.execute("write_file", {"path": "sub/new", "content": "bad"})
    assert result["status"] == "error"
    assert not (tmp_path / "outside/new").exists()


@pytest.mark.parametrize("expression", ["__import__('os')", "().__class__", "[1]*100", "2**1000000", "1e309", "True+1", "(-1)**0.5"])
def test_math_rejects_code_and_resource_abuse(expression):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        arithmetic(expression)


def test_arithmetic_and_bounded_file_ranges(settings):
    assert arithmetic("(12 + 15) * 2 - 4 / 2") == 52
    path = settings.workspace / "code.txt"
    path.write_text("one\ntwo\nthree\nfour\n")
    result = ToolRegistry(settings).execute("read_file", {"path": "code.txt", "start_line": 2, "max_lines": 2})
    assert result["content"] == "2: two\n3: three"
    assert result["truncated"] and result["source"] == str(path)


@pytest.mark.parametrize("text", ["api_key=super-secret-value", "password is abc123", "Bearer abcdefghijklmnop", "sk-abcdefghijklmnopqr", "https://user:password@example.com/a"])
def test_secrets_detected_and_redacted(text):
    assert contains_secret(text)
    assert text not in json.dumps(redact({"message": text}))


def test_query_is_literal_and_skips_escape(settings, tmp_path):
    (settings.workspace / "source.py").write_text("-n[hello]\n")
    (settings.workspace / "escape").symlink_to(tmp_path, target_is_directory=True)
    result = ToolRegistry(settings).execute("search_code", {"directory": ".", "query": "-n[hello]"})
    assert "source.py:1:" in result["content"]
    assert "escape/" not in result["content"]


def test_approved_write_and_large_line_byte_range(settings):
    tools = ToolRegistry(settings, approve=lambda *args: True)
    script = settings.workspace / "script.sh"
    script.write_text("old")
    script.chmod(0o755)
    assert tools.execute("write_file", {"path": "script.sh", "content": "new"})["status"] == "ok"
    assert script.stat().st_mode & 0o777 == 0o755
    assert tools.execute("write_file", {"path": "nested/data", "content": "x" * 20000})["status"] == "ok"
    first = tools.execute("read_file", {"path": "nested/data", "max_chars": 100})
    assert first["truncated"] and len(first["content"]) <= 104
    later = tools.execute("read_file", {"path": "nested/data", "byte_offset": 15000, "max_chars": 100})
    assert later["byte_offset"] == 15000 and len(later["content"]) == 100


def test_git_tools_do_not_read_a_parent_repository(settings, tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = ToolRegistry(settings).execute("git_status", {})
    assert result["status"] == "error" and "workspace" in result["content"]


def test_artifacts_are_bounded_and_expandable(settings):
    class Fetcher:
        def fetch(self, url):
            return {"content": "abcdef" * 4000, "url": url, "truncated": False}
    tools = ToolRegistry(settings, fetcher=Fetcher())
    fetched = tools.execute("fetch_url", {"url": "https://example.com/source"})
    assert fetched["truncated"] and fetched["source"] == "https://example.com/source"
    assert len(fetched["content"]) == settings.output_limit
    expanded = tools.execute("read_artifact", {"artifact_id": fetched["artifact_id"], "offset": 16000, "length": 100})
    assert len(expanded["content"]) == 100 and expanded["source"] == fetched["source"]


def test_swap_after_approval_cannot_create_outside_parents(settings, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sub = settings.workspace / "sub"
    sub.mkdir()
    def approve(*args):
        sub.rmdir()
        sub.symlink_to(outside, target_is_directory=True)
        return True
    result = ToolRegistry(settings, approve=approve).execute("write_file", {"path": "sub/newdir/file", "content": "bad"})
    assert result["status"] == "error"
    assert list(outside.iterdir()) == []


def test_named_pipe_is_rejected_without_blocking(settings):
    import os
    os.mkfifo(settings.workspace / "pipe")
    result = ToolRegistry(settings).execute("read_file", {"path": "pipe"})
    assert result["status"] == "error" and "regular files" in result["content"]


def test_quoted_passphrases_are_fully_redacted():
    assert "horse" not in redact("my password is 'correct horse battery staple'")
    assert "DEF" not in redact('api_key = "ABC DEF"')


def test_protected_path_cannot_be_aliased_by_a_symlink(settings):
    (settings.workspace / "ordinary").write_text("data")
    (settings.workspace / ".env").symlink_to(settings.workspace / "ordinary")
    with pytest.raises(PermissionError):
        workspace_path(settings.workspace, ".env", must_exist=True)
