from dataclasses import replace
import io
import os

from harness.cli import main
from harness.ui import TerminalUI
from conftest import FakeBackend


def test_streamed_answer_is_flushed_without_duplication_or_ansi():
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, interactive=False)
    ui.begin_turn()
    ui.status("generation")
    ui.write("Hello")
    assert stream.getvalue() == "Hello"
    ui.write(" world.")
    ui.status("response_end")
    ui.answer("Hello world.")
    ui.status("memory")
    ui.answer("Hello world.")
    ui.close()
    assert stream.getvalue() == "Hello world.\n"


def test_tool_preamble_does_not_suppress_final_answer():
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, interactive=False)
    ui.write("Checking the file.")
    ui.status("response_end")
    ui.status("generation")
    ui.write("It exists.")
    ui.status("response_end")
    ui.answer("It exists.")
    assert stream.getvalue() == "Checking the file.\nIt exists.\n"


def test_reasoning_is_gray_italic_and_separate_from_answer():
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, interactive=True)
    ui.reasoning("Checking the calculation.")
    ui.write("The answer is 42.")
    ui.status("response_end")
    ui.answer("The answer is 42.")
    assert "\033[90;3mChecking the calculation.\033[0m\n" in stream.getvalue()
    assert stream.getvalue().count("The answer is 42.") == 1
    assert ui.response_text == "The answer is 42."


def test_original_banner_is_packaged_and_rendered(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("harness.ui.shutil.get_terminal_size", lambda: os.terminal_size((120, 40)))
    monkeypatch.setattr("harness.ui.time.sleep", lambda _: None)
    stream = io.StringIO()
    TerminalUI(stream=stream, interactive=True).banner()
    assert "╭" in stream.getvalue() and "\033[38;2;" in stream.getvalue()


def test_cli_connects_backend_stream_and_prints_answer_once(settings, monkeypatch, capsys):
    settings = replace(settings, memory_enabled=False)

    class Backend(FakeBackend):
        def complete(self, request, on_text=None):
            assert on_text is not None
            return super().complete(request, on_text)

        def close(self):
            pass

    backend = Backend(["Streamed final answer."])
    monkeypatch.setattr("harness.cli.load_settings", lambda *args: settings)
    monkeypatch.setattr("harness.backend.ServerManager.start", lambda self: settings)
    monkeypatch.setattr("harness.backend.LlamaBackend", lambda _: backend)
    assert main(["--prompt", "hello", "--plain"]) == 0
    output = capsys.readouterr().out
    assert output.count("Streamed final answer.") == 1
    assert "\033[" not in output
