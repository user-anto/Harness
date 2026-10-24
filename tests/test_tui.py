import asyncio
from dataclasses import replace
import threading

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from harness.tui import ChatUI
from harness.cli import chat, parser
from conftest import FakeBackend, tool_call


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


def test_drag_selection_highlights_and_copies_without_wrap_newlines():
    with create_pipe_input() as keyboard:
        ui = ChatUI(input=keyboard, output=DummyOutput())
        ui.append("hello world\nnext")
        ui.wrapped_transcript(5)
        control = ui.conversation_control
        for kind, point in [(MouseEventType.MOUSE_DOWN, Point(0, 0)),
                            (MouseEventType.MOUSE_MOVE, Point(1, 2)),
                            (MouseEventType.MOUSE_UP, Point(1, 2))]:
            control.mouse_handler(MouseEvent(point, kind, MouseButton.LEFT, frozenset()))
        assert ui.selected_text() == "hello world"
        assert any("reverse" in style for style, text in ui.wrapped_transcript(5) if text)
        assert ui.copy_selection()
        assert ui.app.clipboard.get_data().text == "hello world"
        assert not ui.selected_text() and not ui.cancelled.is_set()


def test_real_mouse_drag_and_ctrl_c_copy_preserve_draft():
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())

            def worker():
                ui.message("hello selectable world")
                ui.read_prompt()
                return 0

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("draft")
            await wait_until(lambda: ui.composer.text == "draft")
            await asyncio.sleep(.15)
            keyboard.send_text("\x1b[<0;1;1M\x1b[<32;6;1M\x1b[<0;6;1m")
            await asyncio.sleep(.15)
            assert ui.selected_text() == "hello", (ui.selection_start, ui.selection_end, ui._wrapped)
            keyboard.send_text("\x03")
            await wait_until(lambda: ui.app.clipboard.get_data().text == "hello")
            assert ui.composer.text == "draft" and not ui.cancelled.is_set()
            keyboard.send_text("\r")
            assert await task == 0

    asyncio.run(scenario())


def test_slash_menu_arrows_insert_command_before_submission():
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            received = []

            def worker():
                received.append(ui.read_prompt())
                return 0

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("/")
            await wait_until(lambda: ui.menu_visible)
            assert [command.name for command in ui.matches][:2] == ["/plan", "/remember"]
            keyboard.send_text("\x1b[B")
            await wait_until(lambda: ui.selected_command == 1)
            keyboard.send_text("\x1b[A")
            await wait_until(lambda: ui.selected_command == 0)
            keyboard.send_text("\x1b[B")
            await wait_until(lambda: ui.selected_command == 1)
            keyboard.send_text("\r")
            await wait_until(lambda: ui.composer.text == "/remember ")
            assert received == [] and not ui.menu_visible
            keyboard.send_text("I prefer concise answers.\r")
            assert await task == 0
            assert received == ["/remember I prefer concise answers."]
            assert "\033[32m› /remember I prefer concise answers.\033[0m" in ui.transcript

    asyncio.run(scenario())


def test_menu_filters_and_escape_preserves_input():
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())

            def worker():
                try:
                    ui.read_prompt()
                except EOFError:
                    return 0
                return 1

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("/mem")
            await wait_until(lambda: ui.menu_visible)
            assert [command.name for command in ui.matches] == ["/memories"]
            keyboard.send_text("\x1b")
            await wait_until(lambda: not ui.menu_visible)
            assert ui.composer.text == "/mem"
            keyboard.send_text("\x03\x04")
            assert await task == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("newline", ["\x1b[13;2u", "\x1b[27;2;13~", "\n"])
def test_shift_enter_adds_newline_without_submitting(newline):
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            received = []

            def worker():
                received.append(ui.read_prompt())
                return 0

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("First line" + newline)
            await wait_until(lambda: "\n" in ui.composer.text)
            assert not received
            keyboard.send_text("Second line\r")
            assert await task == 0
            assert received == ["First line\nSecond line"]

    asyncio.run(scenario())


@pytest.mark.parametrize("keys,expected", [("\r", False), ("y\r", False), ("yes\r", True),
                                            ("\x03", False), ("\x1b", False)])
def test_permission_prompt_is_cyan_and_requires_exact_yes(keys, expected):
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            decisions = []

            def worker():
                decisions.append(ui.approve("touch example.txt", "/workspace", "run_cmd"))
                return 0

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: ui.approval_pending)
            assert "\033[36m" in ui.transcript
            assert "touch example.txt" in ui.transcript and "/workspace" in ui.transcript
            assert "class:approval" in ui.footer_fragments()[0][0]
            assert not ui.menu_visible
            keyboard.send_text(keys)
            assert await task == 0
            assert decisions == [expected]

    asyncio.run(scenario())


def test_bottom_composer_stays_visible_while_answer_streams():
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())

            def worker():
                ui.message("Harness ready")
                ui.read_prompt()
                ui.begin_turn()
                ui.status("generation")
                ui.write("First streamed chunk")
                ui.write(" and second chunk.")
                ui.status("response_end")
                ui.answer("First streamed chunk and second chunk.")
                ui.read_prompt()
                return 0

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("hello\r")
            await wait_until(lambda: "second chunk." in ui.transcript and not ui.busy)
            await asyncio.sleep(.15)
            screen = ui.app.renderer._last_screen
            positions = screen.visible_windows_to_write_positions
            composer = positions[ui.composer.window]
            transcript = positions[ui.transcript_window]
            assert composer.ypos > transcript.ypos + transcript.height
            assert composer.ypos >= ui.app.output.get_size().rows - 5
            assert screen.data_buffer[composer.ypos][1].char == "›"
            placeholder = "".join(screen.data_buffer[composer.ypos][x].char for x in range(3, 33))
            assert placeholder.startswith("Ask Harness")
            assert ui.transcript.count("First streamed chunk and second chunk.") == 1
            keyboard.send_text("done\r")
            assert await task == 0

    asyncio.run(scenario())


def test_interactive_cli_routes_real_tool_approval_without_denied_side_effects(settings, monkeypatch):
    settings = replace(settings, memory_enabled=False)

    class Backend(FakeBackend):
        def close(self):
            pass

    backend = Backend([tool_call("write_file", {"path": "example.txt", "content": "hello"}), "Write denied."])
    monkeypatch.setattr("harness.backend.ServerManager.start", lambda self: settings)
    monkeypatch.setattr("harness.backend.LlamaBackend", lambda _: backend)
    monkeypatch.setattr("harness.tui.ChatUI.banner", lambda self: None)

    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            task = asyncio.create_task(ui.run_async(lambda: chat(parser().parse_args([]), settings, ui)))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("Write hello to example.txt\r")
            await wait_until(lambda: ui.approval_pending)
            assert not (settings.workspace / "example.txt").exists()
            keyboard.send_text("\r")
            await wait_until(lambda: "Write denied." in ui.transcript and not ui.busy)
            assert not (settings.workspace / "example.txt").exists()
            assert "\033[36m" in ui.transcript
            keyboard.send_text("/bye\r\r")
            assert await task == 0
            assert ui.transcript.count("Write denied.") == 1

    asyncio.run(scenario())


def test_mouse_wheel_scrolls_conversation_without_changing_input_or_snapping_back():
    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())

            def worker():
                ui.message("\n".join(f"Conversation line {index}" for index in range(100)))
                ui.message("A wrapped paragraph " * 100)
                ui.read_prompt()
                return 0

            task = asyncio.create_task(ui.run_async(worker))
            await wait_until(lambda: not ui.busy and ui.transcript_window.vertical_scroll > 0)
            keyboard.send_text("Unsent draft")
            await wait_until(lambda: ui.composer.text == "Unsent draft")
            before = ui.transcript_window.vertical_scroll
            keyboard.send_text("\x1b[<64;10;10M")
            await wait_until(lambda: ui.transcript_window.vertical_scroll < before)
            assert ui.transcript_window.vertical_scroll == before - 3
            anchored = ui.transcript_window.vertical_scroll
            ui.append("\nMore streamed output")
            await asyncio.sleep(.15)
            assert ui.transcript_window.vertical_scroll == anchored
            assert ui.composer.text == "Unsent draft"
            keyboard.send_text("\x1b[<65;10;10M")
            await wait_until(lambda: ui.transcript_window.vertical_scroll > anchored)
            keyboard.send_text("\r")
            assert await task == 0

    asyncio.run(scenario())


def test_cli_accepts_second_prompt_while_separate_curator_is_running(settings, monkeypatch):
    from harness.memory import MemoryStore
    from conftest import embed

    settings = replace(settings, memory_enabled=True, memory_background=True)
    started, release = threading.Event(), threading.Event()

    class Extractor:
        def __init__(self, _settings):
            pass

        def extract(self, turns, stopped, audit=None):
            started.set()
            assert release.wait(5)
            return [], turns

        def close(self):
            pass

    class Backend(FakeBackend):
        def close(self):
            pass

    backend = Backend(["First answer", "Second answer"])
    monkeypatch.setattr("harness.curation.OllamaExtractor", Extractor)
    monkeypatch.setattr("harness.backend.ServerManager.start", lambda self: settings)
    monkeypatch.setattr("harness.backend.LlamaBackend", lambda _: backend)
    monkeypatch.setattr("harness.cli.open_memory", lambda settings: MemoryStore(settings.state_dir / "memory", embed))
    monkeypatch.setattr("harness.tui.ChatUI.banner", lambda self: None)

    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            task = asyncio.create_task(ui.run_async(lambda: chat(parser().parse_args([]), settings, ui)))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text("hello\r")
            await wait_until(lambda: started.is_set() and not ui.busy)
            assert not release.is_set()
            assert "Curating memory in background" in ui.footer_fragments()[0][1]
            assert not ui.composer.buffer.read_only()
            keyboard.send_text("next question\r")
            await wait_until(lambda: "Second answer" in ui.transcript and not ui.busy)
            assert not release.is_set() and len(backend.requests) == 2
            release.set()
            keyboard.send_text("/bye\r\r")
            assert await task == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
