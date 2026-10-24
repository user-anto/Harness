import asyncio
from dataclasses import replace

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from conftest import FakeBackend, embed
from harness.cli import chat, parser
from harness.memory import MemoryStore
from harness.tui import ChatUI


async def wait_for(predicate):
    async with asyncio.timeout(4):
        while not predicate():
            await asyncio.sleep(.01)


def setup_cli(settings, monkeypatch, count=3):
    memory = MemoryStore(settings.state_dir / 'memory', embed)
    for index in range(count):
        statement = 'I live in Pune.' if index == 0 else f'I prefer option {index}.'
        memory.save(statement, 'explicit', [{'turn_id': str(index), 'quote': statement}], str(settings.workspace))

    class Backend(FakeBackend):
        def close(self):
            pass

    backend = Backend()
    monkeypatch.setattr('harness.cli.open_memory', lambda _: memory)
    monkeypatch.setattr('harness.backend.ServerManager.start', lambda _: settings)
    monkeypatch.setattr('harness.backend.LlamaBackend', lambda _: backend)
    monkeypatch.setattr('harness.tui.ChatUI.banner', lambda _: None)
    return memory, backend


def test_memories_view_split_search_preservation_and_alt_m_handoff(settings, monkeypatch):
    settings = replace(settings, memory_background=False)
    memory, backend = setup_cli(settings, monkeypatch, count=12)

    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            task = asyncio.create_task(ui.run_async(lambda: chat(parser().parse_args([]), settings, ui)))
            await wait_for(lambda: not ui.busy)
            keyboard.send_text('/memories\r\r')
            browser = ui.memory_browser
            await wait_for(lambda: browser.active and browser.cards_window.render_info is not None)
            first = browser.selected['id']
            keyboard.send_text('\x1b[B' * 8)
            await wait_for(lambda: browser.index == 8)
            await asyncio.sleep(.15)
            assert browser.selected['id'] != first and browser.cards_window.vertical_scroll > 0
            positions = ui.app.renderer._last_screen.visible_windows_to_write_positions
            width = ui.app.output.get_size().columns
            assert positions[browser.cards_window].width == (width - 1) * 7 // 10
            assert positions[browser.details.window].width == width - 1 - (width - 1) * 7 // 10 - 4
            assert browser.selected['id'] in browser.details.text
            keyboard.send_text('\tno-match-xyz')
            await wait_for(lambda: len(browser.matches) == 0)
            await asyncio.sleep(.15)
            assert not task.done() and 'Select a memory' in browser.details.text
            keyboard.send_text('\x15Pune')
            await wait_for(lambda: len(browser.matches) == 1)
            selected = browser.selected['id']
            keyboard.send_text('\x7f')
            await wait_for(lambda: browser.search.text == 'Pun')
            assert browser.pending_delete is None
            keyboard.send_text('\x1b')
            await wait_for(lambda: not browser.active and not ui.busy)
            keyboard.send_text('/memories\r\r')
            await wait_for(lambda: browser.active)
            assert browser.search.text == 'Pun' and browser.selected['id'] == selected
            keyboard.send_text('\x1bm')
            await wait_for(lambda: not browser.active and not ui.busy)
            assert ui.composer.text == f'Modify memory {selected}: '
            assert backend.requests == []
            keyboard.send_text('\x03/bye\r\r')
            assert await task == 0

    asyncio.run(scenario())


def test_backspace_confirms_before_deleting_and_esc_preserves_chat(settings, monkeypatch):
    settings = replace(settings, memory_background=False)
    memory, backend = setup_cli(settings, monkeypatch)

    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            task = asyncio.create_task(ui.run_async(lambda: chat(parser().parse_args([]), settings, ui)))
            await wait_for(lambda: not ui.busy)
            keyboard.send_text('/memories\r\r')
            browser = ui.memory_browser
            await wait_for(lambda: browser.active)
            selected = browser.selected['id']
            keyboard.send_text('\x7f')
            await wait_for(lambda: browser.pending_delete == selected)
            keyboard.send_text('y\r')
            await wait_for(lambda: browser.pending_delete is None)
            assert len(memory.records()) == 3
            keyboard.send_text('\x7fyes\r')
            await wait_for(lambda: len(browser.records) == 2 and not browser.working)
            assert memory.find(selected) is None
            assert not backend.requests
            keyboard.send_text('\x1b')
            await wait_for(lambda: not browser.active and not ui.busy)
            keyboard.send_text('/bye\r\r')
            assert await task == 0

    asyncio.run(scenario())
