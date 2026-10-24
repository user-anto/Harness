import asyncio
from dataclasses import replace
import sqlite3

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from conftest import FakeBackend
from harness.cli import chat, parser
from harness.runtime import AgentRuntime
from harness.sessions import list_sessions, session_name
from harness.tui import ChatUI
from test_memory_browser import wait_for


def test_first_prompt_names_persist_and_legacy_stays_unknown(settings):
    settings = replace(settings, memory_enabled=False)
    runtime = AgentRuntime(settings, FakeBackend(['One', 'Two', 'Three']))
    runtime.turn('Please help me debug Python imports', 'named')
    runtime.turn('An unrelated follow-up', 'named')
    runtime.turn('Old conversation', 'legacy')
    runtime.conn.execute("DELETE FROM session_metadata WHERE id = 'legacy'")
    runtime.conn.commit()
    runtime.close()
    names = {r['id']: r['name'] for r in list_sessions(settings.state_dir)}
    assert names == {'named': 'Help me debug Python imports', 'legacy': 'UNKNOWN'}
    runtime = AgentRuntime(settings, FakeBackend(['Four']))
    try:
        runtime.turn('A new topic', 'legacy')
        assert {r['id']: r['name'] for r in list_sessions(settings.state_dir)} == names
        runtime.delete_session('named')
        assert runtime.snapshot('named') == {}
        assert runtime.conn.execute("SELECT * FROM session_metadata WHERE id='named'").fetchall() == []
        assert runtime.conn.execute("SELECT * FROM session_activity WHERE id='named'").fetchall() == []
    finally:
        runtime.close()
    assert [r['id'] for r in list_sessions(settings.state_dir)] == ['legacy']
    assert len(session_name('Fix ' + 'long prompt ' * 100)) <= 72
    assert '\x1b' not in session_name('\x1b[31mhello')


def test_load_restores_workspace_tools_and_rejects_missing_workspace(settings, tmp_path):
    settings = replace(settings, memory_enabled=False)
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'marker.txt').write_text('other workspace')
    runtime = AgentRuntime(replace(settings, workspace=other), FakeBackend(['Answer']))
    runtime.turn('Other project question', 'other')
    runtime.close()
    runtime = AgentRuntime(settings, FakeBackend())
    try:
        runtime.load_session('other')
        assert runtime.settings.workspace == runtime.tools.settings.workspace == other
        assert 'other workspace' in runtime.tools.execute('read_file', {'path': 'marker.txt'})['content']
        runtime.settings = runtime.tools.settings = settings
        (other / 'marker.txt').unlink()
        other.rmdir()
        with pytest.raises(FileNotFoundError):
            runtime.load_session('other')
        assert runtime.settings.workspace == settings.workspace
        with pytest.raises(ValueError, match='no longer exists'):
            runtime.load_session('missing')
    finally:
        runtime.close()


def test_session_tab_search_load_continuation_and_confirmed_delete(settings, monkeypatch):
    settings = replace(settings, memory_enabled=False)
    runtime = AgentRuntime(settings, FakeBackend(['Python answer', 'Rust answer']))
    runtime.turn('Debug Python imports', 'python')
    runtime.turn('Build Rust service', 'rust')
    runtime.close()

    class Backend(FakeBackend):
        def close(self):
            pass

    backend = Backend(['Current answer', 'Continued answer'])
    monkeypatch.setattr('harness.backend.ServerManager.start', lambda _: settings)
    monkeypatch.setattr('harness.backend.LlamaBackend', lambda _: backend)
    monkeypatch.setattr('harness.tui.ChatUI.banner', lambda _: None)
    args = parser().parse_args(['--session', 'current'])

    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())
            task = asyncio.create_task(ui.run_async(lambda: chat(args, settings, ui)))
            await wait_for(lambda: not ui.busy)
            keyboard.send_text('Current project question\r')
            await wait_for(lambda: not ui.busy and len(backend.requests) == 1)
            keyboard.send_text('/sessions\r\r')
            browser = ui.session_browser
            await wait_for(lambda: browser.active and browser.cards_window.render_info is not None)
            assert browser.cards_window.render_info.window_width == ui.app.output.get_size().columns
            original = browser.selected['id']
            keyboard.send_text('\x1b[B')
            await wait_for(lambda: browser.index == 1)
            assert browser.selected['id'] != original
            keyboard.send_text('\tpython')
            await wait_for(lambda: len(browser.matches) == 1)
            keyboard.send_text('\x7f')
            await wait_for(lambda: browser.search.text == 'pytho')
            assert browser.pending_delete is None
            keyboard.send_text('\r')
            await wait_for(lambda: not browser.active and not ui.busy and args.session == 'python')
            assert 'Python answer' in ui.transcript and 'Current project question' not in ui.transcript
            assert '\033[32m› Debug Python imports\033[0m' in ui.transcript
            keyboard.send_text('Follow-up about imports\r')
            await wait_for(lambda: not ui.busy and len(backend.requests) == 2)
            users = [m['content'] for m in backend.requests[-1]['messages']
                     if m['role'] == 'user' and not m['content'].startswith('<context_data')]
            assert users == ['Debug Python imports', 'Follow-up about imports']
            keyboard.send_text('/sessions\r\r')
            await wait_for(lambda: browser.active)
            assert browser.search.text == 'pytho'
            keyboard.send_text('\x7fy\r')
            await wait_for(lambda: browser.pending_delete is None and browser.notice == 'Deletion cancelled.')
            assert any(r['id'] == 'python' for r in browser.records)
            keyboard.send_text('\x7fyes\r')
            await wait_for(lambda: not browser.working and len(browser.records) == 2)
            assert args.session == 'python'  # CLI receives new ID when the tab closes.
            keyboard.send_text('\x1b')
            await wait_for(lambda: not browser.active and not ui.busy and args.session != 'python')
            assert 'Python answer' not in ui.transcript and ui.composer.text == ''
            keyboard.send_text('/sessions\r\r')
            await wait_for(lambda: browser.active)
            keyboard.send_text('\t\x15current')
            await wait_for(lambda: len(browser.matches) == 1)
            keyboard.send_text('\r')
            await wait_for(lambda: not browser.active and not ui.busy and args.session == 'current')
            assert 'Current project question' in ui.transcript and 'Python answer' not in ui.transcript
            keyboard.send_text('/sessions\r\r')
            await wait_for(lambda: browser.active)
            keyboard.send_text('\x1b')
            await wait_for(lambda: not browser.active and not ui.busy)
            assert args.session == 'current'
            keyboard.send_text('/bye\r\r')
            assert await task == 0

    asyncio.run(scenario())
    with sqlite3.connect(settings.state_dir / 'sessions.sqlite') as connection:
        assert connection.execute("SELECT * FROM checkpoints WHERE thread_id = 'python'").fetchall() == []
    assert len(backend.requests) == 2
