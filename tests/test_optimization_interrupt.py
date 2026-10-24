import asyncio
from dataclasses import replace
import json
import threading

import httpx
import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from conftest import FakeBackend
from harness.backend import LlamaBackend
from harness.cancellation import TurnInterrupted
from harness.cli import chat, parser
from harness.context import ContextAssembler
from harness.curation import OllamaExtractor
from harness.tui import ChatUI
from test_tui import wait_until


@pytest.mark.parametrize('operation', ['generation', 'counting'])
def test_cancel_interrupts_wait_before_response_headers_and_backend_can_be_reused(settings, operation):
    started = threading.Event()
    first = True

    async def handler(request):
        nonlocal first
        if first:
            first = False
            started.set()
            await asyncio.sleep(60)
        if request.url.path.endswith('input_tokens'):
            return httpx.Response(200, json={'input_tokens': 4})
        data = 'data: ' + json.dumps({'choices': [{'delta': {'content': 'Recovered'}}]}) + '\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=data)

    backend = LlamaBackend(settings)
    backend._async_client = httpx.AsyncClient(base_url=settings.server_url, transport=httpx.MockTransport(handler))
    failures = []

    def generate():
        try:
            if operation == 'counting':
                backend.count_tokens({'messages': []})
            else:
                backend.complete({'messages': []}, on_text=lambda _: None)
        except TurnInterrupted:
            failures.append('interrupted')

    worker = threading.Thread(target=generate)
    worker.start()
    try:
        assert started.wait(2)
        backend.cancel()
        worker.join(2)
        assert not worker.is_alive() and failures == ['interrupted']
        backend.reset_cancel()
        assert backend.complete({'messages': []}, on_text=lambda _: None)[0]['content'] == 'Recovered'
    finally:
        backend.cancel()
        worker.join(2)
        backend.close()


def test_esc_interrupts_turn_and_same_cli_session_accepts_next_prompt(settings, monkeypatch):
    settings = replace(settings, memory_enabled=False)
    started = threading.Event()
    received = []

    async def scenario():
        with create_pipe_input() as keyboard:
            ui = ChatUI(input=keyboard, output=DummyOutput())

            class Backend(FakeBackend):
                def complete(self, request, on_text=None):
                    received.append(request)
                    if len(received) == 1:
                        started.set()
                        while not ui.turn_cancelled.wait(.01):
                            pass
                        on_text('Cancelled output')
                        raise AssertionError('Cancelled turn must not return normally')
                    on_text('Second answer')
                    return {'role': 'assistant', 'content': 'Second answer'}, {}

                def close(self):
                    pass

            backend = Backend()
            monkeypatch.setattr('harness.backend.ServerManager.start', lambda _: settings)
            monkeypatch.setattr('harness.backend.LlamaBackend', lambda _: backend)
            monkeypatch.setattr('harness.tui.ChatUI.banner', lambda _: None)
            args = parser().parse_args(['--session', 'esc-test'])
            task = asyncio.create_task(ui.run_async(lambda: chat(args, settings, ui)))
            await wait_until(lambda: not ui.busy)
            keyboard.send_text('First question\r')
            await wait_until(started.is_set)
            keyboard.send_text('\x1b')
            await wait_until(lambda: not ui.busy and 'Turn interrupted' in ui.transcript)
            assert not task.done() and not ui.cancelled.is_set() and args.session == 'esc-test'
            assert 'Cancelled output' not in ui.transcript
            keyboard.send_text('Second question\r')
            await wait_until(lambda: not ui.busy and 'Second answer' in ui.transcript)
            assert len(received) == 2
            keyboard.send_text('/bye\r\r')
            assert await task == 0

    asyncio.run(scenario())


def test_transcript_append_reuses_complete_lines_and_preserves_ansi_selection():
    with create_pipe_input() as keyboard:
        ui = ChatUI(input=keyboard, output=DummyOutput())
        original = ui._wrap_logical_line
        calls = []

        def counted(carry, text, width):
            calls.append(text)
            return original(carry, text, width)

        ui._wrap_logical_line = counted
        ui.append(''.join(f'line {i}\n' for i in range(200)))
        ui.wrapped_transcript(20)
        calls.clear()
        ui.append('\x1b[90;3mwide 界\nnext\x1b[0m')
        rendered = ui.wrapped_transcript(20)
        assert len(calls) == 2
        reference = original('', ui.transcript, 20)[0]
        def flatten(fragments):
            return [(style, char) for style, text in fragments for char in text]
        assert flatten(rendered) == flatten(reference)
        assert 'next' in ''.join(text for style, text in rendered if 'italic' in style)
        ui.wrapped_transcript(12)
        assert flatten(ui._wrapped) == flatten(original('', ui.transcript, 12)[0])


def test_context_reuses_diagnostic_counts_but_always_counts_complete_request(settings):
    backend = FakeBackend()
    calls = []

    def count(request):
        calls.append(request)
        return backend.count_tokens(request)

    assembler = ContextAssembler(settings, count)
    messages = [{'role': 'user', 'content': 'hello', '_id': 'u1'},
                {'role': 'assistant', 'content': 'answer', '_id': 'a1'}]
    assembler.assemble(system='first timestamp', messages=messages)
    calls.clear()
    assembler.assemble(system='new timestamp', messages=messages)
    assert len(calls) == 1 and calls[0]['messages'][0]['content'] == 'new timestamp'
    messages[-1]['content'] = 'changed answer'
    calls.clear()
    assembler.assemble(system='new timestamp', messages=messages)
    assert len(calls) == 2


def test_card_navigation_reuses_layout_instead_of_rebuilding_every_card():
    with create_pipe_input() as keyboard:
        ui = ChatUI(input=keyboard, output=DummyOutput())
        browser = ui.memory_browser
        browser.set_records([{'id': f'{i:032x}', 'kind': 'explicit', 'statement': f'Fact {i}',
                              'updated_at': '2026-10-24T00:00:00+00:00'} for i in range(500)])
        calls = []
        original = browser.cards.render_card

        def counted(record, selected):
            calls.append(record['id'])
            return original(record, selected)

        browser.cards.render_card = counted
        browser.cards.create_content(80, 20)
        calls.clear()
        browser.select(100)
        content = browser.cards.create_content(80, 20)
        assert len(calls) == 1
        assert any('Fact' in text for _, text in content.get_line(browser.cards.cursor_row + 1))


def test_curation_retries_with_diagnostic_feedback_and_accepts_complete_fenced_json(settings):
    requests, events = [], []

    class Audit:
        def log(self, kind, content):
            events.append((kind, content))

    valid = {'candidates': [{'statement': 'I live in Pune.', 'fact_key': 'user.location',
             'kind': 'user_stated', 'confidence': .96, 'evidence': [{'turn_id': 'u1', 'quote': 'I live in Pune.'}]}]}

    def handler(request):
        if request.url.path == '/api/show':
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        requests.append(body)
        text = '{"candidates": [' if len(requests) == 1 else '```json\n' + json.dumps(valid) + '\n```'
        return httpx.Response(200, text=json.dumps({'done': True, 'done_reason': 'stop', 'message': {'content': text}})+'\n')

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    try:
        records, _ = extractor.extract([{'id': 'u1', 'content': 'I live in Pune.'}], lambda: False, Audit())
        assert records == valid['candidates']
        assert 'invalid_json' in requests[1]['messages'][0]['content']
        invalid = [data for kind, data in events if kind == 'memory_extraction_invalid']
        assert invalid[0]['reason'] == 'invalid_json' and 'Pune' not in json.dumps(invalid)
    finally:
        extractor.close()


def test_curation_does_not_salvage_partial_or_invalid_records(settings):
    def handler(request):
        if request.url.path == '/api/show':
            return httpx.Response(200, json={})
        return httpx.Response(200, text=json.dumps({'done': True, 'message': {'content': '{"candidates": ['}})+'\n')

    extractor = OllamaExtractor(settings, client=httpx.Client(base_url=settings.memory_server_url,
        transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(ValueError, match='invalid_json'):
            extractor.extract([{'id': 'u1', 'content': 'I live in Pune.'}], lambda: False)
    finally:
        extractor.close()


def test_cancelled_tool_response_never_executes_and_next_turn_does_not_replay(settings):
    from conftest import tool_call
    from harness.runtime import AgentRuntime
    stop = threading.Event()

    class Backend(FakeBackend):
        def complete(self, request, on_text=None):
            result = super().complete(request, on_text)
            if len(self.requests) == 1:
                stop.set()
            return result

    def check():
        if stop.is_set():
            raise TurnInterrupted()

    backend = Backend([tool_call('write_file', {'path': 'must-not-exist.txt', 'content': 'bad'}), 'Continued'])
    runtime = AgentRuntime(replace(settings, memory_enabled=False), backend,
                           approve=lambda *_: True, check_cancelled=check)
    try:
        with pytest.raises(TurnInterrupted):
            runtime.turn('First request')
        assert not (settings.workspace / 'must-not-exist.txt').exists()
        stop.clear()
        assert runtime.turn('Continue safely') == 'Continued'
        assert not (settings.workspace / 'must-not-exist.txt').exists()
        outcomes = [json.loads(m['content']) for m in runtime.snapshot('default')['messages'] if m['role'] == 'tool']
        assert outcomes[0]['status'] == 'interrupted'
    finally:
        runtime.close()
