import pytest

from conftest import FakeBackend, embed, tool_call
from harness.memory import MemoryStore
from harness.runtime import AgentRuntime


def test_agent_updates_selected_memory_automatically_with_stable_id_and_vector(settings):
    embedded = []

    def tracked_embed(text):
        embedded.append(text)
        return embed(text)

    memory = MemoryStore(settings.state_dir / 'memory', tracked_embed)
    old = memory.save('I live in Pune.', 'explicit', [{'turn_id': 'u0', 'quote': 'I live in Pune.'}], str(settings.workspace))
    backend = FakeBackend([tool_call('read_memory', {'memory_id': old['id']}),
                           tool_call('update_memory', {'memory_id': old['id'], 'statement': 'I live in Mumbai.'}),
                           'Updated your memory.'])
    runtime = AgentRuntime(settings, backend, memory=memory,
        approve=lambda *args: pytest.fail('Memory edits must not request a second approval'))
    try:
        runtime.turn(f"Modify memory {old['id']}: I moved to Mumbai; update the location.")
        updated = memory.find(old['id'])
        assert updated['id'] == old['id'] and updated['statement'] == 'I live in Mumbai.'
        assert embedded[-1] == 'I live in Mumbai.'
        assert updated['edit_history'][-1]['statement'] == 'I live in Pune.'
        assert updated['evidence'][0]['evidence_type'] == 'edit_request'
        assert updated['edited_by'] == 'user_requested_agent'
        assert all(event['result']['status'] == 'ok' for event in runtime.last_trace if event['type'] == 'tool')
        assert len(backend.requests) == 3
        assert runtime.tools.memory_edit is None
        assert runtime.snapshot('default')['memories'] == []
    finally:
        runtime.close()
        memory.close()


def test_unrequested_or_different_memory_updates_are_refused(settings):
    memory = MemoryStore(settings.state_dir / 'memory', embed)
    one = memory.save('First fact', 'explicit', [{'turn_id': 'u0', 'quote': 'First fact'}], 'workspace')
    two = memory.save('Second fact', 'explicit', [{'turn_id': 'u1', 'quote': 'Second fact'}], 'workspace')
    backend = FakeBackend([tool_call('update_memory', {'memory_id': one['id'], 'statement': 'Bad update'}),
                           'No update.', '{"candidates": []}',
                           tool_call('update_memory', {'memory_id': two['id'], 'statement': 'Wrong target'}),
                           'Wrong target refused.'])
    runtime = AgentRuntime(settings, backend, memory=memory)
    try:
        runtime.turn('Hello')
        assert runtime.last_trace[1]['result']['status'] == 'error'
        runtime.turn(f"Modify memory {one['id']}: Update this fact.")
        result = next(event['result'] for event in runtime.last_trace if event['type'] == 'tool')
        assert result['status'] == 'error' and 'only the selected memory' in result['content']
        assert memory.find(one['id'])['statement'] == 'First fact'
        assert memory.find(two['id'])['statement'] == 'Second fact'
    finally:
        runtime.close()
        memory.close()


def test_stale_edits_and_secrets_cannot_overwrite_memory(settings):
    memory = MemoryStore(settings.state_dir / 'memory', embed)
    old = memory.save('A fact', 'explicit', [{'turn_id': 'u0', 'quote': 'A fact'}], 'workspace')
    try:
        with pytest.raises(ValueError, match='changed during this edit'):
            memory.update_statement(old['id'], 'New fact', instruction='Fix it', turn_id='u1', expected_updated_at='stale')
        with pytest.raises(ValueError, match='secrets'):
            memory.update_statement(old['id'], 'password: SuperSecret123', instruction='Fix it', turn_id='u1',
                                    expected_updated_at=old['updated_at'])
        assert memory.find(old['id'])['statement'] == 'A fact'
    finally:
        memory.close()
