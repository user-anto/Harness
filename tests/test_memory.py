from datetime import datetime, timedelta, timezone

import pytest
from qdrant_client.models import Distance, VectorParams

from harness.memory import COLLECTION, MemoryStore
from conftest import embed


def evidence(turn="u1", quote="I prefer concise answers"):
    return [{"turn_id": turn, "quote": quote}]


def test_explicit_persists_independent_of_logs(settings):
    path = settings.state_dir / "memory"
    store = MemoryStore(path, embed)
    saved = store.save("I prefer concise answers", "explicit", evidence(), "project-a")
    store.close()
    store = MemoryStore(path, embed)
    try:
        retrieved = store.retrieve("concise answers", "project-b")
        assert retrieved[0]["id"] == saved["id"]
        assert retrieved[0]["workspace"] == "project-a"
        assert not retrieved[0]["tentative"]
    finally:
        store.close()


def test_legacy_collection_untouched_and_not_searched(settings):
    store = MemoryStore(settings.state_dir / "memory", embed)
    try:
        store.client.create_collection("conversation_history", vectors_config=VectorParams(size=128, distance=Distance.COSINE))
        assert store.retrieve("anything", "a") == []
        assert store.client.collection_exists("conversation_history")
        assert not store.client.collection_exists(COLLECTION)
    finally:
        store.close()


def test_conflicts_precedence_and_forget_erases_versions(settings):
    store = MemoryStore(settings.state_dir / "memory", embed)
    try:
        store.save("I prefer detailed answers", "user_stated", evidence(), "a")
        explicit = store.save("I prefer concise answers", "explicit", evidence("u2"), "a")
        refused = store.save("I prefer detailed answers", "inferred", evidence() + evidence("u3"), "a", confidence=.8)
        assert refused["id"] == explicit["id"]
        assert len(store.records()) == 1
        assert len(store.records(active_only=False)) == 2
        assert store.forget(explicit["id"])
        assert store.records(active_only=False) == []
    finally:
        store.close()


def test_inference_requires_two_real_user_turns_and_expires(settings):
    now = datetime.now(timezone.utc)
    store = MemoryStore(settings.state_dir / "memory", embed, now=lambda: now)
    turns = [{"id": "a", "content": "I use Python every day"}, {"id": "b", "content": "Python is my main language"}]
    candidate = {"statement": "The user likely codes primarily in Python", "kind": "inferred",
                 "fact_key": "user.language", "confidence": .9,
                 "evidence": [{"turn_id": t["id"], "quote": t["content"]} for t in turns]}
    try:
        assert store.save_candidates([candidate], turns[:1], "project") == []
        saved = store.save_candidates([candidate], turns, "project")
        assert saved[0]["confidence"] == .85
        assert store.retrieve("Python main language", "project")[0]["tentative"]
        now += timedelta(days=91)
        assert store.records() == []
    finally:
        store.close()


def test_hallucinated_evidence_secret_and_irrelevant_recall(settings):
    store = MemoryStore(settings.state_dir / "memory", embed)
    try:
        with pytest.raises(ValueError, match="secrets"):
            store.save("password is hunter22", "explicit", evidence(quote="password is hunter22"), "a")
        candidate = {"statement": "Likes tea", "fact_key": "user.drink", "kind": "user_stated", "confidence": 1,
                     "evidence": [{"turn_id": "missing", "quote": "I like tea"}]}
        assert store.save_candidates([candidate], [{"id": "u", "content": "hello"}], "a") == []
        store.save("I prefer concise answers", "explicit", evidence(), "a")
        assert store.retrieve("quantum zebra telescope", "b") == []
    finally:
        store.close()


def test_duplicate_facts_reinforce_without_growing_versions(settings):
    store = MemoryStore(settings.state_dir / "memory", embed)
    try:
        first = store.save("I prefer concise answers", "explicit", evidence(), "a")
        second = store.save("I prefer concise answers", "explicit", evidence("u2"), "a")
        assert first["id"] == second["id"]
        assert len(store.records(active_only=False)) == 1
        assert len(second["evidence"]) == 2
    finally:
        store.close()
