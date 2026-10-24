"""Curated, self-contained Qdrant memories, opened only by an explicit runtime."""

from datetime import datetime, timedelta, timezone
import hashlib
from functools import wraps
import re
import threading
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from .config import private_dir
from .security import contains_secret

COLLECTION = "harness_curated_v1"
KINDS = {"explicit": 3, "user_stated": 2, "inferred": 1, "answer_note": 0}


def serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("Memory store is closed")
            return method(self, *args, **kwargs)
    return wrapped


def fact_key(statement: str) -> str:
    text = statement.lower().strip().rstrip(".!?")
    for pattern, key in [(r"^(?:my name is|i am called)\b", "user.name"),
                         (r"^i live in\b", "user.location"),
                         (r"^my (?:primary )?programming language is\b", "user.programming_language"),
                         (r"^i prefer (?:concise|brief|detailed|long|short) (?:answers|responses)\b", "user.answer_length")]:
        if re.search(pattern, text):
            return key
    return "statement." + hashlib.sha256(text.encode()).hexdigest()[:24]


class MemoryStore:
    def __init__(self, path, embed, *, client=None, now=None, embedding_model="test"):
        self.embed = embed
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.embedding_model = embedding_model
        self._lock = threading.RLock()
        self._closed = False
        self.client = client or QdrantClient(path=str(private_dir(path)), force_disable_check_same_thread=True)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.client.close()

    def _ensure(self, dimension):
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(COLLECTION, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
        else:
            vectors = self.client.get_collection(COLLECTION).config.params.vectors
            if vectors.size != dimension:
                raise ValueError("Embedding dimension changed. Restore the configured model or export and rebuild curated memory.")
            existing, _ = self.client.scroll(COLLECTION, limit=1, with_vectors=False)
            if existing and existing[0].payload.get("embedding_model") != self.embedding_model:
                raise ValueError("Embedding model changed. Restore it or export and rebuild curated memory.")

    @serialized
    def records(self, *, active_only=True):
        if not self.client.collection_exists(COLLECTION):
            return []
        result, offset = [], None
        while True:
            points, offset = self.client.scroll(COLLECTION, offset=offset, limit=100, with_vectors=False)
            for point in points:
                record = dict(point.payload)
                if (not active_only or record["status"] == "active") and not self._expired(record):
                    result.append(record)
            if offset is None:
                break
        return result

    def _expired(self, record):
        return bool(record.get("expires_at") and datetime.fromisoformat(record["expires_at"]) <= self.now())

    def save(self, statement, kind, evidence, workspace, *, key=None, confidence=1.0, before_commit=None,
             sources=None, user_prompt="", observed_at=None):
        statement = statement.strip()
        if not statement or len(statement) > 1000 or kind not in KINDS:
            raise ValueError("Invalid memory statement or kind")
        if (contains_secret(statement) or "[REDACTED]" in statement or
                any(contains_secret(e.get("quote", "")) or "[REDACTED]" in e.get("quote", "") for e in evidence)):
            raise ValueError("Credentials and secrets cannot be saved as memories")
        evidence = list({e["turn_id"]: e for e in evidence if e.get("turn_id") and e.get("quote")}.values())
        if not evidence or (kind == "inferred" and len(evidence) < 2):
            raise ValueError("Memories require user evidence; inferred insights require two distinct turns")
        key = key or fact_key(statement)
        note = None
        if kind == "answer_note":
            if contains_secret(user_prompt) or "[REDACTED]" in user_prompt:
                raise ValueError("Credentials and secrets cannot be saved as memories")
            key = key if key.startswith("answer_note.") else "answer_note." + key[:108]
            observed = datetime.fromisoformat(observed_at) if observed_at else self.now()
            if observed.tzinfo is None:
                raise ValueError("Answer-note dates must include a timezone")
            note = {"verification": "assistant_reported", "requires_revalidation": True,
                    "observed_at": observed.isoformat(),
                    "user_prompt": user_prompt[:4000], "related_sources": list(sources or [])[:5]}
        with self._lock:
            if self._closed or (before_commit and not before_commit()):
                return None
            existing = self._reinforce(statement, kind, evidence, self._active(key), note)
            if existing:
                return existing
        # Network embedding runs outside the database lock, so foreground recall can proceed.
        vector = self.embed(statement)
        with self._lock:
            if self._closed or (before_commit and not before_commit()):
                return None
            active = self._active(key)
            existing = self._reinforce(statement, kind, evidence, active, note)
            if existing:
                return existing
            return self._insert(statement, kind, evidence, workspace, key, confidence, vector, active, note)

    def _active(self, key):
        active = []
        if self.client.collection_exists(COLLECTION):
            offset = None
            while True:
                points, offset = self.client.scroll(COLLECTION, offset=offset, limit=100, with_vectors=False,
                    scroll_filter=Filter(must=[FieldCondition(key="fact_key", match=MatchValue(value=key)),
                                               FieldCondition(key="status", match=MatchValue(value="active"))]))
                active.extend(dict(p.payload) for p in points if not self._expired(p.payload))
                if offset is None:
                    break
        return active

    def _reinforce(self, statement, kind, evidence, active, note=None):
        for existing in active:
            if KINDS[existing["kind"]] > KINDS[kind]:
                return existing
        now = self.now()
        for existing in active:
            if existing["statement"].casefold() == statement.casefold() and existing["kind"] == kind:
                if existing.get("embedding_model") != self.embedding_model:
                    raise ValueError("Embedding model changed; restore it or export and rebuild curated memory")
                merged = list({e["turn_id"]: e for e in existing["evidence"] + evidence}.values())[-8:]
                existing.update(evidence=merged, updated_at=now.isoformat())
                if kind == "inferred":
                    existing["expires_at"] = (now + timedelta(days=90)).isoformat()
                if kind == "answer_note":
                    existing.update(note or {})
                    existing["expires_at"] = (datetime.fromisoformat(existing["observed_at"]) + timedelta(days=30)).isoformat()
                self.client.set_payload(COLLECTION, existing, points=[existing["id"]], wait=True)
                return existing

    def _insert(self, statement, kind, evidence, workspace, key, confidence, vector, active, note=None):
        now = self.now()
        self._ensure(len(vector))
        for existing in active:
            if existing.get("embedding_model") != self.embedding_model:
                raise ValueError("Embedding model changed; reindex first")
        # Upsert the new record first; readers resolve precedence if a later cleanup fails.
        record = {"schema_version": 1, "id": str(uuid.uuid4()), "statement": statement,
                  "fact_key": key, "kind": kind, "evidence": evidence,
                  "workspace": workspace, "created_at": now.isoformat(), "updated_at": now.isoformat(),
                  "confidence": min(float(confidence), 0.85) if kind in {"inferred", "answer_note"} else float(confidence),
                  "expires_at": (now + timedelta(days=90)).isoformat() if kind == "inferred" else None,
                  "status": "active", "embedding_model": self.embedding_model}
        if kind == "answer_note":
            record.update(note or {})
            record["expires_at"] = (datetime.fromisoformat(record["observed_at"]) + timedelta(days=30)).isoformat()
        self.client.upsert(COLLECTION, [PointStruct(id=record["id"], vector=vector, payload=record)], wait=True)
        for previous in active:
            self.client.set_payload(COLLECTION, {"status": "superseded", "superseded_by": record["id"]},
                                    points=[previous["id"]], wait=True)
        return record

    def save_candidates(self, candidates, user_turns, workspace, *, before_commit=None, sources=None, observed_at=None):
        """User facts need user quotes; historical answer notes need assistant quotes."""
        lookup = {turn["id"]: turn["content"] for turn in user_turns}
        if not lookup:
            return []
        newest = user_turns[-1]["id"]
        saved = []
        for candidate in candidates[:5]:
            try:
                kind = candidate["kind"]
                evidence = candidate["evidence"]
                if kind not in {"user_stated", "inferred", "answer_note"} or not 0 <= candidate["confidence"] <= 1:
                    continue
                if kind == "answer_note":
                    reply = user_turns[-1].get("assistant_reply", "")
                    personal_claim = re.compile(
                        r"\b(?:you\s+(?:are|have|use|live|prefer|work|own|like)|"
                        r"the user\s+(?:is|has|uses|lives|prefers|works|owns|likes))\b", re.I)
                    note_text = candidate["statement"] + " " + " ".join(e.get("quote", "") for e in evidence)
                    if (not reply or candidate["confidence"] < .7 or not evidence
                            or personal_claim.search(note_text)
                            or any(e["turn_id"] != newest or not e["quote"] or e["quote"] not in reply for e in evidence)
                            or not re.fullmatch(r"[a-zA-Z0-9_.:-]{3,120}", candidate["fact_key"])
                            or any(record["kind"] == "answer_note" for record in saved)):
                        continue
                    record = self.save(candidate["statement"], kind,
                        [dict(e, role="assistant") for e in evidence], workspace,
                        key=candidate["fact_key"], confidence=candidate["confidence"], before_commit=before_commit,
                        sources=sources, user_prompt=lookup[newest], observed_at=observed_at)
                    if record:
                        saved.append(record)
                    continue
                if not any(e["turn_id"] == newest for e in evidence):
                    continue
                if any(e["turn_id"] not in lookup or not e["quote"] or
                       e["quote"] not in lookup[e["turn_id"]] for e in evidence):
                    continue
                if kind == "user_stated" and candidate["confidence"] < .9:
                    continue
                if kind == "inferred" and candidate["confidence"] < .7:
                    continue
                if not re.fullmatch(r"[a-zA-Z0-9_.:-]{3,120}", candidate["fact_key"]):
                    continue
                # Quoted documents/code are not an automatic-memory source.
                if any("```" in lookup[e["turn_id"]] or lookup[e["turn_id"]].lstrip().startswith(">") for e in evidence):
                    continue
                key = candidate["fact_key"]
                canonical = fact_key(candidate["statement"])
                if kind == "user_stated" and not canonical.startswith("statement."):
                    key = canonical
                record = self.save(candidate["statement"], kind, evidence, workspace,
                                   key=key, confidence=candidate["confidence"],
                                   before_commit=before_commit)
                if record:
                    saved.append(record)
            except (KeyError, ValueError, TypeError):
                continue
        return saved

    def retrieve(self, query, workspace, limit=5):
        with self._lock:
            if self._closed or not self.client.collection_exists(COLLECTION):
                return []
        query_vector = self.embed(query)
        return self._retrieve(query, workspace, limit, query_vector)

    @serialized
    def _retrieve(self, query, workspace, limit, query_vector):
        self._ensure(len(query_vector))
        hits = self.client.query_points(COLLECTION, query=query_vector, limit=20,
                query_filter=Filter(must=[FieldCondition(key="status", match=MatchValue(value="active"))])).points
        words = set(re.findall(r"\w+", query.lower()))
        ranked = []
        for hit in hits:
            record = dict(hit.payload)
            if self._expired(record):
                continue
            if record.get("embedding_model") != self.embedding_model:
                raise ValueError("Embedding model changed; reindex first")
            overlap = len(words & set(re.findall(r"\w+", record["statement"].lower()))) / max(1, len(words))
            # Reject weak semantic matches, even if their memory type is explicit.
            if (hit.score < .35 and overlap < .25) or (hit.score < .65 and overlap == 0):
                continue
            age_days = max(0, (self.now() - datetime.fromisoformat(record["updated_at"])).total_seconds() / 86400)
            score = .70 * hit.score + .15 * overlap + .05 * KINDS[record["kind"]] / 3
            score += .05 / (1 + age_days / 30) + .05 * (record["workspace"] == workspace)
            ranked.append((score, record))
        seen, result = set(), []
        # Collapse conflicts by kind precedence first, then most recent user evidence.
        winners = {}
        for score, record in ranked:
            old = winners.get(record["fact_key"])
            if old is None or (KINDS[record["kind"]], record["updated_at"]) > (KINDS[old[1]["kind"]], old[1]["updated_at"]):
                winners[record["fact_key"]] = (score, record)
        for score, record in sorted(winners.values(), key=lambda pair: (-pair[0], pair[1]["id"])):
            if record["statement"].lower() in seen:
                continue
            seen.add(record["statement"].lower())
            recalled = {"id": record["id"], "statement": record["statement"], "kind": record["kind"],
                           "confidence": record["confidence"], "workspace": record["workspace"],
                           "updated_at": record["updated_at"], "tentative": record["kind"] in {"inferred", "answer_note"}}
            if record["kind"] == "answer_note":
                recalled.update({key: record.get(key) for key in
                    ("observed_at", "expires_at", "verification", "requires_revalidation", "related_sources", "user_prompt")})
                recalled["user_prompt"] = recalled.get("user_prompt", "")[:500]
            result.append(recalled)
            if len(result) >= min(limit, 5):
                break
        return result

    @serialized
    def find(self, ident):
        if not self.client.collection_exists(COLLECTION):
            return None
        try:
            full = str(uuid.UUID(ident))
        except ValueError:
            if not re.fullmatch(r"[a-fA-F0-9]{6,32}", ident):
                raise ValueError("Use a displayed short memory ID or its full UUID") from None
            matches = [record for record in self.records(active_only=False)
                       if record["id"].replace("-", "").startswith(ident.lower())]
            if len(matches) > 1:
                raise ValueError("That short ID matches multiple memories; use the full UUID")
            return matches[0] if matches else None
        points = self.client.retrieve(COLLECTION, [full])
        return dict(points[0].payload) if points else None

    def update_statement(self, ident, statement, *, instruction, turn_id, expected_updated_at, sources=None):
        statement = statement.strip()
        if not statement or len(statement) > 1000:
            raise ValueError("Memory statements must contain 1–1000 characters")
        if contains_secret(statement) or contains_secret(instruction) or "[REDACTED]" in statement:
            raise ValueError("Credentials and secrets cannot be saved as memories")
        with self._lock:
            record = self.find(ident)
            if not record or record["status"] != "active":
                raise ValueError("Active memory not found")
            if record["updated_at"] != expected_updated_at:
                raise ValueError("Memory changed during this edit; read it again before updating")
        vector = self.embed(statement)
        with self._lock:
            current = self.find(record["id"])
            if not current or current["updated_at"] != expected_updated_at or current["status"] != "active":
                raise ValueError("Memory changed during this edit; read it again before updating")
            self._ensure(len(vector))
            now = self.now()
            current["edit_history"] = (current.get("edit_history", []) + [
                {"statement": current["statement"], "updated_at": current["updated_at"]}])[-10:]
            current.update(statement=statement, updated_at=now.isoformat(), edited_by="user_requested_agent",
                evidence=[{"turn_id": turn_id, "quote": instruction[:1000], "role": "user", "evidence_type": "edit_request"}])
            if current["kind"] == "answer_note":
                current.update(observed_at=now.isoformat(), expires_at=(now + timedelta(days=30)).isoformat(),
                               related_sources=list(sources or [])[:5], requires_revalidation=True)
            elif current["kind"] == "inferred":
                current["expires_at"] = (now + timedelta(days=90)).isoformat()
            self.client.upsert(COLLECTION, [PointStruct(id=current["id"], vector=vector, payload=current)], wait=True)
            return current

    @serialized
    def forget(self, ident):
        found = self.find(ident)
        if found:
            # Erase every version of the fact, so an old superseded copy cannot return.
            key = found["fact_key"]
            self.client.delete(COLLECTION, Filter(must=[FieldCondition(key="fact_key", match=MatchValue(value=key))]), wait=True)
        return bool(found)

