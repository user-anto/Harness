"""Pure memory validation and bounded answer-note recovery eligibility."""

import re
import hashlib

from .security import contains_secret


REQUIRED_FIELDS = {"statement", "fact_key", "kind", "confidence", "evidence"}
PERSONAL_CLAIM = re.compile(
    r"\b(?:you\s+(?:are|have|use|live|prefer|work|own|like)|"
    r"the user\s+(?:is|has|uses|lives|prefers|works|owns|likes))\b", re.I)


def candidate_field_errors(candidate):
    """Report only fixed field paths/codes, never candidate text or unknown keys."""
    if not isinstance(candidate, dict):
        return [{"field": "candidate", "reason": "invalid_type"}]
    if set(candidate) != REQUIRED_FIELDS:
        errors = [{"field": name, "reason": "missing_field"} for name in sorted(REQUIRED_FIELDS - set(candidate))]
        if set(candidate) - REQUIRED_FIELDS:
            errors.append({"field": "candidate", "reason": "unexpected_fields"})
        return errors
    errors = []
    for field, minimum, maximum in (("statement", 1, 1000), ("fact_key", 3, 120)):
        value = candidate[field]
        if not isinstance(value, str):
            errors.append({"field": field, "reason": "invalid_type"})
        elif not minimum <= len(value) <= maximum:
            errors.append({"field": field, "reason": "invalid_length"})
        elif field == "fact_key" and not re.fullmatch(r"[a-zA-Z0-9_.:-]{3,120}", value):
            errors.append({"field": field, "reason": "invalid_format"})
    if candidate["kind"] not in ("user_stated", "inferred", "answer_note"):
        errors.append({"field": "kind", "reason": "invalid_kind"})
    if type(candidate["confidence"]) not in (int, float) or not 0 <= candidate["confidence"] <= 1:
        errors.append({"field": "confidence", "reason": "invalid_score"})
    evidence = candidate["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
        errors.append({"field": "evidence", "reason": "invalid_size"})
    else:
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"turn_id", "quote"}:
                errors.append({"field": "evidence", "reason": "invalid_fields"})
                continue
            for field in ("turn_id", "quote"):
                value = item[field]
                if not isinstance(value, str) or not value:
                    errors.append({"field": "evidence." + field, "reason": "invalid_type_or_empty"})
                elif field == "quote" and len(value) > 1000:
                    errors.append({"field": "evidence.quote", "reason": "invalid_length"})
    return errors


def candidate_policy_errors(candidate, turns):
    errors = candidate_field_errors(candidate)
    if errors:
        return errors
    if not turns:
        return [{"field": "evidence", "reason": "missing_source_turns"}]
    kind, evidence = candidate["kind"], candidate["evidence"]
    texts = [candidate["statement"], *(item["quote"] for item in evidence)]
    if kind == "answer_note":
        texts.append(turns[-1]["content"])
    if any(contains_secret(text) or "[REDACTED]" in text for text in texts):
        errors.append({"field": "candidate", "reason": "sensitive_content"})
    if candidate["confidence"] < (.9 if kind == "user_stated" else .7):
        errors.append({"field": "confidence", "reason": "below_threshold"})
    newest = turns[-1]["id"]
    if kind == "answer_note":
        reply = turns[-1].get("assistant_reply", "")
        if not reply:
            errors.append({"field": "evidence", "reason": "missing_assistant_reply"})
        elif any(e["turn_id"] != newest or e["quote"] not in reply for e in evidence):
            errors.append({"field": "evidence", "reason": "assistant_quote_or_turn_mismatch"})
        if PERSONAL_CLAIM.search(candidate["statement"] + " " + " ".join(e["quote"] for e in evidence)):
            errors.append({"field": "statement", "reason": "assistant_personal_guess"})
    else:
        lookup = {turn["id"]: turn["content"] for turn in turns}
        if not any(e["turn_id"] == newest for e in evidence):
            errors.append({"field": "evidence.turn_id", "reason": "newest_turn_not_cited"})
        if any(e["turn_id"] not in lookup or e["quote"] not in lookup.get(e["turn_id"], "") for e in evidence):
            errors.append({"field": "evidence", "reason": "user_quote_or_turn_mismatch"})
        if any("```" in lookup.get(e["turn_id"], "") or lookup.get(e["turn_id"], "").lstrip().startswith(">") for e in evidence):
            errors.append({"field": "evidence", "reason": "quoted_document_or_code"})
        if kind == "inferred" and len({e["turn_id"] for e in evidence}) < 2:
            errors.append({"field": "evidence", "reason": "two_distinct_turns_required"})
    return errors


def answer_note_expected(turns):
    """Conservative retry signal, not a decision to save or verify a fact."""
    if not turns:
        return False
    question = turns[-1].get("content", "")
    reply = turns[-1].get("assistant_reply", "")
    if (not question.strip() or question.lstrip().startswith(("/", ">")) or "```" in question
            or any(contains_secret(text) or "[REDACTED]" in text for text in (question, reply))):
        return False
    # Remove conversational invitations before checking for speculative personal claims.
    if re.fullmatch(r"\s*(?:hi|hello|hey|thanks|thank you|ok|okay)[.!\s]*", question, re.I):
        return False
    core = re.split(r"\b(?:would you like|let me know if)\b", reply, flags=re.I)[0]
    if len(core) < 200 or len(re.findall(r"\b\w+\b", core)) < 35 or PERSONAL_CLAIM.search(core):
        return False
    failure = re.compile(r"(?:i (?:cannot|can't|couldn't|am unable|was unable)|i'm unable|"
                         r"(?:the )?(?:web )?(?:search|tool|request) (?:failed|is unavailable)|"
                         r"(?:i (?:have )?)?(?:saved|remembered|updated|deleted) (?:your |the |this )?(?:memory|fact)|"
                         r"appears to be a typo|could you (?:clarify|rephrase)|did you mean)", re.I)
    return not failure.search(core[:200])



def note_excerpts(reply):
    """Number exact bounded source slices; prefer uncertainty over invitations."""
    candidates = []
    for index, match in enumerate(re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", reply)):
        quote = match[0].strip()[:200].rstrip()
        if len(quote) < 40 or re.search(r"\b(?:would you like|let me know)\b", quote, re.I):
            continue
        uncertain = bool(re.search(r"\b(?:unconfirmed|not (?:yet )?confirmed|candidate|possible|if confirmed|verification)\b", quote, re.I))
        candidates.append((not uncertain, index, quote))
    seen, excerpts = set(), []
    for _, _, quote in sorted(candidates):
        if quote in seen:
            continue
        seen.add(quote)
        excerpts.append({"id": "q" + str(len(excerpts) + 1), "quote": quote})
        if len(excerpts) == 6:
            break
    if not excerpts and len(reply.strip()) >= 40:
        excerpts.append({"id": "q1", "quote": reply.strip()[:200].rstrip()})
    return excerpts


def resolve_note_candidate(selection, turn, excerpts):
    """Keep provenance code-owned; the model can select only a supplied source."""
    if not isinstance(selection, dict) or set(selection) != {"statement", "confidence", "evidence_id"}:
        return None, [{"field": "candidate", "reason": "invalid_recovery_fields"}]
    identifier = selection["evidence_id"]
    lookup = {excerpt["id"]: excerpt["quote"] for excerpt in excerpts}
    if not isinstance(identifier, str) or identifier not in lookup:
        return None, [{"field": "evidence_id", "reason": "unknown_excerpt"}]
    if isinstance(selection["statement"], str) and len(selection["statement"]) > 300:
        return None, [{"field": "statement", "reason": "invalid_length"}]
    candidate = {"statement": selection["statement"], "confidence": selection["confidence"],
        "kind": "answer_note", "fact_key": "answer_note.exchange." + hashlib.sha256(turn["id"].encode()).hexdigest()[:24],
        "evidence": [{"turn_id": turn["id"], "quote": lookup[identifier]}]}
    errors = candidate_policy_errors(candidate, [turn])
    return candidate, errors
