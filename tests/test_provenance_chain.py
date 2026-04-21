"""Audit chain tests."""
import json

import pytest

from beet.provenance.chain import AuditChain, GENESIS


def test_first_entry_genesis(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    first = chain.append({"submission_id": "a1"})
    assert first["_prev_hash"] == GENESIS
    assert "_hash" in first


def test_chain_links_forward(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    e1 = chain.append({"submission_id": "a1"})
    e2 = chain.append({"submission_id": "a2"})
    e3 = chain.append({"submission_id": "a3"})
    assert e2["_prev_hash"] == e1["_hash"]
    assert e3["_prev_hash"] == e2["_hash"]


def test_validate_clean_chain(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    for i in range(5):
        chain.append({"submission_id": f"a{i}"})
    ok, errors = chain.validate()
    assert ok and errors == []


def test_validate_detects_mutation(tmp_path):
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path)
    chain.append({"submission_id": "a1"})
    chain.append({"submission_id": "a2"})

    # Tamper: modify the first entry
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["submission_id"] = "tampered"
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")

    ok, errors = chain.validate()
    assert not ok
    assert any("_hash" in e for e in errors)


def test_validate_detects_prev_hash_break(tmp_path):
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path)
    chain.append({"submission_id": "a1"})
    chain.append({"submission_id": "a2"})
    chain.append({"submission_id": "a3"})

    # Delete middle entry → forward hash chain breaks
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")

    ok, errors = chain.validate()
    assert not ok
    assert any("_prev_hash" in e for e in errors)


def test_len(tmp_path):
    chain = AuditChain(tmp_path / "audit.jsonl")
    for i in range(3):
        chain.append({"submission_id": f"a{i}"})
    assert len(chain) == 3
