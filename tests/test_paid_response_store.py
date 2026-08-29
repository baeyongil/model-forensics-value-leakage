from __future__ import annotations

from pathlib import Path

import pytest

from model_forensics.io import read_json, write_json
from model_forensics.paid_response_store import (
    PaidResponseStore,
    PaidResponseStoreError,
)


def _body(content: str = '{"status":"KNOWN","value":"42"}') -> dict:
    return {
        "id": "provider-response-1",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.001},
    }


def test_store_round_trip_is_content_authenticated_and_secret_safe(tmp_path: Path) -> None:
    store = PaidResponseStore(tmp_path)
    key = store.key(request_id="unit-1", model_id="vendor/model", purpose="final")
    fingerprint = store.fingerprint(
        endpoint="https://example.invalid/api",
        model_id="vendor/model",
        purpose="final",
        system_prompt=None,
        user_content="blind case",
        decoding={"temperature": 0},
    )
    stored = store.commit(
        key=key,
        request_fingerprint=fingerprint,
        logical_request_hash="sha256:" + "a" * 64,
        model_id="vendor/model",
        purpose="final",
        http_status=200,
        response_body=_body(),
    )
    loaded = store.load(key=key, request_fingerprint=fingerprint)
    assert loaded == stored
    assert loaded is not None
    assert loaded["response_body_hash"].startswith("sha256:")
    artifact = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "blind case" not in artifact
    assert "Bearer" not in artifact


def test_store_rejects_same_logical_route_with_changed_request(tmp_path: Path) -> None:
    store = PaidResponseStore(tmp_path)
    key = store.key(request_id="unit-1", model_id="vendor/model", purpose="final")
    first = store.fingerprint(
        endpoint="https://example.invalid/api",
        model_id="vendor/model",
        purpose="final",
        system_prompt=None,
        user_content="first",
        decoding={"temperature": 0},
    )
    second = store.fingerprint(
        endpoint="https://example.invalid/api",
        model_id="vendor/model",
        purpose="final",
        system_prompt=None,
        user_content="changed",
        decoding={"temperature": 0},
    )
    store.commit(
        key=key,
        request_fingerprint=first,
        logical_request_hash="sha256:" + "a" * 64,
        model_id="vendor/model",
        purpose="final",
        http_status=200,
        response_body=_body(),
    )
    with pytest.raises(PaidResponseStoreError, match="fingerprint"):
        store.load(key=key, request_fingerprint=second)


def test_store_rejects_tampered_body_or_record_hash(tmp_path: Path) -> None:
    store = PaidResponseStore(tmp_path)
    key = store.key(request_id="unit-1", model_id="vendor/model", purpose="final")
    fingerprint = store.fingerprint(
        endpoint="https://example.invalid/api",
        model_id="vendor/model",
        purpose="final",
        system_prompt=None,
        user_content="first",
        decoding={"temperature": 0},
    )
    store.commit(
        key=key,
        request_fingerprint=fingerprint,
        logical_request_hash="sha256:" + "a" * 64,
        model_id="vendor/model",
        purpose="final",
        http_status=200,
        response_body=_body(),
    )
    path = next(tmp_path.glob("*.json"))
    row = read_json(path)
    row["response_body"]["choices"][0]["message"]["content"] = "tampered"
    write_json(path, row)
    with pytest.raises(PaidResponseStoreError, match="hash"):
        store.load(key=key, request_fingerprint=fingerprint)


def test_existing_identical_commit_is_idempotent(tmp_path: Path) -> None:
    store = PaidResponseStore(tmp_path)
    key = store.key(request_id="unit-1", model_id="vendor/model", purpose="final")
    fingerprint = store.fingerprint(
        endpoint="https://example.invalid/api",
        model_id="vendor/model",
        purpose="final",
        system_prompt=None,
        user_content="first",
        decoding={"temperature": 0},
    )
    kwargs = {
        "key": key,
        "request_fingerprint": fingerprint,
        "logical_request_hash": "sha256:" + "a" * 64,
        "model_id": "vendor/model",
        "purpose": "final",
        "http_status": 200,
        "response_body": _body(),
    }
    assert store.commit(**kwargs) == store.commit(**kwargs)
