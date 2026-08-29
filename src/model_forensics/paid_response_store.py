"""Durable, content-authenticated storage for paid provider responses.

Requests themselves are represented only by hashes.  A successful HTTP body is
committed before downstream parsing or scientific use, allowing a resumed run
to reuse the paid response without another network call—even when its completion
is malformed and must become a terminal invalid measurement.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from model_forensics.io import canonical_json, read_json, stable_hash, write_json

PAID_RESPONSE_STORE_PROTOCOL = "paid-response-checkpoint-v1"
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PaidResponseStoreError(RuntimeError):
    """A paid-response checkpoint is corrupt or disagrees with its request."""


def _without_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "record_hash"}


@contextmanager
def _exclusive_lock(path: Path):  # type: ignore[no-untyped-def]
    """Serialize checkpoint updates across local resume processes."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        # fdopen owns the descriptor on the normal path; this fallback matters
        # only if fdopen itself raises.
        try:
            os.close(descriptor)
        except OSError:
            pass


class PaidResponseStore:
    """Private checkpoint directory keyed by logical route and request hash."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.directory.chmod(0o700)
        except OSError:  # pragma: no cover - permission model dependent
            pass
        self._lock_path = self.directory / ".checkpoint.lock"

    @staticmethod
    def key(*, request_id: str, model_id: str, purpose: str) -> str:
        if not request_id or not model_id or not purpose:
            raise ValueError("request_id, model_id, and purpose must be non-empty")
        return stable_hash(
            {
                "logical_request_hash": stable_hash(request_id),
                "model_id": model_id,
                "purpose": purpose,
            }
        )

    @staticmethod
    def fingerprint(
        *,
        endpoint: str,
        model_id: str,
        purpose: str,
        system_prompt: str | None,
        user_content: str,
        decoding: Mapping[str, Any],
    ) -> str:
        if not endpoint or not model_id or not purpose or not user_content:
            raise ValueError("paid request fingerprint fields must be non-empty")
        return stable_hash(
            {
                "endpoint": endpoint,
                "model_id": model_id,
                "purpose": purpose,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "decoding": dict(decoding),
            }
        )

    def _path(self, key: str) -> Path:
        if _HASH_RE.fullmatch(key) is None:
            raise ValueError("paid response store key must be a namespaced SHA-256")
        return self.directory / f"{key.split(':', 1)[1]}.json"

    @staticmethod
    def _validate_record(
        record: Any,
        *,
        key: str,
        request_fingerprint: str,
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise PaidResponseStoreError("paid response checkpoint must be an object")
        if record.get("schema_version") != 1 or record.get(
            "protocol_version"
        ) != PAID_RESPONSE_STORE_PROTOCOL:
            raise PaidResponseStoreError("paid response checkpoint protocol mismatch")
        if record.get("store_key") != key:
            raise PaidResponseStoreError("paid response checkpoint key mismatch")
        if record.get("request_fingerprint") != request_fingerprint:
            raise PaidResponseStoreError("paid response request fingerprint mismatch")
        body = record.get("response_body")
        if not isinstance(body, Mapping):
            raise PaidResponseStoreError("paid response body is not an object")
        if record.get("response_body_hash") != stable_hash(dict(body)):
            raise PaidResponseStoreError("paid response body hash mismatch")
        if record.get("record_hash") != stable_hash(_without_hash(record)):
            raise PaidResponseStoreError("paid response record hash mismatch")
        return record

    def load(
        self,
        *,
        key: str,
        request_fingerprint: str,
    ) -> dict[str, Any] | None:
        path = self._path(key)
        with _exclusive_lock(self._lock_path):
            if not path.exists():
                return None
            return self._validate_record(
                read_json(path),
                key=key,
                request_fingerprint=request_fingerprint,
            )

    def commit(
        self,
        *,
        key: str,
        request_fingerprint: str,
        logical_request_hash: str,
        model_id: str,
        purpose: str,
        http_status: int,
        response_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self._path(key)
        if _HASH_RE.fullmatch(request_fingerprint) is None or _HASH_RE.fullmatch(
            logical_request_hash
        ) is None:
            raise ValueError("request hashes must be namespaced SHA-256 values")
        if (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 100 <= http_status <= 599
        ):
            raise ValueError("HTTP status must be an integer from 100 to 599")
        if not model_id or not purpose:
            raise ValueError("model_id and purpose must be non-empty")
        try:
            body = json.loads(canonical_json(dict(response_body)))
        except (TypeError, ValueError) as exc:
            raise PaidResponseStoreError("paid response body is not canonical JSON") from exc
        record: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": PAID_RESPONSE_STORE_PROTOCOL,
            "store_key": key,
            "request_fingerprint": request_fingerprint,
            "logical_request_hash": logical_request_hash,
            "model_id": model_id,
            "purpose": purpose,
            "http_status": http_status,
            "response_body": body,
            "response_body_hash": stable_hash(body),
        }
        record["record_hash"] = stable_hash(record)
        with _exclusive_lock(self._lock_path):
            if path.exists():
                observed = self._validate_record(
                    read_json(path),
                    key=key,
                    request_fingerprint=request_fingerprint,
                )
                if observed != record:
                    raise PaidResponseStoreError(
                        "paid response checkpoint already exists with different content"
                    )
                return observed
            write_json(path, record)
            try:
                path.chmod(0o600)
            except OSError:  # pragma: no cover - permission model dependent
                pass
        return record


__all__ = [
    "PAID_RESPONSE_STORE_PROTOCOL",
    "PaidResponseStore",
    "PaidResponseStoreError",
]
