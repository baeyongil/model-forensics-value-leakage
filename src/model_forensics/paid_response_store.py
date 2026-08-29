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
import secrets
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from model_forensics.io import canonical_json, stable_hash

PAID_RESPONSE_STORE_PROTOCOL = "paid-response-checkpoint-v1"
UNCERTAIN_PAID_ATTEMPT_PROTOCOL = "uncertain-paid-attempt-v1"
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PaidResponseStoreError(RuntimeError):
    """A paid-response checkpoint is corrupt or disagrees with its request."""


def _without_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "record_hash"}


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_LOCK_OPEN_FLAGS = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_LOCK_CREATE_FLAGS = _LOCK_OPEN_FLAGS | os.O_CREAT | os.O_EXCL


def _assert_directory_descriptor(descriptor: int, *, label: str) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise PaidResponseStoreError(f"{label} is not a directory")
    return observed


def _open_directory_at(parent_descriptor: int, name: str, *, label: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    except OSError as exc:
        raise PaidResponseStoreError(f"{label} is unavailable or unsafe") from exc
    try:
        _assert_directory_descriptor(descriptor, label=label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_json_at(directory_descriptor: int, name: str, *, label: str) -> Any:
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_descriptor)
    except OSError as exc:
        raise PaidResponseStoreError(f"{label} is unavailable or unsafe") from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise PaidResponseStoreError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            return json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaidResponseStoreError(f"{label} is not valid UTF-8 JSON") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _entry_exists_at(directory_descriptor: int, name: str, *, label: str) -> bool:
    try:
        observed = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PaidResponseStoreError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise PaidResponseStoreError(f"{label} is not a regular file")
    return True


def _write_json_at(directory_descriptor: int, name: str, value: Any, *, label: str) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary_name = f".{name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            _WRITE_FLAGS,
            0o600,
            dir_fd=directory_descriptor,
        )
        written = 0
        while written < len(payload):
            chunk = os.write(descriptor, payload[written:])
            if chunk <= 0:  # pragma: no cover - defensive POSIX invariant
                raise OSError("short write while persisting paid-response JSON")
            written += chunk
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    except OSError as exc:
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except OSError:
            pass
        raise PaidResponseStoreError(f"could not durably write {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _exclusive_lock_at(directory_descriptor: int, name: str):  # type: ignore[no-untyped-def]
    """Serialize updates using a lock opened relative to an anchored directory."""

    import fcntl

    descriptor = -1
    last_error: OSError | None = None
    for _ in range(3):
        try:
            descriptor = os.open(name, _LOCK_OPEN_FLAGS, dir_fd=directory_descriptor)
            break
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    name,
                    _LOCK_CREATE_FLAGS,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                break
            except FileExistsError as exc:
                # Another process won the first-create race. Re-open the exact
                # regular file instead of failing a safe local synchronization.
                last_error = exc
                continue
            except OSError as exc:
                last_error = exc
                break
        except OSError as exc:
            last_error = exc
            break
    if descriptor < 0:
        raise PaidResponseStoreError("paid-response lock is unavailable or unsafe") from last_error
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise PaidResponseStoreError("paid-response lock is not a regular file")
        with os.fdopen(descriptor, "a+") as handle:
            descriptor = -1
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class PaidResponseStore:
    """Private checkpoint directory keyed by logical route and request hash."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._directory_fd = -1
        self._claim_directory_fd = -1
        self._uncertain_directory_fd = -1
        if self.directory.is_symlink():
            raise PaidResponseStoreError("paid-response store must not be a symbolic link")
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.directory.is_dir() or self.directory.is_symlink():
            raise PaidResponseStoreError("paid-response store is not a safe directory")
        try:
            self._directory_fd = os.open(self.directory, _DIRECTORY_FLAGS)
        except OSError as exc:
            raise PaidResponseStoreError(
                "paid-response store directory cannot be anchored"
            ) from exc
        root_stat = _assert_directory_descriptor(self._directory_fd, label="paid-response store")
        try:
            os.fchmod(self._directory_fd, 0o700)
        except OSError:  # pragma: no cover - platform permission model
            pass
        self._resolved_directory = str(self.directory.resolve(strict=True))
        try:
            parent_descriptor = os.open(self.directory.parent, _DIRECTORY_FLAGS)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError as exc:
            self._close_descriptors()
            raise PaidResponseStoreError(
                "paid-response store directory creation is not crash-durable"
            ) from exc
        for child in (".claims", ".uncertain"):
            try:
                os.mkdir(child, mode=0o700, dir_fd=self._directory_fd)
            except FileExistsError:
                pass
            except OSError as exc:
                self._close_descriptors()
                raise PaidResponseStoreError(
                    "paid-response control directory could not be created"
                ) from exc
            child_descriptor = _open_directory_at(
                self._directory_fd,
                child,
                label="paid-response control directory",
            )
            try:
                os.fchmod(child_descriptor, 0o700)
            except OSError:  # pragma: no cover - platform permission model
                pass
            if child == ".claims":
                self._claim_directory_fd = child_descriptor
            else:
                self._uncertain_directory_fd = child_descriptor
            try:
                os.fsync(self._directory_fd)
            except OSError as exc:
                self._close_descriptors()
                raise PaidResponseStoreError(
                    "paid-response control directory creation is not crash-durable"
                ) from exc
        claim_stat = _assert_directory_descriptor(
            self._claim_directory_fd,
            label="paid-response claim directory",
        )
        uncertain_stat = _assert_directory_descriptor(
            self._uncertain_directory_fd,
            label="uncertain paid-attempt directory",
        )
        self._anchored_identity = (
            self._resolved_directory,
            int(root_stat.st_dev),
            int(root_stat.st_ino),
            int(claim_stat.st_dev),
            int(claim_stat.st_ino),
            int(uncertain_stat.st_dev),
            int(uncertain_stat.st_ino),
        )
        self._lock_path = self.directory / ".checkpoint.lock"
        self._claim_directory = self.directory / ".claims"
        self._uncertain_directory = self.directory / ".uncertain"

    def _close_descriptors(self) -> None:
        for attribute in (
            "_uncertain_directory_fd",
            "_claim_directory_fd",
            "_directory_fd",
        ):
            descriptor = getattr(self, attribute, -1)
            if isinstance(descriptor, int) and descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, attribute, -1)

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown timing
        self._close_descriptors()

    def identity(self) -> tuple[str, int, int, int, int, int, int]:
        """Return the anchored identity, rejecting path replacement or symlinks."""

        try:
            current = os.stat(self.directory, follow_symlinks=False)
            resolved = str(self.directory.resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise PaidResponseStoreError(
                "paid-response store path identity is unavailable"
            ) from exc
        anchored = _assert_directory_descriptor(self._directory_fd, label="paid-response store")
        claim_anchored = _assert_directory_descriptor(
            self._claim_directory_fd,
            label="paid-response claim directory",
        )
        uncertain_anchored = _assert_directory_descriptor(
            self._uncertain_directory_fd,
            label="uncertain paid-attempt directory",
        )
        try:
            claim_current = os.stat(
                ".claims",
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            uncertain_current = os.stat(
                ".uncertain",
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise PaidResponseStoreError(
                "paid-response control directory identity is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or not stat.S_ISDIR(claim_current.st_mode)
            or not stat.S_ISDIR(uncertain_current.st_mode)
            or resolved != self._resolved_directory
            or int(current.st_dev) != int(anchored.st_dev)
            or int(current.st_ino) != int(anchored.st_ino)
            or int(claim_current.st_dev) != int(claim_anchored.st_dev)
            or int(claim_current.st_ino) != int(claim_anchored.st_ino)
            or int(uncertain_current.st_dev) != int(uncertain_anchored.st_dev)
            or int(uncertain_current.st_ino) != int(uncertain_anchored.st_ino)
        ):
            raise PaidResponseStoreError(
                "paid-response store path no longer names its anchored directory"
            )
        return self._anchored_identity

    @contextmanager
    def request_claim(self, *, key: str, request_fingerprint: str):  # type: ignore[no-untyped-def]
        """Hold one cross-process request claim through replay or settlement."""

        self._path(key)
        if _HASH_RE.fullmatch(request_fingerprint) is None:
            raise ValueError("request fingerprint must be a namespaced SHA-256")
        digest = key.split(":", 1)[1]
        self.identity()
        with _exclusive_lock_at(self._claim_directory_fd, f"{digest}.lock"):
            self.identity()
            yield

    def _uncertain_path(self, key: str) -> Path:
        self._path(key)
        return self._uncertain_directory / f"{key.split(':', 1)[1]}.json"

    @staticmethod
    def _validate_uncertain_record(
        record: Any,
        *,
        key: str,
        request_fingerprint: str,
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise PaidResponseStoreError("uncertain paid-attempt marker must be an object")
        if (
            record.get("schema_version") != 1
            or record.get("protocol_version") != UNCERTAIN_PAID_ATTEMPT_PROTOCOL
        ):
            raise PaidResponseStoreError("uncertain paid-attempt protocol mismatch")
        if record.get("store_key") != key:
            raise PaidResponseStoreError("uncertain paid-attempt key mismatch")
        if record.get("request_fingerprint") != request_fingerprint:
            raise PaidResponseStoreError("uncertain paid-attempt fingerprint mismatch")
        if record.get("attempt_state") != "dispatch_started_outcome_unknown":
            raise PaidResponseStoreError("uncertain paid-attempt state is invalid")
        if record.get("record_hash") != stable_hash(_without_hash(record)):
            raise PaidResponseStoreError("uncertain paid-attempt record hash mismatch")
        return record

    def load_uncertain_attempt(
        self,
        *,
        key: str,
        request_fingerprint: str,
    ) -> dict[str, Any] | None:
        self.identity()
        filename = self._uncertain_path(key).name
        with _exclusive_lock_at(self._directory_fd, ".checkpoint.lock"):
            if not _entry_exists_at(
                self._uncertain_directory_fd,
                filename,
                label="uncertain paid-attempt marker",
            ):
                return None
            record = self._validate_uncertain_record(
                _read_json_at(
                    self._uncertain_directory_fd,
                    filename,
                    label="uncertain paid-attempt marker",
                ),
                key=key,
                request_fingerprint=request_fingerprint,
            )
        self.identity()
        return record

    def mark_uncertain_attempt(
        self,
        *,
        key: str,
        request_fingerprint: str,
        logical_request_hash: str,
        model_id: str,
        purpose: str,
        reservation_id: str,
    ) -> dict[str, Any]:
        for value in (request_fingerprint, logical_request_hash, reservation_id):
            if _HASH_RE.fullmatch(value) is None:
                raise ValueError("uncertain paid-attempt hashes must be namespaced SHA-256")
        if not model_id or not purpose:
            raise ValueError("model_id and purpose must be non-empty")
        self.identity()
        filename = self._uncertain_path(key).name
        record: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": UNCERTAIN_PAID_ATTEMPT_PROTOCOL,
            "store_key": key,
            "request_fingerprint": request_fingerprint,
            "logical_request_hash": logical_request_hash,
            "model_id": model_id,
            "purpose": purpose,
            "reservation_id": reservation_id,
            "attempt_state": "dispatch_started_outcome_unknown",
        }
        record["record_hash"] = stable_hash(record)
        with _exclusive_lock_at(self._directory_fd, ".checkpoint.lock"):
            if _entry_exists_at(
                self._uncertain_directory_fd,
                filename,
                label="uncertain paid-attempt marker",
            ):
                observed = self._validate_uncertain_record(
                    _read_json_at(
                        self._uncertain_directory_fd,
                        filename,
                        label="uncertain paid-attempt marker",
                    ),
                    key=key,
                    request_fingerprint=request_fingerprint,
                )
                if observed != record:
                    raise PaidResponseStoreError(
                        "uncertain paid-attempt marker already exists with different content"
                    )
                return observed
            _write_json_at(
                self._uncertain_directory_fd,
                filename,
                record,
                label="uncertain paid-attempt marker",
            )
        self.identity()
        return record

    def resolve_uncertain_attempt(
        self,
        *,
        key: str,
        request_fingerprint: str,
        expected_record_hash: str,
    ) -> None:
        """Clear write-ahead state only after an authenticated checkpoint exists."""

        if _HASH_RE.fullmatch(expected_record_hash) is None:
            raise ValueError("uncertain attempt record hash must be a namespaced SHA-256")
        self.identity()
        marker_filename = self._uncertain_path(key).name
        checkpoint_filename = self._path(key).name
        with _exclusive_lock_at(self._directory_fd, ".checkpoint.lock"):
            if not _entry_exists_at(
                self._uncertain_directory_fd,
                marker_filename,
                label="uncertain paid-attempt marker",
            ):
                return
            record = self._validate_uncertain_record(
                _read_json_at(
                    self._uncertain_directory_fd,
                    marker_filename,
                    label="uncertain paid-attempt marker",
                ),
                key=key,
                request_fingerprint=request_fingerprint,
            )
            if record["record_hash"] != expected_record_hash:
                raise PaidResponseStoreError("uncertain paid-attempt reconciliation hash mismatch")
            if not _entry_exists_at(
                self._directory_fd,
                checkpoint_filename,
                label="paid response checkpoint",
            ):
                raise PaidResponseStoreError(
                    "uncertain paid attempt cannot resolve without a durable checkpoint"
                )
            self._validate_record(
                _read_json_at(
                    self._directory_fd,
                    checkpoint_filename,
                    label="paid response checkpoint",
                ),
                key=key,
                request_fingerprint=request_fingerprint,
            )
            try:
                os.unlink(marker_filename, dir_fd=self._uncertain_directory_fd)
                os.fsync(self._uncertain_directory_fd)
            except OSError as exc:
                raise PaidResponseStoreError(
                    "uncertain paid-attempt marker could not be durably resolved"
                ) from exc
        self.identity()

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
        if (
            record.get("schema_version") != 1
            or record.get("protocol_version") != PAID_RESPONSE_STORE_PROTOCOL
        ):
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
        self.identity()
        filename = self._path(key).name
        with _exclusive_lock_at(self._directory_fd, ".checkpoint.lock"):
            if not _entry_exists_at(
                self._directory_fd,
                filename,
                label="paid response checkpoint",
            ):
                return None
            record = self._validate_record(
                _read_json_at(
                    self._directory_fd,
                    filename,
                    label="paid response checkpoint",
                ),
                key=key,
                request_fingerprint=request_fingerprint,
            )
        self.identity()
        return record

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
        self.identity()
        filename = self._path(key).name
        if (
            _HASH_RE.fullmatch(request_fingerprint) is None
            or _HASH_RE.fullmatch(logical_request_hash) is None
        ):
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
        with _exclusive_lock_at(self._directory_fd, ".checkpoint.lock"):
            if _entry_exists_at(
                self._directory_fd,
                filename,
                label="paid response checkpoint",
            ):
                observed = self._validate_record(
                    _read_json_at(
                        self._directory_fd,
                        filename,
                        label="paid response checkpoint",
                    ),
                    key=key,
                    request_fingerprint=request_fingerprint,
                )
                if observed != record:
                    raise PaidResponseStoreError(
                        "paid response checkpoint already exists with different content"
                    )
                return observed
            _write_json_at(
                self._directory_fd,
                filename,
                record,
                label="paid response checkpoint",
            )
        self.identity()
        return record


__all__ = [
    "PAID_RESPONSE_STORE_PROTOCOL",
    "UNCERTAIN_PAID_ATTEMPT_PROTOCOL",
    "PaidResponseStore",
    "PaidResponseStoreError",
]
