"""Derive a canonical private session path from an authenticated reservation."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from model_forensics.gpu_budget import load_gpu_phase_budget_reservation

_PHASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}\Z")
_SESSION_HASH_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")


class RunpodSessionPathError(RuntimeError):
    """A reservation cannot safely identify a canonical host session path."""


def canonical_host_session_directory(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir() or _PHASE_RE.fullmatch(phase) is None:
        raise RunpodSessionPathError("project root or GPU phase is invalid")
    supplied = Path(reservation_path).resolve(strict=True)
    expected = (root / ".runpod" / "reservations" / f"{phase}.json").resolve()
    if supplied != expected:
        raise RunpodSessionPathError("reservation path is not canonical for the GPU phase")
    try:
        reservation = load_gpu_phase_budget_reservation(supplied)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RunpodSessionPathError("GPU reservation is not authenticated") from exc
    match = _SESSION_HASH_RE.fullmatch(reservation.session_hash)
    if reservation.phase != phase or match is None:
        raise RunpodSessionPathError("GPU reservation phase or session hash is invalid")

    sessions_root = root / ".runpod" / "sessions"
    if os.path.lexists(sessions_root):
        details = sessions_root.lstat()
        if sessions_root.is_symlink() or not stat.S_ISDIR(details.st_mode):
            raise RunpodSessionPathError("private sessions root is unsafe")
    session = sessions_root / match.group(1)
    if session.parent != sessions_root or not session.is_relative_to(root / ".runpod"):
        raise RunpodSessionPathError("derived GPU session path escapes private state")
    if os.path.lexists(session):
        details = session.lstat()
        if session.is_symlink() or not stat.S_ISDIR(details.st_mode):
            raise RunpodSessionPathError("existing GPU session path is unsafe")
    return session


__all__ = ["RunpodSessionPathError", "canonical_host_session_directory"]
