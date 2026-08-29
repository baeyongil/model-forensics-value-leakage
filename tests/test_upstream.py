from __future__ import annotations

from model_forensics.upstream import detect_license_files


def test_license_detection_is_case_insensitive_and_root_only(tmp_path) -> None:
    (tmp_path / "LICENSE.md").write_text("x", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/LICENSE").write_text("nested", encoding="utf-8")
    assert detect_license_files(tmp_path) == ["LICENSE.md"]
