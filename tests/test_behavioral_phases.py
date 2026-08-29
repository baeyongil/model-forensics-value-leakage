from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from model_forensics.behavioral_phases import (
    BehavioralPhaseError,
    build_behavioral_generation_environment_identity,
    freeze_behavioral_generation_plan,
    load_behavioral_generation_phase,
    run_behavioral_generation_phase,
)
from model_forensics.io import read_json, read_jsonl, write_json, write_jsonl
from model_forensics.sampling import (
    FakeBackend,
    GenerationRequest,
    SamplingParameters,
    build_requests,
)


def _prompt_builder(task: str, condition: str, threshold: float | None) -> str:
    return f"task={task};condition={condition};threshold={threshold}"


def _requests(*, condition: str = "baseline", count: int = 5) -> list[GenerationRequest]:
    return build_requests(
        task="giraffe",
        condition=condition,
        count=count,
        threshold=None if condition == "baseline" else 41_000_000,
        master_seed=19,
        prompt_builder=_prompt_builder,
        parameters=SamplingParameters(max_new_tokens=32),
        randomize=False,
    )


class CountingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, ...]] = []

    def generate(self, requests: Sequence[GenerationRequest]):  # type: ignore[no-untyped-def]
        self.calls.append(tuple(request.request_id for request in requests))
        return super().generate(requests)


def test_production_environment_identity_binds_approval_and_full_runtime_provenance() -> None:
    execution = {
        "container_image_digest": "vllm/example@sha256:" + "1" * 64,
        "gpu_family": "H100_80GB",
        "gpu_count": 8,
        "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "vllm_wheel_sha256": "2" * 64,
    }
    backend = {
        "backend": "vllm_offline",
        "model_id": "Qwen/Qwen3.5-122B-A10B",
        "model_revision": "3" * 40,
        "tokenizer_id": "Qwen/Qwen3.5-122B-A10B",
        "tokenizer_revision": "3" * 40,
        "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "max_model_len": 32_768,
        "chat_template_kwargs_hash": "sha256:" + "4" * 64,
        "detokenization_kwargs_hash": "sha256:" + "5" * 64,
        "chat_template_hash": "sha256:" + "6" * 64,
        "vllm_version": "0.28.0",
        "transformers_version": "5.5.3",
    }

    identity = build_behavioral_generation_environment_identity(
        execution_environment=execution,
        backend_provenance=backend,
    )
    assert identity["execution_environment"] == execution
    assert identity["backend_provenance"] == backend
    assert identity["identity_hash"].startswith("sha256:")

    incomplete = dict(backend)
    incomplete.pop("chat_template_hash")
    with pytest.raises(BehavioralPhaseError, match="chat_template_hash"):
        build_behavioral_generation_environment_identity(
            execution_environment=execution,
            backend_provenance=incomplete,
        )


def test_freeze_plan_happens_before_backend_factory_and_is_content_addressed(
    tmp_path: Path,
) -> None:
    requests = _requests(count=3)
    constructed: list[bool] = []
    authorized: list[str] = []

    def factory() -> CountingBackend:
        assert (tmp_path / "generation_plan.json").exists()
        assert authorized
        constructed.append(True)
        return CountingBackend()

    result = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=factory,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=2,
        expected_backend_provenance=FakeBackend().provenance,
        before_backend=lambda plan: authorized.append(str(plan["plan_hash"])),
    )

    assert constructed == [True]
    assert authorized == [plan_hash := read_json(tmp_path / "generation_plan.json")["plan_hash"]]
    assert len(result.rows) == 3
    plan = read_json(tmp_path / "generation_plan.json")
    assert plan["plan_hash"] == plan_hash
    assert plan["phase"] == "baseline"
    assert plan["request_count"] == 3
    assert plan["plan_hash"].startswith("sha256:")


def test_generation_is_microbatched_resumable_and_does_not_repeat_completed_batches(
    tmp_path: Path,
) -> None:
    requests = _requests(count=5)
    first = CountingBackend()
    initial = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: first,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=2,
        expected_backend_provenance=first.provenance,
        max_new_batches=2,
    )
    assert initial.complete is False
    assert [len(call) for call in first.calls] == [2, 2]

    second = CountingBackend()
    resumed = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: second,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=2,
        expected_backend_provenance=second.provenance,
    )
    assert resumed.complete is True
    assert [len(call) for call in second.calls] == [1]
    assert [row["dispatch_order"] for row in resumed.rows] == list(range(5))
    assert len({row["run_id"] for row in resumed.rows}) == 5


def test_complete_resume_does_not_construct_backend(tmp_path: Path) -> None:
    requests = _requests(count=2)
    backend = CountingBackend()
    run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: backend,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=1,
        expected_backend_provenance=backend.provenance,
    )

    def forbidden_factory() -> CountingBackend:
        raise AssertionError("completed phase must not reconstruct the model")

    resumed = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=forbidden_factory,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=1,
        expected_backend_provenance=backend.provenance,
    )
    assert resumed.complete is True


def test_plan_mismatch_fails_before_backend_construction(tmp_path: Path) -> None:
    original = _requests(count=2)
    freeze_behavioral_generation_plan(
        requests=original,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=1,
        expected_backend_provenance=FakeBackend().provenance,
    )
    constructed = False

    def factory() -> CountingBackend:
        nonlocal constructed
        constructed = True
        return CountingBackend()

    with pytest.raises(BehavioralPhaseError, match="frozen generation plan mismatch"):
        run_behavioral_generation_phase(
            requests=_requests(count=3),
            backend_factory=factory,
            phase="baseline",
            checkpoint_dir=tmp_path,
            batch_size=1,
            expected_backend_provenance=FakeBackend().provenance,
        )
    assert constructed is False


def test_corrupt_checkpoint_fails_closed_without_regeneration(tmp_path: Path) -> None:
    requests = _requests(count=2)
    backend = CountingBackend()
    run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: backend,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=1,
        expected_backend_provenance=backend.provenance,
    )
    batch_path = tmp_path / "batches" / "batch-00000.jsonl"
    rows = read_jsonl(batch_path)
    rows[0]["answer"] = "tampered"
    write_jsonl(batch_path, rows)

    with pytest.raises(BehavioralPhaseError, match="record hash"):
        run_behavioral_generation_phase(
            requests=requests,
            backend_factory=lambda: (_ for _ in ()).throw(
                AssertionError("must not regenerate corrupt paid output")
            ),
            phase="baseline",
            checkpoint_dir=tmp_path,
            batch_size=1,
            expected_backend_provenance=backend.provenance,
        )


def test_phase_contract_rejects_baseline_treatment_mix(tmp_path: Path) -> None:
    with pytest.raises(BehavioralPhaseError, match="baseline phase"):
        freeze_behavioral_generation_plan(
            requests=[*_requests(count=1), *_requests(condition="above_good", count=1)],
            phase="baseline",
            checkpoint_dir=tmp_path,
            batch_size=1,
            expected_backend_provenance=FakeBackend().provenance,
        )


def test_load_phase_requires_final_manifest_and_exact_batch_inventory(tmp_path: Path) -> None:
    requests = _requests(count=2)
    backend = CountingBackend()
    run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: backend,
        phase="baseline",
        checkpoint_dir=tmp_path,
        batch_size=1,
        expected_backend_provenance=backend.provenance,
    )
    loaded = load_behavioral_generation_phase(tmp_path)
    assert loaded.complete is True
    assert len(loaded.rows) == 2

    manifest = read_json(tmp_path / "generation_manifest.json")
    manifest["batch_files"].append(dict(manifest["batch_files"][0]))
    write_json(tmp_path / "generation_manifest.json", manifest)
    with pytest.raises(BehavioralPhaseError, match=r"manifest hash|batch inventory"):
        load_behavioral_generation_phase(tmp_path)
