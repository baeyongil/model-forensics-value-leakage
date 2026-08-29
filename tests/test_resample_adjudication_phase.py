from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from model_forensics.adjudication import AdjudicationRequest, JudgeProvenance
from model_forensics.anchors import (
    AnchorCandidate,
    AnchorManifest,
    select_frozen_anchors,
    sentence_spans,
)
from model_forensics.io import (
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from model_forensics.record_checkpoint import RecordCheckpointStore
from model_forensics.resample_adjudication_phase import (
    ResampleAdjudicationGateError,
    ResampleAdjudicationPhaseError,
    load_authenticated_resample_generation,
    run_resample_adjudication_phase,
)
from model_forensics.resample_phases import generate_sentence_resampling_intermediates
from model_forensics.resample_runner import (
    BaseTrace,
    NeutralControlSpec,
    RawPrefixGenerationRequest,
    RawPrefixGenerationResult,
    ReplacementClassificationError,
    ReplacementClassificationRequest,
    ReplacementClassificationResult,
    ReplacementTokenTolerance,
    ResampleAllocationManifest,
    build_fixed_stage_two_allocation_manifest,
    build_initial_allocation_manifest,
)
from model_forensics.token_spans import token_stream_hash


@dataclass(frozen=True)
class FrozenInputs:
    directory: Path
    anchors: AnchorManifest
    bases: Mapping[str, BaseTrace]
    initial: ResampleAllocationManifest
    stage_two: ResampleAllocationManifest
    terminal_invalid_id: str


def _frozen_design() -> tuple[AnchorManifest, dict[str, BaseTrace]]:
    sentences = {
        "accuracy_commitment": "Accuracy should not be swayed by threshold 100.",
        "value_threshold_planning": "I should revise until the estimate crosses threshold 100.",
        "epistemic_control": "Multiplying the two observed quantities gives another estimate.",
    }
    candidates: list[AnchorCandidate] = []
    bases: dict[str, BaseTrace] = {}
    index = 0
    for sentence_class, sentence in sentences.items():
        for direction in ("above_good", "below_good"):
            for cell_index in range(4):
                trace_id = f"trace-{index:02d}"
                trace = f"Opening thought {index}. {sentence} Old downstream estimate 90."
                span = sentence_spans(trace)[1]
                candidates.append(
                    AnchorCandidate(
                        trace_id=trace_id,
                        sentence_class=sentence_class,
                        direction=direction,
                        sentence_index=span.index,
                        sentence_text=span.text,
                        char_start=span.start,
                        char_end=span.end,
                        initial_side="bad" if cell_index % 2 == 0 else "good",
                        final_flip=cell_index >= 2,
                        provenance={"fixture": trace_id},
                    )
                )
                bases[trace_id] = BaseTrace(
                    base_trace_id=trace_id,
                    prompt="Estimate the quantity.",
                    trace=trace,
                    threshold=100.0,
                    condition=direction,
                    task="giraffe",
                    provenance={"fixture": trace_id},
                )
                index += 1
    manifest = select_frozen_anchors(candidates, seed="resample-adjudication-test")
    return manifest, bases


class FixtureBackend:
    def __init__(self, *, malformed_request_id: str) -> None:
        self.malformed_request_id = malformed_request_id

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": "deterministic-test",
            "model_id": "no-network",
            "revision": "fixed",
            "synthetic_smoke": False,
        }

    def encode_prefix(
        self,
        messages: Sequence[Mapping[str, Any]],
        raw_thinking_prefix: str,
    ) -> Sequence[int]:
        del messages
        return (101, len(raw_thinking_prefix), 202)

    def encode_continuation(self, raw_text: str) -> Sequence[int]:
        if "genuinely different" in raw_text:
            return (301, 302, 303, 304, 305)
        return (301, 302, 303)

    def generate(
        self, requests: Sequence[RawPrefixGenerationRequest]
    ) -> Sequence[RawPrefixGenerationResult]:
        results: list[RawPrefixGenerationResult] = []
        for request in requests:
            if request.request_id == self.malformed_request_id:
                text = "Unclosed but preserved raw output."
            elif request.arm == "retain":
                text = " Continued reasoning.</think>Final answer: 80"
            else:
                text = "A genuinely different replacement. More.</think>Final answer: 120"
            completion_ids = tuple(ord(character) for character in text)
            results.append(
                RawPrefixGenerationResult(
                    request_id=request.request_id,
                    generated_text=text,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_tokens=len(request.prompt_token_ids),
                    completion_tokens=len(completion_ids),
                    backend_metadata={
                        "completion_token_ids": list(completion_ids),
                        "completion_token_ids_hash": token_stream_hash(
                            completion_ids, stream="completion"
                        ),
                        "seed": request.seed,
                    },
                )
            )
        return results


class FixtureEmbedder:
    def __init__(self, cosine: float = 0.0) -> None:
        self.cosine = cosine
        self.calls = 0

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "model_id": "fixture-embedder",
            "revision": "fixed",
            "device": "cpu",
        }

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        assert len(texts) == 2
        return ([1.0, 0.0], [self.cosine, (1.0 - self.cosine**2) ** 0.5])


class ReplayClassifier:
    def __init__(self, *, malformed_paid_index: int | None = None) -> None:
        self.malformed_paid_index = malformed_paid_index
        self.requests: list[str] = []
        self.paid_calls = 0
        self._cache: dict[str, ReplacementClassificationResult | Exception] = {}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "provider": "test",
            "classifier_version": "two-route-test-v1",
            "routes": ["classifier-a", "classifier-b"],
            "synthetic_smoke": False,
        }

    def classify(
        self, request: ReplacementClassificationRequest
    ) -> ReplacementClassificationResult:
        self.requests.append(request.request_hash)
        cached = self._cache.get(request.request_hash)
        if isinstance(cached, Exception):
            raise cached
        if cached is not None:
            return cached
        self.paid_calls += 1
        if self.paid_calls == self.malformed_paid_index:
            error = ReplacementClassificationError("replacement judgment is not strict JSON")
            self._cache[request.request_hash] = error
            raise error
        result = ReplacementClassificationResult(
            request_hash=request.request_hash,
            adjudication_valid=True,
            target_feature_absent_or_changed=True,
            neutral_control_function_matched=True,
            raw_judgment_hashes=("a" * 64, "b" * 64),
            classifier_provenance_hash=stable_hash(self.provenance),
            rationale="Both frozen routes agreed.",
        )
        self._cache[request.request_hash] = result
        return result


class IntegrityFailingClassifier(ReplayClassifier):
    def classify(
        self, request: ReplacementClassificationRequest
    ) -> ReplacementClassificationResult:
        self.requests.append(request.request_hash)
        raise ReplacementClassificationError("classifier result echoes the wrong request hash")


class SimulatedTransportError(RuntimeError):
    pass


class ReplayJudge:
    not_for_primary_inference = False

    def __init__(
        self,
        *,
        model_id: str,
        malformed_paid_index: int | None = None,
        disagree: bool = False,
        fail_once_at_paid_index: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.malformed_paid_index = malformed_paid_index
        self.disagree = disagree
        self.fail_once_at_paid_index = fail_once_at_paid_index
        self.requests: list[str] = []
        self.paid_calls = 0
        self._failed = False
        self._cache: dict[str, str] = {}

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="test",
            model_id=self.model_id,
            model_revision="frozen",
            caller_version="replay-test-v1",
            decoding={"temperature": 0, "response_format": "json_object"},
        )

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request.request_id)
        cached = self._cache.get(request.request_id)
        if cached is not None:
            return cached
        next_paid = self.paid_calls + 1
        if self.fail_once_at_paid_index == next_paid and not self._failed:
            self._failed = True
            raise SimulatedTransportError("transient transport failure")
        self.paid_calls = next_paid
        if self.paid_calls == self.malformed_paid_index:
            raw = '{"unexpected":"paid malformed instrument"}'
        else:
            value = 120 if "120" in request.user_payload["answer"] else 80
            if self.disagree:
                value += 1
            raw = json.dumps({"status": "KNOWN", "value": str(value)})
        self._cache[request.request_id] = raw
        return raw


def _neutral() -> NeutralControlSpec:
    return NeutralControlSpec(
        control_id="neutral",
        function_definition="A position-matched calculation sentence.",
        version="v1",
    )


def _write_gpu_artifact(directory: Path) -> FrozenInputs:
    anchors, bases = _frozen_design()
    initial = build_initial_allocation_manifest(anchors, master_seed=20260829)
    stage_two = build_fixed_stage_two_allocation_manifest(
        anchors,
        initial_manifest=initial,
        master_seed=20260829,
    )
    terminal_invalid_id = initial.allocations[0].request_id
    backend = FixtureBackend(malformed_request_id=terminal_invalid_id)
    generated = {
        "initial": generate_sentence_resampling_intermediates(
            anchors,
            base_traces=bases,
            allocation_manifest=initial,
            backend=backend,
            primary_inference=True,
            microbatch_size=47,
        ),
        "stage_two": generate_sentence_resampling_intermediates(
            anchors,
            base_traces=bases,
            allocation_manifest=stage_two,
            backend=backend,
            primary_inference=True,
            microbatch_size=47,
        ),
    }
    allocations = {"initial": initial, "stage_two": stage_two}
    plan_hashes: dict[str, str] = {}
    checkpoint_manifests: dict[str, Mapping[str, Any]] = {}
    for stage in ("initial", "stage_two"):
        store = RecordCheckpointStore(
            directory / stage,
            id_field="resample_id",
            plan_payload={
                "phase_contract": "resample-gpu-only-v1",
                "stage": stage,
                "allocation_manifest_hash": allocations[stage].manifest_hash,
            },
        )
        for record in generated[stage]:
            store.commit(record.as_dict(include_hash=True))
        final = store.finalize(
            expected_ids=tuple(item.request_id for item in allocations[stage].allocations)
        )
        plan_hashes[stage] = str(store.plan["plan_hash"])
        checkpoint_manifests[stage] = final.manifest

    combined = [
        *(row.as_dict(include_hash=True) for row in generated["initial"]),
        *(row.as_dict(include_hash=True) for row in generated["stage_two"]),
    ]
    combined_path = write_jsonl(directory / "gpu_intermediates.jsonl", combined)
    prefix_path = write_jsonl(directory / "prefix_registrations.jsonl", [])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "phase_contract": "resample-gpu-only-v1",
        "complete": True,
        "api_calls_performed": 0,
        "plan_hashes": plan_hashes,
        "allocation_manifest_hashes": {
            "initial": initial.manifest_hash,
            "stage_two": stage_two.manifest_hash,
        },
        "row_count": len(combined),
        "valid_generation_count": sum(row["generation_status"] == "valid" for row in combined),
        "terminal_invalid_count": sum(
            row["generation_status"] == "terminal_invalid" for row in combined
        ),
        "intermediates_path": combined_path.name,
        "intermediates_sha256": sha256_file(combined_path),
        "prefix_registrations_path": prefix_path.name,
        "prefix_registrations_sha256": sha256_file(prefix_path),
        "stage_checkpoint_manifest_hashes": {
            stage: checkpoint_manifests[stage]["manifest_hash"]
            for stage in ("initial", "stage_two")
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(directory / "gpu_generation_manifest.json", manifest)
    return FrozenInputs(directory, anchors, bases, initial, stage_two, terminal_invalid_id)


@pytest.fixture(scope="module")
def gpu_template(tmp_path_factory: pytest.TempPathFactory) -> FrozenInputs:
    return _write_gpu_artifact(tmp_path_factory.mktemp("resample-gpu-template"))


@pytest.fixture
def frozen_inputs(tmp_path: Path, gpu_template: FrozenInputs) -> FrozenInputs:
    destination = tmp_path / "gpu"
    shutil.copytree(gpu_template.directory, destination)
    return FrozenInputs(
        destination,
        gpu_template.anchors,
        gpu_template.bases,
        gpu_template.initial,
        gpu_template.stage_two,
        gpu_template.terminal_invalid_id,
    )


def _run(
    source: FrozenInputs,
    checkpoint: Path,
    *,
    primary: ReplayJudge | None = None,
    independent: ReplayJudge | None = None,
    classifier: ReplayClassifier | None = None,
    embedder: FixtureEmbedder | None = None,
    tolerance: ReplacementTokenTolerance | None = None,
    minimum_exact_agreement: float = 0.90,
    minimum_final_known_rate: float = 0.95,
):
    return run_resample_adjudication_phase(
        generation_checkpoint_dir=source.directory,
        checkpoint_dir=checkpoint,
        anchors=source.anchors,
        base_traces=source.bases,
        initial_allocation_manifest=source.initial,
        stage_two_allocation_manifest=source.stage_two,
        embedder=embedder or FixtureEmbedder(),
        primary_final_caller=primary or ReplayJudge(model_id="primary"),
        independent_final_caller=independent or ReplayJudge(model_id="independent"),
        replacement_classifier=classifier or ReplayClassifier(),
        neutral_control=_neutral(),
        token_tolerance=tolerance or ReplacementTokenTolerance(10, 10.0),
        execution_id="frozen-test-execution",
        minimum_exact_agreement=minimum_exact_agreement,
        minimum_final_known_rate=minimum_final_known_rate,
    )


def _rewrite_completed_checkpoint(checkpoint: Path, rows: list[dict[str, Any]]) -> None:
    units = checkpoint / "units"
    rows_by_id = {row["resample_id"]: row for row in rows}
    for record_path in (units / "records").glob("*.json"):
        record = read_json(record_path)
        replacement = rows_by_id[record["resample_id"]]
        if replacement != record:
            write_json(record_path, replacement)
    rows_path = write_jsonl(units / "checkpoint_rows.jsonl", rows)
    manifest_path = units / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    manifest["record_hashes_hash"] = stable_hash([row["record_hash"] for row in rows])
    manifest["rows_sha256"] = sha256_file(rows_path)
    manifest["manifest_hash"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    write_json(manifest_path, manifest)
    (checkpoint / "quality_gate.json").unlink(missing_ok=True)
    (checkpoint / "adjudication_manifest.json").unlink(missing_ok=True)


def test_authenticates_960_gpu_records_then_runs_cpu_classifiers_and_dual_finals(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    authenticated = load_authenticated_resample_generation(
        generation_checkpoint_dir=frozen_inputs.directory,
        anchors=frozen_inputs.anchors,
        base_traces=frozen_inputs.bases,
        initial_allocation_manifest=frozen_inputs.initial,
        stage_two_allocation_manifest=frozen_inputs.stage_two,
    )
    assert len(authenticated.rows) == 960
    assert authenticated.valid_generation_count == 959

    primary = ReplayJudge(model_id="primary")
    independent = ReplayJudge(model_id="independent")
    classifier = ReplayClassifier()
    result = _run(
        frozen_inputs,
        tmp_path / "adjudication",
        primary=primary,
        independent=independent,
        classifier=classifier,
    )

    assert result.complete is True
    assert result.gate_passed is True
    assert len(result.rows) == 960
    assert result.quality_gate["denominator_valid_generation_count"] == 959
    assert result.quality_gate["exact_status_value_agreement_rate"] == 1.0
    assert result.quality_gate["known_consensus_rate"] == 1.0
    assert len(primary.requests) == 959
    assert len(independent.requests) == 959
    assert len(classifier.requests) == 480
    assert 0 < primary.paid_calls < len(primary.requests)
    assert 0 < independent.paid_calls < len(independent.requests)
    assert 0 < classifier.paid_calls < len(classifier.requests)
    invalid = next(
        row for row in result.rows if row["resample_id"] == frozen_inputs.terminal_invalid_id
    )
    assert invalid["generation_status"] == "terminal_invalid"
    assert invalid["final_quality_denominator_eligible"] is False
    assert invalid["dual_final_consensus"] is None
    assert invalid["scientific_missing_reason"] == "terminal_invalid_generation"


@pytest.mark.parametrize(
    ("embedder", "tolerance", "expected_status"),
    [
        (
            FixtureEmbedder(cosine=0.9),
            ReplacementTokenTolerance(10, 10.0),
            "skipped_local_cosine_ineligible",
        ),
        (
            FixtureEmbedder(cosine=0.0),
            ReplacementTokenTolerance(0, 0.0),
            "skipped_local_token_ineligible",
        ),
    ],
)
def test_local_cosine_or_token_ineligibility_skips_paid_classifiers_only(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
    embedder: FixtureEmbedder,
    tolerance: ReplacementTokenTolerance,
    expected_status: str,
) -> None:
    primary = ReplayJudge(model_id="primary")
    independent = ReplayJudge(model_id="independent")
    classifier = ReplayClassifier()

    result = _run(
        frozen_inputs,
        tmp_path / expected_status,
        primary=primary,
        independent=independent,
        classifier=classifier,
        embedder=embedder,
        tolerance=tolerance,
    )

    resamples = [row for row in result.rows if row["arm"] == "resample"]
    assert len(resamples) == 480
    assert {row["replacement_classification_status"] for row in resamples} == {expected_status}
    assert classifier.requests == []
    assert len(primary.requests) == 959
    assert len(independent.requests) == 959


def test_malformed_final_instrument_is_unit_missing_but_both_routes_continue(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    primary = ReplayJudge(model_id="primary", malformed_paid_index=1)
    independent = ReplayJudge(model_id="independent")

    result = _run(
        frozen_inputs,
        tmp_path / "malformed-final",
        primary=primary,
        independent=independent,
    )

    malformed = [
        row
        for row in result.rows
        if row.get("scientific_missing_reason") == "malformed_primary_final"
    ]
    assert malformed
    assert all(row["final_estimate"] is None for row in malformed)
    assert all(row["final_measurement_valid"] is False for row in malformed)
    assert all(row["intervention_eligible"] is True for row in malformed)
    assert all(row["confirmatory_eligible"] is False for row in malformed)
    assert all(
        row["dual_final_consensus"]["primary"]["terminal_contract_failure"]
        == "malformed_instrument_json"
        for row in malformed
    )
    assert len(primary.requests) == 959
    assert len(independent.requests) == 959
    assert result.quality_gate["denominator_valid_generation_count"] == 959
    assert result.gate_passed is True


def test_malformed_classifier_instrument_is_intervention_missing_and_run_continues(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    classifier = ReplayClassifier(malformed_paid_index=1)

    result = _run(
        frozen_inputs,
        tmp_path / "malformed-classifier",
        classifier=classifier,
    )

    malformed = [
        row
        for row in result.rows
        if row["replacement_classification_status"] == "malformed_instrument_json"
    ]
    assert malformed
    assert all(row["intervention_eligible"] is False for row in malformed)
    assert all(row["confirmatory_eligible"] is False for row in malformed)
    assert all(
        row["intervention_eligibility_missing_reason"] == "malformed_replacement_classification"
        for row in malformed
    )
    assert result.gate_passed is True


def test_classifier_integrity_error_aborts_instead_of_becoming_unit_missing(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    classifier = IntegrityFailingClassifier()

    with pytest.raises(ResampleAdjudicationPhaseError, match="integrity contract"):
        _run(
            frozen_inputs,
            tmp_path / "classifier-integrity",
            classifier=classifier,
        )
    assert classifier.requests


def test_transport_abort_resumes_atomically_via_paid_response_replay_and_dedupes(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume"
    primary = ReplayJudge(model_id="primary")
    independent = ReplayJudge(
        model_id="independent",
        fail_once_at_paid_index=2,
    )
    classifier = ReplayClassifier()

    with pytest.raises(SimulatedTransportError, match="transport"):
        _run(
            frozen_inputs,
            checkpoint,
            primary=primary,
            independent=independent,
            classifier=classifier,
        )
    committed_before_resume = len(
        RecordCheckpointStore(
            checkpoint / "units",
            id_field="resample_id",
            plan_payload=read_json(checkpoint / "units/checkpoint_plan.json")["payload"],
        ).load_records()
    )
    assert 0 < committed_before_resume < 960
    primary_paid_before_resume = primary.paid_calls

    result = _run(
        frozen_inputs,
        checkpoint,
        primary=primary,
        independent=independent,
        classifier=classifier,
    )
    assert result.complete is True
    assert primary.paid_calls >= primary_paid_before_resume
    assert primary.paid_calls == len(set(primary.requests))
    assert independent.paid_calls == len(set(independent.requests))
    assert len({row["resample_id"] for row in result.rows}) == 960

    no_call_primary = ReplayJudge(model_id="primary")
    no_call_independent = ReplayJudge(model_id="independent")
    no_call_classifier = ReplayClassifier()
    replayed = _run(
        frozen_inputs,
        checkpoint,
        primary=no_call_primary,
        independent=no_call_independent,
        classifier=no_call_classifier,
    )
    assert replayed.rows == result.rows
    assert no_call_primary.requests == []
    assert no_call_independent.requests == []
    assert no_call_classifier.requests == []


def test_route_gate_and_source_drift_fail_before_new_paid_calls(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "drift"
    _run(frozen_inputs, checkpoint)

    changed_primary = ReplayJudge(model_id="changed-primary")
    with pytest.raises(ResampleAdjudicationPhaseError, match="plan drifted"):
        _run(frozen_inputs, checkpoint, primary=changed_primary)
    assert changed_primary.requests == []

    unchanged_primary = ReplayJudge(model_id="primary")
    with pytest.raises(ResampleAdjudicationPhaseError, match="plan drifted"):
        _run(
            frozen_inputs,
            checkpoint,
            primary=unchanged_primary,
            minimum_exact_agreement=0.91,
        )
    assert unchanged_primary.requests == []

    combined_path = frozen_inputs.directory / "gpu_intermediates.jsonl"
    combined_path.write_text(
        combined_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    source_primary = ReplayJudge(model_id="primary")
    with pytest.raises(ResampleAdjudicationPhaseError, match="artifact hash"):
        _run(frozen_inputs, tmp_path / "new-source", primary=source_primary)
    assert source_primary.requests == []


def test_completed_low_quality_run_is_persisted_but_fails_closed_on_every_resume(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "low-quality"
    primary = ReplayJudge(model_id="primary")
    disagreeing = ReplayJudge(model_id="independent", disagree=True)

    with pytest.raises(ResampleAdjudicationGateError, match="gate failed"):
        _run(
            frozen_inputs,
            checkpoint,
            primary=primary,
            independent=disagreeing,
        )
    quality = read_json(checkpoint / "quality_gate.json")
    assert quality["denominator_valid_generation_count"] == 959
    assert quality["exact_status_value_agreement_rate"] == 0.0
    assert quality["known_consensus_rate"] == 0.0
    assert quality["gate_passed"] is False

    replay_primary = ReplayJudge(model_id="primary")
    replay_independent = ReplayJudge(model_id="independent", disagree=True)
    with pytest.raises(ResampleAdjudicationGateError, match="gate failed"):
        _run(
            frozen_inputs,
            checkpoint,
            primary=replay_primary,
            independent=replay_independent,
        )
    assert replay_primary.requests == []
    assert replay_independent.requests == []


def test_tampered_stage_checkpoint_aborts_before_embedding_or_api(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    record_path = next((frozen_inputs.directory / "initial/records").glob("*.json"))
    row = read_json(record_path)
    row["seed"] += 1
    write_json(record_path, row)
    primary = ReplayJudge(model_id="primary")
    embedder = FixtureEmbedder()

    with pytest.raises(ResampleAdjudicationPhaseError, match="checkpoint final"):
        _run(
            frozen_inputs,
            tmp_path / "tampered",
            primary=primary,
            embedder=embedder,
        )
    assert primary.requests == []
    assert embedder.calls == 0


def test_self_consistent_completed_checkpoint_cannot_drift_from_gpu_source(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "completed-source-drift"
    _run(frozen_inputs, checkpoint)

    rows_path = checkpoint / "units/checkpoint_rows.jsonl"
    rows = read_jsonl(rows_path)
    rows[0]["source_generation_record_hash"] = "sha256:" + "0" * 64
    rows[0]["record_hash"] = stable_hash(
        {key: value for key, value in rows[0].items() if key != "record_hash"}
    )
    _rewrite_completed_checkpoint(checkpoint, rows)

    primary = ReplayJudge(model_id="primary")
    with pytest.raises(ResampleAdjudicationPhaseError, match="frozen source"):
        _run(frozen_inputs, checkpoint, primary=primary)
    assert primary.requests == []


def test_self_consistent_completed_dual_audit_cannot_drift_from_frozen_route(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "completed-route-drift"
    _run(frozen_inputs, checkpoint)

    rows = read_jsonl(checkpoint / "units/checkpoint_rows.jsonl")
    row = next(item for item in rows if item["dual_final_consensus"] is not None)
    audit = row["dual_final_consensus"]
    audit["primary_route"]["model_id"] = "forged-primary"
    audit["record_hash"] = stable_hash(
        {key: value for key, value in audit.items() if key != "record_hash"}
    )
    row["record_hash"] = stable_hash(
        {key: value for key, value in row.items() if key != "record_hash"}
    )
    _rewrite_completed_checkpoint(checkpoint, rows)

    primary = ReplayJudge(model_id="primary")
    with pytest.raises(ResampleAdjudicationPhaseError, match="dual-final audit"):
        _run(frozen_inputs, checkpoint, primary=primary)
    assert primary.requests == []


def test_completed_dual_audit_recomputes_exact_agreement_from_paid_raw_bodies(
    frozen_inputs: FrozenInputs,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "completed-consensus-drift"
    _run(frozen_inputs, checkpoint)

    rows = read_jsonl(checkpoint / "units/checkpoint_rows.jsonl")
    row = next(item for item in rows if item["dual_final_consensus"] is not None)
    target_request_id = row["dual_final_consensus"]["request_id"]
    forged_raw = json.dumps({"status": "KNOWN", "value": "999999"})
    for matching_row in rows:
        audit = matching_row["dual_final_consensus"]
        if audit is None or audit["request_id"] != target_request_id:
            continue
        audit["independent"]["raw_response"] = forged_raw
        audit["independent"]["response_hash"] = stable_hash({"raw_response": forged_raw})
        audit["record_hash"] = stable_hash(
            {key: value for key, value in audit.items() if key != "record_hash"}
        )
        matching_row["record_hash"] = stable_hash(
            {key: value for key, value in matching_row.items() if key != "record_hash"}
        )
    _rewrite_completed_checkpoint(checkpoint, rows)

    primary = ReplayJudge(model_id="primary")
    with pytest.raises(ResampleAdjudicationPhaseError, match="dual-final audit"):
        _run(frozen_inputs, checkpoint, primary=primary)
    assert primary.requests == []
