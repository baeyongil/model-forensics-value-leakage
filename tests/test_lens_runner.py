from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from model_forensics.io import read_jsonl
from model_forensics.lens import (
    ARTIFACT_SOURCE_LAYERS,
    DEFAULT_CONCEPT_WORDS,
    EXPECTED_D_MODEL,
    EXPECTED_MODEL_ID,
    EXPECTED_N_LAYERS,
    EXPECTED_TARGET_LAYER,
    FITTED_LAYERS,
    LensProvenance,
    LoadedLens,
    ModelRuntime,
    ProvenanceError,
)
from model_forensics.lens_runner import (
    FROZEN_PROBE_TOKEN_IDS,
    JLENS_REVISION,
    POSITION_ORDER,
    PRIMARY_MODEL_PIN,
    SMOKE_D_MODEL,
    SMOKE_MODEL_ID,
    SMOKE_N_LAYERS,
    TRANSFORMERS_REVISION,
    CapturedActivations,
    LensArtifactPin,
    LensTraceInput,
    PrimaryCompatibilityFailure,
    causal_probe_collisions,
    download_and_load_lens_pair,
    execute_lens_traces,
    freeze_causal_probe_design,
    freeze_prefix_absent_probes,
    load_pinned_text_runtime,
    run_4b_compatibility_smoke,
    run_122b_preflight,
    run_ordered_compatibility_gate,
    verify_software_revisions,
)
from model_forensics.token_spans import token_stream_hash, token_stream_manifest


def _probe_encoding() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for concept, polarities in DEFAULT_CONCEPT_WORDS.items():
        for polarity, words in polarities.items():
            for word, token_id in zip(
                words, FROZEN_PROBE_TOKEN_IDS[concept][polarity], strict=True
            ):
                result[word] = [token_id]
    return result


class _Tokenizer:
    def __init__(self, *, decoded: str = "neutral arithmetic trace") -> None:
        self.encodings = _probe_encoding()
        self.decoded = decoded

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.encodings[text])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return self.decoded


class _Wrapped:
    d_model = EXPECTED_D_MODEL
    n_layers = EXPECTED_N_LAYERS


def _runtime(*, tokenizer: Any | None = None) -> ModelRuntime:
    return ModelRuntime(
        model_id=EXPECTED_MODEL_ID,
        model=_Wrapped(),
        tokenizer=tokenizer or _Tokenizer(),
        device_map={"": "cuda:0"},
        compile=False,
    )


def _provenance(character: str) -> LensProvenance:
    return LensProvenance(
        model_id=EXPECTED_MODEL_ID,
        d_model=EXPECTED_D_MODEL,
        target_layer=EXPECTED_TARGET_LAYER,
        source_layers=ARTIFACT_SOURCE_LAYERS,
        file_sha256=character * 64,
        artifact_path=f"/tmp/{character}.pt",
    )


class _Lens:
    def __init__(self, name: str) -> None:
        self.name = name


class _SparseVector:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __len__(self) -> int:
        return 100_000

    def __getitem__(self, token_id: int) -> float:
        positive = {
            token_id
            for polarities in FROZEN_PROBE_TOKEN_IDS.values()
            for token_id in polarities["positive"]
        }
        return self.offset + (3.0 if token_id in positive else 1.0)


class _SameForwardFake:
    def __init__(self) -> None:
        self.capture_calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
        self.transport_calls: list[tuple[str, int, int]] = []

    def capture_once(
        self,
        runtime: ModelRuntime,
        *,
        input_token_ids: tuple[int, ...],
        layers: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> CapturedActivations:
        del runtime
        ids = tuple(input_token_ids)
        normalized_layers = tuple(layers)
        normalized_positions = tuple(positions)
        self.capture_calls.append((ids, normalized_layers, normalized_positions))
        return CapturedActivations(
            ids,
            normalized_positions,
            {layer: object() for layer in normalized_layers},
            forward_count=1,
        )

    def transport_and_unembed(
        self,
        runtime: ModelRuntime,
        lens: LoadedLens,
        *,
        layer: int,
        residual: object,
    ) -> list[_SparseVector]:
        del runtime
        self.transport_calls.append((lens.lens_type, layer, id(residual)))
        offset = 1.0 if lens.lens_type == "J" else 2.0
        return [_SparseVector(offset) for _ in range(5)]


def _trace(*, direction: int = -1) -> LensTraceInput:
    streams = token_stream_manifest(
        prompt_token_ids=(10, 11), completion_token_ids=(12, 13, 14, 15)
    )
    return LensTraceInput.from_token_stream_manifest(
        trace_id="trace-001",
        token_streams=streams,
        position_indices={
            "prompt_end": 1,
            "first_estimate_pre": 2,
            "anchor_pre": 3,
            "anchor_post": 4,
            "final_answer_pre": 5,
        },
        good_side_direction=direction,  # type: ignore[arg-type]
    )


def _probe_design(trace: LensTraceInput, *, tokenizer: Any | None = None):
    return freeze_causal_probe_design(
        tokenizer or _Tokenizer(),
        traces=(trace,),
        candidate_probe_manifest_hash="sha256:" + "1" * 64,
        candidate_probe_manifest_sha256="2" * 64,
        anchor_manifest_hash="sha256:" + "3" * 64,
        anchor_selection_hash="sha256:" + "4" * 64,
        rollout_manifest_hash="sha256:" + "5" * 64,
        position_manifest_hash="sha256:" + "6" * 64,
    )


def test_exact_generation_tokens_are_forwarded_once_to_both_lenses(tmp_path: Path) -> None:
    handles = (
        LoadedLens("J", _Lens("j"), _provenance("a")),
        LoadedLens("R", _Lens("r"), _provenance("b")),
    )
    backend = _SameForwardFake()
    trace = _trace(direction=-1)
    output = tmp_path / "lens.jsonl"

    records = execute_lens_traces(
        (trace,),
        runtime=_runtime(),
        lenses=handles,
        backend=backend,
        probe_design=_probe_design(trace),
        output_path=output,
        layers=(4, 19),
    )

    assert len(backend.capture_calls) == 1
    assert backend.capture_calls[0][0] == (10, 11, 12, 13, 14, 15)
    for layer in (4, 19):
        residual_ids = {
            residual_id
            for _, observed_layer, residual_id in backend.transport_calls
            if observed_layer == layer
        }
        assert len(residual_ids) == 1
    assert len(records) == 2 * 2 * 5 * 3
    direction = next(record for record in records if record.contrast == "direction")
    valence = next(record for record in records if record.contrast == "valence")
    assert direction.raw_mean_logit_contrast == 2.0
    assert direction.signed_mean_logit_contrast == -2.0
    assert valence.raw_mean_logit_contrast == 2.0
    assert valence.signed_mean_logit_contrast == 2.0
    written = read_jsonl(output)
    assert len(written) == len(records)
    assert all(row["record_hash"].startswith("sha256:") for row in written)
    assert all(row["causal_claim"] is False for row in written)


def test_frozen_probe_ids_are_verified_then_exact_prefix_collisions_are_filtered() -> None:
    colliding = FROZEN_PROBE_TOKEN_IDS["direction"]["positive"][0]
    concepts = freeze_prefix_absent_probes(
        _Tokenizer(),
        exact_prefix_token_ids=(1, colliding, 2),
    )
    assert concepts["direction"].positive_ids == FROZEN_PROBE_TOKEN_IDS["direction"]["positive"][1:]

    tokenizer = _Tokenizer()
    tokenizer.encodings[" upward"] = [999]
    with pytest.raises(ProvenanceError, match="token ID changed"):
        freeze_prefix_absent_probes(tokenizer, exact_prefix_token_ids=(1, 2))


def test_causal_probe_design_keeps_fixed_universe_and_never_looks_past_a_cell() -> None:
    collision = FROZEN_PROBE_TOKEN_IDS["direction"]["positive"][0]
    streams = token_stream_manifest(
        prompt_token_ids=(10, 11),
        completion_token_ids=(12, 13, collision, 15, collision),
    )
    trace = LensTraceInput.from_token_stream_manifest(
        trace_id="trace-causal",
        token_streams=streams,
        position_indices={
            "prompt_end": 1,
            "first_estimate_pre": 2,
            "anchor_pre": 3,
            "anchor_post": 4,
            "final_answer_pre": 5,
        },
        good_side_direction=1,
    )

    design = freeze_causal_probe_design(
        _Tokenizer(),
        traces=(trace,),
        candidate_probe_manifest_hash="sha256:" + "1" * 64,
        candidate_probe_manifest_sha256="2" * 64,
        anchor_manifest_hash="sha256:" + "3" * 64,
        anchor_selection_hash="sha256:" + "4" * 64,
        rollout_manifest_hash="sha256:" + "5" * 64,
        position_manifest_hash="sha256:" + "6" * 64,
    )

    before = design.cell_for("trace-causal", "anchor_pre", "direction")
    after = design.cell_for("trace-causal", "anchor_post", "direction")
    assert before.probe_eligible is True
    assert before.collisions == ()
    assert after.probe_eligible is False
    assert after.probe_ineligibility_reason == "causal_prefix_probe_collision"
    assert design.concepts["direction"].positive_ids == (
        FROZEN_PROBE_TOKEN_IDS["direction"]["positive"]
    )
    assert len(design.cells) == len(POSITION_ORDER) * len(DEFAULT_CONCEPT_WORDS)


def test_shared_collision_recomputation_detects_lexical_copy_without_probe_token() -> None:
    tokenizer = _Tokenizer(decoded="neutral reasoning moves upward carefully")
    trace = _trace(direction=1)
    design = _probe_design(trace, tokenizer=tokenizer)
    collisions = causal_probe_collisions(
        tokenizer,
        causal_token_ids=(10, 11),
        probes=design.concepts["direction"],
    )
    upward = next(item for item in collisions if item.word.strip() == "upward")
    assert upward.exact_token_id_present is False
    assert upward.lexical_word_present is True


def test_fixed_probe_execution_emits_null_collision_rows_and_truncates_future_suffix() -> None:
    collision = FROZEN_PROBE_TOKEN_IDS["direction"]["positive"][0]
    streams = token_stream_manifest(
        prompt_token_ids=(10, 11),
        completion_token_ids=(12, 13, collision, 15, 16),
    )
    trace = LensTraceInput.from_token_stream_manifest(
        trace_id="trace-grid",
        token_streams=streams,
        position_indices={
            "prompt_end": 1,
            "first_estimate_pre": 2,
            "anchor_pre": 3,
            "anchor_post": 4,
            "final_answer_pre": 5,
        },
        good_side_direction=1,
    )
    design = freeze_causal_probe_design(
        _Tokenizer(),
        traces=(trace,),
        candidate_probe_manifest_hash="sha256:" + "1" * 64,
        candidate_probe_manifest_sha256="2" * 64,
        anchor_manifest_hash="sha256:" + "3" * 64,
        anchor_selection_hash="sha256:" + "4" * 64,
        rollout_manifest_hash="sha256:" + "5" * 64,
        position_manifest_hash="sha256:" + "6" * 64,
    )
    handles = (
        LoadedLens("J", _Lens("j"), _provenance("a")),
        LoadedLens("R", _Lens("r"), _provenance("b")),
    )
    backend = _SameForwardFake()

    records = execute_lens_traces(
        (trace,),
        runtime=_runtime(),
        lenses=handles,
        backend=backend,
        probe_design=design,
        layers=(4,),
    )

    assert backend.capture_calls[0][0] == trace.sequence_token_ids[:6]
    assert len(records) == 2 * 1 * 5 * 3
    collided = [
        record
        for record in records
        if record.position_name == "anchor_post" and record.contrast == "direction"
    ]
    assert len(collided) == 2
    assert all(record.probe_eligible is False for record in collided)
    assert all(record.raw_mean_logit_contrast is None for record in collided)
    assert all(record.signed_mean_logit_contrast is None for record in collided)
    assert all(
        record.positive_token_ids == FROZEN_PROBE_TOKEN_IDS["direction"]["positive"]
        for record in collided
    )


class _CheckpointLens:
    def __init__(self) -> None:
        self.d_model = EXPECTED_D_MODEL
        self.source_layers = list(ARTIFACT_SOURCE_LAYERS)


class _JacobianLens:
    @classmethod
    def load(cls, path: str) -> _CheckpointLens:
        assert Path(path).is_file()
        return _CheckpointLens()


class _Jlens:
    JacobianLens = _JacobianLens


def _checkpoint(estimator: str) -> dict[str, Any]:
    return {
        "J": {},
        "d_model": EXPECTED_D_MODEL,
        "n_prompts": 25,
        "source_layers": list(ARTIFACT_SOURCE_LAYERS),
        "provenance": {
            "model_id": EXPECTED_MODEL_ID,
            "dataset_id": "NeelNanda/pile-10k",
            "target_layer": EXPECTED_TARGET_LAYER,
            "skip_first": FITTED_LAYERS[0],
            "config_json": json.dumps({"estimator": estimator}),
        },
    }


def test_lens_download_requires_pinned_size_hash_and_embedded_geometry(tmp_path: Path) -> None:
    paths = {"J": tmp_path / "j.pt", "R": tmp_path / "r.pt"}
    paths["J"].write_bytes(b"j-lens")
    paths["R"].write_bytes(b"r-lens")
    pins = tuple(
        LensArtifactPin(
            lens_type=lens_type,  # type: ignore[arg-type]
            repository="org/lenses",
            revision="d" * 40,
            filename=f"{lens_type.lower()}.pt",
            sha256=hashlib.sha256(paths[lens_type].read_bytes()).hexdigest(),
            size_bytes=paths[lens_type].stat().st_size,
        )
        for lens_type in ("J", "R")
    )
    calls: list[dict[str, Any]] = []

    def downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(paths[kwargs["filename"][0].upper()])

    pair = download_and_load_lens_pair(
        pins,
        cache_dir=tmp_path / "cache",
        downloader=downloader,
        checkpoint_reader=lambda path: _checkpoint("standard" if path == paths["J"] else "relp"),
        jlens_module=_Jlens,
    )
    assert [call["revision"] for call in calls] == ["d" * 40, "d" * 40]
    assert pair.j.provenance.metadata["n_prompts"] == 25
    assert pair.r.provenance.metadata["fit_config"] == {"estimator": "relp"}

    bad = list(pins)
    bad[0] = LensArtifactPin(
        lens_type="J",
        repository="org/lenses",
        revision="d" * 40,
        filename="j.pt",
        sha256=pins[0].sha256,
        size_bytes=pins[0].size_bytes + 1,
    )
    with pytest.raises(ProvenanceError, match="size mismatch"):
        download_and_load_lens_pair(
            bad,
            cache_dir=tmp_path / "cache",
            downloader=downloader,
            checkpoint_reader=lambda _: _checkpoint("standard"),
            jlens_module=_Jlens,
        )


class FakeForCausalLM:
    def __init__(self, *, device: str = "cuda:0") -> None:
        self.config = SimpleNamespace(
            hidden_size=EXPECTED_D_MODEL,
            num_hidden_layers=EXPECTED_N_LAYERS,
        )
        self.hf_device_map = {"model": device, "lm_head": device}

    def named_parameters(self):
        return [("weight", SimpleNamespace(device="cuda:0"))]

    def named_buffers(self):
        return []


class _AutoTokenizer:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _Tokenizer:
        cls.calls.append((model_id, kwargs))
        return _Tokenizer()


class _AutoModel:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    device = "cuda:0"

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeForCausalLM:
        cls.calls.append((model_id, kwargs))
        return FakeForCausalLM(device=cls.device)


class _Transformers:
    AutoTokenizer = _AutoTokenizer
    AutoModelForCausalLM = _AutoModel


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 8


class _Torch:
    cuda = _Cuda()
    bfloat16 = "bf16"


class _JlensRuntime:
    @staticmethod
    def from_hf(model: Any, tokenizer: Any, *, compile: bool) -> _Wrapped:
        assert isinstance(model, FakeForCausalLM)
        assert isinstance(tokenizer, _Tokenizer)
        assert compile is False
        return _Wrapped()


def test_text_only_loader_is_revision_pinned_bf16_and_forbids_offload() -> None:
    _AutoModel.calls.clear()
    _AutoTokenizer.calls.clear()
    loaded = load_pinned_text_runtime(
        PRIMARY_MODEL_PIN,
        required_cuda_devices=8,
        per_gpu_memory_gib=78,
        transformers_module=_Transformers,
        torch_module=_Torch,
        jlens_module=_JlensRuntime,
        verify_dependencies=False,
    )
    assert loaded.model_class == "FakeForCausalLM"
    model_id, kwargs = _AutoModel.calls[0]
    assert model_id == EXPECTED_MODEL_ID
    assert kwargs["revision"] == PRIMARY_MODEL_PIN.revision
    assert kwargs["dtype"] == "bf16"
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {index: "78GiB" for index in range(8)}
    assert kwargs["offload_state_dict"] is False

    _AutoModel.device = "cpu"
    with pytest.raises(ProvenanceError, match="offload"):
        load_pinned_text_runtime(
            PRIMARY_MODEL_PIN,
            required_cuda_devices=8,
            per_gpu_memory_gib=78,
            transformers_module=_Transformers,
            torch_module=_Torch,
            jlens_module=_JlensRuntime,
            verify_dependencies=False,
        )
    _AutoModel.device = "cuda:0"


def test_software_source_revisions_are_hard_gated() -> None:
    revisions = {"transformers": TRANSFORMERS_REVISION, "jlens": JLENS_REVISION}
    assert verify_software_revisions(revision_reader=revisions.get) == revisions
    revisions["jlens"] = "0" * 40
    with pytest.raises(ProvenanceError, match="jlens"):
        verify_software_revisions(revision_reader=revisions.get)


def test_four_b_smoke_captures_boundary_layers_once() -> None:
    class SmokeWrapped:
        d_model = SMOKE_D_MODEL
        n_layers = SMOKE_N_LAYERS

    runtime = ModelRuntime(
        model_id=SMOKE_MODEL_ID,
        model=SmokeWrapped(),
        tokenizer=_Tokenizer(),
        device_map={"": "cuda:0"},
        compile=False,
    )
    backend = _SameForwardFake()
    details = run_4b_compatibility_smoke(runtime, token_ids=(1, 2), backend=backend)
    assert details["forward_count"] == 1
    assert backend.capture_calls[0][1] == (0, 31)


def test_primary_preflight_uses_one_forward_for_both_transports() -> None:
    handles = (
        LoadedLens("J", _Lens("j"), _provenance("a")),
        LoadedLens("R", _Lens("r"), _provenance("b")),
    )
    backend = _SameForwardFake()
    details = run_122b_preflight(
        _runtime(),
        handles,
        token_ids=(1, 2),
        backend=backend,
    )
    assert details["forward_count"] == 1
    assert len(backend.capture_calls) == 1
    assert len(backend.transport_calls) == 4
    assert details["lens_file_sha256"] == {"J": "a" * 64, "R": "b" * 64}


def test_gate_orders_4b_then_two_limited_primary_attempts_without_fallback() -> None:
    order: list[str] = []

    def smoke(ids: Any) -> dict[str, Any]:
        order.append(f"4b:{len(ids)}")
        return {"ok": True}

    def primary(ids: Any) -> dict[str, Any]:
        order.append(f"122b:{len(ids)}")
        if len(ids) > 2:
            raise RuntimeError("full prefix incompatible")
        return {"ok": True}

    manifest = run_ordered_compatibility_gate(
        four_b_prefix_token_ids=(1,),
        primary_prefix_token_ids=(1, 2, 3),
        shortened_primary_prefix_token_ids=(1, 2),
        four_b_runner=smoke,
        primary_runner=primary,
    )
    assert order == ["4b:1", "122b:3", "122b:2"]
    assert manifest.primary_ready is True
    assert manifest.fallback_model_used is False
    assert [attempt.status for attempt in manifest.attempts] == ["passed", "failed", "passed"]
    assert [attempt.prefix_token_ids_hash for attempt in manifest.attempts] == [
        token_stream_hash((1,), stream="lens_compatibility_attempt_prefix"),
        token_stream_hash((1, 2, 3), stream="lens_compatibility_attempt_prefix"),
        token_stream_hash((1, 2), stream="lens_compatibility_attempt_prefix"),
    ]

    def always_fail(ids: Any) -> dict[str, Any]:
        raise RuntimeError(f"failed {len(ids)}")

    with pytest.raises(PrimaryCompatibilityFailure) as error:
        run_ordered_compatibility_gate(
            four_b_prefix_token_ids=(1,),
            primary_prefix_token_ids=(1, 2, 3),
            shortened_primary_prefix_token_ids=(1, 2),
            four_b_runner=smoke,
            primary_runner=always_fail,
        )
    failed = error.value.manifest
    primary_attempts = [item for item in failed.attempts if item.stage == "122b_preflight"]
    assert len(primary_attempts) == 2
    assert all(item.status == "failed" for item in primary_attempts)
    assert [item.prefix_token_ids_hash for item in primary_attempts] == [
        token_stream_hash((1, 2, 3), stream="lens_compatibility_attempt_prefix"),
        token_stream_hash((1, 2), stream="lens_compatibility_attempt_prefix"),
    ]
    assert failed.primary_ready is False
    assert failed.fallback_model_used is False


def test_trace_rejects_tampered_original_token_hash() -> None:
    streams = token_stream_manifest(prompt_token_ids=(1,), completion_token_ids=(2, 3, 4, 5, 6))
    streams["prompt_token_ids_hash"] = "sha256:tampered"
    with pytest.raises(Exception, match="failed validation"):
        LensTraceInput.from_token_stream_manifest(
            trace_id="trace",
            token_streams=streams,
            position_indices=dict(
                zip(
                    (
                        "prompt_end",
                        "first_estimate_pre",
                        "anchor_pre",
                        "anchor_post",
                        "final_answer_pre",
                    ),
                    range(5),
                    strict=True,
                )
            ),
            good_side_direction=1,
        )
