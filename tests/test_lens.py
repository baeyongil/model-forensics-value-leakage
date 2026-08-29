from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

import pytest

from model_forensics.lens import (
    ARTIFACT_SOURCE_LAYERS,
    DEFAULT_CONCEPT_WORDS,
    EXPECTED_D_MODEL,
    EXPECTED_MODEL_ID,
    EXPECTED_TARGET_LAYER,
    FITTED_LAYERS,
    ConceptTokenIds,
    ConceptValidationError,
    LensExecutionError,
    LensProvenance,
    LoadedLens,
    ModelRuntime,
    OptionalDependencyError,
    PositionMappingError,
    ProvenanceError,
    build_model_runtime,
    fitted_layer_tertiles,
    layer_band,
    load_local_lens,
    map_named_positions,
    run_lens_analysis,
    signed_mean_logit_contrasts,
    validate_concept_tokens,
)


class _FakeLoadedLens:
    def __init__(self, *, d_model: int, source_layers: tuple[int, ...]) -> None:
        self.d_model = d_model
        self.source_layers = list(source_layers)


class _FakeJacobianLens:
    loaded_paths: ClassVar[list[str]] = []

    @classmethod
    def load(cls, path: str) -> _FakeLoadedLens:
        cls.loaded_paths.append(path)
        return _FakeLoadedLens(d_model=EXPECTED_D_MODEL, source_layers=ARTIFACT_SOURCE_LAYERS)


class _FakeJlens:
    JacobianLens = _FakeJacobianLens


def _checkpoint(estimator: str = "standard") -> dict[str, object]:
    return {
        "d_model": EXPECTED_D_MODEL,
        "n_prompts": 25,
        "source_layers": list(ARTIFACT_SOURCE_LAYERS),
        "provenance": {
            "model_id": EXPECTED_MODEL_ID,
            "dataset_id": "NeelNanda/pile-10k",
            "target_layer": EXPECTED_TARGET_LAYER,
            "skip_first": FITTED_LAYERS[0],
            "config_json": '{"estimator":"' + estimator + '"}',
        },
    }


def test_local_lens_provenance_is_validated_before_official_load(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "lens.pt"
    artifact.write_bytes(b"an immutable local lens artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _FakeJacobianLens.loaded_paths.clear()

    handle = load_local_lens(
        artifact,
        lens_type="J",
        expected_sha256=digest,
        checkpoint_reader=lambda _: _checkpoint(),
        jlens_module=_FakeJlens,
    )

    assert handle.lens_type == "J"
    assert handle.provenance.file_sha256 == digest
    assert handle.provenance.model_id == EXPECTED_MODEL_ID
    assert _FakeJacobianLens.loaded_paths == [str(artifact)]

    _FakeJacobianLens.loaded_paths.clear()
    checkpoint_reads: list[Path] = []

    def unexpected_reader(path: Path) -> dict[str, object]:
        checkpoint_reads.append(path)
        return _checkpoint()

    with pytest.raises(ProvenanceError, match="SHA256"):
        load_local_lens(
            artifact,
            lens_type="J",
            expected_sha256="0" * 64,
            checkpoint_reader=unexpected_reader,
            jlens_module=_FakeJlens,
        )
    assert checkpoint_reads == []
    assert _FakeJacobianLens.loaded_paths == []


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("model_id", "Qwen/Qwen3.5-27B", "model_id"),
        ("d_model", 5120, "d_model"),
        ("target_layer", 45, "target_layer"),
        ("source_layers", list(range(4, 46)), "source_layers"),
    ],
)
def test_local_lens_rejects_incompatible_provenance(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    artifact = tmp_path / "lens.pt"
    artifact.write_bytes(b"lens")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    checkpoint = _checkpoint(estimator="relp")
    if field in {"model_id", "target_layer"}:
        provenance = dict(checkpoint["provenance"])  # type: ignore[arg-type]
        provenance[field] = bad_value
        checkpoint["provenance"] = provenance
    else:
        checkpoint[field] = bad_value

    with pytest.raises(ProvenanceError, match=message):
        load_local_lens(
            artifact,
            lens_type="R",
            expected_sha256=digest,
            checkpoint_reader=lambda _: checkpoint,
            jlens_module=_FakeJlens,
        )


class _FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.encodings[text])


def test_concepts_are_single_token_disjoint_and_absent_from_every_prefix() -> None:
    assert set(DEFAULT_CONCEPT_WORDS) == {"direction", "valence", "epistemic"}
    concepts = {
        "direction": {"positive": (" raise", " upward"), "negative": (" lower",)},
        "valence": {"positive": (" helpful",), "negative": (" harmful",)},
        "epistemic": {"positive": (" accurate",), "negative": (" biased",)},
    }
    tokenizer = _FakeTokenizer(
        {
            " raise": [10],
            " upward": [18],
            " lower": [11],
            " helpful": [12],
            " harmful": [13],
            " accurate": [14],
            " biased": [15],
            "neutral prefix one": [1, 2],
            "neutral prefix two": [2, 3],
            "split": [16, 17],
            "prefix containing probe token": [1, 10, 2],
        }
    )

    tokenized = validate_concept_tokens(
        tokenizer,
        prefixes=("neutral prefix one", "neutral prefix two"),
        concept_words=concepts,
    )
    assert tokenized["direction"].positive_ids == (10, 18)
    assert tokenized["epistemic"].negative_ids == (15,)

    split_concepts = {
        **concepts,
        "direction": {"positive": ("split",), "negative": (" lower",)},
    }
    with pytest.raises(ConceptValidationError, match="exactly one token"):
        validate_concept_tokens(
            tokenizer,
            prefixes=("neutral prefix one",),
            concept_words=split_concepts,
        )

    filtered = validate_concept_tokens(
        tokenizer,
        prefixes=("prefix containing probe token",),
        concept_words=concepts,
    )
    assert filtered["direction"].positive_ids == (18,)

    no_fallback = {
        **concepts,
        "direction": {"positive": (" raise",), "negative": (" lower",)},
    }
    with pytest.raises(ConceptValidationError, match="no prefix-absent"):
        validate_concept_tokens(
            tokenizer,
            prefixes=("prefix containing probe token",),
            concept_words=no_fallback,
        )


def test_named_token_positions_and_fitted_layer_tertiles_are_fixed() -> None:
    positions = map_named_positions(
        sequence_length=25,
        prompt_end=4,
        first_estimate_start=8,
        anchor_start=12,
        anchor_end=15,
        final_answer_start=20,
    )
    assert positions == {
        "prompt_end": 4,
        "first_estimate_pre": 7,
        "anchor_pre": 11,
        "anchor_post": 14,
        "final_answer_pre": 19,
    }

    bands = fitted_layer_tertiles()
    assert bands == {
        "early": tuple(range(4, 19)),
        "middle": tuple(range(19, 33)),
        "late": tuple(range(33, 47)),
    }
    assert tuple(layer for layers in bands.values() for layer in layers) == FITTED_LAYERS
    assert layer_band(4) == "early"
    assert layer_band(19) == "middle"
    assert layer_band(46) == "late"

    with pytest.raises(PositionMappingError, match="anchor"):
        map_named_positions(
            sequence_length=25,
            prompt_end=4,
            first_estimate_start=8,
            anchor_start=12,
            anchor_end=12,
            final_answer_start=20,
        )
    with pytest.raises(PositionMappingError, match="fitted"):
        layer_band(47)


def test_signed_mean_logit_contrasts_use_equal_word_weighting() -> None:
    concepts = {
        "direction": ConceptTokenIds(
            positive_ids=(1, 2),
            negative_ids=(3, 4),
            positive_words=(" raise", " increase"),
            negative_words=(" lower", " decrease"),
        ),
        "valence": ConceptTokenIds(
            positive_ids=(5,),
            negative_ids=(6,),
            positive_words=(" helpful",),
            negative_words=(" harmful",),
        ),
    }
    logits = [0.0, 4.0, 6.0, 1.0, 3.0, 8.5, 2.5]

    assert signed_mean_logit_contrasts(logits, concepts) == {
        "direction": 3.0,
        "valence": 6.0,
    }

    with pytest.raises(LensExecutionError, match="vocabulary"):
        signed_mean_logit_contrasts(logits[:6], concepts)


class _FakeTextConfig:
    _name_or_path = EXPECTED_MODEL_ID
    hidden_size = EXPECTED_D_MODEL
    num_hidden_layers = 48

    def get_text_config(self) -> _FakeTextConfig:
        return self


class _FakeHfModel:
    config = _FakeTextConfig()


class _FakeWrappedModel:
    d_model = EXPECTED_D_MODEL
    n_layers = 48


class _FakeJlensRuntime:
    calls: ClassVar[list[tuple[object, object, bool]]] = []

    @classmethod
    def from_hf(cls, model: object, tokenizer: object, *, compile: bool) -> object:
        cls.calls.append((model, tokenizer, compile))
        return _FakeWrappedModel()


def test_model_runtime_is_injected_for_auto_device_map_without_compilation() -> None:
    loader_calls: list[tuple[str, str]] = []
    hf_model = _FakeHfModel()
    tokenizer = object()

    def loader(model_id: str, *, device_map: str) -> tuple[object, object]:
        loader_calls.append((model_id, device_map))
        return hf_model, tokenizer

    _FakeJlensRuntime.calls.clear()
    runtime = build_model_runtime(
        model_loader=loader,
        device_map="auto",
        compile=False,
        jlens_module=_FakeJlensRuntime,
    )

    assert loader_calls == [(EXPECTED_MODEL_ID, "auto")]
    assert _FakeJlensRuntime.calls == [(hf_model, tokenizer, False)]
    assert runtime.model_id == EXPECTED_MODEL_ID
    assert runtime.device_map == "auto"
    assert runtime.compile is False
    assert runtime.model.d_model == EXPECTED_D_MODEL

    with pytest.raises(LensExecutionError, match="compile=False"):
        build_model_runtime(
            model_loader=loader,
            device_map="auto",
            compile=True,
            jlens_module=_FakeJlensRuntime,
        )

    with pytest.raises(OptionalDependencyError, match="model_loader"):
        build_model_runtime(model_loader=None, jlens_module=_FakeJlensRuntime)


class _FakeAnalysisLens:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls: list[dict[str, object]] = []

    def apply(
        self,
        model: object,
        prompt: str,
        *,
        layers: tuple[int, ...],
        positions: list[int],
        max_seq_len: int,
        use_jacobian: bool,
    ) -> tuple[dict[int, list[list[float]]], list[object], list[int]]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "layers": layers,
                "positions": positions,
                "max_seq_len": max_seq_len,
                "use_jacobian": use_jacobian,
            }
        )
        result: dict[int, list[list[float]]] = {}
        for layer in layers:
            rows: list[list[float]] = []
            for position in positions:
                logits = [0.0] * 40
                for positive_id, negative_id in ((30, 31), (32, 33), (34, 35)):
                    logits[positive_id] = self.offset + layer + position
                    logits[negative_id] = position
                rows.append(logits)
            result[layer] = rows
        return result, [], list(range(25))


def _fake_provenance(digest_character: str) -> LensProvenance:
    return LensProvenance(
        model_id=EXPECTED_MODEL_ID,
        d_model=EXPECTED_D_MODEL,
        target_layer=EXPECTED_TARGET_LAYER,
        source_layers=ARTIFACT_SOURCE_LAYERS,
        file_sha256=digest_character * 64,
        artifact_path=f"/tmp/{digest_character}-lens.pt",
    )


def test_analysis_emits_observational_long_form_j_and_r_records() -> None:
    prefix = "a tokenized neutral reasoning prefix"
    tokenizer = _FakeTokenizer(
        {
            " raise": [30],
            " lower": [31],
            " helpful": [32],
            " harmful": [33],
            " accurate": [34],
            " biased": [35],
            prefix: list(range(25)),
        }
    )
    concepts = {
        "direction": {"positive": (" raise",), "negative": (" lower",)},
        "valence": {"positive": (" helpful",), "negative": (" harmful",)},
        "epistemic": {"positive": (" accurate",), "negative": (" biased",)},
    }
    model = _FakeWrappedModel()
    runtime = ModelRuntime(
        model_id=EXPECTED_MODEL_ID,
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        compile=False,
    )
    j_lens = _FakeAnalysisLens(offset=1.0)
    r_lens = _FakeAnalysisLens(offset=2.0)
    handles = (
        LoadedLens("J", j_lens, _fake_provenance("a")),
        LoadedLens("R", r_lens, _fake_provenance("b")),
    )
    positions = map_named_positions(
        sequence_length=25,
        prompt_end=4,
        first_estimate_start=8,
        anchor_start=12,
        anchor_end=15,
        final_answer_start=20,
    )
    layers = (4, 19, 33, 46)

    records = run_lens_analysis(
        trace_id="trace-001",
        prefix=prefix,
        runtime=runtime,
        lenses=handles,
        position_indices=positions,
        good_side_direction=1,
        concept_words=concepts,
        layers=layers,
        max_seq_len=64,
    )

    assert len(records) == 2 * len(layers) * len(positions) * 3
    assert {record.lens_type for record in records} == {"J", "R"}
    assert {record.contrast for record in records} == {
        "direction",
        "valence",
        "epistemic",
    }
    assert {record.layer_band for record in records} == {"early", "middle", "late"}
    assert all(record.evidence_scope == "observational_readout" for record in records)
    assert all(record.causal_claim is False for record in records)
    assert all(record.model_id == EXPECTED_MODEL_ID for record in records)
    prompt_end_j_direction = next(
        record
        for record in records
        if record.lens_type == "J"
        and record.layer == 4
        and record.position_name == "prompt_end"
        and record.contrast == "direction"
    )
    assert prompt_end_j_direction.signed_mean_logit_contrast == 5.0
    assert prompt_end_j_direction.raw_mean_logit_contrast == 5.0
    assert prompt_end_j_direction.token_index == 4
    assert j_lens.calls[0]["positions"] == [4, 7, 11, 14, 19]
    assert j_lens.calls[0]["use_jacobian"] is True
    assert r_lens.calls[0]["layers"] == layers

    negative_direction_records = run_lens_analysis(
        trace_id="trace-002",
        prefix=prefix,
        runtime=runtime,
        lenses=handles,
        position_indices=positions,
        good_side_direction=-1,
        concept_words=concepts,
        layers=(4,),
        max_seq_len=64,
    )
    negative_direction = next(
        record
        for record in negative_direction_records
        if record.lens_type == "J"
        and record.position_name == "prompt_end"
        and record.contrast == "direction"
    )
    assert negative_direction.raw_mean_logit_contrast == 5.0
    assert negative_direction.signed_mean_logit_contrast == -5.0

    with pytest.raises(LensExecutionError, match="both J and R"):
        run_lens_analysis(
            trace_id="trace-001",
            prefix=prefix,
            runtime=runtime,
            lenses=handles[:1],
            position_indices=positions,
            good_side_direction=1,
            concept_words=concepts,
            layers=layers,
        )
