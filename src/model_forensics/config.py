"""Strict project and preregistration configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_forensics.io import sha256_file, stable_hash, write_json


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    revision: str | None = None
    require_pinned_revision: bool = True
    dtype: str = "bfloat16"
    tensor_parallel_size: int = Field(ge=1)
    max_model_len: int = Field(ge=1024)
    language_model_only: bool = True

    @model_validator(mode="after")
    def require_revision_when_frozen(self) -> ModelConfig:
        # Loading an unfrozen authoring config is allowed; execution calls
        # ``assert_execution_ready`` after the Hub revision is resolved.
        return self


class LensConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    revision: str | None = None
    require_pinned_revision: bool = True
    j_filename: str
    r_filename: str
    j_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    r_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    j_size_bytes: int | None = Field(default=None, gt=0)
    r_size_bytes: int | None = Field(default=None, gt=0)


class UpstreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    cache_dir: Path


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_dir: Path
    interim_dir: Path
    manifest_dir: Path
    figure_dir: Path
    report_dir: Path


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    secret_env: dict[str, str] = Field(default_factory=dict)
    gpu_cost_hard_stop_usd: float = Field(ge=0)
    api_cost_hard_stop_usd: float = Field(ge=0)
    total_cost_hard_stop_usd: float = Field(ge=0)
    terminate_compute_after_sync: bool = True

    @model_validator(mode="after")
    def validate_total_budget(self) -> ExecutionConfig:
        if self.total_cost_hard_stop_usd < (
            self.gpu_cost_hard_stop_usd + self.api_cost_hard_stop_usd
        ):
            raise ValueError("total hard stop must cover GPU plus API hard stops")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile: str
    preregistration: Path
    model: ModelConfig
    lenses: LensConfig
    upstream: UpstreamConfig
    paths: PathConfig
    execution: ExecutionConfig
    source_path: Path | None = None

    def assert_execution_ready(self) -> None:
        if self.model.require_pinned_revision and not self.model.revision:
            raise ValueError("model revision must be resolved and frozen before primary execution")
        if self.lenses.require_pinned_revision and not self.lenses.revision:
            raise ValueError("lens revision must be resolved and frozen before primary execution")
        if self.lenses.require_pinned_revision and not all(
            (
                self.lenses.j_sha256,
                self.lenses.r_sha256,
                self.lenses.j_size_bytes,
                self.lenses.r_size_bytes,
            )
        ):
            raise ValueError("lens files must have frozen SHA-256 hashes and byte sizes")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path).resolve()
    config = RunConfig.model_validate(_load_yaml_mapping(source))
    config.source_path = source
    return config


def load_preregistration(config: RunConfig) -> dict[str, Any]:
    path = config.preregistration
    if not path.is_absolute():
        base = config.source_path.parent.parent if config.source_path else Path.cwd()
        path = (base / path).resolve()
    return _load_yaml_mapping(path)


def freeze_configuration(
    config: RunConfig,
    destination: str | Path,
    *,
    resolved_model_revision: str,
    resolved_lens_revision: str,
) -> Path:
    """Write an immutable execution manifest after resolving Hub revisions."""

    if not resolved_model_revision or not resolved_lens_revision:
        raise ValueError("both resolved revisions are required")
    preregistration = load_preregistration(config)
    payload = config.model_dump(mode="json", exclude={"source_path"})
    payload["model"]["revision"] = resolved_model_revision
    payload["lenses"]["revision"] = resolved_lens_revision
    payload["preregistration_sha256"] = sha256_file(
        (config.source_path.parent.parent / config.preregistration).resolve()
        if config.source_path and not config.preregistration.is_absolute()
        else config.preregistration
    )
    payload["preregistration_hash"] = stable_hash(preregistration)
    payload["manifest_hash"] = stable_hash(payload)
    return write_json(destination, payload)
