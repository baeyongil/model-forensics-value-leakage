"""Preregistered, cluster-aware estimands and conclusion rules.

The causal unit in the sentence-resampling experiment is the frozen base trace,
not an individual continuation. Accordingly, arm means are first computed
within each base trace and every inferential procedure below resamples or
permutes those base-trace contrasts. Missing outcomes are never silently
recoded as favorable observations: complete-case inference and explicit
worst/best-case bounds are separate, labelled results.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

Conclusion = Literal["positive", "negative", "practically_null", "inconclusive"]
PermutationMethod = Literal["exact_sign_flip", "monte_carlo_sign_flip"]


@dataclass(frozen=True)
class PermutationResult:
    """Two-sided randomization result for paired cluster contrasts."""

    p_value: float
    method: PermutationMethod
    assignments: int
    nonzero_clusters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectEstimate:
    """Equal-base-trace-weighted arm contrast with cluster-aware inference."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    clusters: int
    retain_observations: int
    resample_observations: int
    p_value: float | None
    conclusion: Conclusion
    analysis_population: str = "base_trace_complete_case"
    bootstrap_method: str = "percentile_resample_base_traces"
    permutation_method: str | None = None
    permutation_assignments: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedEffectEstimate:
    """Paired first-to-final change estimated at the rollout level."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    units: int
    p_value: float
    permutation_method: PermutationMethod
    permutation_assignments: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissingnessSensitivity:
    """Complete-case effect, missingness outcome, and binary-outcome bounds.

    ``worst_case_bound`` assigns every missing retain outcome to 0 and every
    missing resample outcome to 1. ``best_case_bound`` makes the reverse
    assignments. These are deliberately labelled sensitivity bounds, not a
    primary treatment-effect estimate.
    """

    complete_case: EffectEstimate | None
    missingness_effect: EffectEstimate | None
    total_clusters: int
    complete_case_clusters: int
    excluded_from_complete_case_clusters: int
    retain_observations: int
    resample_observations: int
    retain_missing: int
    resample_missing: int
    retain_missing_rate: float
    resample_missing_rate: float
    worst_case_bound: float
    best_case_bound: float
    bounds_population: str = "all_generated_rows_in_base_traces_with_both_arms"
    bounds_assumption: str = "worst: missing retain=0/resample=1; best: missing retain=1/resample=0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["complete_case"] = (
            None if self.complete_case is None else self.complete_case.to_dict()
        )
        payload["missingness_effect"] = (
            None if self.missingness_effect is None else self.missingness_effect.to_dict()
        )
        return payload


def classify_effect(ci_low: float, ci_high: float, *, rope: float = 0.10) -> Conclusion:
    """Classify an interval against the preregistered symmetric ROPE."""

    if rope <= 0:
        raise ValueError("rope must be positive")
    if not math.isfinite(ci_low) or not math.isfinite(ci_high):
        raise ValueError("CI endpoints must be finite")
    if ci_high < ci_low:
        raise ValueError("CI high must not be below CI low")
    if ci_low > rope:
        return "positive"
    if ci_high < -rope:
        return "negative"
    if ci_low >= -rope and ci_high <= rope:
        return "practically_null"
    return "inconclusive"


def ci_half_width(ci_low: float, ci_high: float) -> float:
    if ci_high < ci_low:
        raise ValueError("CI high must not be below CI low")
    return (ci_high - ci_low) / 2


def _coerce_outcome(value: Any, *, binary: bool) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"outcome must be numeric or missing, got {value!r}") from exc
    if not math.isfinite(number):
        # JSON artifacts must not contain NaN, but treating a stray NaN as
        # missing here prevents pandas round-trips from changing the estimand.
        if math.isnan(number):
            return None
        raise ValueError("outcome must be finite")
    if binary and number not in {0.0, 1.0}:
        raise ValueError(f"risk-difference outcome must be binary, got {number!r}")
    return number


def _collect_cluster_arms(
    rows: Iterable[Mapping[str, Any]],
    *,
    cluster_key: str,
    arm_key: str,
    outcome_key: str,
    retain_label: str,
    resample_label: str,
    binary: bool,
) -> dict[Any, dict[str, list[float | None]]]:
    grouped: dict[Any, dict[str, list[float | None]]] = defaultdict(
        lambda: {retain_label: [], resample_label: []}
    )
    for index, row in enumerate(rows):
        if arm_key not in row:
            raise ValueError(f"row {index} lacks arm key {arm_key!r}")
        arm = str(row[arm_key])
        if arm not in {retain_label, resample_label}:
            continue
        if cluster_key not in row:
            raise ValueError(f"row {index} lacks cluster key {cluster_key!r}")
        grouped[row[cluster_key]][arm].append(_coerce_outcome(row.get(outcome_key), binary=binary))
    if not grouped:
        raise ValueError("no rows from either requested arm")
    structural = [
        cluster
        for cluster, arms in grouped.items()
        if not arms[retain_label] or not arms[resample_label]
    ]
    if structural:
        raise ValueError(f"clusters lack both arms: {sorted(structural, key=str)}")
    return dict(grouped)


def _complete_cluster_differences(
    grouped: Mapping[Any, Mapping[str, Sequence[float | None]]],
    *,
    retain_label: str,
    resample_label: str,
) -> tuple[np.ndarray, int, int, tuple[Any, ...]]:
    differences: list[float] = []
    retain_n = 0
    resample_n = 0
    included: list[Any] = []
    for cluster, arms in grouped.items():
        retain = [value for value in arms[retain_label] if value is not None]
        resample = [value for value in arms[resample_label] if value is not None]
        if not retain or not resample:
            continue
        differences.append(float(np.mean(retain) - np.mean(resample)))
        retain_n += len(retain)
        resample_n += len(resample)
        included.append(cluster)
    return np.asarray(differences, dtype=float), retain_n, resample_n, tuple(included)


def _cluster_differences(
    rows: Iterable[Mapping[str, Any]],
    *,
    cluster_key: str,
    arm_key: str,
    outcome_key: str,
    retain_label: str,
    resample_label: str,
) -> tuple[np.ndarray, int, int]:
    """Compatibility helper used by the public risk-difference API."""

    grouped = _collect_cluster_arms(
        rows,
        cluster_key=cluster_key,
        arm_key=arm_key,
        outcome_key=outcome_key,
        retain_label=retain_label,
        resample_label=resample_label,
        binary=True,
    )
    differences, retain_n, resample_n, included = _complete_cluster_differences(
        grouped,
        retain_label=retain_label,
        resample_label=resample_label,
    )
    incomplete = set(grouped).difference(included)
    if incomplete:
        raise ValueError(
            "clusters lack both arms after missing outcomes are removed: "
            f"{sorted(incomplete, key=str)}"
        )
    if differences.size < 2:
        raise ValueError("at least two complete base-trace clusters are required")
    return differences, retain_n, resample_n


def paired_cluster_permutation(
    cluster_differences: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260829,
    exact_max_assignments: int = 1_048_576,
) -> PermutationResult:
    """Return a two-sided paired sign-flip test.

    All assignments are enumerated when feasible. Otherwise, Monte Carlo signs
    are drawn with the observed assignment included through the standard +1
    correction. Zero contrasts are retained in the reported cluster count but
    removed from the enumeration because their signs cannot change the statistic.
    """

    differences = np.asarray(cluster_differences, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("cluster differences must be a nonempty one-dimensional sequence")
    if not np.isfinite(differences).all():
        raise ValueError("cluster differences must be finite")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if exact_max_assignments <= 0:
        raise ValueError("exact_max_assignments must be positive")

    observed = abs(float(differences.mean()))
    nonzero = differences[~np.isclose(differences, 0.0, rtol=0.0, atol=1e-15)]
    assignment_count = 1 << int(nonzero.size)
    tolerance = 1e-15
    if assignment_count <= exact_max_assignments:
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=int(nonzero.size)):
            signed_sum = float(np.dot(np.asarray(signs), nonzero)) if nonzero.size else 0.0
            statistic = abs(signed_sum / differences.size)
            extreme += statistic >= observed - tolerance
        return PermutationResult(
            p_value=float(extreme / assignment_count),
            method="exact_sign_flip",
            assignments=assignment_count,
            nonzero_clusters=int(nonzero.size),
        )

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(replicates, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    p_value = float((np.count_nonzero(null >= observed - tolerance) + 1) / (replicates + 1))
    return PermutationResult(
        p_value=p_value,
        method="monte_carlo_sign_flip",
        assignments=replicates,
        nonzero_clusters=int(nonzero.size),
    )


def paired_cluster_permutation_pvalue(
    cluster_differences: Sequence[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260829,
    exact_max_assignments: int = 1_048_576,
) -> float:
    """Compatibility wrapper returning only the paired-test p-value."""

    return paired_cluster_permutation(
        cluster_differences,
        replicates=replicates,
        seed=seed,
        exact_max_assignments=exact_max_assignments,
    ).p_value


def _cluster_effect_from_differences(
    differences: np.ndarray,
    *,
    retain_n: int,
    resample_n: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
    confidence_level: float,
    rope: float,
    seed: int,
    analysis_population: str,
) -> EffectEstimate:
    if differences.ndim != 1 or differences.size < 2:
        raise ValueError("at least two complete base-trace clusters are required")
    if not np.isfinite(differences).all():
        raise ValueError("cluster differences must be finite")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(bootstrap_replicates, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    alpha = 1 - confidence_level
    ci_low, ci_high = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    permutation = paired_cluster_permutation(
        differences,
        replicates=permutation_replicates,
        seed=seed ^ 0xA5A5A5A5,
    )
    return EffectEstimate(
        estimate=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        clusters=int(differences.size),
        retain_observations=retain_n,
        resample_observations=resample_n,
        p_value=permutation.p_value,
        conclusion=classify_effect(float(ci_low), float(ci_high), rope=rope),
        analysis_population=analysis_population,
        permutation_method=permutation.method,
        permutation_assignments=permutation.assignments,
    )


def cluster_mean_difference(
    rows: Iterable[Mapping[str, Any]],
    *,
    cluster_key: str = "base_trace_id",
    arm_key: str = "arm",
    outcome_key: str,
    retain_label: str = "retain",
    resample_label: str = "resample",
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    confidence_level: float = 0.95,
    rope: float = 0.10,
    seed: int = 20260829,
) -> EffectEstimate:
    """Estimate a continuous retain-minus-resample contrast by base trace."""

    grouped = _collect_cluster_arms(
        rows,
        cluster_key=cluster_key,
        arm_key=arm_key,
        outcome_key=outcome_key,
        retain_label=retain_label,
        resample_label=resample_label,
        binary=False,
    )
    differences, retain_n, resample_n, included = _complete_cluster_differences(
        grouped,
        retain_label=retain_label,
        resample_label=resample_label,
    )
    incomplete = set(grouped).difference(included)
    if incomplete:
        raise ValueError(
            "clusters lack both arms after missing outcomes are removed: "
            f"{sorted(incomplete, key=str)}"
        )
    return _cluster_effect_from_differences(
        differences,
        retain_n=retain_n,
        resample_n=resample_n,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        confidence_level=confidence_level,
        rope=rope,
        seed=seed,
        analysis_population="base_trace_complete_case",
    )


def cluster_risk_difference(
    rows: Iterable[Mapping[str, Any]],
    *,
    cluster_key: str = "base_trace_id",
    arm_key: str = "arm",
    outcome_key: str = "final_good_side",
    retain_label: str = "retain",
    resample_label: str = "resample",
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    confidence_level: float = 0.95,
    rope: float = 0.10,
    seed: int = 20260829,
) -> EffectEstimate:
    """Estimate the complete-case binary risk difference by base trace."""

    differences, retain_n, resample_n = _cluster_differences(
        rows,
        cluster_key=cluster_key,
        arm_key=arm_key,
        outcome_key=outcome_key,
        retain_label=retain_label,
        resample_label=resample_label,
    )
    return _cluster_effect_from_differences(
        differences,
        retain_n=retain_n,
        resample_n=resample_n,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        confidence_level=confidence_level,
        rope=rope,
        seed=seed,
        analysis_population="base_trace_complete_case",
    )


def cluster_missingness_sensitivity(
    rows: Iterable[Mapping[str, Any]],
    *,
    cluster_key: str = "base_trace_id",
    arm_key: str = "arm",
    outcome_key: str = "final_good_side",
    retain_label: str = "retain",
    resample_label: str = "resample",
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    confidence_level: float = 0.95,
    rope: float = 0.10,
    seed: int = 20260829,
) -> MissingnessSensitivity:
    """Report missing outcomes without quietly assigning them to either side.

    Primary inference uses base traces with at least one observed outcome in
    both arms. A separate risk difference treats missingness itself as the
    binary outcome. The final two point estimates are logically extreme
    outcome bounds over every generated row in structurally complete clusters.
    """

    grouped = _collect_cluster_arms(
        rows,
        cluster_key=cluster_key,
        arm_key=arm_key,
        outcome_key=outcome_key,
        retain_label=retain_label,
        resample_label=resample_label,
        binary=True,
    )
    if len(grouped) < 2:
        raise ValueError("at least two base-trace clusters are required")

    differences, retain_observed, resample_observed, included = _complete_cluster_differences(
        grouped,
        retain_label=retain_label,
        resample_label=resample_label,
    )
    complete_case: EffectEstimate | None = None
    if differences.size >= 2:
        complete_case = _cluster_effect_from_differences(
            differences,
            retain_n=retain_observed,
            resample_n=resample_observed,
            bootstrap_replicates=bootstrap_replicates,
            permutation_replicates=permutation_replicates,
            confidence_level=confidence_level,
            rope=rope,
            seed=seed,
            analysis_population="base_trace_complete_case",
        )

    retain_total = sum(len(arms[retain_label]) for arms in grouped.values())
    resample_total = sum(len(arms[resample_label]) for arms in grouped.values())
    retain_missing = sum(value is None for arms in grouped.values() for value in arms[retain_label])
    resample_missing = sum(
        value is None for arms in grouped.values() for value in arms[resample_label]
    )

    missing_rows: list[dict[str, Any]] = []
    worst_differences: list[float] = []
    best_differences: list[float] = []
    for cluster, arms in grouped.items():
        for arm in (retain_label, resample_label):
            missing_rows.extend(
                {
                    cluster_key: cluster,
                    arm_key: arm,
                    "__outcome_missing": int(value is None),
                }
                for value in arms[arm]
            )
        retain_worst = [0.0 if value is None else value for value in arms[retain_label]]
        resample_worst = [1.0 if value is None else value for value in arms[resample_label]]
        retain_best = [1.0 if value is None else value for value in arms[retain_label]]
        resample_best = [0.0 if value is None else value for value in arms[resample_label]]
        worst_differences.append(float(np.mean(retain_worst) - np.mean(resample_worst)))
        best_differences.append(float(np.mean(retain_best) - np.mean(resample_best)))

    missingness_effect = cluster_risk_difference(
        missing_rows,
        cluster_key=cluster_key,
        arm_key=arm_key,
        outcome_key="__outcome_missing",
        retain_label=retain_label,
        resample_label=resample_label,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        confidence_level=confidence_level,
        rope=rope,
        seed=seed ^ 0x5A5A5A5A,
    )

    worst = float(np.mean(worst_differences))
    best = float(np.mean(best_differences))
    if worst > best + 1e-15:  # pragma: no cover - mathematical invariant
        raise AssertionError("worst-case effect cannot exceed best-case effect")
    return MissingnessSensitivity(
        complete_case=complete_case,
        missingness_effect=missingness_effect,
        total_clusters=len(grouped),
        complete_case_clusters=len(included),
        excluded_from_complete_case_clusters=len(grouped) - len(included),
        retain_observations=retain_observed,
        resample_observations=resample_observed,
        retain_missing=retain_missing,
        resample_missing=resample_missing,
        retain_missing_rate=retain_missing / retain_total,
        resample_missing_rate=resample_missing / resample_total,
        worst_case_bound=worst,
        best_case_bound=best,
    )


def paired_effect(
    differences: Sequence[float],
    *,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260829,
) -> PairedEffectEstimate:
    """Estimate a paired mean change with unit bootstrap and sign-flip test."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two paired units are required")
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_replicates, values.size))
    draws = values[indices].mean(axis=1)
    alpha = 1 - confidence_level
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    permutation = paired_cluster_permutation(
        values,
        replicates=permutation_replicates,
        seed=seed ^ 0xC3C3C3C3,
    )
    return PairedEffectEstimate(
        estimate=float(values.mean()),
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        units=int(values.size),
        p_value=permutation.p_value,
        permutation_method=permutation.method,
        permutation_assignments=permutation.assignments,
    )


def holm_adjust_pvalues(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-adjust p-values while preserving positions and missing values."""

    indexed: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        if value is None:
            continue
        number = float(value)
        if not 0 <= number <= 1 or not math.isfinite(number):
            raise ValueError(f"invalid p-value at index {index}: {value!r}")
        indexed.append((index, number))
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(sorted(indexed, key=lambda pair: pair[1])):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def signed_log_ratio(estimate: float, threshold: float, direction: int) -> float:
    """Return direction times log(estimate / threshold).

    ``direction=+1`` is used when values above the threshold are beneficial and
    ``direction=-1`` when values at or below it are beneficial. Positive values
    therefore always point toward the condition's good side (except the exact
    threshold, which is zero under this continuous transform).
    """

    if estimate <= 0 or threshold <= 0:
        raise ValueError("estimate and threshold must be positive")
    if not math.isfinite(estimate) or not math.isfinite(threshold):
        raise ValueError("estimate and threshold must be finite")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or +1")
    return float(direction * np.log(estimate / threshold))


def direction_interaction(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome_key: str = "final_good_side",
    direction_key: str = "direction",
) -> float:
    """Difference in mean good-side outcome between +1 and -1 conditions.

    The outcome must already be encoded relative to each condition's good side;
    this diagnostic catches direction-specific asymmetry rather than defining the
    primary motivated-reasoning effect.
    """

    grouped: dict[int, list[float]] = {-1: [], 1: []}
    for row in rows:
        direction = int(row[direction_key])
        outcome = _coerce_outcome(row.get(outcome_key), binary=True)
        if direction in grouped and outcome is not None:
            grouped[direction].append(outcome)
    if not grouped[-1] or not grouped[1]:
        raise ValueError("both incentive directions are required")
    return float(np.mean(grouped[1]) - np.mean(grouped[-1]))


__all__ = [
    "Conclusion",
    "EffectEstimate",
    "MissingnessSensitivity",
    "PairedEffectEstimate",
    "PermutationResult",
    "ci_half_width",
    "classify_effect",
    "cluster_mean_difference",
    "cluster_missingness_sensitivity",
    "cluster_risk_difference",
    "direction_interaction",
    "holm_adjust_pvalues",
    "paired_cluster_permutation",
    "paired_cluster_permutation_pvalue",
    "paired_effect",
    "signed_log_ratio",
]
