"""Deterministic numeric parsing and estimate-trajectory features.

This parser is intentionally transparent and auditable.  It handles the common
formats in Fermi traces, records exact character spans, and exposes an optional
adjudicated sequence so blind manual review can replace heuristic extraction
without changing downstream analysis code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from model_forensics.prompts import Condition, is_good_outcome
from model_forensics.schemas import NumericEstimate, TrajectoryFeatures, TrajectoryRecord

Number = int | float

_NUMBER_PATTERN = re.compile(
    r"""
    (?<![\w.])
    (?P<number>
        [+-]?
        (?:
            \d{1,3}(?:,\d{3})+(?:\.\d+)?
            |
            \d+(?:\.\d*)?
            |
            \.\d+
        )
        (?:[eE][+-]?\d+)?
    )
    (?:
        \s*
        (?P<multiplier>thousand|million|billion|[kKmMbB])
        \b
    )?
    (?!\w)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MULTIPLIERS = {
    None: 1.0,
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}

_FINAL_LABEL = re.compile(
    r"(?:final\s+(?:answer|estimate)|answer|best\s+estimate|point\s+estimate)"
    r"\s*(?:is|of|=|:|-)?\s*(?:\*{1,2}|_{1,2})?\s*$",
    re.IGNORECASE,
)
_COMMITMENT_LABEL = re.compile(
    r"(?:settl(?:e|ing)\s+on|commit(?:ting)?\s+to|go\s+with|lock(?:ing)?\s+in)"
    r"(?:\s+(?:approximately|about|roughly))?\s*$",
    re.IGNORECASE,
)
_FINAL_SUFFIX = re.compile(
    r"^\s*(?:\*{1,2}|_{1,2})?\s*(?:is\s+)?(?:my\s+)?"
    r"final\s+(?:answer|estimate)\b",
    re.IGNORECASE,
)


def extract_numeric_estimates(text: str, *, source: str = "trace") -> tuple[NumericEstimate, ...]:
    """Extract finite numbers with spans and normalize magnitude suffixes.

    Supported forms include comma grouping, decimals, ``k/m/b`` and written
    thousand/million/billion suffixes, and scientific notation.  Malformed or
    overflowing values are skipped rather than poisoning a rollout record.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not source:
        raise ValueError("source must not be empty")
    estimates: list[NumericEstimate] = []
    for match in _NUMBER_PATTERN.finditer(text):
        number_text = match.group("number").replace(",", "")
        suffix = match.group("multiplier")
        multiplier = _MULTIPLIERS[suffix.lower() if suffix else None]
        try:
            value = float(number_text) * multiplier
        except (OverflowError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        estimates.append(
            NumericEstimate(
                value=value,
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
                source=source,
                multiplier=multiplier,
            )
        )
    return tuple(estimates)


def _answer_priority(text: str, estimate: NumericEstimate) -> int:
    """Score how strongly local wording marks a number as the committed answer."""

    prefix = text[max(0, estimate.start - 100) : estimate.start]
    suffix = text[estimate.end : estimate.end + 80]
    if _FINAL_LABEL.search(prefix):
        return 4
    if _FINAL_SUFFIX.search(suffix):
        return 4
    if _COMMITMENT_LABEL.search(prefix):
        return 3
    if (
        text[max(0, estimate.start - 2) : estimate.start] == "**"
        and text[estimate.end : estimate.end + 2] == "**"
    ):
        return 2
    return 0


def select_final_estimate(text: str, *, source: str = "answer") -> NumericEstimate | None:
    """Select the committed answer, preferring explicit final-answer labels.

    Among equally strong candidates, the last mention wins.  With no explicit
    commitment language, the final numeric mention is returned.
    """

    candidates = extract_numeric_estimates(text, source=source)
    if not candidates:
        return None
    return max(enumerate(candidates), key=lambda item: (_answer_priority(text, item[1]), item[0]))[
        1
    ]


def _numeric_values(
    estimates: Iterable[Number | NumericEstimate],
) -> list[float]:
    values: list[float] = []
    for estimate in estimates:
        value = estimate.value if isinstance(estimate, NumericEstimate) else float(estimate)
        if not math.isfinite(value):
            raise ValueError("trajectory estimates must be finite")
        values.append(float(value))
    return values


def _deduplicate_revisions(values: Sequence[float]) -> list[float]:
    revisions: list[float] = []
    for value in values:
        if not revisions or not math.isclose(value, revisions[-1], rel_tol=1e-12, abs_tol=0.0):
            revisions.append(value)
    return revisions


def derive_trajectory_features(
    estimates: Iterable[Number | NumericEstimate],
    threshold: Number | None,
    condition: str | Condition,
) -> TrajectoryFeatures:
    """Derive first/final, revision, crossing, and stopping features.

    Repeated mentions of the same value do not count as revisions.  Crossing
    indices refer to this consecutive-deduplicated sequence.  Above-good uses
    strict ``> threshold``; below-good uses ``<= threshold``.
    """

    values = _numeric_values(estimates)
    revisions = _deduplicate_revisions(values)
    first = revisions[0] if revisions else None
    final = revisions[-1] if revisions else None
    revision_count = max(0, len(revisions) - 1)

    try:
        condition_value = condition if isinstance(condition, Condition) else Condition(condition)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown condition: {condition!r}") from exc

    crossing: int | None = None
    stop_after_good: bool | None = None
    revisions_after_good: int | None = None
    if condition_value in {Condition.ABOVE_GOOD, Condition.BELOW_GOOD}:
        if threshold is None:
            raise ValueError(f"threshold is required for {condition_value.value}")
        for index, value in enumerate(revisions):
            if is_good_outcome(condition_value, value, threshold):
                crossing = index
                break
        if crossing is not None:
            revisions_after_good = len(revisions) - crossing - 1
            stop_after_good = revisions_after_good == 0

    return TrajectoryFeatures(
        first_estimate=first,
        final_estimate=final,
        revision_count=revision_count,
        first_good_side_crossing_index=crossing,
        stopped_after_first_good_side_crossing=stop_after_good,
        revisions_after_good=revisions_after_good,
        estimate_count=len(revisions),
    )


def _adjudicated_mentions(values: Iterable[Number]) -> tuple[NumericEstimate, ...]:
    """Represent blind-adjudicated values without pretending to text spans."""

    mentions = []
    for index, value in enumerate(values):
        rendered = format(float(value), ".15g")
        mentions.append(
            NumericEstimate(
                value=float(value),
                raw=rendered,
                start=0,
                end=len(rendered),
                source=f"adjudicated:{index}",
            )
        )
    return tuple(mentions)


def parse_trajectory(
    trace: str,
    answer: str,
    threshold: Number | None,
    condition: str | Condition,
    *,
    adjudicated_estimates: Iterable[Number] | None = None,
    review_notes: str | None = None,
) -> TrajectoryRecord:
    """Build a canonical trajectory from a reasoning trace and visible answer.

    Only the selected final-answer number is imported from ``answer`` so trailing
    calculations cannot overwrite it.  Passing ``adjudicated_estimates`` replaces
    heuristic extraction and records the trajectory as manually reviewed.
    """

    if adjudicated_estimates is not None:
        estimates = _adjudicated_mentions(adjudicated_estimates)
        manually_reviewed = True
    else:
        trace_estimates = extract_numeric_estimates(trace, source="trace")
        answer_estimate = select_final_estimate(answer, source="answer")
        estimates = trace_estimates + ((answer_estimate,) if answer_estimate else ())
        manually_reviewed = False

    features = derive_trajectory_features(estimates, threshold, condition)
    return TrajectoryRecord(
        estimates=estimates,
        features=features,
        requires_manual_review=not manually_reviewed and features.final_estimate is None,
        review_notes=review_notes,
    )


# Explicit aliases ease migration from notebooks and keep call sites readable.
extract_numbers = extract_numeric_estimates
extract_final_estimate = select_final_estimate
trajectory_features = derive_trajectory_features


__all__ = [
    "derive_trajectory_features",
    "extract_final_estimate",
    "extract_numbers",
    "extract_numeric_estimates",
    "parse_trajectory",
    "select_final_estimate",
    "trajectory_features",
]
