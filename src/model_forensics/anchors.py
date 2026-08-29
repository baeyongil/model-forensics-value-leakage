"""Deterministic sentence spans and frozen anchor selection.

The module deliberately uses only the Python standard library.  Model-specific
classification belongs upstream; this file receives audited candidate labels and
freezes a reproducible, outcome-balanced subset.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

_TERMINALS = frozenset(".!?\u3002\uff01\uff1f")
_CLOSERS = frozenset("\"'\u201d\u2019)]}\u300b\u300d\u3011")
_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "e.g.",
        "i.e.",
        "jr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "st.",
        "vs.",
    }
)
_LIST_MARKER_ONLY = re.compile(r"(?:[-+*]|\d{1,3}[.)])\Z")
_MARKDOWN_HEADING = re.compile(r"#{1,6}\s+")
_MARKDOWN_DECORATION = re.compile(r"[*_`~>#\[\]()]")
_ALPHABETIC_TOKEN = re.compile(r"[A-Za-z]{2,}")


def eligible_sentence_text(text: str) -> bool:
    """Freeze Markdown-aware minimum-content rules for anchor candidates.

    Headings, standalone list markers, and one-word formatting fragments are
    excluded before classification.  The original text is never normalized, so
    accepted spans still map exactly onto generated completion tokens.
    """

    stripped = text.strip()
    if not stripped or _LIST_MARKER_ONLY.fullmatch(stripped):
        return False
    if _MARKDOWN_HEADING.match(stripped):
        return False
    visible = _MARKDOWN_DECORATION.sub("", stripped).strip()
    tokens = _ALPHABETIC_TOKEN.findall(visible)
    return len(tokens) >= 2 and sum(len(token) for token in tokens) >= 8


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """One sentence-like unit with half-open offsets into the source string."""

    index: int
    start: int
    end: int
    text: str
    boundary: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("sentence index must be non-negative")
        if not 0 <= self.start < self.end:
            raise ValueError("sentence span must be non-empty and ordered")
        if self.boundary not in {"terminal", "newline", "end"}:
            raise ValueError(f"unsupported sentence boundary: {self.boundary!r}")

    @property
    def terminated(self) -> bool:
        """Whether the unit ended at punctuation or an explicit line break."""

        return self.boundary != "end"

    def validate_against(self, source: str) -> None:
        """Raise if the recorded text and offsets do not match ``source``."""

        if self.end > len(source) or source[self.start : self.end] != self.text:
            raise ValueError("sentence span does not match its source text")


def _is_decimal_point(text: str, position: int) -> bool:
    return (
        text[position] == "."
        and position > 0
        and position + 1 < len(text)
        and text[position - 1].isdigit()
        and text[position + 1].isdigit()
    )


def _is_abbreviation(text: str, sentence_start: int, period: int) -> bool:
    prefix = text[sentence_start : period + 1]
    match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", prefix)
    if match is None:
        return False
    token = match.group(0).lower()
    return token in _ABBREVIATIONS or bool(re.fullmatch(r"[a-z]\.", token))


def sentence_spans(text: str) -> tuple[SentenceSpan, ...]:
    """Split ``text`` into deterministic, offset-preserving sentence spans.

    This conservative scanner treats newlines and sentence-final punctuation as
    boundaries, while protecting decimal points and a small fixed abbreviation
    list.  Leading and trailing whitespace is excluded from each span but never
    normalized, so slicing the original text always reconstructs ``span.text``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    spans: list[SentenceSpan] = []
    length = len(text)
    cursor = 0

    while cursor < length:
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break

        start = cursor
        boundary = "end"
        end = length

        while cursor < length:
            character = text[cursor]

            if character in "\r\n":
                end = cursor
                boundary = "newline"
                break

            if character in _TERMINALS and not _is_decimal_point(text, cursor):
                punctuation_end = cursor + 1
                while punctuation_end < length and text[punctuation_end] in _TERMINALS:
                    punctuation_end += 1
                while punctuation_end < length and text[punctuation_end] in _CLOSERS:
                    punctuation_end += 1

                follows_boundary = punctuation_end == length or text[punctuation_end].isspace()
                abbreviation = character == "." and _is_abbreviation(text, start, cursor)
                if follows_boundary and not abbreviation:
                    end = punctuation_end
                    boundary = "terminal"
                    break

            cursor += 1

        trimmed_end = end
        while trimmed_end > start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_end > start and eligible_sentence_text(text[start:trimmed_end]):
            spans.append(
                SentenceSpan(
                    index=len(spans),
                    start=start,
                    end=trimmed_end,
                    text=text[start:trimmed_end],
                    boundary=boundary,
                )
            )

        if boundary == "end":
            break
        cursor = end
        while cursor < length and text[cursor] in "\r\n":
            cursor += 1

    return tuple(spans)


DEFAULT_SENTENCE_CLASSES = (
    "accuracy_commitment",
    "value_threshold_planning",
    "epistemic_control",
)
DEFAULT_DIRECTIONS = ("above_good", "below_good")
ANCHORS_PER_CELL = 4
ANCHOR_SCHEMA_VERSION = "1"


class AnchorSelectionError(ValueError):
    """Raised when an exact, auditable anchor manifest cannot be produced."""


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    """A labeled sentence eligible for preregistered anchor selection."""

    trace_id: str
    sentence_class: str
    direction: str
    sentence_index: int
    sentence_text: str
    char_start: int
    char_end: int
    initial_side: str
    final_flip: bool
    eligible: bool = True
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("trace_id", "sentence_class", "direction", "initial_side"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if self.sentence_index < 0:
            raise ValueError("sentence_index must be non-negative")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("candidate character span must be non-empty and ordered")
        if self.char_end - self.char_start != len(self.sentence_text):
            raise ValueError("candidate span length must equal sentence_text length")
        if type(self.final_flip) is not bool:
            raise TypeError("final_flip must be a bool")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def stratum(self) -> tuple[str, bool]:
        return (self.initial_side, self.final_flip)

    def selection_payload(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "sentence_class": self.sentence_class,
            "direction": self.direction,
            "sentence_index": self.sentence_index,
            "sentence_text": self.sentence_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "initial_side": self.initial_side,
            "final_flip": self.final_flip,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class FrozenAnchor:
    """An immutable selected anchor with a content-addressed identifier."""

    anchor_id: str
    trace_id: str
    sentence_class: str
    direction: str
    sentence_index: int
    sentence_text: str
    char_start: int
    char_end: int
    initial_side: str
    final_flip: bool
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def stratum(self) -> tuple[str, bool]:
        return (self.initial_side, self.final_flip)

    def as_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "trace_id": self.trace_id,
            "sentence_class": self.sentence_class,
            "direction": self.direction,
            "sentence_index": self.sentence_index,
            "sentence_text": self.sentence_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "initial_side": self.initial_side,
            "final_flip": self.final_flip,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class AnchorManifest:
    """A frozen 3 x 2 x 4 anchor allocation and its canonical digest."""

    anchors: tuple[FrozenAnchor, ...]
    sentence_classes: tuple[str, ...]
    directions: tuple[str, ...]
    per_cell: int
    seed: str
    selection_hash: str
    schema_version: str = ANCHOR_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sentence_classes": list(self.sentence_classes),
            "directions": list(self.directions),
            "per_cell": self.per_cell,
            "seed": self.seed,
            "anchors": [anchor.as_dict() for anchor in self.anchors],
            "selection_hash": self.selection_hash,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _candidate_anchor_id(candidate: AnchorCandidate) -> str:
    return _sha256({"schema_version": ANCHOR_SCHEMA_VERSION, **candidate.selection_payload()})


def _freeze_candidate(candidate: AnchorCandidate) -> FrozenAnchor:
    return FrozenAnchor(
        anchor_id=_candidate_anchor_id(candidate),
        trace_id=candidate.trace_id,
        sentence_class=candidate.sentence_class,
        direction=candidate.direction,
        sentence_index=candidate.sentence_index,
        sentence_text=candidate.sentence_text,
        char_start=candidate.char_start,
        char_end=candidate.char_end,
        initial_side=candidate.initial_side,
        final_flip=candidate.final_flip,
        provenance=candidate.provenance,
    )


def _manifest_digest(manifest: AnchorManifest) -> str:
    return _sha256(
        {
            "schema_version": manifest.schema_version,
            "sentence_classes": list(manifest.sentence_classes),
            "directions": list(manifest.directions),
            "per_cell": manifest.per_cell,
            "seed": manifest.seed,
            "anchors": [anchor.as_dict() for anchor in manifest.anchors],
        }
    )


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    candidate: AnchorCandidate | None = None


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: int,
    *,
    candidate: AnchorCandidate | None = None,
) -> None:
    forward = _FlowEdge(
        to=target,
        reverse=len(graph[target]),
        capacity=capacity,
        cost=cost,
        candidate=candidate,
    )
    backward = _FlowEdge(
        to=source,
        reverse=len(graph[source]),
        capacity=0,
        cost=-cost,
    )
    graph[source].append(forward)
    graph[target].append(backward)


def _minimum_cost_flow(graph: list[list[_FlowEdge]], source: int, sink: int, required: int) -> int:
    """Send exact unit flow using deterministic Bellman-Ford augmentations."""

    sent = 0
    node_count = len(graph)
    infinity = 10**100

    while sent < required:
        distances = [infinity] * node_count
        previous_node = [-1] * node_count
        previous_edge = [-1] * node_count
        distances[source] = 0

        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == infinity:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate_distance = distances[node] + edge.cost
                    if candidate_distance < distances[edge.to]:
                        distances[edge.to] = candidate_distance
                        previous_node[edge.to] = node
                        previous_edge[edge.to] = edge_index
                        changed = True
            if not changed:
                break

        if distances[sink] == infinity:
            break

        node = sink
        while node != source:
            parent = previous_node[node]
            edge_index = previous_edge[node]
            if parent < 0 or edge_index < 0:
                raise RuntimeError("invalid residual path while selecting anchors")
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        sent += 1

    return sent


def _outcome_blind_rank(candidate: AnchorCandidate, seed: str) -> str:
    """Rank within a declared stratum without using either outcome label."""

    return _sha256(
        {
            "seed": seed,
            "trace_id": candidate.trace_id,
            "sentence_class": candidate.sentence_class,
            "direction": candidate.direction,
            "sentence_index": candidate.sentence_index,
            "sentence_text": candidate.sentence_text,
            "char_start": candidate.char_start,
            "char_end": candidate.char_end,
        }
    )


def _validate_design(
    sentence_classes: Sequence[str], directions: Sequence[str], per_cell: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    classes = tuple(sentence_classes)
    direction_values = tuple(directions)
    if len(classes) != 3 or len(set(classes)) != 3 or any(not value for value in classes):
        raise AnchorSelectionError("anchor design requires exactly three distinct classes")
    if (
        len(direction_values) != 2
        or len(set(direction_values)) != 2
        or any(not value for value in direction_values)
    ):
        raise AnchorSelectionError("anchor design requires exactly two distinct directions")
    if per_cell != ANCHORS_PER_CELL:
        raise AnchorSelectionError("anchor design requires exactly four traces per cell")
    return classes, direction_values


def select_frozen_anchors(
    candidates: Iterable[AnchorCandidate],
    *,
    sentence_classes: Sequence[str] = DEFAULT_SENTENCE_CLASSES,
    directions: Sequence[str] = DEFAULT_DIRECTIONS,
    per_cell: int = ANCHORS_PER_CELL,
    seed: str = "anchor-selection-v1",
) -> AnchorManifest:
    """Select the preregistered 24 anchors with global trace uniqueness.

    A minimum-cost bipartite allocation enforces one anchor per base trace.  Its
    convex stratum cost spreads each cell across ``initial_side x final_flip``
    strata whenever the candidate graph permits it.  Hash-based tie breaking is
    intentionally blind to those outcome fields after stratification.
    """

    classes, direction_values = _validate_design(sentence_classes, directions, per_cell)
    if not isinstance(seed, str) or not seed:
        raise AnchorSelectionError("selection seed must be a non-empty string")

    cells = tuple(
        (sentence_class, direction) for sentence_class in classes for direction in direction_values
    )
    allowed_cells = set(cells)
    materialized = [
        candidate
        for candidate in candidates
        if candidate.eligible and (candidate.sentence_class, candidate.direction) in allowed_cells
    ]
    if not materialized:
        raise AnchorSelectionError("no eligible anchor candidates were supplied")

    trace_properties: dict[str, tuple[str, str, bool]] = {}
    for candidate in materialized:
        properties = (
            candidate.direction,
            candidate.initial_side,
            candidate.final_flip,
        )
        previous = trace_properties.setdefault(candidate.trace_id, properties)
        if previous != properties:
            raise AnchorSelectionError(
                f"trace {candidate.trace_id!r} has inconsistent direction/outcome strata"
            )

    # At most one sentence from a trace can represent a given cell.  The choice is
    # made without reference to initial/final outcome fields.
    by_cell_trace: dict[tuple[str, str, str], AnchorCandidate] = {}
    for candidate in materialized:
        key = (candidate.sentence_class, candidate.direction, candidate.trace_id)
        incumbent = by_cell_trace.get(key)
        if incumbent is None or (
            _outcome_blind_rank(candidate, seed),
            candidate.sentence_index,
            candidate.char_start,
        ) < (
            _outcome_blind_rank(incumbent, seed),
            incumbent.sentence_index,
            incumbent.char_start,
        ):
            by_cell_trace[key] = candidate

    cell_candidates: dict[tuple[str, str], list[AnchorCandidate]] = {cell: [] for cell in cells}
    for candidate in by_cell_trace.values():
        cell_candidates[(candidate.sentence_class, candidate.direction)].append(candidate)
    for cell in cells:
        cell_candidates[cell].sort(
            key=lambda item: (_outcome_blind_rank(item, seed), item.trace_id)
        )
        if len(cell_candidates[cell]) < per_cell:
            raise AnchorSelectionError(
                f"cell {cell!r} has {len(cell_candidates[cell])} distinct eligible traces; "
                f"{per_cell} required"
            )

    unique_traces = sorted({candidate.trace_id for candidate in by_cell_trace.values()})
    required = len(cells) * per_cell
    if len(unique_traces) < required:
        raise AnchorSelectionError(
            f"only {len(unique_traces)} distinct traces are available; {required} required"
        )

    node_keys: list[object] = [("source",), ("sink",)]
    node_keys.extend(("cell", *cell) for cell in cells)
    cell_strata: dict[tuple[str, str], tuple[tuple[str, bool], ...]] = {}
    for cell in cells:
        strata = tuple(sorted({candidate.stratum for candidate in cell_candidates[cell]}))
        cell_strata[cell] = strata
        node_keys.extend(("stratum", *cell, *stratum) for stratum in strata)
    node_keys.extend(("trace", trace_id) for trace_id in unique_traces)
    node_index = {key: index for index, key in enumerate(node_keys)}
    graph: list[list[_FlowEdge]] = [[] for _ in node_keys]
    source = node_index[("source",)]
    sink = node_index[("sink",)]

    max_tie_cost = 2**48
    balance_weight = max_tie_cost * (required + 1)
    for cell in cells:
        cell_node = node_index[("cell", *cell)]
        _add_flow_edge(graph, source, cell_node, per_cell, 0)
        for stratum in cell_strata[cell]:
            stratum_node = node_index[("stratum", *cell, *stratum)]
            # Convex marginal costs make a 1/1/1/1 allocation preferable to a
            # concentrated allocation whenever global trace uniqueness allows it.
            for marginal_index in range(per_cell):
                _add_flow_edge(
                    graph,
                    cell_node,
                    stratum_node,
                    1,
                    marginal_index * balance_weight,
                )

        for candidate in cell_candidates[cell]:
            stratum_node = node_index[("stratum", *cell, *candidate.stratum)]
            trace_node = node_index[("trace", candidate.trace_id)]
            tie_cost = int(_outcome_blind_rank(candidate, seed)[:12], 16)
            _add_flow_edge(
                graph,
                stratum_node,
                trace_node,
                1,
                tie_cost,
                candidate=candidate,
            )

    for trace_id in unique_traces:
        _add_flow_edge(graph, node_index[("trace", trace_id)], sink, 1, 0)

    sent = _minimum_cost_flow(graph, source, sink, required)
    if sent != required:
        raise AnchorSelectionError(
            "candidate cells cannot be allocated to 24 globally distinct base traces"
        )

    selected: list[AnchorCandidate] = []
    for cell in cells:
        for stratum in cell_strata[cell]:
            stratum_node = node_index[("stratum", *cell, *stratum)]
            selected.extend(
                edge.candidate
                for edge in graph[stratum_node]
                if edge.candidate is not None and edge.capacity == 0
            )

    cell_order = {cell: index for index, cell in enumerate(cells)}
    selected.sort(
        key=lambda candidate: (
            cell_order[(candidate.sentence_class, candidate.direction)],
            candidate.initial_side,
            candidate.final_flip,
            _outcome_blind_rank(candidate, seed),
            candidate.trace_id,
        )
    )
    frozen = tuple(_freeze_candidate(candidate) for candidate in selected)
    manifest = AnchorManifest(
        anchors=frozen,
        sentence_classes=classes,
        directions=direction_values,
        per_cell=per_cell,
        seed=seed,
        selection_hash="",
    )
    manifest = replace(manifest, selection_hash=_manifest_digest(manifest))
    validate_anchor_manifest(manifest)
    return manifest


def validate_anchor_manifest(manifest: AnchorManifest) -> None:
    """Validate cardinality, global independence, content IDs, and frozen hash."""

    classes, directions = _validate_design(
        manifest.sentence_classes, manifest.directions, manifest.per_cell
    )
    required = len(classes) * len(directions) * manifest.per_cell
    if len(manifest.anchors) != required:
        raise AnchorSelectionError(
            f"manifest contains {len(manifest.anchors)} anchors; {required} required"
        )

    trace_ids = [anchor.trace_id for anchor in manifest.anchors]
    if len(set(trace_ids)) != len(trace_ids):
        raise AnchorSelectionError("manifest must contain one anchor per base trace")

    expected_cells = {
        (sentence_class, direction) for sentence_class in classes for direction in directions
    }
    cell_counts = {cell: 0 for cell in expected_cells}
    for anchor in manifest.anchors:
        cell = (anchor.sentence_class, anchor.direction)
        if cell not in cell_counts:
            raise AnchorSelectionError(f"anchor contains undeclared cell {cell!r}")
        cell_counts[cell] += 1
        candidate = AnchorCandidate(
            trace_id=anchor.trace_id,
            sentence_class=anchor.sentence_class,
            direction=anchor.direction,
            sentence_index=anchor.sentence_index,
            sentence_text=anchor.sentence_text,
            char_start=anchor.char_start,
            char_end=anchor.char_end,
            initial_side=anchor.initial_side,
            final_flip=anchor.final_flip,
            provenance=anchor.provenance,
        )
        if anchor.anchor_id != _candidate_anchor_id(candidate):
            raise AnchorSelectionError(f"anchor {anchor.trace_id!r} has an invalid content ID")
    if any(count != manifest.per_cell for count in cell_counts.values()):
        raise AnchorSelectionError(
            f"every class/direction cell must contain {manifest.per_cell} anchors"
        )

    expected_hash = _manifest_digest(replace(manifest, selection_hash=""))
    if manifest.selection_hash != expected_hash:
        raise AnchorSelectionError("anchor manifest selection_hash does not match its contents")
