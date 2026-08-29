# Lens position input contract

The primary J/R-lens job will not infer token positions from decoded numbers.
It requires one canonical `lens_positions.jsonl` row for every selected frozen
anchor. Rows are produced by `model_forensics.lens_positions.build_lens_position_row`
after a blind external first-estimate span adjudication.

Each row must contain:

- `schema_version: 1`, the selected `trace_id` and `anchor_id`;
- the canonical full `anchor_manifest_hash` and source `rollout_record_hash`;
- `first_estimate_span_record_hash`, instrument ID
  `target-first-estimate-span-v1`, and `first_estimate_span_primary_inference: true`;
- hashes of the exact original prompt, completion, and combined token streams;
- the fixed five-name `position_order` and corresponding `position_indices`;
- `position_evidence` containing original-token spans for `first_estimate`,
  `anchor`, and `answer_first_token`;
- `good_side_direction`, `causal_claim: false`, and a canonical `record_hash`.

Every evidence span must round-trip, authenticate the same completion token
stream, and contain the exact persisted token-ID slice. The command recomputes
all five positions from those spans and rejects a supplied index that differs.
It also rejects locally parsed, smoke-only, unknown, non-primary, incomplete,
or hash-mismatched first-estimate evidence before loading either GPU model.

The complete machine-readable shape is available from
`model_forensics.lens_command.required_position_record_schema()`.

## Fixed common probes and causal-prefix eligibility

Before any 122B forward, and only after the immutable paid-plan receipt and
active-session gate pass, the command writes
`data/manifests/lens_probe_design_manifest.json`. The probe universe is the
same preregistered three positive and three negative single-token IDs for every
trace, position, layer, and lens. Probe words are never selected from a
completed answer and never dropped separately.

Eligibility is evaluated for each trace × position × concept cell using only
the exact combined token stream through that authenticated position,
inclusive. If any one of the six probes appears by exact token ID or a decoded,
case-folded lexical boundary, the whole cell is ineligible. The complete
30,960-row Cartesian grid is still written; every J/R/layer row for that cell
has explicit null raw and signed contrasts plus the same authenticated
collision record. Scientific forwards stop after `final_answer_pre`, so later
answer tokens cannot affect eligibility or activations.

Analysis does not trust the stored lexical flags. It repeats the complete
pinned-tokenizer recomputation from the exact causal token prefixes and requires
field-for-field equality of every collision, eligibility decision, cell hash,
and grid count before statistics can run.

Analysis uses only eligible cells. Temporal contrasts use the intersection of
traces eligible at both positions. J is primary, R is a same-form sensitivity
readout, and the three fixed layer bands receive equal one-third weights when
collapsed for the exploratory accuracy-anchor association.

The ordered compatibility gate also writes
`data/manifests/lens_compatibility_prefix_manifest.json`. It records the exact
token count and content hash for the pinned 4B phrase, the full first
manifest-ordered 122B prefix, and its strict shortened prefix. Every attempt has
the same exact-prefix hash in its own canonical schema; analysis re-tokenizes
and re-derives all three prefixes rather than accepting declared counts.

`data/manifests/lens_release_authorization.json` links the canonical paid plan,
immutable paid receipt, approval-bindings hash, active GPU session gate, and
probe-design manifest. Both the success execution root and the two-failure root
must bind this same authorization hash and file SHA-256. Revalidating a completed
release is idempotent and does not require reconstructing a new live-session
receipt.

For a verdict-facing lens criterion, corroboration passes only with association
status `available`, the exact 4+4 trace universe and all 576 permutations, and
strictly positive J and R tau-a. The exact p-value and leave-one-trace-out range
remain reported robustness descriptors, not gate requirements. If the predicate
fails, raw diagnostic tables remain visible but every verdict-facing lens
criterion is `unknown` with an explicit reason. Report validation recomputes the
association from its hash-linked raw resampling and lens inputs.

If the passed 4B smoke is followed by failures of both bounded 122B attempts,
the command writes `data/manifests/lens_failure_manifest.json`. That manifest
binds the failure evidence to the frozen rollouts, anchors, positions,
candidate probes, and probe design. Downstream analysis then remains
behavior-only: lens criteria are unavailable, not zero or negative, and the
heatmap is omitted. Any optional 27B artifact is methodology support only and
cannot enter 122B criteria or conclusions.
