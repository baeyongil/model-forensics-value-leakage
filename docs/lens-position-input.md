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
