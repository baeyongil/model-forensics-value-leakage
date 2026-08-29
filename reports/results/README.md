# Public aggregate results

This directory is the narrow public bridge between the private authenticated analysis and
credential-free result reproduction. A completed primary release contains only these exact files:

- `released_evidence.json`: content-addressed behavior, sentence-effect, and (when available)
  common-population lens aggregates;
- `results_manifest.json`: hashes, row counts, and exact field inventories for every aggregate;
- `tables/behavior_stage_summary.jsonl`;
- `tables/sentence_effects.jsonl`;
- `tables/lens_direction_heatmap.jsonl`.

Run `make reproduce-results` to verify the evidence and manifest, rebuild all three sanitized
tables, and regenerate the available core figures. The command is deliberately separate from raw
generation and does not load credentials, call a provider, or read `data/raw/` or `data/interim/`.

The release schema has no field for model reasoning, prompts, provider request/response bodies,
provider or infrastructure identifiers, or per-trace identifiers. `make release-check` applies an
independent exact-path and field-level audit before publication.
