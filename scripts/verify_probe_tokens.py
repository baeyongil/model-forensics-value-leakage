#!/usr/bin/env python3
"""Verify and record the frozen Qwen lens probe token IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_forensics.io import stable_hash, write_json
from model_forensics.lens import DEFAULT_CONCEPT_WORDS
from model_forensics.lens_runner import FROZEN_PROBE_TOKEN_IDS, PRIMARY_MODEL_PIN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/lens_probe_token_verification.json"),
    )
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Transformers is required for probe verification") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL_PIN.model_id,
        revision=PRIMARY_MODEL_PIN.revision,
        trust_remote_code=False,
        use_fast=True,
    )
    records = []
    for concept, polarities in DEFAULT_CONCEPT_WORDS.items():
        for polarity, words in polarities.items():
            declared = FROZEN_PROBE_TOKEN_IDS[concept][polarity]
            for word, expected_id in zip(words, declared, strict=True):
                observed = tuple(tokenizer.encode(word, add_special_tokens=False))
                if observed != (expected_id,):
                    raise SystemExit(
                        f"probe {word!r} changed: expected {(expected_id,)}, observed {observed}"
                    )
                records.append(
                    {
                        "concept": concept,
                        "polarity": polarity,
                        "word": word,
                        "token_id": expected_id,
                        "single_token": True,
                    }
                )
    payload = {
        "schema_version": 1,
        "model_id": PRIMARY_MODEL_PIN.model_id,
        "tokenizer_revision": PRIMARY_MODEL_PIN.revision,
        "trust_remote_code": False,
        "probe_count": len(records),
        "all_exact_single_token": True,
        "records": records,
    }
    payload["manifest_hash"] = stable_hash(payload)
    write_json(args.output, payload)
    print(f"verified {len(records)} exact single-token probes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
