# Third-party notices and redistribution boundary

This file records external dependencies and data sources used by the reproducibility workflow.
It is not legal advice and does not replace the license or terms supplied by each owner.

## Value Leakage reproduction repository

- Repository: <https://github.com/adsingh-64/value-leakage>
- Pinned commit: `16d129859e1f0e281363fb4f5910bcaeea316b10`
- Observed license boundary: no license file was present at the pinned repository root when this
  project inspected it.
- Redistribution decision: no source files, prompt files, figures, or raw traces from this
  repository are included here. `data/upstream/value-leakage/` is an ignored local checkout.
  Only independently generated hashes and compact derived reference statistics are eligible for
  release.

Absence of a license is not permission to redistribute. Anyone using `make reproduce` obtains a
separate local copy directly from the upstream host and is responsible for its terms.

## Jacobian Lens

- Repository: <https://github.com/anthropics/jacobian-lens>
- Pinned commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Upstream license: Apache License 2.0, as stated in the pinned repository's `LICENSE` and package
  metadata.
- Use here: installed as an external dependency on the GPU host. Its source is not vendored in
  this repository.

The separately downloaded workspace lens weights are referenced by repository revision,
filename, size, and SHA-256 in `config/run_122b.yaml` and `config/gpu_lock.yaml`. The weight files
are not redistributed here and remain subject to the terms at their source.

## Transformers and vLLM

- Transformers repository: <https://github.com/huggingface/transformers>, pinned by the
  bootstrap script to commit `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`.
- vLLM repository: <https://github.com/vllm-project/vllm>. The production environment accepts
  only an exact wheel URL plus verified SHA-256 and records the installed version.
- Upstream licenses: both pinned local checkouts state Apache License 2.0.
- Use here: external runtime dependencies; neither source tree nor wheel is released from this
  repository.

Other Python packages installed from `pyproject.toml` or `scripts/bootstrap_gpu.sh` retain their
respective upstream licenses. The captured GPU environment manifest records exact installed
versions for the run.

## Models and external services

Qwen model weights, tokenizer files, and downloaded lens weights are never committed. Their
source-specific licenses and acceptable-use terms apply independently. RunPod, Hugging Face, and
OpenRouter are external services; their credentials, billing records, and provider payloads are
not repository content.

## This repository

Code authored specifically for this repository is released under the MIT License in `LICENSE`.
That license applies only to original repository content and does not relicense any external
dependency, model, lens, upstream dataset, or service output.
