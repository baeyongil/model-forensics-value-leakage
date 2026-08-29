SHELL := /bin/bash

PYTHON ?= .venv/bin/python
GPU_PYTHON ?= .venv-gpu/bin/python
PIP ?= .venv/bin/pip
PYTHON_BOOTSTRAP ?= python3
CONFIG ?= config/run_122b.yaml

# Immutable paid-run inputs. Model routes and all provider prices are read only
# from these content-addressed locks and the exact user approval; they are not
# accepted as Make or CLI overrides.
GPU_LOCK ?= config/gpu_lock.yaml
GPU_QUOTE_LOCK ?= .runpod/gpu_quote_lock.json
API_QUOTE_LOCK ?= .runpod/api_route_quote_lock.json
PAID_APPROVAL ?= .runpod/paid_run_approval.json
PAID_RECEIPT_DIR ?= .runpod/paid_phase_receipts

COST_LEDGER ?= data/manifests/cost_ledger.yaml
ROLLOUTS ?= data/raw/qwen35_122b/rollouts.jsonl
SAMPLING_MANIFEST ?= data/manifests/sampling_manifest.json
BEHAVIOR_THRESHOLDS ?= data/manifests/behavioral_thresholds.json
BEHAVIOR_BASELINE_GENERATION_CHECKPOINT ?= data/interim/qwen35_122b/checkpoints/behavior/baseline_generation
BEHAVIOR_BASELINE_ADJUDICATION_CHECKPOINT ?= data/interim/qwen35_122b/checkpoints/behavior/baseline_adjudication
BEHAVIOR_TREATMENT_GENERATION_CHECKPOINT ?= data/interim/qwen35_122b/checkpoints/behavior/treatment_generation
BEHAVIOR_TREATMENT_ADJUDICATION_CHECKPOINT ?= data/interim/qwen35_122b/checkpoints/behavior/treatment_adjudication
ANCHOR_CANDIDATES ?= data/interim/qwen35_122b/anchor_candidates.jsonl
ANCHOR_MANIFEST ?= data/manifests/anchor_manifest.json
LENS_POSITIONS ?= data/manifests/lens_positions.jsonl
LENS_PROBE_CANDIDATES ?= data/manifests/lens_probe_token_verification.json
LENS_PROBE_DESIGN ?= data/manifests/lens_probe_design_manifest.json
RESAMPLE_GENERATION_CHECKPOINT ?= data/interim/qwen35_122b/checkpoints/resampling/gpu
RESAMPLE_ADJUDICATION_CHECKPOINT ?= data/interim/qwen35_122b/checkpoints/resampling/adjudication
RESAMPLE_INPUT ?= data/interim/qwen35_122b/resampling.jsonl
LENS_INPUT ?= data/interim/qwen35_122b/lens.jsonl
LENS_CACHE_DIR ?= data/cache/lenses
PER_GPU_MEMORY_GIB ?= 76
PUBLIC_RESULTS_DIR ?= reports/results
PUBLIC_RESULTS_FIGURE_DIR ?= reports/figures
QWEN4B_SMOKE_OUTPUT ?= data/manifests/qwen4b_prefix_gpu_smoke.json

# GPU_PHASE must be one exact canonical phase. The quote lock supplies the GPU
# family, rate, timestamp, source, and this phase's approved runtime. Each phase
# uses a fresh opaque nonce supplied only through GPU_BUDGET_SESSION_ID.
GPU_PHASE ?=
GPU_RESERVATION_RECEIPT ?= .runpod/reservations/$(GPU_PHASE).json
PROVIDER_INCURRED_USD ?=

TIME_LEDGER ?= data/manifests/time_ledger.yaml
TIME_CATEGORY ?=
TIME_DESCRIPTION ?=
TIME_STATUS ?=

# Optional local credentials. This file is gitignored. Provider credentials are
# unexported by default and exposed only to the exact paid targets below.
-include .env.local
LOCAL_HF := $(HF_TOKEN)
LOCAL_OPENROUTER := $(OPENROUTER_API_KEY)
LOCAL_RUNPOD := $(RUNPOD_API_KEY)
unexport HF_TOKEN OPENROUTER_API_KEY RUNPOD_API_KEY

export PYTHONPATH := $(CURDIR)/src

define require-var
$(if $(strip $($1)),,$(error $1 is required for target '$@'))
endef

PAID_APPROVAL_ARGS = \
	--gpu-lock "$(GPU_LOCK)" \
	--gpu-quote-lock "$(GPU_QUOTE_LOCK)" \
	--api-quote-lock "$(API_QUOTE_LOCK)" \
	--paid-approval "$(PAID_APPROVAL)" \
	--paid-receipt-dir "$(PAID_RECEIPT_DIR)"

# Recipes set $$session_dir by deriving it from the authenticated receipt hash.
# The raw GPU_BUDGET_SESSION_ID never appears in argv, a path, or an artifact.
GPU_ACTIVE_SESSION_ARGS = \
	--gpu-budget-reservation "$(GPU_RESERVATION_RECEIPT)" \
	--gpu-session-directory "$$session_dir" \
	--gpu-session-id-env GPU_BUDGET_SESSION_ID \
	--cost-ledger "$(COST_LEDGER)"

define derive-session-directory
session_dir="$$( $(PYTHON_BOOTSTRAP) -c 'import json, sys; from pathlib import Path; receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); value = str(receipt["session_hash"]); prefix = "sha256:"; assert value.startswith(prefix) and len(value) == len(prefix) + 64; print((Path.cwd() / ".runpod" / "sessions" / value.removeprefix(prefix)).resolve())' "$(GPU_RESERVATION_RECEIPT)" )"
endef

.PHONY: setup test lint format reproduce reproduce-results stage-results-release \
	behavior-baseline-generate behavior-baseline-adjudicate \
	behavior-treatment-generate behavior-treatment-adjudicate sample \
	anchors positions resample-generate resample-adjudicate resample lens \
	analyze report smoke qwen4b-gpu-integration-smoke clean \
	gpu-reserve gpu-bootstrap gpu-active-verify gpu-settle \
	time-start time-stop time-status release-check

HF_TARGETS := behavior-baseline-generate behavior-treatment-generate \
	resample-generate lens qwen4b-gpu-integration-smoke gpu-bootstrap
OPENROUTER_TARGETS := behavior-baseline-adjudicate behavior-treatment-adjudicate \
	anchors positions resample-adjudicate

$(HF_TARGETS): export HF_TOKEN := $(LOCAL_HF)
$(OPENROUTER_TARGETS): export OPENROUTER_API_KEY := $(LOCAL_OPENROUTER)
gpu-bootstrap: export RUNPOD_API_KEY := $(LOCAL_RUNPOD)

setup:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e '.[analysis,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

reproduce:
	$(PYTHON) -m model_forensics.cli reproduce --config "$(CONFIG)"

# Public, credential-free reproduction from the aggregate-only released
# evidence. This never invokes a model backend, provider, or raw-data command.
reproduce-results:
	$(PYTHON) scripts/reproduce_results.py \
		--results-dir "$(PUBLIC_RESULTS_DIR)" \
		--figure-dir "$(PUBLIC_RESULTS_FIGURE_DIR)"

# Maintainer-only release staging. The source analysis summary and every
# linked artifact are authenticated before field-level aggregation.
stage-results-release:
	$(PYTHON) scripts/prepare_results_release.py \
		--config "$(CONFIG)" \
		--results-dir "$(PUBLIC_RESULTS_DIR)" \
		--figure-dir "$(PUBLIC_RESULTS_FIGURE_DIR)"

behavior-baseline-generate: override GPU_PHASE := behavior_baseline_gpu
behavior-baseline-generate:
	$(derive-session-directory); \
	$(GPU_PYTHON) -m model_forensics.cli behavior-generate \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		$(GPU_ACTIVE_SESSION_ARGS) \
		--phase baseline \
		--checkpoint-dir "$(BEHAVIOR_BASELINE_GENERATION_CHECKPOINT)"

behavior-baseline-adjudicate:
	$(PYTHON) -m model_forensics.cli behavior-adjudicate \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		--phase baseline \
		--generation-checkpoint-dir "$(BEHAVIOR_BASELINE_GENERATION_CHECKPOINT)" \
		--checkpoint-dir "$(BEHAVIOR_BASELINE_ADJUDICATION_CHECKPOINT)"

behavior-treatment-generate: override GPU_PHASE := behavior_treatment_gpu
behavior-treatment-generate:
	$(derive-session-directory); \
	$(GPU_PYTHON) -m model_forensics.cli behavior-generate \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		$(GPU_ACTIVE_SESSION_ARGS) \
		--phase treatment \
		--checkpoint-dir "$(BEHAVIOR_TREATMENT_GENERATION_CHECKPOINT)" \
		--thresholds "$(BEHAVIOR_THRESHOLDS)"

behavior-treatment-adjudicate:
	$(PYTHON) -m model_forensics.cli behavior-adjudicate \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		--phase treatment \
		--generation-checkpoint-dir "$(BEHAVIOR_TREATMENT_GENERATION_CHECKPOINT)" \
		--checkpoint-dir "$(BEHAVIOR_TREATMENT_ADJUDICATION_CHECKPOINT)" \
		--baseline-adjudication-checkpoint-dir "$(BEHAVIOR_BASELINE_ADJUDICATION_CHECKPOINT)"

# Validation-only alias. It cannot create a backend or provider client.
sample:
	$(PYTHON) -m model_forensics.cli sample \
		--config "$(CONFIG)" \
		--output "$(ROLLOUTS)" \
		--sampling-manifest "$(SAMPLING_MANIFEST)"

anchors:
	$(PYTHON) -m model_forensics.cli anchors \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		--candidates "$(ANCHOR_CANDIDATES)" \
		--output "$(ANCHOR_MANIFEST)" \
		--rollouts "$(ROLLOUTS)" \
		--sampling-manifest "$(SAMPLING_MANIFEST)"

positions:
	$(PYTHON) -m model_forensics.cli positions \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		--rollouts "$(ROLLOUTS)" \
		--anchors "$(ANCHOR_MANIFEST)" \
		--output "$(LENS_POSITIONS)"

resample-generate: override GPU_PHASE := resample_gpu
resample-generate:
	$(derive-session-directory); \
	$(GPU_PYTHON) -m model_forensics.cli resample-generate \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		$(GPU_ACTIVE_SESSION_ARGS) \
		--rollouts "$(ROLLOUTS)" \
		--anchors "$(ANCHOR_MANIFEST)" \
		--sampling-manifest "$(SAMPLING_MANIFEST)" \
		--checkpoint-dir "$(RESAMPLE_GENERATION_CHECKPOINT)"

resample-adjudicate:
	$(PYTHON) -m model_forensics.cli resample-adjudicate \
		--config "$(CONFIG)" \
		$(PAID_APPROVAL_ARGS) \
		--generation-checkpoint-dir "$(RESAMPLE_GENERATION_CHECKPOINT)" \
		--checkpoint-dir "$(RESAMPLE_ADJUDICATION_CHECKPOINT)" \
		--rollouts "$(ROLLOUTS)" \
		--anchors "$(ANCHOR_MANIFEST)" \
		--sampling-manifest "$(SAMPLING_MANIFEST)" \
		--output "$(RESAMPLE_INPUT)"

# Validation-only alias. It cannot generate a continuation or call a judge.
resample:
	$(PYTHON) -m model_forensics.cli resample \
		--config "$(CONFIG)" \
		--input "$(RESAMPLE_INPUT)"

lens: override GPU_PHASE := lens_gpu
lens:
	@if [[ -f "$(LENS_INPUT)" ]]; then \
		$(PYTHON) -m model_forensics.cli lens \
			--config "$(CONFIG)" \
			--input "$(LENS_INPUT)" \
			--probe-candidates "$(LENS_PROBE_CANDIDATES)" \
			--probe-design "$(LENS_PROBE_DESIGN)"; \
	else \
		$(derive-session-directory); \
		$(GPU_PYTHON) -m model_forensics.cli lens \
			--config "$(CONFIG)" \
			$(PAID_APPROVAL_ARGS) \
			$(GPU_ACTIVE_SESSION_ARGS) \
			--input "$(LENS_INPUT)" \
			--rollouts "$(ROLLOUTS)" \
			--anchors "$(ANCHOR_MANIFEST)" \
			--positions "$(LENS_POSITIONS)" \
			--probe-candidates "$(LENS_PROBE_CANDIDATES)" \
			--probe-design "$(LENS_PROBE_DESIGN)" \
			--cache-dir "$(LENS_CACHE_DIR)" \
			--per-gpu-memory-gib "$(PER_GPU_MEMORY_GIB)"; \
	fi

analyze:
	$(PYTHON) -m model_forensics.cli analyze --config "$(CONFIG)"

# Stages result_context.json/.md only; DOCX/Google Docs production is separate.
report:
	$(PYTHON) -m model_forensics.cli report --config "$(CONFIG)"

smoke:
	$(PYTHON) -m model_forensics.cli smoke --config config/smoke.yaml

# Bounded real-model integration only. It is non-primary, calls no paid API,
# and records the explicit no-matched-4B-lens transport boundary.
qwen4b-gpu-integration-smoke:
	$(GPU_PYTHON) scripts/qwen4b_prefix_smoke.py \
		--output "$(QWEN4B_SMOKE_OUTPUT)" \
		--tensor-parallel-size 1 \
		--max-model-len 4096 \
		--rollout-max-tokens 1024 \
		--continuation-max-tokens 256

clean:
	$(PYTHON) -m model_forensics.cli clean --config "$(CONFIG)"

# GPU lifecycle. Rate, hardware, source, timestamp, and per-phase runtime are
# extracted from the authenticated quote lock; none can be supplied ad hoc.
gpu-reserve:
	$(call require-var,GPU_PHASE)
	@mkdir -p "$(dir $(GPU_RESERVATION_RECEIPT))"
	@$(PYTHON) scripts/runpod_storage_budget.py reserve \
		--cost-ledger "$(COST_LEDGER)" \
		--amount-usd 5 \
		--gpu-hard-stop-usd 220 \
		--api-hard-stop-usd 100 \
		--total-hard-stop-usd 325
	@quote_values="$$( $(PYTHON) -c 'import sys; from model_forensics.execution_bindings import load_gpu_quote_lock; from model_forensics.gpu_budget import approved_gpu_phase_maximum_usd; quote = load_gpu_quote_lock(sys.argv[1]); phase = sys.argv[2]; allocation = next((item for item in quote.phase_runtime_allocations if item.command_phase == phase), None); assert allocation is not None, f"phase absent from quote lock: {phase}"; maximum = approved_gpu_phase_maximum_usd(gpu_count=quote.gpu_count, quote_hourly_per_gpu_usd=quote.usd_per_gpu_hour, running_storage_hourly_usd=quote.running_storage_usd_per_hour, approved_runtime_hours=allocation.maximum_runtime_hours); print(quote.gpu_count, quote.usd_per_gpu_hour, quote.running_storage_usd_per_hour, allocation.maximum_runtime_hours, maximum, sep="\t")' "$(GPU_QUOTE_LOCK)" "$(GPU_PHASE)" )"; \
	IFS=$$'\t' read -r gpu_count per_gpu_rate storage_rate runtime_hours phase_maximum <<< "$$quote_values"; \
	$(PYTHON) scripts/gpu_budget_reserve.py \
		--cost-ledger "$(COST_LEDGER)" \
		--phase "$(GPU_PHASE)" \
		--session-id-env GPU_BUDGET_SESSION_ID \
		--approved-phase-runtime-hours "$$runtime_hours" \
		--approved-phase-maximum-usd "$$phase_maximum" \
		--gpu-count "$$gpu_count" \
		--quote-hourly-per-gpu-usd "$$per_gpu_rate" \
		--running-storage-hourly-usd "$$storage_rate" \
		--gpu-hard-stop-usd 220 \
		--api-hard-stop-usd 100 \
		--total-hard-stop-usd 325 \
		--receipt "$(GPU_RESERVATION_RECEIPT)"

gpu-bootstrap:
	$(call require-var,GPU_PHASE)
	@bootstrap_values="$$( $(PYTHON_BOOTSTRAP) scripts/extract_gpu_bootstrap_inputs.py --gpu-quote-lock "$(GPU_QUOTE_LOCK)" --gpu-lock "$(GPU_LOCK)" --phase "$(GPU_PHASE)" )"; \
	IFS=$$'\t' read -r gpu_family provider_gpu_id allowed_cuda_versions data_center_ids storage_rate per_gpu_rate runtime_hours price_source price_checked_at container_digest wheel_url wheel_sha256 semantic_wheel_url semantic_wheel_sha256 semantic_version semantic_stack_lock_hash bootstrap_constraints_path bootstrap_constraints_sha256 bootstrap_distribution_lock_hash <<< "$$bootstrap_values"; \
	bash scripts/bootstrap_gpu.sh \
		"$$gpu_family" \
		"$$provider_gpu_id" \
		"$$allowed_cuda_versions" \
		"$$data_center_ids" \
		"$$storage_rate" \
		"$$per_gpu_rate" \
		"$$runtime_hours" \
		"$$price_source" \
		"$$price_checked_at" \
		"$$container_digest" \
		"$$wheel_url" \
		"$$wheel_sha256" \
		"$$semantic_wheel_url" \
		"$$semantic_wheel_sha256" \
		"$$semantic_version" \
		"$$semantic_stack_lock_hash" \
		"$$bootstrap_constraints_path" \
		"$$bootstrap_constraints_sha256" \
		"$$bootstrap_distribution_lock_hash" \
		"$(GPU_PHASE)" \
		"$(GPU_RESERVATION_RECEIPT)" \
		"$(COST_LEDGER)"

gpu-active-verify:
	$(call require-var,GPU_PHASE)
	@$(derive-session-directory); \
	$(GPU_PYTHON) scripts/runpod_active_session_verify.py \
		--session-directory "$$session_dir" \
		--reservation-receipt "$(GPU_RESERVATION_RECEIPT)" \
		--cost-ledger "$(COST_LEDGER)" \
		--phase "$(GPU_PHASE)" \
		--session-id-env GPU_BUDGET_SESSION_ID \
		--gpu-hard-stop-usd 220 \
		--api-hard-stop-usd 100 \
		--total-hard-stop-usd 325

gpu-settle:
	$(call require-var,GPU_PHASE)
	$(call require-var,PROVIDER_INCURRED_USD)
	@$(derive-session-directory); \
	$(PYTHON) scripts/gpu_budget_settle.py \
		--reservation-receipt "$(GPU_RESERVATION_RECEIPT)" \
		--cost-ledger "$(COST_LEDGER)" \
		--watchdog-state "$$session_dir/runpod_watchdog.json" \
		--session-id-env GPU_BUDGET_SESSION_ID \
		--provider-incurred-usd "$(PROVIDER_INCURRED_USD)" \
		--gpu-hard-stop-usd 220 \
		--api-hard-stop-usd 100 \
		--total-hard-stop-usd 325 \
		--output "$$session_dir/settlement.json"

time-start:
	$(call require-var,TIME_CATEGORY)
	$(call require-var,TIME_DESCRIPTION)
	$(call require-var,TIME_STATUS)
	$(PYTHON) scripts/investigation_timer.py start \
		--ledger "$(TIME_LEDGER)" \
		--category "$(TIME_CATEGORY)" \
		--description "$(TIME_DESCRIPTION)" \
		--status "$(TIME_STATUS)"

time-stop:
	$(PYTHON) scripts/investigation_timer.py stop --ledger "$(TIME_LEDGER)"

time-status:
	$(PYTHON) scripts/investigation_timer.py status --ledger "$(TIME_LEDGER)"

release-check: lint test
	$(PYTHON) scripts/release_audit.py --root "$(CURDIR)"
