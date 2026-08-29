#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 22 ]]; then
  echo "usage: $0 GPU_FAMILY PROVIDER_GPU_ID ALLOWED_CUDA_VERSIONS_CSV DATA_CENTER_IDS_CSV RUNNING_STORAGE_USD_PER_HOUR HOURLY_PER_GPU_USD APPROVED_PHASE_RUNTIME_HOURS PRICE_SOURCE PRICE_CHECKED_AT CONTAINER_IMAGE_DIGEST VLLM_WHEEL_URL VLLM_WHEEL_SHA256 SEMANTIC_WHEEL_URL SEMANTIC_WHEEL_SHA256 SEMANTIC_DISTRIBUTION_VERSION SEMANTIC_STACK_LOCK_HASH BOOTSTRAP_CONSTRAINTS_PATH BOOTSTRAP_CONSTRAINTS_SHA256 BOOTSTRAP_DISTRIBUTION_LOCK_HASH GPU_PHASE GPU_RESERVATION_RECEIPT COST_LEDGER" >&2
  exit 2
fi

GPU_FAMILY="$1"
PROVIDER_GPU_ID="$2"
ALLOWED_CUDA_VERSIONS_CSV="$3"
DATA_CENTER_IDS_CSV="$4"
RUNNING_STORAGE_USD_PER_HOUR="$5"
HOURLY_PER_GPU_USD="$6"
APPROVED_PHASE_RUNTIME_HOURS="$7"
PRICE_SOURCE="$8"
PRICE_CHECKED_AT="$9"
CONTAINER_IMAGE_DIGEST="${10}"
VLLM_WHEEL_URL="${11}"
VLLM_WHEEL_SHA256="${12}"
SEMANTIC_WHEEL_URL="${13}"
SEMANTIC_WHEEL_SHA256="${14}"
SEMANTIC_DISTRIBUTION_VERSION="${15}"
SEMANTIC_STACK_LOCK_HASH="${16}"
BOOTSTRAP_CONSTRAINTS_PATH="${17}"
BOOTSTRAP_CONSTRAINTS_SHA256="${18}"
BOOTSTRAP_DISTRIBUTION_LOCK_HASH="${19}"
GPU_PHASE="${20}"
GPU_RESERVATION_RECEIPT="${21}"
COST_LEDGER="${22}"

if [[ "$ALLOWED_CUDA_VERSIONS_CSV" != "12.8" ]]; then
  echo "approved CUDA host version set must be exactly 12.8" >&2
  exit 2
fi
CUDA_ARGS=(--allowed-cuda-version "$ALLOWED_CUDA_VERSIONS_CSV")
IFS=',' read -r -a DATA_CENTER_IDS <<< "$DATA_CENTER_IDS_CSV"
if [[ "${#DATA_CENTER_IDS[@]}" -eq 0 ]]; then
  echo "at least one frozen RunPod data center id is required" >&2
  exit 2
fi
DATA_CENTER_ARGS=()
for data_center_id in "${DATA_CENTER_IDS[@]}"; do
  if [[ ! "$data_center_id" =~ ^[A-Z0-9][A-Z0-9-]{2,31}$ ]]; then
    echo "invalid frozen RunPod data center id" >&2
    exit 2
  fi
  DATA_CENTER_ARGS+=(--allowed-data-center-id "$data_center_id")
done

umask 077
mkdir -p .runpod
RUNPOD_SESSIONS_ROOT=".runpod/sessions"
WATCHDOG_STATE=".runpod/emergency_watchdog.json"
WATCHDOG_PID_FILE=".runpod/emergency_watchdog.pid"
WATCHDOG_LOG=".runpod/emergency_watchdog.log"
STOP_REQUEST_PATH=".runpod/emergency_stop.request"
GPU_BUDGET_BOOTSTRAP_STATE=".runpod/gpu_budget_bootstrap.pending.json"
LIFECYCLE_STATE=".runpod/pod_lifecycle.json"
EXPECTED_SESSION_HASH=""
GPU_BOOTSTRAP_TMP=""
BOOTSTRAP_SUCCEEDED=0
WATCHDOG_PID=""
# Inherited environment variables retain Bash's export attribute after a plain
# assignment.  Remove any caller-provided private aliases before recreating
# them so the in-memory credential copies can never become child environment
# variables accidentally.
unset GPU_BUDGET_SESSION_ID_PRIVATE RUNPOD_API_KEY_PRIVATE HF_TOKEN_PRIVATE || true
GPU_BUDGET_SESSION_ID_PRIVATE=""
RUNPOD_API_KEY_PRIVATE=""
HF_TOKEN_PRIVATE=""

secret_free() {
  env -u RUNPOD_API_KEY -u HF_TOKEN -u GPU_BUDGET_SESSION_ID \
    -u GPU_BUDGET_SESSION_ID_PRIVATE -u RUNPOD_API_KEY_PRIVATE \
    -u HF_TOKEN_PRIVATE "$@"
}

with_hf_token() (
  printf -v HF_TOKEN '%s' "$HF_TOKEN_PRIVATE"
  export HF_TOKEN
  unset RUNPOD_API_KEY GPU_BUDGET_SESSION_ID GPU_BUDGET_SESSION_ID_PRIVATE \
    RUNPOD_API_KEY_PRIVATE HF_TOKEN_PRIVATE
  exec "$@"
)

with_gpu_session_id() (
  printf -v GPU_BUDGET_SESSION_ID '%s' "$GPU_BUDGET_SESSION_ID_PRIVATE"
  export GPU_BUDGET_SESSION_ID
  unset RUNPOD_API_KEY HF_TOKEN GPU_BUDGET_SESSION_ID_PRIVATE \
    RUNPOD_API_KEY_PRIVATE HF_TOKEN_PRIVATE
  exec "$@"
)

# The exact regular/emergency watchdog is the sole non-model exception that
# receives both provider credentials: it binds the live Pod's HF credential
# hash while retaining stop capability.  No setup, download, or build child
# uses this wrapper.
with_watchdog_credentials() (
  printf -v RUNPOD_API_KEY '%s' "$RUNPOD_API_KEY_PRIVATE"
  printf -v HF_TOKEN '%s' "$HF_TOKEN_PRIVATE"
  export RUNPOD_API_KEY HF_TOKEN
  unset GPU_BUDGET_SESSION_ID GPU_BUDGET_SESSION_ID_PRIVATE \
    RUNPOD_API_KEY_PRIVATE HF_TOKEN_PRIVATE
  exec "$@"
)
case "$GPU_FAMILY" in
  H100 | H100_80GB | A100 | A100_80GB) EMERGENCY_GPU_FAMILY="$GPU_FAMILY" ;;
  *) EMERGENCY_GPU_FAMILY="H100_80GB" ;;
esac

cleanup_on_exit() {
  local status="$?"
  if [[ "$BOOTSTRAP_SUCCEEDED" != "1" ]]; then
    : > "$STOP_REQUEST_PATH" || true
    local watchdog_identity_live=0
    if [[ -n "$WATCHDOG_PID" && -f "$WATCHDOG_PID_FILE" ]] \
      && [[ -f scripts/runpod_process_identity.py ]] \
      && secret_free env PYTHONPATH="$PWD/src" \
        python3 scripts/runpod_process_identity.py verify \
        --identity "$WATCHDOG_PID_FILE" >/dev/null 2>&1; then
      watchdog_identity_live=1
    fi
    if [[ "$watchdog_identity_live" != "1" ]] \
      && [[ -n "${RUNPOD_POD_ID:-}" ]] \
      && [[ -n "$RUNPOD_API_KEY_PRIVATE" ]] \
      && [[ -n "$HF_TOKEN_PRIVATE" ]] \
      && [[ -n "$EXPECTED_SESSION_HASH" ]] \
      && [[ -f "$LIFECYCLE_STATE" ]] \
      && [[ -f scripts/runpod_watchdog.py ]]; then
      # Receipt validation happens before the regular watchdog is configured.
      # If it fails, arm an immediate one-shot watchdog so this exact Pod does
      # not continue billing. Hardware/rate mismatch also takes the tested stop
      # path in run_watchdog.
      with_watchdog_credentials env PYTHONPATH="$PWD/src" \
        python3 scripts/runpod_watchdog.py \
        --pod-id "$RUNPOD_POD_ID" \
        --lifecycle-state "$LIFECYCLE_STATE" \
        --expected-session-hash "$EXPECTED_SESSION_HASH" \
        --expected-phase "$GPU_PHASE" \
        --expected-gpu-family "$EMERGENCY_GPU_FAMILY" \
        --expected-provider-gpu-id "$PROVIDER_GPU_ID" \
        "${DATA_CENTER_ARGS[@]}" \
        "${CUDA_ARGS[@]}" \
        --expected-container-image "$CONTAINER_IMAGE_DIGEST" \
        --expected-gpu-count 8 \
        --maximum-approved-hourly-per-gpu-usd 1000000 \
        --maximum-approved-storage-hourly-usd "$RUNNING_STORAGE_USD_PER_HOUR" \
        --gpu-hard-stop-usd 220 \
        --maximum-runtime-hours 0.000001 \
        --safety-margin-fraction 0.03 \
        --poll-seconds 1 \
        --state ".runpod/emergency_stop.json" \
        >/dev/null 2>&1 || true
    fi
  fi
  unset GPU_BUDGET_SESSION_ID || true
  unset GPU_BUDGET_SESSION_ID_PRIVATE || true
  unset RUNPOD_API_KEY RUNPOD_API_KEY_PRIVATE HF_TOKEN HF_TOKEN_PRIVATE || true
  if [[ -n "$GPU_BOOTSTRAP_TMP" && -d "$GPU_BOOTSTRAP_TMP" ]]; then
    rm -rf -- "$GPU_BOOTSTRAP_TMP" || true
  fi
  return "$status"
}
trap cleanup_on_exit EXIT

if [[ ! -f pyproject.toml || ! -f scripts/runpod_watchdog.py ]]; then
  echo "run bootstrap from the repository root" >&2
  exit 2
fi
if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
  echo "RunPod-provided RUNPOD_POD_ID is required" >&2
  exit 2
fi
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RunPod-provided RUNPOD_API_KEY is required" >&2
  exit 2
fi
if [[ -z "${GPU_BUDGET_SESSION_ID:-}" ]]; then
  echo "GPU_BUDGET_SESSION_ID must contain the opaque pre-launch session nonce" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for in-memory provider credential binding" >&2
  exit 2
fi
GPU_BUDGET_SESSION_ID_PRIVATE="$GPU_BUDGET_SESSION_ID"
RUNPOD_API_KEY_PRIVATE="$RUNPOD_API_KEY"
HF_TOKEN_PRIVATE="$HF_TOKEN"
unset GPU_BUDGET_SESSION_ID RUNPOD_API_KEY HF_TOKEN
EXPECTED_SESSION_HASH="$(with_gpu_session_id env PYTHONPATH="$PWD/src" python3 -c 'import os; from model_forensics.io import stable_hash; print(stable_hash({"opaque_gpu_session_id": os.environ["GPU_BUDGET_SESSION_ID"]}))')"
if [[ ! "$EXPECTED_SESSION_HASH" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "could not derive a safe GPU session identity hash" >&2
  exit 2
fi
if [[ ! -f "$GPU_RESERVATION_RECEIPT" || ! -f "$COST_LEDGER" ]]; then
  echo "authenticated GPU reservation receipt and canonical cost ledger must be synced" >&2
  exit 2
fi
if [[ -n "${RUNPOD_GPU_COUNT:-}" && "$RUNPOD_GPU_COUNT" != "8" ]]; then
  echo "RunPod-provided RUNPOD_GPU_COUNT must equal 8" >&2
  exit 2
fi
for stale_path in "$WATCHDOG_STATE" "$WATCHDOG_PID_FILE" "$STOP_REQUEST_PATH" "$GPU_BUDGET_BOOTSTRAP_STATE"; do
  if [[ -e "$stale_path" ]]; then
    echo "refusing to reuse stale watchdog artifact: $stale_path" >&2
    exit 2
  fi
done

# Authenticate the locally created one-use reservation before the watchdog,
# downloads, model load, or experiment backend. The opaque session nonce is
# read from the environment and never placed in a process command line.
with_gpu_session_id env PYTHONPATH="$PWD/src" python3 scripts/gpu_budget_preflight.py \
  --reservation-receipt "$GPU_RESERVATION_RECEIPT" \
  --cost-ledger "$COST_LEDGER" \
  --phase "$GPU_PHASE" \
  --session-id-env GPU_BUDGET_SESSION_ID \
  --expected-approved-runtime-hours "$APPROVED_PHASE_RUNTIME_HOURS" \
  --expected-live-hourly-total-usd "$(secret_free python3 -c 'import sys; print(8 * float(sys.argv[1]) + float(sys.argv[2]))' "$HOURLY_PER_GPU_USD" "$RUNNING_STORAGE_USD_PER_HOUR")" \
  --gpu-hard-stop-usd 220 \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325 \
  --output "$GPU_BUDGET_BOOTSTRAP_STATE"

IFS=$'\t' read -r GPU_HARD_STOP_USD MAXIMUM_SAFE_RUNTIME_HOURS SAFETY_MARGIN_FRACTION PRIOR_COMMITTED_GPU_USD < <(
  secret_free python3 - "$GPU_BUDGET_BOOTSTRAP_STATE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    payload["global_gpu_hard_stop_usd"],
    payload["maximum_safe_runtime_hours"],
    payload["safety_margin_fraction"],
    payload["prior_committed_gpu_usd"],
    sep="\t",
)
PY
)
if [[ -z "$GPU_HARD_STOP_USD" || -z "$MAXIMUM_SAFE_RUNTIME_HOURS" || -z "$SAFETY_MARGIN_FRACTION" || -z "$PRIOR_COMMITTED_GPU_USD" ]]; then
  echo "GPU budget preflight did not emit complete watchdog limits" >&2
  exit 2
fi

# A new session directory can be claimed only when every prior phase has both
# stopped_confirmed watchdog state and an exact settlement in the canonical
# ledger. The pending bootstrap receipt is atomically moved into that private
# directory, so the same phase/session identity cannot be re-armed.
SESSION_DIR="$(secret_free env PYTHONPATH="$PWD/src" python3 scripts/runpod_session_prepare.py \
  --sessions-root "$RUNPOD_SESSIONS_ROOT" \
  --pending-budget-bootstrap "$GPU_BUDGET_BOOTSTRAP_STATE" \
  --cost-ledger "$COST_LEDGER" \
  --gpu-hard-stop-usd "$GPU_HARD_STOP_USD" \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325)"
if [[ "sha256:${SESSION_DIR##*/}" != "$EXPECTED_SESSION_HASH" ]]; then
  echo "authenticated session directory disagrees with the in-memory session identity" >&2
  exit 2
fi
case "$SESSION_DIR" in
  "$PWD"/.runpod/sessions/*) ;;
  *)
    echo "private RunPod session preparation returned an unsafe path" >&2
    exit 2
    ;;
esac
WATCHDOG_STATE="$SESSION_DIR/runpod_watchdog.json"
WATCHDOG_PID_FILE="$SESSION_DIR/runpod_watchdog.pid"
WATCHDOG_LOG="$SESSION_DIR/runpod_watchdog.log"
STOP_REQUEST_PATH="$SESSION_DIR/runpod_stop.request"
GPU_PREFLIGHT_STATE="$SESSION_DIR/gpu_preflight.json"
GPU_SETUP_DIR=".runpod/setup"
GPU_SETUP_LOCK="$GPU_SETUP_DIR/setup_lock.json"
GPU_ENVIRONMENT_MANIFEST="$GPU_SETUP_DIR/gpu_environment.json"
QWEN4B_SMOKE_MANIFEST="$GPU_SETUP_DIR/qwen4b_prefix_smoke.json"
GPU_SETUP_VALIDATION="$SESSION_DIR/gpu_setup_validation.json"
POST_SETUP_ACTIVE_SESSION="$SESSION_DIR/post_setup_active_session.json"
TRANSFORMERS_COMMIT="42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
JLENS_COMMIT="581d398613e5602a5af361e1c34d3a92ea82ba8e"

# Arm a separate, nohup-protected process before any wheel or model download.
with_watchdog_credentials env PYTHONPATH="$PWD/src" nohup \
  python3 scripts/runpod_watchdog.py \
  --pod-id "$RUNPOD_POD_ID" \
  --lifecycle-state "$LIFECYCLE_STATE" \
  --expected-session-hash "$EXPECTED_SESSION_HASH" \
  --expected-phase "$GPU_PHASE" \
  --expected-gpu-family "$GPU_FAMILY" \
  --expected-provider-gpu-id "$PROVIDER_GPU_ID" \
  "${DATA_CENTER_ARGS[@]}" \
  "${CUDA_ARGS[@]}" \
  --expected-container-image "$CONTAINER_IMAGE_DIGEST" \
  --expected-gpu-count 8 \
  --maximum-approved-hourly-per-gpu-usd "$HOURLY_PER_GPU_USD" \
  --maximum-approved-storage-hourly-usd "$RUNNING_STORAGE_USD_PER_HOUR" \
  --gpu-hard-stop-usd "$GPU_HARD_STOP_USD" \
  --maximum-runtime-hours "$MAXIMUM_SAFE_RUNTIME_HOURS" \
  --safety-margin-fraction "$SAFETY_MARGIN_FRACTION" \
  --prior-committed-gpu-usd "$PRIOR_COMMITTED_GPU_USD" \
  --poll-seconds 15 \
  --state "$WATCHDOG_STATE" \
  --stop-request "$STOP_REQUEST_PATH" \
  > "$WATCHDOG_LOG" 2>&1 &
WATCHDOG_PID="$!"
secret_free env PYTHONPATH="$PWD/src" python3 scripts/runpod_process_identity.py record \
  --pid "$WATCHDOG_PID" \
  --output "$WATCHDOG_PID_FILE" \
  --required-cmdline-token scripts/runpod_watchdog.py \
  --required-cmdline-token "$WATCHDOG_STATE" \
  --required-cmdline-token "$STOP_REQUEST_PATH" \
  >/dev/null

watchdog_process_is_live() {
  secret_free env PYTHONPATH="$PWD/src" python3 scripts/runpod_process_identity.py verify \
    --identity "$WATCHDOG_PID_FILE" >/dev/null 2>&1
}

watchdog_is_armed() {
  secret_free python3 - "$WATCHDOG_STATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, OSError, json.JSONDecodeError):
    raise SystemExit(1)
metadata = payload.get("live_metadata")
expected_gaps = {
    "cuda_version",
    "global_networking_enabled",
    "interruptible",
    "locked",
    "runtime_gpu_count",
}
unavailable = metadata.get("provider_evidence_unavailable") if isinstance(metadata, dict) else None
ready = (
    payload.get("schema_version") == 2
    and payload.get("watchdog_version") == "runpod-gpu-cost-watchdog-v2"
    and payload.get("status") == "armed"
    and isinstance(metadata, dict)
    and metadata.get("provider_api") == "rest-v1"
    and isinstance(unavailable, list)
    and all(isinstance(item, str) for item in unavailable)
    and len(unavailable) == len(set(unavailable))
    and set(unavailable) == expected_gaps
    and all(field in metadata and metadata[field] is None for field in expected_gaps)
)
raise SystemExit(0 if ready else 1)
PY
}

WATCHDOG_READY=0
for _ in $(seq 1 45); do
  if ! watchdog_process_is_live; then
    echo "RunPod watchdog process identity changed before arming; inspect $WATCHDOG_LOG" >&2
    exit 2
  fi
  if watchdog_is_armed; then
    WATCHDOG_READY=1
    break
  fi
  sleep 1
done
if [[ "$WATCHDOG_READY" != "1" ]]; then
  echo "RunPod watchdog did not arm within 45 seconds" >&2
  exit 2
fi

with_gpu_session_id env PYTHONPATH="$PWD/src" python3 scripts/runpod_preflight.py \
  --required-gpus 8 \
  --minimum-memory-gib 79 \
  --minimum-free-disk-gib 520 \
  --expected-gpu-family "$GPU_FAMILY" \
  --expected-provider-gpu-id "$PROVIDER_GPU_ID" \
  "${DATA_CENTER_ARGS[@]}" \
  --allowed-cuda-version "$ALLOWED_CUDA_VERSIONS_CSV" \
  --pod-id "$RUNPOD_POD_ID" \
  --watchdog-state "$WATCHDOG_STATE" \
  --watchdog-pid-file "$WATCHDOG_PID_FILE" \
  --hourly-per-gpu-usd "$HOURLY_PER_GPU_USD" \
  --approved-storage-hourly-usd "$RUNNING_STORAGE_USD_PER_HOUR" \
  --approved-phase-runtime-hours "$APPROVED_PHASE_RUNTIME_HOURS" \
  --planned-hours "$MAXIMUM_SAFE_RUNTIME_HOURS" \
  --gpu-budget-usd "$GPU_HARD_STOP_USD" \
  --prior-committed-gpu-usd "$PRIOR_COMMITTED_GPU_USD" \
  --gpu-budget-reservation "$GPU_RESERVATION_RECEIPT" \
  --cost-ledger "$COST_LEDGER" \
  --gpu-phase "$GPU_PHASE" \
  --gpu-session-id-env GPU_BUDGET_SESSION_ID \
  --api-budget-usd 100 \
  --total-budget-usd 325 \
  --price-source "$PRICE_SOURCE" \
  --price-checked-at "$PRICE_CHECKED_AT" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
  --vllm-wheel-url "$VLLM_WHEEL_URL" \
  --vllm-wheel-sha256 "$VLLM_WHEEL_SHA256" \
  --output "$GPU_PREFLIGHT_STATE"

# Raw credentials were copied into non-exported shell variables before any
# watchdog or setup child.  Keep the public names absent for defense in depth.
unset GPU_BUDGET_SESSION_ID RUNPOD_API_KEY HF_TOKEN

mkdir -p "$GPU_SETUP_DIR"
if [[ -e .venv-gpu ]]; then
  if [[ ! -x .venv-gpu/bin/python || ! -f "$GPU_SETUP_LOCK" || ! -f "$GPU_ENVIRONMENT_MANIFEST" || ! -f "$QWEN4B_SMOKE_MANIFEST" ]]; then
    echo "existing GPU environment is incomplete and cannot be re-armed" >&2
    exit 2
  fi
  secret_free env PYTHONPATH="$PWD/src" python3 scripts/gpu_setup_lock.py validate \
    --lock "$GPU_SETUP_LOCK" \
    --environment-manifest "$GPU_ENVIRONMENT_MANIFEST" \
    --qwen4b-smoke-manifest "$QWEN4B_SMOKE_MANIFEST" \
    --venv-python .venv-gpu/bin/python \
    --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
    --vllm-wheel-url "$VLLM_WHEEL_URL" \
    --vllm-wheel-sha256 "$VLLM_WHEEL_SHA256" \
    --transformers-commit "$TRANSFORMERS_COMMIT" \
    --jlens-commit "$JLENS_COMMIT" \
    --semantic-wheel-url "$SEMANTIC_WHEEL_URL" \
    --semantic-wheel-sha256 "$SEMANTIC_WHEEL_SHA256" \
    --semantic-distribution-version "$SEMANTIC_DISTRIBUTION_VERSION" \
    --semantic-stack-lock-hash "$SEMANTIC_STACK_LOCK_HASH" \
    --bootstrap-constraints-sha256 "$BOOTSTRAP_CONSTRAINTS_SHA256" \
    --bootstrap-distribution-lock-hash "$BOOTSTRAP_DISTRIBUTION_LOCK_HASH" \
    > "$GPU_SETUP_VALIDATION"
else
  if [[ -e "$GPU_SETUP_LOCK" || -e "$GPU_ENVIRONMENT_MANIFEST" || -e "$QWEN4B_SMOKE_MANIFEST" ]]; then
    echo "GPU setup lock exists without its virtual environment" >&2
    exit 2
  fi
  GPU_BOOTSTRAP_TMP="$(mktemp -d)"
  if [[ "$BOOTSTRAP_CONSTRAINTS_PATH" != "config/gpu_bootstrap_constraints.txt" ]] \
    || [[ ! -f "$BOOTSTRAP_CONSTRAINTS_PATH" ]]; then
    echo "frozen GPU bootstrap constraints are absent" >&2
    exit 2
  fi
  ACTUAL_CONSTRAINTS_SHA256="$(secret_free shasum -a 256 "$BOOTSTRAP_CONSTRAINTS_PATH" | awk '{print $1}')"
  if [[ "$ACTUAL_CONSTRAINTS_SHA256" != "$BOOTSTRAP_CONSTRAINTS_SHA256" ]]; then
    echo "GPU bootstrap constraints SHA-256 mismatch" >&2
    exit 2
  fi
  VLLM_WHEEL_NAME="${VLLM_WHEEL_URL%%\?*}"
  VLLM_WHEEL_NAME="${VLLM_WHEEL_NAME##*/}"
  if [[ "$VLLM_WHEEL_NAME" != *.whl ]]; then
    echo "vLLM URL must resolve to a named .whl file" >&2
    exit 2
  fi
  VLLM_WHEEL_PATH="$GPU_BOOTSTRAP_TMP/$VLLM_WHEEL_NAME"
  secret_free curl --fail --location --retry 3 --output "$VLLM_WHEEL_PATH" "$VLLM_WHEEL_URL"
  ACTUAL_VLLM_SHA256="$(secret_free shasum -a 256 "$VLLM_WHEEL_PATH" | awk '{print $1}')"
  if [[ "$ACTUAL_VLLM_SHA256" != "$VLLM_WHEEL_SHA256" ]]; then
    echo "vLLM wheel SHA-256 mismatch" >&2
    exit 2
  fi
  SEMANTIC_WHEEL_NAME="${SEMANTIC_WHEEL_URL%%\?*}"
  SEMANTIC_WHEEL_NAME="${SEMANTIC_WHEEL_NAME##*/}"
  if [[ "$SEMANTIC_WHEEL_NAME" != *.whl ]]; then
    echo "semantic runtime URL must resolve to a named .whl file" >&2
    exit 2
  fi
  SEMANTIC_WHEEL_PATH="$GPU_BOOTSTRAP_TMP/$SEMANTIC_WHEEL_NAME"
  secret_free curl --fail --location --retry 3 --output "$SEMANTIC_WHEEL_PATH" "$SEMANTIC_WHEEL_URL"
  ACTUAL_SEMANTIC_SHA256="$(secret_free shasum -a 256 "$SEMANTIC_WHEEL_PATH" | awk '{print $1}')"
  if [[ "$ACTUAL_SEMANTIC_SHA256" != "$SEMANTIC_WHEEL_SHA256" ]]; then
    echo "sentence-transformers wheel SHA-256 mismatch" >&2
    exit 2
  fi

  secret_free python3 -m venv .venv-gpu
  secret_free .venv-gpu/bin/python -m pip install \
    --constraint "$BOOTSTRAP_CONSTRAINTS_PATH" \
    "setuptools==80.9.0" "wheel==0.46.3"
  secret_free .venv-gpu/bin/python -m pip install --no-build-isolation \
    --constraint "$BOOTSTRAP_CONSTRAINTS_PATH" \
    "$VLLM_WHEEL_PATH" "$SEMANTIC_WHEEL_PATH" \
    "transformers @ git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_COMMIT" \
    "jlens @ git+https://github.com/anthropics/jacobian-lens.git@$JLENS_COMMIT" \
    "accelerate==1.12.0" "safetensors==0.7.0" \
    "pandas==3.0.3" "pydantic==2.12.5" "PyYAML==6.0.3" "matplotlib==3.10.8" \
    "huggingface-hub==1.29.0" \
    "numpy==2.5.2" \
    "scikit-learn==1.9.0" \
    "scipy==1.18.1" \
    "tokenizers==0.23.1" \
    "torch==2.13.0"
  secret_free .venv-gpu/bin/python -m pip install --no-build-isolation -e . --no-deps
  secret_free .venv-gpu/bin/python scripts/capture_environment.py \
    --output "$GPU_ENVIRONMENT_MANIFEST" \
    --vllm-wheel "$VLLM_WHEEL_PATH" \
    --semantic-wheel "$SEMANTIC_WHEEL_PATH" \
    --bootstrap-constraints "$BOOTSTRAP_CONSTRAINTS_PATH"
  with_hf_token .venv-gpu/bin/python scripts/qwen4b_prefix_smoke.py \
    --output "$QWEN4B_SMOKE_MANIFEST" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --rollout-max-tokens 1024 \
    --continuation-max-tokens 256
  secret_free env PYTHONPATH="$PWD/src" python3 scripts/gpu_setup_lock.py create \
    --lock "$GPU_SETUP_LOCK" \
    --environment-manifest "$GPU_ENVIRONMENT_MANIFEST" \
    --qwen4b-smoke-manifest "$QWEN4B_SMOKE_MANIFEST" \
    --venv-python .venv-gpu/bin/python \
    --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
    --vllm-wheel-url "$VLLM_WHEEL_URL" \
    --vllm-wheel-sha256 "$VLLM_WHEEL_SHA256" \
    --transformers-commit "$TRANSFORMERS_COMMIT" \
    --jlens-commit "$JLENS_COMMIT" \
    --semantic-wheel-url "$SEMANTIC_WHEEL_URL" \
    --semantic-wheel-sha256 "$SEMANTIC_WHEEL_SHA256" \
    --semantic-distribution-version "$SEMANTIC_DISTRIBUTION_VERSION" \
    --semantic-stack-lock-hash "$SEMANTIC_STACK_LOCK_HASH" \
    --bootstrap-constraints-sha256 "$BOOTSTRAP_CONSTRAINTS_SHA256" \
    --bootstrap-distribution-lock-hash "$BOOTSTRAP_DISTRIBUTION_LOCK_HASH" \
    > "$GPU_SETUP_VALIDATION"
fi

# Re-run the complete receipt/ledger/watchdog/process/preflight gate after all
# setup and smoke work.  A live PID plus an `armed` string is not sufficient:
# the PID may have been recycled and any one of the bound artifacts may have
# changed during the long installation window.
with_gpu_session_id env PYTHONPATH="$PWD/src" \
  python3 scripts/runpod_active_session_verify.py \
  --session-directory "$SESSION_DIR" \
  --reservation-receipt "$GPU_RESERVATION_RECEIPT" \
  --cost-ledger "$COST_LEDGER" \
  --phase "$GPU_PHASE" \
  --gpu-hard-stop-usd "$GPU_HARD_STOP_USD" \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325 \
  > "$POST_SETUP_ACTIVE_SESSION"
unset GPU_BUDGET_SESSION_ID_PRIVATE
unset RUNPOD_API_KEY_PRIVATE HF_TOKEN_PRIVATE
BOOTSTRAP_SUCCEEDED=1
