# Production RunPod lifecycle gate

`scripts/runpod_pod_lifecycle.py` is the only supported creator/re-arm path for the paid Pod. It
is intentionally not a general RunPod client: there is no delete, terminate, stop, restart, or
arbitrary update operation. Stopping remains the independently armed watchdog's responsibility;
that process uses only official REST v1 GET and non-destructive `/stop` operations.

The helper authenticates, before constructing an HTTP client:

- the content-addressed GPU quote and API quote;
- the exact schema-v2 paid-run approval and command phase;
- the full config, preregistration, and GPU/software lock bindings;
- the phase's active, pre-created reservation and opaque nonce against the canonical cost ledger;
- the all-in rate and maximum: `8 × GPU rate + running-storage rate`;
- the exact H100 SXM provider ID, Secure cloud, CUDA 12.8, the privately quote-bound data-center
  allowlist, 50 GB container disk, and 650 GB host-local persistent `/workspace` mount.

Creation is `POST https://api.runpod.io/v2/pods` using the v2-native shape. Provider list/live
verification remains read-only v1 `GET https://rest.runpod.io/v1/pods...`. Re-arm uses v2 `PATCH
/pods/{id}` followed by v2 `POST /pods/{id}/action` with exactly `{"action":"start"}`.

## Secret boundary

Set these only in the local process environment:

```bash
read -rsp 'RunPod API key: ' RUNPOD_API_KEY; echo; export RUNPOD_API_KEY
read -rsp 'Read-only Hugging Face token: ' HF_TOKEN; echo; export HF_TOKEN
GPU_BUDGET_SESSION_ID="$(openssl rand -hex 32)"
export GPU_BUDGET_SESSION_ID
```

Never put their values in an argument, tracked file, report, or chat. The account RunPod key is
used only in the HTTP Authorization header and is never sent into the Pod. The create request's
caller-supplied Pod environment is restricted to `HF_TOKEN`, `GPU_BUDGET_SESSION_ID`, the fixed
Hugging Face/vLLM cache directories, and `VLLM_ENABLE_CUDA_COMPATIBILITY=1`. RunPod may add its
non-secret `PUBLIC_KEY` value for `startSsh=true`; re-arm preserves that value unchanged.

Provider responses can echo all Pod environment values. The helper validates those values in
memory, then persists only a reduced record. `.runpod/pod_lifecycle.json` is non-symlinked, owned
by the current user, mode `0600`, content-addressed, and ignored by Git. It contains the private
Pod ID and SSH details, but no environment object or raw secret. CLI output prints only a Pod-ID
hash and whether private SSH details exist.

## First creation

Freeze and explicitly approve the current quote/bundle, then reserve the exact phase with the
same fresh nonce:

```bash
GPU_PHASE='behavior_baseline_gpu'
make gpu-reserve GPU_PHASE="$GPU_PHASE"

PYTHONPATH=src .venv/bin/python scripts/runpod_pod_lifecycle.py \
  --project-root . \
  create \
  --phase "$GPU_PHASE" \
  --reservation ".runpod/reservations/$GPU_PHASE.json"
```

By default, before POST the helper refuses creation if the account has any existing nonterminal
Pod. A user-confirmed unrelated Pod may coexist only when its exact provider ID has been hashed
locally as `runpod-pod-id-sha256:` followed by the lowercase SHA-256 of the raw UTF-8 ID, and that
hash is supplied once with the create-only repeatable option below:

```bash
PYTHONPATH=src .venv/bin/python scripts/runpod_pod_lifecycle.py \
  --project-root . \
  create \
  --phase "$GPU_PHASE" \
  --reservation ".runpod/reservations/$GPU_PHASE.json" \
  --allow-existing-pod-id-hash 'runpod-pod-id-sha256:<64-lowercase-hex>'
```

The raw unrelated Pod ID must never be put in argv, lifecycle state, or output. Every live
nonterminal Pod must match exactly one supplied hash; duplicate hashes, stale/extra hashes,
terminal-Pod hashes, malformed live IDs, and any unacknowledged live Pod all fail closed before
POST. The canonical sorted hash set is included in the secret-safe launch-intent hash and private
create authorization for audit. This exception never bypasses the local lifecycle claim: an
existing `.runpod/pod_lifecycle.json` still prevents a second model-forensics create.

After the account-level coexistence gate, the helper claims a local `create_intent` before the
paid request, preventing a retry after any uncertain
network outcome. It verifies the v2 response and polls v1 for at most ten minutes, with no sleep
longer than 30 seconds, until the exact Pod is running and direct SSH is ready. Pending, terminal,
verification-failed, and timeout states remain claimed. Do **not** rerun `create`; inspect status.

```bash
PYTHONPATH=src .venv/bin/python scripts/runpod_pod_lifecycle.py \
  --project-root . \
  status
```

Once ready, obtain the Pod ID privately for the existing bootstrap/watchdog workflow without
printing it:

```bash
read -r RUNPOD_POD_ID < <(jq -r '.pod.id' .runpod/pod_lifecycle.json)
export RUNPOD_POD_ID
make gpu-bootstrap GPU_PHASE="$GPU_PHASE"
```

The watchdog must remain armed for the entire paid phase. It performs stop—not this helper—and the
reservation must be settled after the stopped state and provider cost are confirmed.

The watchdog's private state protocol remains `schema_version: 2` /
`runpod-gpu-cost-watchdog-v2`; that is an internal record-format version, not a claim that the
provider endpoint is v2. `live_metadata.provider_api: rest-v1` identifies the authoritative
provider source. REST v1 cannot report CUDA placement, runtime GPU inventory, global-networking
state, or the lock flag, so those values remain null and are disclosed as unavailable. Bootstrap
supplies local `nvidia-smi` inventory and CUDA compatibility evidence; networking and lock remain
explicit provider-evidence limitations rather than inferred values.

## Re-arm the same stopped Pod

After the prior reservation is settled, create a new nonce and reservation for the next exact
phase. Re-arm refuses unless live v1 metadata proves this is the same `EXITED` Pod, on the same
machine, with the approved image/GPU/cloud/data center/storage/ports/networking and unchanged
allow-listed environment. It replaces the complete allow-listed environment with values that
differ only at `GPU_BUDGET_SESSION_ID`, then starts that Pod. It never creates another Pod.

```bash
GPU_PHASE='behavior_treatment_gpu'
GPU_BUDGET_SESSION_ID="$(openssl rand -hex 32)"
export GPU_BUDGET_SESSION_ID
make gpu-reserve GPU_PHASE="$GPU_PHASE"

PYTHONPATH=src .venv/bin/python scripts/runpod_pod_lifecycle.py \
  --project-root . \
  rearm \
  --phase "$GPU_PHASE" \
  --reservation ".runpod/reservations/$GPU_PHASE.json"
```

Immediately repeat the private `RUNPOD_POD_ID` extraction and `make gpu-bootstrap` step so the new
phase's watchdog and local active-session gate are armed. If any lifecycle operation ends in an
intent/pending/failed state, use `status` and audit the existing Pod; never delete it through this
helper and never attempt a second create.
