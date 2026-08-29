# Bounded RunPod execution

No paid Pod is authorized by this document. Launch only after the user confirms the exact
machine, current price, planned maximum runtime, and estimated charge.

Pod creation and stopped-Pod re-arming must use the production-safe workflow in
[`docs/runpod_lifecycle.md`](docs/runpod_lifecycle.md). It records the official v2 request shape,
private secret boundary, duplicate-spend lock, bounded readiness polling, and recovery rules.
The only coexistence exception is the create-only, hash-bound acknowledgement documented there:
every unrelated nonterminal Pod must be explicitly matched, while duplicate, missing, or extra
hashes and any existing model-forensics lifecycle claim still fail closed.

## Prepare without starting compute

1. Create a RunPod account and add only the intended payment balance or spending control.
2. Create a Hugging Face read-only token and an OpenRouter key. Store them as RunPod Secrets named
   `HF_TOKEN` and `OPENROUTER_API_KEY`. The Pod must also receive `RUNPOD_POD_ID` and a Pod-scoped
   `RUNPOD_API_KEY`; verify both are present, but do not replace the scoped key with an account-wide
   credential.
3. Do not paste secret values into chat, shell history, a notebook, Make variables, a report, or
   a tracked file. `.env.example` contains names only.
4. Immediately before launch, verify the current RunPod price and OpenRouter model prices from
   their provider pages. The hardware-price timestamp must be timezone-aware and less than six
   hours old.
5. Freeze an immutable container image digest and the exact URL and SHA-256 of the compatible
   vLLM wheel. A floating image tag or unverified wheel fails preflight.

Before any reservation, use the two-step private-bundle workflow in
[`README.md`](README.md#freeze-and-review-the-private-paid-bundle). `preview` either exclusively
creates both content-addressed quote locks from independently reviewed private specifications or
authenticates locks that already exist. It prints a non-authorizing cost/binding summary. Only
after the user explicitly approves that exact summary may `approve` create schema-v2
`.runpod/paid_run_approval.json` with an explicit identifier, timestamp, and phase list. Neither
action contacts a provider or starts paid work. The approval binds the canonical hash of the full
software/GPU lock in addition to the exact quote, route, runtime, and budget fields.

## Exact launch shape

The approved primary launch is an 8-GPU Secure Pod with this exact immutable image:
`runpod/pytorch@sha256:e855789ff7e4b1ad76698171b1974a99a5c48c5b3e80a908976987938b090992`
(observed tag `1.0.7-cu1300-torch291-ubuntu2404`, OCI config
`sha256:18d90716b6bb9f4bed820c08e5b80cf6e0a99eb14e611b3446db82729b2d0b18`). The image includes
`cuda-compat-13-0` and `/start.sh`; do not override its entrypoint or start command.

The v2 create helper must bind and hash this request shape before launch; values marked “from
quote” come only from the authenticated GPU quote lock:

```json
{
  "cloudType": "SECURE",
  "computeType": "GPU",
  "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
  "gpuCount": 8,
  "allowedCudaVersions": ["12.8"],
  "dataCenterIds": ["FROM_FROZEN_QUOTE"],
  "imageName": "runpod/pytorch@sha256:e855789ff7e4b1ad76698171b1974a99a5c48c5b3e80a908976987938b090992",
  "containerDiskInGb": 50,
  "volumeInGb": 650,
  "volumeMountPath": "/workspace",
  "ports": ["22/tcp"],
  "supportPublicIp": true,
  "dockerEntrypoint": [],
  "dockerStartCmd": [],
  "env": {"VLLM_ENABLE_CUDA_COMPATIBILITY": "1"}
}
```

Do not attach a network volume: the watchdog uses the non-destructive stop operation, while a
network-volume Pod may require termination. Inject `RUNPOD_API_KEY`, `HF_TOKEN`, and any other
secret using RunPod Secrets; never serialize their values into the launch spec or its hash.

Creation and watchdog observation both use the approved v2 base
`https://api.runpod.io/v2`. The watchdog reads the exact Pod with
`GET /v2/pods/{podId}` and performs only the non-destructive
`POST /v2/pods/{podId}/action` body `{"action":"stop"}`. The v2 resource is required because it
exposes the approval-bound image, exact GPU/count, Secure cloud, data center, CUDA host version,
disks/mounts, ports, networking state, environment shape, SSH readiness, runtime inventory, start
time, and current compute rate in one authoritative response. Secret values and SSH routes are
validated in memory and never persisted.

## Hard budget and hardware gate

- Preferred hardware: one homogeneous 8× H100 80GB Pod.
- Fallback hardware: one homogeneous 8× A100 80GB Pod.
- Exactly eight unique full-GPU UUIDs are required; MIG and mixed GPU families are rejected.
- The quote and approval also freeze the exact provider GPU ID, `SECURE` cloud, allowed data
  centers, CUDA `12.8`, 50 GB container disk, 650 GB volume, and running-storage hourly rate.
- At least 79 GiB visible memory per GPU and 520 GiB free disk are required. A 650 GiB volume is
  recommended.
- GPU hard stop: USD 220.
- External judge/classifier hard stop: USD 100.
- Unallocated reserve: USD 5.
- Total hard stop: USD 325.

The frozen quality-first API design sends every valid final to both Claude Opus 5 and Gemini 3.1
Pro Preview; it does not substitute entry-tier or economy models. This does not relax the USD 100
API hard stop or authorize a call before the exact paid bundle is explicitly approved.

The approval input uses the provider-displayed **per-GPU hourly price**. The watchdog independently
reads the live Pod-level v2 `cost` from RunPod and refuses a live compute rate above the frozen
all-in compute-plus-storage ceiling. The approved GPU projection includes running
storage. Under the currently reviewed USD 0.10/GB-month schedule, the frozen 50 GB container plus
650 GB volume is encoded as `700 × 0.10 ÷ 720 ≈ 0.0972222222` USD/hour; the authenticated quote must
record that derivation's source and timestamp.

## Reserve once before starting a phase

Create the reservation in the canonical cost ledger before starting or restarting the Pod. Use a
new cryptographically random `GPU_BUDGET_SESSION_ID` for every phase. The raw value is an ephemeral
secret: put it only in the environment (and the corresponding RunPod Secret), never in an argument,
filename, tracked file, or chat. Select one exact canonical phase; the Make target loads the GPU
identity, cloud type, count, rate, quote provenance, and approved runtime from the authenticated
quote lock and accepts no free-form price or runtime override:

```bash
GPU_PHASE='behavior_baseline_gpu'
GPU_BUDGET_SESSION_ID="$(openssl rand -hex 32)"
export GPU_BUDGET_SESSION_ID

make gpu-reserve GPU_PHASE="$GPU_PHASE"
```

`make gpu-reserve` first places the preregistered USD 5 storage estimate in the canonical ledger,
then invokes `scripts/gpu_budget_reserve.py` with the exact lock-derived values. The approved GPU
maximum is `(GPU count × quoted per-GPU rate + running storage rate) × approved runtime`, rounded
upward to six decimal places. The reservation atomically accounts for all prior incurred and unresolved costs,
reserves the entire phase maximum, derives the remaining safe runtime at the quoted rate, and
refuses a duplicate or overlapping GPU reservation.

The receipt is created atomically under ignored `.runpod/`; an existing path is never overwritten.
Sync that receipt and the exact canonical ledger to the Pod without changing either file. Reserve
only immediately before a committed launch. If the Pod is never started, do not fabricate a
`stopped_confirmed` record or settlement; the outstanding reservation remains fail-closed until a
reviewed no-launch recovery is performed.

## Bootstrap or re-arm the stopped Pod

After the approved Pod starts, copy or clone only this repository, restore the reservation receipt
and ledger at their original paths, and inject the same `GPU_BUDGET_SESSION_ID`. From the repository
root, run the lock-derived fail-closed bootstrap:

```bash
make gpu-bootstrap GPU_PHASE="$GPU_PHASE"
```

The Make target reads the exact quote and `config/gpu_lock.yaml`, then supplies the current
15-argument interface to `scripts/bootstrap_gpu.sh`. It does not accept an image, wheel, price,
hardware identity, cloud type, or runtime through a Make variable.

Before a wheel, model, or experiment backend starts, bootstrap authenticates the receipt against
the canonical ledger and nonce, claims `.runpod/sessions/<session-hash>/`, and starts a separate
`nohup` watchdog. It passes the receipt's `prior_committed_gpu_usd` and
`maximum_safe_runtime_hours` to that watchdog and to hardware preflight. It then verifies eight
full GPUs, MIG status, disk, live prices, price freshness, image digest, and wheel hash. Pod ID, GPU
UUIDs, watchdog state, live billing state, and preflight stay only below ignored `.runpod/`; they
must not be copied to `data/manifests/`.

On the first phase, bootstrap installs `.venv-gpu`, then runs one real pinned Qwen3.5-4B rollout
and one exact raw-token-prefix continuation before any 122B backend can be constructed. Its passed
manifest stays at `.runpod/setup/qwen4b_prefix_smoke.json`; the setup lock binds both the manifest's
canonical hash and file SHA-256. On later phases using the same stopped Pod/volume, bootstrap does not reinstall or rerun the smoke: it
requires the exact container digest, 64-character wheel SHA-256, pinned 40-character Transformers
and Jacobian Lens Git commits, environment-manifest hash, smoke hashes, and live `pip freeze` to match. A stale
active session, an incomplete prior stop, or an unsettled prior reservation blocks re-arm.

The watchdog derives current incurred cost and two absolute deadlines from the provider's live v2
`startedAt` and `cost`: the 97% safe-budget deadline and approved maximum
runtime deadline. It uses the earlier one, never extends the deadline after a reported rate drop,
and stops early if live metadata becomes unsafe. The private persisted state intentionally omits
the API response's `env` object and all credentials.

## Authenticate every paid GPU command

Bootstrap alone is not permission to construct a model backend. Immediately before every paid GPU
CLI phase (and before a resume that would reload a backend), authenticate the live private session.
Derive the directory from the non-secret receipt hash, not from the raw nonce:

```bash
make gpu-active-verify GPU_PHASE="$GPU_PHASE"
```

This target derives the private session directory from the authenticated reservation receipt and
invokes `scripts/runpod_active_session_verify.py` with the exact receipt, ledger, phase, nonce
environment name, and hard caps.

Only `passed: true` authorizes backend construction. The verifier requires the same active ledger
reservation, receipt, phase and nonce hash; a fresh `armed` watchdog with a live PID; exact
cumulative budget/runtime/rate bindings; and the bound hardware preflight. The raw nonce never
appears in argv or output. Production callers must invoke the equivalent
`validate_active_runpod_session(...)` library gate before constructing `VLLMOfflineBackend` or
`VLLMRawPrefixBackend`; running a GPU CLI directly without that gate is unauthorized.

## Ordered execution gates

Run the deterministic local smoke before paid model work, then run the active-session verifier
again immediately before the production backend:

```bash
.venv-gpu/bin/python -m pytest
.venv-gpu/bin/python -m model_forensics.cli smoke --config config/smoke.yaml
```

The production order is split by resource boundary:

1. Run the local tests and smoke command above.
2. Reserve, bootstrap, and actively verify `behavior_baseline_gpu`; run
   `make behavior-baseline-generate`; checksum and sync its durable checkpoint; stop the Pod,
   confirm `stopped_confirmed`, and settle the reservation.
3. With no GPU backend loaded, run `make behavior-baseline-adjudicate` under the separately
   approved `behavior_baseline_api` phase. This double-judges the baseline outcomes, measures the
   trajectories, enforces the gates, and freezes the baseline-derived thresholds.
4. Repeat the complete reserve→bootstrap→verify→generate→sync→stop→settle lifecycle for
   `behavior_treatment_gpu` and `make behavior-treatment-generate`. Then run
   `make behavior-treatment-adjudicate` under `behavior_treatment_api` and run `make sample` to
   validate the completed behavioral release.
5. Run `make anchors`, which validates an existing manifest or authenticates `anchors_api` before
   the two frozen classifiers are called. Then run `make positions`, which similarly validates a
   complete bundle or authenticates `positions_api` before external span adjudication.
6. Repeat the GPU lifecycle for `resample_gpu` and run `make resample-generate`. After verified
   sync, stop and settle GPU compute before running `make resample-adjudicate` under
   `resample_api`. Run `make resample` to validate the completed canonical resampling artifact.
7. Repeat the GPU lifecycle for `lens_gpu` and run `make lens` only when the lens artifact is
   absent. If it already exists, the same target uses the local Python environment and performs
   validation only; it does not require a GPU session. The paid branch runs the 4B compatibility
   gate, no more than two limited 122B compatibility attempts, and matched J/R records from one
   activation capture.
8. Run `make analyze`, then `make report`, and perform the local release audit.

`make sample` and `make resample` are always validation-only aliases. They never create a model
backend or provider client, and an absent canonical artifact is an error. Paid work exists only in
the explicit `*-generate`, `*-adjudicate`, conditional `anchors`/`positions`, and absent-artifact
`lens` branches described above. Use only the non-secret path variables shown in `README.md`;
route identities, prices, hardware, cloud type, and runtimes come from authenticated locks. An
interrupted checkpoint directory must be audited and recovered explicitly; commands fail rather
than risk duplicating paid work.

The 27B lens is not an automatic fallback. It may be run only after both predeclared 122B
compatibility attempts fail, and must be labelled methodology support rather than evidence about
the 122B model's internal state.

## Checkpoint, stop, settle, and start the next phase

- Keep paid API outputs and cost-ledger writes checkpointed after every successful call.
- Keep raw and interim artifacts under ignored directories. Never copy the unlicensed upstream
  checkout into a release bundle.
- Sync needed outputs to the local workspace and verify SHA-256 hashes before declaring the sync
  complete.
- Derive the private session directory from the authenticated receipt's non-secret hash:

  ```bash
  GPU_RESERVATION_RECEIPT=".runpod/reservations/${GPU_PHASE}.json"
  SESSION_HASH_HEX="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_hash"].removeprefix("sha256:"))' "$GPU_RESERVATION_RECEIPT")"
  SESSION_DIR="$PWD/.runpod/sessions/$SESSION_HASH_HEX"
  ```

- After verified sync, request an immediate non-destructive stop with
  `touch "$SESSION_DIR/runpod_stop.request"`; do not wait for the watchdog deadline.
- Wait until `$SESSION_DIR/runpod_watchdog.json` reports `stopped_confirmed`. The watchdog calls
  only `POST /v2/pods/{podId}/action` with `{"action":"stop"}` and confirms `status=EXITED`; it
  never calls DELETE.
- Confirm in the RunPod UI that GPU billing stopped. A stopped Pod may still incur volume/storage
  charges, so remove no-longer-needed storage through the UI only after the local checksum audit.
- Read the authoritative provider-incurred GPU charge, then reconcile the exact reservation. The
  settlement command is idempotent only for the identical charge and authenticated receipt:

```bash
make gpu-settle \
  GPU_PHASE="$GPU_PHASE" \
  PROVIDER_INCURRED_USD="$AUTHORITATIVE_POST_STOP_PROVIDER_CHARGE"
```

The target derives the session and invokes `scripts/gpu_budget_settle.py` with the authenticated
receipt, canonical ledger, stopped watchdog state, nonce environment name, and hard caps. The
provider-incurred amount is reconciliation evidence after confirmed stop; it is not a rate,
runtime, or budget override.

The 650 GB volume costs USD 65/month while running and USD 130/month while stopped at the current
official rates; the 50 GB container disk adds USD 5/month only while running. The USD 5 storage
ledger reservation therefore covers at most about 27.7 stopped hours at the current 720-hour
billing-month convention. After the final checksum audit, terminate the Pod promptly, read the
authoritative cumulative storage charge, and settle the reserved storage entry exactly once:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/runpod_storage_budget.py settle \
  --cost-ledger data/manifests/cost_ledger.yaml \
  --amount-usd "$AUTHORITATIVE_STORAGE_CHARGE" \
  --gpu-hard-stop-usd 220 \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325
```

If the exact storage charge is unavailable, leave the USD 5 estimate outstanding and do not claim
final budget reconciliation. Never silently remove or relabel it.

An active session may resume only by reusing its existing receipt and the same nonce through the
active-session verifier; never call reserve or bootstrap again for that session. After
`stopped_confirmed` and `settled`, the session can never authorize more GPU work. The next phase
must use a new nonce and receipt, and re-arm validates that every prior private session is both
stopped and settled. Preserve the private `.runpod/sessions/` records locally for audit but publish
only hashes and aggregate costs, never Pod IDs, GPU UUIDs, live billing state, or the raw nonce.

The watchdog is a last-resort stop request, not proof of provider billing. Reconcile the final
provider invoice with the local cost ledger and report any difference without retroactively
changing the preregistered analysis.

Official API references: [RunPod v2 OpenAPI](https://api.runpod.io/v2/openapi.yaml),
[Pod API overview](https://docs.runpod.io/api-reference/pods/overview), and
[Pod billing history](https://docs.runpod.io/api-reference/billing/GET/billing/pods).
