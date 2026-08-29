# Bounded RunPod execution

No paid Pod is authorized by this document. Launch only after the user confirms the exact
machine, current price, planned maximum runtime, and estimated charge.

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

## Hard budget and hardware gate

- Preferred hardware: one homogeneous 8× H100 80GB Pod.
- Fallback hardware: one homogeneous 8× A100 80GB Pod.
- Exactly eight unique full-GPU UUIDs are required; MIG and mixed GPU families are rejected.
- At least 79 GiB visible memory per GPU and 520 GiB free disk are required. A 650 GiB volume is
  recommended.
- GPU hard stop: USD 220.
- External judge/classifier hard stop: USD 100.
- Unallocated reserve: USD 5.
- Total hard stop: USD 325.

The approval input uses the provider-displayed **per-GPU hourly price**. The watchdog independently
reads the live Pod-level `costPerHr` and `adjustedCostPerHr` from RunPod and refuses a nominal total
above `8 ×` the approved per-GPU quote.

## Reserve once before starting a phase

Create the reservation in the canonical cost ledger before starting or restarting the Pod. Use a
new cryptographically random `GPU_BUDGET_SESSION_ID` for every phase. The raw value is an ephemeral
secret: put it only in the environment (and the corresponding RunPod Secret), never in an argument,
filename, tracked file, or chat. The following are placeholders, not current prices or approved values:

```bash
GPU_FAMILY='H100_80GB'
PER_GPU_RATE='PROVIDER_DISPLAYED_USD_PER_GPU_HOUR'
PLANNED_HOURS='APPROVED_MAXIMUM_HOURS'
PHASE_MAXIMUM_USD='EXACT_8_X_RATE_X_HOURS'
GPU_PHASE='APPROVED_PHASE_IDENTIFIER'
PRICE_SOURCE='PROVIDER_PRICE_PAGE_URL'
PRICE_CHECKED_AT='ISO_8601_TIMESTAMP_WITH_TIMEZONE'
CONTAINER_DIGEST='registry/image@sha256:64_HEX_DIGEST'
VLLM_WHEEL_URL='HTTPS_URL_TO_EXACT_WHEEL'
VLLM_WHEEL_SHA256='64_LOWERCASE_HEX_DIGEST'
COST_LEDGER='data/manifests/cost_ledger.yaml'
GPU_RESERVATION_RECEIPT=".runpod/reservations/${GPU_PHASE}.json"
GPU_BUDGET_SESSION_ID="$(openssl rand -hex 32)"
export GPU_BUDGET_SESSION_ID
```

The approved maximum must exactly equal `8 × PER_GPU_RATE × PLANNED_HOURS`, rounded upward to six
decimal places. This command atomically accounts for all prior incurred and unresolved costs,
reserves the entire phase maximum, derives the remaining safe runtime at the quoted 8-GPU rate,
and refuses a duplicate or overlapping GPU reservation:

```bash
PYTHONPATH="$PWD/src" python3 scripts/gpu_budget_reserve.py \
  --cost-ledger "$COST_LEDGER" \
  --phase "$GPU_PHASE" \
  --approved-phase-runtime-hours "$PLANNED_HOURS" \
  --approved-phase-maximum-usd "$PHASE_MAXIMUM_USD" \
  --gpu-count 8 \
  --quote-hourly-per-gpu-usd "$PER_GPU_RATE" \
  --gpu-hard-stop-usd 220 \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325 \
  --receipt "$GPU_RESERVATION_RECEIPT"
```

The receipt is created atomically under ignored `.runpod/`; an existing path is never overwritten.
Sync that receipt and the exact canonical ledger to the Pod without changing either file. Reserve
only immediately before a committed launch. If the Pod is never started, do not fabricate a
`stopped_confirmed` record or settlement; the outstanding reservation remains fail-closed until a
reviewed no-launch recovery is performed.

## Bootstrap or re-arm the stopped Pod

After the approved Pod starts, copy or clone only this repository, restore the reservation receipt
and ledger at their original paths, and inject the same `GPU_BUDGET_SESSION_ID`. Run the 11-argument
fail-closed bootstrap command:

```bash
bash scripts/bootstrap_gpu.sh \
  "$GPU_FAMILY" \
  "$PER_GPU_RATE" \
  "$PLANNED_HOURS" \
  "$PRICE_SOURCE" \
  "$PRICE_CHECKED_AT" \
  "$CONTAINER_DIGEST" \
  "$VLLM_WHEEL_URL" \
  "$VLLM_WHEEL_SHA256" \
  "$GPU_PHASE" \
  "$GPU_RESERVATION_RECEIPT" \
  "$COST_LEDGER"
```

Before a wheel, model, or experiment backend starts, bootstrap authenticates the receipt against
the canonical ledger and nonce, claims `.runpod/sessions/<session-hash>/`, and starts a separate
`nohup` watchdog. It passes the receipt's `prior_committed_gpu_usd` and
`maximum_safe_runtime_hours` to that watchdog and to hardware preflight. It then verifies eight
full GPUs, MIG status, disk, live prices, price freshness, image digest, and wheel hash. Pod ID, GPU
UUIDs, watchdog state, live billing state, and preflight stay only below ignored `.runpod/`; they
must not be copied to `data/manifests/`.

On the first phase, bootstrap installs `.venv-gpu` and creates an authenticated setup lock under
`.runpod/setup/`. On later phases using the same stopped Pod/volume, it does not reinstall: it
requires the exact container digest, 64-character wheel SHA-256, pinned 40-character Transformers
and Jacobian Lens Git commits, environment-manifest hash, and live `pip freeze` to match. A stale
active session, an incomplete prior stop, or an unsettled prior reservation blocks re-arm.

The watchdog derives current incurred cost and two absolute deadlines from the provider's live
`lastStartedAt` and effective `adjustedCostPerHr`: the 97% safe-budget deadline and approved maximum
runtime deadline. It uses the earlier one, never extends the deadline after a reported rate drop,
and stops early if live metadata becomes unsafe. The private persisted state intentionally omits
the API response's `env` object and all credentials.

## Authenticate every paid GPU command

Bootstrap alone is not permission to construct a model backend. Immediately before every paid GPU
CLI phase (and before a resume that would reload a backend), authenticate the live private session.
Derive the directory from the non-secret receipt hash, not from the raw nonce:

```bash
SESSION_HASH_HEX="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_hash"].removeprefix("sha256:"))' "$GPU_RESERVATION_RECEIPT")"
SESSION_DIR="$PWD/.runpod/sessions/$SESSION_HASH_HEX"

PYTHONPATH="$PWD/src" python3 scripts/runpod_active_session_verify.py \
  --session-directory "$SESSION_DIR" \
  --reservation-receipt "$GPU_RESERVATION_RECEIPT" \
  --cost-ledger "$COST_LEDGER" \
  --phase "$GPU_PHASE" \
  --session-id-env GPU_BUDGET_SESSION_ID \
  --gpu-hard-stop-usd 220 \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325
```

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

The production order is:

1. `sample`: BF16 behavioral rollouts plus blind external final/trajectory measurement;
2. `anchors`: outcome-blind prefilter, two distinct classifier routes, exact token spans, and
   frozen 24-anchor manifest;
3. `resample`: freeze both full allocation manifests, generate 10 paired samples per arm in
   stage one and an unconditional 10 more in stage two, and blind-judge every final outcome;
4. `positions`: blind exact first-estimate-span adjudication and five frozen token positions;
5. `lens`: 4B compatibility gate, then no more than two limited 122B compatibility attempts,
   followed by matched J/R records from the same activation capture;
6. `analyze`, then sync and validate artifacts locally.

Use the Make variables shown in `README.md`; never put credentials in those variables. When its
canonical JSONL is absent, `make resample` runs the frozen raw-token-prefix GPU intervention and
external measurement workflow, checkpointing generations and non-secret API usage after each
successful operation. If the canonical JSONL already exists, the CLI validates it and refuses
to overwrite it. An interrupted checkpoint directory must be audited and recovered explicitly;
the command fails rather than risk duplicating paid work.

The 27B lens is not an automatic fallback. It may be run only after both predeclared 122B
compatibility attempts fail, and must be labelled methodology support rather than evidence about
the 122B model's internal state.

## Checkpoint, stop, settle, and start the next phase

- Keep paid API outputs and cost-ledger writes checkpointed after every successful call.
- Keep raw and interim artifacts under ignored directories. Never copy the unlicensed upstream
  checkout into a release bundle.
- Sync needed outputs to the local workspace and verify SHA-256 hashes before declaring the sync
  complete.
- After verified sync, request an immediate non-destructive stop with
  `touch "$SESSION_DIR/runpod_stop.request"`; do not wait for the watchdog deadline.
- Wait until `$SESSION_DIR/runpod_watchdog.json` reports `stopped_confirmed`. The watchdog calls
  only `POST /v1/pods/{podId}/stop` and confirms `desiredStatus=EXITED`; it never calls DELETE.
- Confirm in the RunPod UI that GPU billing stopped. A stopped Pod may still incur volume/storage
  charges, so remove no-longer-needed storage through the UI only after the local checksum audit.
- Read the authoritative provider-incurred GPU charge, then reconcile the exact reservation. The
  settlement command is idempotent only for the identical charge and authenticated receipt:

```bash
PROVIDER_INCURRED_USD='AUTHORITATIVE_PROVIDER_CHARGE_FOR_THIS_SESSION'
PYTHONPATH="$PWD/src" python3 scripts/gpu_budget_settle.py \
  --reservation-receipt "$GPU_RESERVATION_RECEIPT" \
  --cost-ledger "$COST_LEDGER" \
  --watchdog-state "$SESSION_DIR/runpod_watchdog.json" \
  --session-id-env GPU_BUDGET_SESSION_ID \
  --provider-incurred-usd "$PROVIDER_INCURRED_USD" \
  --gpu-hard-stop-usd 220 \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325 \
  --output "$SESSION_DIR/settlement.json"
```

An active session may resume only by reusing its existing receipt and the same nonce through the
active-session verifier; never call reserve or bootstrap again for that session. After
`stopped_confirmed` and `settled`, the session can never authorize more GPU work. The next phase
must use a new nonce and receipt, and re-arm validates that every prior private session is both
stopped and settled. Preserve the private `.runpod/sessions/` records locally for audit but publish
only hashes and aggregate costs, never Pod IDs, GPU UUIDs, live billing state, or the raw nonce.

The watchdog is a last-resort stop request, not proof of provider billing. Reconcile the final
provider invoice with the local cost ledger and report any difference without retroactively
changing the preregistered analysis.

Official API references: [find a Pod by ID](https://docs.runpod.io/api-reference/pods/GET/pods/podId),
[stop a Pod](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop), and
[Pod billing history](https://docs.runpod.io/api-reference/billing/GET/billing/pods).
