# Bounded RunPod execution

No paid Pod is authorized by this document. Launch only after the user confirms the exact
machine, current price, planned maximum runtime, and estimated charge.

This frozen study uses only the already authenticated, stopped research Pod. The creation shape
below remains documentation of the capability-limited implementation; it is not authorization to
create another Pod. Every study phase, including baseline, starts through the guarded re-arm flow.

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
[`README.md`](README.md#freeze-and-review-the-fresh-private-paid-bundle). `preview` either exclusively
creates both content-addressed quote locks from independently reviewed private specifications or
authenticates locks that already exist. It prints a non-authorizing cost/binding summary. Only
after the user explicitly approves that exact summary may `approve` create schema-v4
`.runpod/paid_run_approval.json` with an explicit identifier, timestamp, and phase list. Neither
action contacts a provider or starts paid work. The approval binds the canonical hash of the full
software/GPU lock in addition to the exact quote, route, runtime, and budget fields.

## Exact launch shape

The approved primary launch is an 8-GPU Secure Pod with this exact immutable image:
`runpod/pytorch@sha256:e855789ff7e4b1ad76698171b1974a99a5c48c5b3e80a908976987938b090992`
(observed tag `1.0.7-cu1300-torch291-ubuntu2404`, OCI config
`sha256:18d90716b6bb9f4bed820c08e5b80cf6e0a99eb14e611b3446db82729b2d0b18`). The image includes
`cuda-compat-13-0` and `/start.sh`; do not override its entrypoint or start command.

The disabled v2 create implementation binds and hashes this secret-redacted request shape; values
marked “from quote” come only from the authenticated GPU quote lock. This is audit documentation,
not a launch command:

```json
{
  "name": "model-forensics-<approved-phase>",
  "image": "runpod/pytorch@sha256:e855789ff7e4b1ad76698171b1974a99a5c48c5b3e80a908976987938b090992",
  "disk": 50,
  "ports": ["22/tcp"],
  "env": {
    "HF_TOKEN": "<secret>",
    "GPU_BUDGET_SESSION_ID": "<secret>",
    "HF_HOME": "/workspace/.cache/huggingface",
    "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
    "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
    "VLLM_ENABLE_CUDA_COMPATIBILITY": "1"
  },
  "cloud": "SECURE",
  "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 8},
  "dataCenterIds": ["FROM_FROZEN_QUOTE"],
  "globalNetworking": false,
  "mounts": {"persistent": {"size": 650, "path": "/workspace"}},
  "startJupyter": false,
  "startSsh": true
}
```

Do not attach a network volume: the watchdog uses the non-destructive stop operation, while a
network-volume Pod may require termination. The launch environment contains only the allow-listed
Hugging Face token, session nonce, and fixed cache/runtime values. RunPod must separately expose
its provider-managed Pod identity and Pod-scoped API key at runtime; never serialize those values
into the launch environment, lifecycle state, or report.

Creation uses the approval-bound v2 request described above. The independent watchdog observes
and stops the Pod only through RunPod's official REST v1 Pod API: it reads
`GET /v1/pods/{podId}?includeMachine=true&includeNetworkVolume=true&includeTemplate=true` and can
send only the non-destructive `POST /v1/pods/{podId}/stop` request with no body. A live v1 response attests the
Pod identity, desired status, image, GPU/count, Secure cloud, machine/data center, disks/mount,
ports, allow-listed environment, direct-SSH readiness, start time, and current `costPerHr`.
Secret values, machine IDs, public addresses, and SSH routes are validated in memory and never
persisted; only one-way identity/endpoint hashes survive in the private watchdog state.

REST v1 does **not** expose the placement CUDA version, runtime GPU inventory, global-networking
flag, or Pod lock flag. The watchdog records those four fields as null and lists them under
`provider_evidence_unavailable`; it never turns absence into a false provider attestation.
Bootstrap independently verifies eight unique full GPUs with `nvidia-smi`, records driver
versions, and verifies the pinned CUDA 13 forward-compatibility libraries. Global networking and
lock state remain explicit live-provider evidence gaps, bounded by the content-addressed launch
approval/lifecycle record, exact SSH-only port and environment checks, and the watchdog's
fail-closed stop retries. They are not described as provider-verified facts.

## Hard budget and hardware gate

- Authorized hardware: the existing homogeneous 8× H100 80GB research Pod. A100 is not an
  approved fallback for this frozen execution.
- Exactly eight unique full-GPU UUIDs are required; MIG and mixed GPU families are rejected.
- The quote and approval also freeze the exact provider GPU ID, `SECURE` cloud, allowed data
  centers, CUDA `12.8`, 50 GB container disk, 650 GB volume, and running-storage hourly rate.
- At least 79 GiB visible memory per GPU and 520 GiB free disk are required. A 650 GiB volume is
  recommended.
- GPU hard stop: USD 220.
- External judge/classifier hard stop: USD 100.
- RunPod storage reserve: USD 5.
- Total hard stop: USD 325.

The frozen quality-first API design sends every valid final to both Claude Opus 5 and Gemini 3.1
Pro Preview; it does not substitute entry-tier or economy models. This does not relax the USD 100
API hard stop or authorize a call before the exact paid bundle is explicitly approved.

The approval input uses the provider-displayed **per-GPU hourly price**. The watchdog independently
reads the live Pod-level v1 `costPerHr` from RunPod and refuses a live compute rate above the frozen
compute-only ceiling. It then adds the separately frozen running-storage rate when deriving cost
and deadline ceilings. Under the currently reviewed USD 0.10/GB-month schedule, the frozen 50 GB
container plus 650 GB volume is encoded as `700 × 0.10 ÷ 720 ≈ 0.0972222222` USD/hour; the
authenticated quote must record that derivation's source and timestamp.

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

## Guarded re-arm and bootstrap of the stopped Pod

For every paid study phase, including the baseline phase on the existing stopped research Pod,
keep `make gpu-host-watch-rearm GPU_PHASE="$GPU_PHASE"` running in a dedicated host terminal.
After its private acknowledgement appears, run `make gpu-rearm GPU_PHASE="$GPU_PHASE"` in a second
terminal. The re-arm path verifies the host watcher acknowledgement and live PID immediately
before provider start; direct invocation without that guard fails closed. Full commands and
recovery rules are in
[`docs/runpod_lifecycle.md`](docs/runpod_lifecycle.md#re-arm-the-same-stopped-pod).

After start, run the one-shot `make gpu-sync` command documented below. It clones the exact
approved and pushed source commit into a new clean remote stage, transfers only the authenticated
lifecycle/reservation/ledger/approval controls and validated completed-session evidence, verifies
the stage, then recoverably promotes it at exactly
`/workspace/model-forensics-value-leakage`. The old checkout is preserved under
`/workspace/.model-forensics-sync-archive/`; the current host-watch session is never copied.
The transfer requires a live phase/session/Pod-bound acknowledgement, a watcher heartbeat
no older than 20 seconds, no pending stop request, and exact incurred-ledger coverage. A failure
after the one-shot claim durably requests an immediate host-watcher stop and independently calls
the exact-Pod provider stop path, which must confirm `EXITED` or `TERMINATED` before returning.
Keep the local `RUNPOD_API_KEY` and `HF_TOKEN` exported through `gpu-sync`; they never enter argv or
the remote bundle and are used only to preflight that independent stop capability before first SSH.

Only a `rearmed` running lifecycle is eligible: the first `created` path has no equivalent
independently armed host-guard producer and must be stopped and settled before a fresh guarded
re-arm. Heartbeat timestamps and derived counters may then advance normally, while the
acknowledgement, watcher process-start identity, execution/limit/deadline invariants, and
provider-derived direct SSH endpoint hash remain exact. Never fabricate a first-create
acknowledgement to bypass this gate.

The transfer accepts only the exact `root@<canonical IPv4>` direct-SSH target and numeric public
port returned for this Pod. Before materialization, claiming, or any SSH/rsync command, it hashes
that IP/port with the same domain-separated encoding as the live watchdog and compares it in
constant time with the authenticated sync-plan guard. Hostnames, aliases, alternate users, or an
endpoint belonging to another Pod fail without issuing a remote command or requesting a stop.

After `gpu-sync` succeeds, open the already host-key-pinned direct SSH session and run the
lock-derived fail-closed bootstrap from the exact installed destination. Do not forward local
credentials or override the Pod's provider-managed environment. The sync manifest expires
five minutes after creation; do not leave a manual gap between sync and bootstrap. If the initial
manifest/source verifier fails or expires, a pre-armed standard-library failure handler
prefers the exact lifecycle/reservation/session binding. Missing or corrupt local controls switch
to an independent read-only provider gate: the ambient Pod id, in-memory session nonce, exact
research name, pinned image, 8x H100 Secure Cloud hardware, approved location, local storage,
SSH-only endpoint, allow-listed environment, and no network volume must all agree before any stop
POST. A mismatch sends no POST; a successful stop must be confirmed as exactly `EXITED` before the
failed bootstrap returns:

```bash
ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -p "$RUNPOD_SSH_PORT" "$RUNPOD_SSH_HOST"

# Run the remaining commands inside the Pod.
cd /workspace/model-forensics-value-leakage
GPU_PHASE='behavior_baseline_gpu'  # advance this value for later frozen phases
test -n "${RUNPOD_POD_ID:-}" && test -n "${RUNPOD_API_KEY:-}" \
  && test -n "${GPU_BUDGET_SESSION_ID:-}" && test -n "${HF_TOKEN:-}"
make gpu-bootstrap GPU_PHASE="$GPU_PHASE"
```

The Make target reads the exact quote and `config/gpu_lock.yaml`, then supplies the current
22-argument interface to `scripts/bootstrap_gpu.sh`. The additional lock-derived fields bind
the sentence-transformers wheel URL/hash, distribution version, complete semantic-stack hash,
and the exact top-level bootstrap constraints/hash.
It does not accept an image, wheel, price, hardware identity, cloud type, or runtime through a Make
variable.

Before a wheel, model, or experiment backend starts, bootstrap authenticates the receipt against
the canonical ledger and nonce, claims `.runpod/sessions/<session-hash>/`, and starts a separate
`nohup` watchdog. It passes the receipt's `prior_committed_gpu_usd` and
`maximum_safe_runtime_hours` to that watchdog and to hardware preflight. It then verifies eight
full GPUs, MIG status, disk, live prices, price freshness, image digest, and wheel hash. Pod ID, GPU
UUIDs, watchdog state, live billing state, and preflight stay only below ignored `.runpod/`; they
must not be copied to `data/manifests/`.

Bootstrap copies credentials only into non-exported shell variables and immediately removes their
public environment names. The exact regular/emergency watchdog receives `RUNPOD_API_KEY` plus
`HF_TOKEN` because its live metadata gate binds the Pod's HF credential hash; the exact Qwen3.5-4B
smoke command receives only `HF_TOKEN`. Curl, pip, build, environment-capture, setup-lock, and
active-session verification children receive none of those credentials or private aliases.

On the first phase, bootstrap installs `.venv-gpu` and authenticates the exact semantic inference
stack: version-pinned distributions, every SHA-256-bearing installed file named by `RECORD`, the
sentence-transformers wheel's PEP-610 archive SHA-256, and the pinned Transformers Git source.
Bootstrap uses the container/venv pip without self-upgrade, installs exact top-level versions,
and then installs this project editable with `--no-deps`. Transitive PyPI wheels are not fully
hash-locked; the pinned container plus PyPI TLS is therefore an explicit first-install trust
boundary, while observed top-level versions and semantic installed files fail closed on drift.
Primary embedding eligibility later also requires the frozen model revision plus tokenizer/model
runtime evidence in every semantic resampling record.

Bootstrap then runs one real pinned Qwen3.5-4B rollout and exact raw-token-prefix retain/resample
continuations before any 122B backend can be constructed. The bounded non-primary gate also checks
a deterministic parser/trajectory fixture, exact anchor span-to-token mapping, and the complete
5-position × 3-concept probe-design grid. There are no matched Qwen3.5-4B J/R lens artifacts or
same-forward vLLM activation interface, so the manifest must record that transport boundary, zero
fabricated lens rows, and a forbidden analysis handoff. Its passed manifest stays at
`.runpod/setup/qwen4b_prefix_smoke.json`; the setup lock binds both the manifest's canonical hash
and file SHA-256. On later phases using the same stopped Pod/volume, bootstrap does not reinstall
or rerun the smoke: it requires the exact container digest, both 64-character wheel SHA-256 values,
pinned 40-character Transformers and Jacobian Lens Git commits, environment/semantic-runtime
hashes, smoke hashes, and live `pip freeze` to match. A stale active session, an incomplete prior
stop, or an unsettled prior reservation blocks re-arm.

The watchdog derives current incurred cost and two absolute deadlines from the provider's live v1
`lastStartedAt` and `costPerHr`: the 97% safe-budget deadline and approved maximum
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
- After verified output sync, request the immediate non-destructive stop from the independently
  running **host** watcher. This target derives the exact private session from the authenticated
  reservation, requires a heartbeat no older than 20 seconds and the original live watcher
  process identity, then atomically creates only the canonical local request:

  ```bash
  make gpu-stop-request GPU_PHASE="$GPU_PHASE"
  ```

- Keep Terminal A open until its local
  `.runpod/sessions/<session-hash>/host_rearm_watchdog.json` reports
  `stopped_confirmed` with `stop_reason=external_stop_request`. The host watcher calls only
  `POST /v1/pods/{podId}/stop` with no request body and confirms `desiredStatus=EXITED`; it never
  calls DELETE. Do not depend on SSH after provider stop and do not retrieve or fabricate a
  remote `runpod_watchdog.json`.
- Confirm in the RunPod UI that GPU billing stopped. A stopped Pod may still incur volume/storage
  charges, so remove no-longer-needed storage through the UI only after the local checksum audit.
- Convert the provider-confirmed host stop into a read-only external-stop receipt and a CAS-bound
  local lifecycle transition to `stopped` / `EXITED`, then settle schema v2 from that receipt:

```bash
make gpu-recover-stop GPU_PHASE="$GPU_PHASE"
make gpu-settle-external GPU_PHASE="$GPU_PHASE"
```

`gpu-recover-stop` first authenticates the exact canonical reservation, local request, and host
watcher stop record. Its provider client has GET capability only: it verifies the lifecycle-bound
Pod is exactly `EXITED`, reads the exact billing row, writes
`external_stop_receipt.json`, and content-addresses the host control evidence before atomically
advancing the local lifecycle. If the billing row has not appeared, it fails closed; rerun after
the provider posts the row. `gpu-settle-external` derives its amount only from that authenticated
receipt and requires the receipt's stopped-lifecycle hash to equal the current canonical
lifecycle. The target invokes `scripts/gpu_budget_settle.py`; its legacy `--watchdog-state` plus
caller-supplied amount path is disabled for new settlements.

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
active-session verifier, while the original approval and quote remain fresh; never call reserve or
bootstrap again for that session. After `stopped_confirmed` and `settled`, the session can never
authorize more GPU work. A fresh-approval same-phase re-arm is allowed only when the failed attempt
never created that phase's immutable paid-plan receipt. Once that receipt exists, the phase is
terminal in this release and re-arm rejects before provider mutation. Otherwise the next phase
must use a new nonce and receipt, and re-arm validates that every prior private session is both
stopped and settled. Preserve the private `.runpod/sessions/` records locally for audit but publish
only hashes and aggregate costs, never Pod IDs, GPU UUIDs, live billing state, or the raw nonce.

The watchdog is a last-resort stop request, not proof of provider billing. Reconcile the final
provider invoice with the local cost ledger and report any difference without retroactively
changing the preregistered analysis.

Official API references: [RunPod v2 OpenAPI](https://api.runpod.io/v2/openapi.yaml),
[Pod API overview](https://docs.runpod.io/api-reference/pods/overview), and
[Pod billing history](https://docs.runpod.io/api-reference/billing/GET/billing/pods).
