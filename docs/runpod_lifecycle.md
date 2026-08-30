# Production RunPod lifecycle gate

`scripts/runpod_pod_lifecycle.py` is the only supported re-arm path for the paid Pod. Its frozen
CLI rejects `create` and `recover-create` before reading credentials or constructing a provider
client. It is intentionally not a general RunPod client: there is no delete, terminate, stop,
restart, or arbitrary update operation. Stopping remains the independently armed watchdog's
responsibility; that process uses only official REST v1 GET and non-destructive `/stop` operations.

The helper authenticates, before constructing an HTTP client:

- the content-addressed GPU quote and API quote;
- the exact schema-v4 paid-run approval and command phase;
- the full config, preregistration, and GPU/software lock bindings;
- the phase's active, pre-created reservation and opaque nonce against the canonical cost ledger;
- the all-in rate and maximum: `8 × GPU rate + running-storage rate`;
- the exact H100 SXM provider ID, Secure cloud, CUDA 12.8, the privately quote-bound data-center
  allowlist, 50 GB container disk, and 650 GB host-local persistent `/workspace` mount.

Creation is `POST https://api.runpod.io/v2/pods` using the v2-native shape. Provider list/live
verification remains read-only REST v1 `GET https://rest.runpod.io/v1/pods...`. Re-arm uses REST
v1 `PATCH /pods/{id}` to replace the allow-listed environment and REST v1 `POST /pods/{id}/start`
with no request body.

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

## Fresh creation boundary

The frozen take-home execution does **not** authorize creation of another Pod. It resumes the
already authenticated, stopped research Pod through the guarded re-arm procedure below. Although
the capability-limited lifecycle module retains a tested `create` implementation, a first
`created` lifecycle has no independently armed host-guard producer and is therefore rejected by
selective sync and bootstrap. Do not invoke `create`, fabricate a host acknowledgement, or consume
one of the four study-phase reservations for provisioning.

A future from-scratch run would require a separately preregistered provisioning phase, a quoted
budget allocation, a first-create host-guard producer, tests, and a new explicit approval before
the provider POST. Those are outside this frozen execution.

## Bootstrap guarantees

The first bootstrap authenticates and installs the exact sentence-transformers wheel plus its
inference-critical version set. The environment record hashes each installed distribution's
`METADATA` and `RECORD`, the wheel's PEP-610 archive SHA-256, and the pinned Transformers source
commit. Primary semantic eligibility additionally requires the exact embedding-model revision and
tokenizer/model runtime provenance; re-arm recomputes this identity and fails closed on drift.

Before any 122B model load, bootstrap also runs the bounded real Qwen3.5-4B integration gate. It
uses one rollout and two short raw-prefix continuations, exercises deterministic parsing,
trajectory, anchor span-to-token mapping, and the complete 5-position × 3-concept probe-design
plumbing, then writes a non-primary manifest. The 4B generation runtime has no matched J/R lens
weights or same-forward activation contract, so the manifest must state that exact boundary,
contain zero fabricated lens rows, and forbid analysis ingestion. Failure of this gate prevents the
reusable setup lock from being created.

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
The same approval content hash cannot authorize a second fresh reservation for the same phase.
An active attempt may resume only through its existing session path while its original approval and
quote remain fresh. A fresh-approval re-arm retry is supported only if that phase failed before its
immutable paid-plan receipt was created. Once the receipt exists and the session is stopped, this
release treats that phase as terminal and rejects re-arm before any provider mutation; it does not
silently replace provenance under a newly reviewed approval.

```bash
GPU_PHASE='behavior_baseline_gpu'  # advance this value for later frozen phases
GPU_BUDGET_SESSION_ID="$(openssl rand -hex 32)"
export GPU_BUDGET_SESSION_ID
make gpu-reserve GPU_PHASE="$GPU_PHASE"

# Terminal A: keep this independent host stop-capable watcher in the foreground.
make gpu-host-watch-rearm GPU_PHASE="$GPU_PHASE"

# Terminal B: only after the private host_rearm_watchdog_ack.json exists.
make gpu-rearm GPU_PHASE="$GPU_PHASE"
```

`gpu-host-watch-rearm` first binds the authenticated stopped lifecycle, then performs a
read-only provider check for `desiredStatus=EXITED`. Only then does it atomically create the
fresh-session acknowledgement. `gpu-rearm` authenticates that acknowledgement, the watcher's
boot/process-start identity (not merely its reusable PID), its current heartbeat, current
phase/session, stopped-lifecycle hash, and Pod-ID hash both before PATCH and again
immediately before `POST /start`. A missing, stale, tampered, wrong-session, or dead-watcher
acknowledgement blocks start. The watcher bounds the complete start/readiness observation to five
minutes; provider GET failures enter the stop-and-confirm loop instead of sleeping forever.

Before remote bootstrap, run the guarded one-shot transfer from the host. The approved source
commit must already be pushed to the canonical public repository, and strict host-key checking
must already know the RunPod SSH host:

```bash
export RUNPOD_SSH_HOST='root@RUNPOD_DIRECT_SSH_CANONICAL_IPV4'
export RUNPOD_SSH_PORT='RUNPOD_DIRECT_SSH_PORT'
make gpu-sync GPU_PHASE="$GPU_PHASE" \
  RUNPOD_SSH_HOST="$RUNPOD_SSH_HOST" RUNPOD_SSH_PORT="$RUNPOD_SSH_PORT"
```

Keep the host shell's `RUNPOD_API_KEY` and `HF_TOKEN` exported for this command. They are never
placed in argv or copied remotely. Before the durable one-shot claim and first SSH, the transfer
authenticates an independent exact-Pod stop client; after the claim, every remote command first
revalidates the live host guard, and any failure triggers both the durable host-watcher stop request
and the bounded provider stop-and-confirm path.

Use the literal canonical IPv4 and mapped numeric port from this Pod's direct-SSH endpoint. Do
not substitute a DNS name, SSH alias, another user, or another Pod's endpoint. The transfer
recomputes the watchdog's domain-separated IP/port hash and constant-time compares it with the
authenticated `current_host_guard.direct_ssh_endpoint_hash` before bundle materialization, claim,
stop signaling, or any SSH/rsync command.

The helper requires a clean local checkout whose `origin` is exactly
`https://github.com/baeyongil/model-forensics-value-leakage.git` and whose HEAD equals the
approval/manifest source commit. It clones that exact commit into a new clean directory under
`/workspace`, authenticates the source before claiming the transfer, then copies only the
manifest inventory. Both the staged and installed checkout are verified with standard-library
code before bootstrap or any download. Promotion archives the entire old destination outside the
project and atomically installs the verified stage; failure rolls the old destination back.

The one-shot claim is created before the first SSH or private-state transfer. Any failure after that
claim atomically creates the current host session's `runpod_stop.request` and independently stops
and confirms the exact Pod through the provider API. Never manually rsync `.runpod`,
`.runpod/sessions`, or the
repository: broad copying can duplicate the current host claim or mix stale persistent-workspace
state. The bundle contains no `.env.local`; inject the nonce and credentials through the remote
environment. `scripts/bootstrap_gpu.sh` re-verifies the exact installed manifest and source as
its first action, before creating `.runpod` artifacts, provider GETs, downloads, or imports. The
manifest has a strict five-minute lifetime, so start bootstrap immediately after `gpu-sync` with
no manual pause. A verifier failure or expiry runs the already-armed standard-library emergency
handler. It first prefers the authenticated lifecycle/reservation binding. If either synced file
is missing or corrupt, the fallback trusts neither one: a read-only REST-v1 GET must bind the
provider-managed ambient `RUNPOD_POD_ID` and in-memory session nonce to the exact research Pod
name, pinned image, 8x H100 Secure Cloud placement, approved data center, local storage, SSH-only
endpoint, allow-listed environment, and absence of a network volume. Any mismatch issues zero
POSTs. Only after every independent field agrees may it request the non-destructive stop, and the
failed bootstrap does not return until a subsequent provider GET confirms exactly `EXITED`.

Immediately open the already host-key-pinned direct SSH session; do not forward local credentials
or override the Pod's provider-managed environment:

```bash
# Host: use the same authenticated endpoint values accepted by gpu-sync.
ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -p "$RUNPOD_SSH_PORT" "$RUNPOD_SSH_HOST"

# Pod: the re-arm environment already carries the exact nonce, HF token,
# provider-managed Pod id, and Pod-scoped RunPod key.
cd /workspace/model-forensics-value-leakage
GPU_PHASE='behavior_baseline_gpu'  # advance this value for later frozen phases
test -n "${RUNPOD_POD_ID:-}" && test -n "${RUNPOD_API_KEY:-}" \
  && test -n "${GPU_BUDGET_SESSION_ID:-}" && test -n "${HF_TOKEN:-}"
make gpu-bootstrap GPU_PHASE="$GPU_PHASE"
```

Keep the original host watcher terminal running for the entire phase. If any lifecycle operation
ends in an intent/pending/failed state, use `status` and audit the existing Pod; never delete it
through this helper and never attempt a second create.

## Stop, attest, and settle from the host

After the phase outputs have been checksummed and synced, keep the original host watcher running
and execute:

```bash
make gpu-stop-request GPU_PHASE="$GPU_PHASE"
# Wait for local host_rearm_watchdog.json: stopped_confirmed.
make gpu-recover-stop GPU_PHASE="$GPU_PHASE"
make gpu-settle-external GPU_PHASE="$GPU_PHASE"
```

`gpu-stop-request` accepts only the canonical reservation-derived host session. It requires the
watcher heartbeat to be at most 20 seconds old, authenticates the acknowledgement's live
boot/process-start identity, and verifies the exact cumulative and split compute/storage rate
limits before creating the zero-byte request without following or overwriting a path.

The host watcher owns the stop request and provider confirmation. Do not touch a remote stop file,
depend on post-stop SSH, or copy a remote watchdog record back to the host. Once the local watcher
records `stopped_confirmed` for `external_stop_request`, `gpu-recover-stop` hashes that local state
and request into a GET-only provider attestation, then CAS-transitions the lifecycle to
`stopped` / `EXITED`. `gpu-settle-external` accepts only the canonical receipt, canonical stopped
lifecycle, reservation and ledger and creates a schema-v2 settlement. New settlement invocations
cannot use a remote watchdog record or a caller-supplied incurred amount.

If the billing row is not yet available, recovery fails without changing the lifecycle or ledger;
retry it after the provider posts the row. The no-start reconciliation remains a separate path and
must not be used for a session whose provider `lastStartedAt` advanced.
