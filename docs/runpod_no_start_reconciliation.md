# RunPod no-start reconciliation

Use this path only after a re-arm invocation has ended before provider start
was confirmed. It has no provider mutation method: both observations use the
official RunPod REST-v1 GET client.

Eligible lifecycle operations are `rearm_intent`, `rearm_patched`, and
`rearm_start_intent`. The lifecycle persists the provider's raw
`lastStartedAt` value before PATCH. Reconciliation requires all of the
following:

- the same Pod, name, image, machine, GPU type/count, Secure Cloud placement,
  data center, storage, mount, ports, hourly-price ceiling, and frozen
  environment;
- exact `desiredStatus=EXITED` and byte-for-byte unchanged `lastStartedAt`;
- an empty, bounded REST-v1 billing query;
- for the uncertain `rearm_start_intent` boundary, a second identical Pod
  observation after a 30–300 second quiet window. This handles a crash just
  after `POST /start`: any RUNNING/STARTING state, timestamp advance, identity
  change, billing row, or uncertain GET fails closed.

`rearm_intent` can legitimately expose either the prior or current session
environment. The durable intent is written before PATCH, and the next durable
state is written only after PATCH verification; therefore either side of that
single-field PATCH is possible, while provider start is not yet reachable.
Later eligible operations require the current environment exactly.

The command writes `no_start_receipt.json`, fsyncs its session directory, and
then compare-and-swaps the authenticated lifecycle to `stopped`. If the process
crashes between those steps, rerunning the command replays only the already
bound local transition and performs no new provider query.

```bash
PYTHONPATH=src .venv/bin/python scripts/runpod_reconcile_no_start.py \
  --project-root . \
  --quiet-window-seconds 60
```

Only after the receipt exists **and** the canonical lifecycle has completed its
compare-and-swap to exact `stopped` / `EXITED`, reconcile the active estimate to
incurred zero. A crash leaving the receipt beside `rearm_patched` or another
non-stopped lifecycle fails settlement and keeps the estimate active. Replace
the receipt paths and hard-stop values with the already approved private bundle
values.

```bash
PYTHONPATH=src .venv/bin/python scripts/gpu_budget_settle.py \
  --reservation-receipt .runpod/reservations/PHASE.json \
  --cost-ledger data/manifests/cost_ledger.yaml \
  --no-start-receipt .runpod/sessions/SESSION_DIGEST/no_start_receipt.json \
  --lifecycle-state .runpod/pod_lifecycle.json \
  --gpu-hard-stop-usd 220 \
  --api-hard-stop-usd 100 \
  --total-hard-stop-usd 325 \
  --output .runpod/sessions/SESSION_DIGEST/settlement.json
```

The settlement records `billing_status=not_started`,
`evidence_kind=provider_no_start`, and both provider/accounted GPU amounts as
exactly zero. The receipt and settlement together satisfy completed-session
validation; a new reservation still requires a new session identity.
