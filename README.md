# Model Forensics: Value Leakage

This repository is an independent, preregistered investigation of value leakage in
`Qwen/Qwen3.5-122B-A10B`. It asks one narrow question: when the model says a donation threshold
must not affect its estimate, does that statement causally constrain the answer, or does it
narrate a value-directed search already in progress?

The repository currently contains the frozen design, implementation, validation tests, and
synthetic smoke artifacts. It does **not** contain a completed primary run or empirical
conclusion. Files under `data/**/smoke` and `reports/figures/smoke` are deterministic fixtures,
not evidence about Qwen3.5-122B-A10B.

## Evidence design

The investigation combines three evidence lines:

1. behavioral decomposition of first versus final estimates, revision, threshold crossing,
   and stopping;
2. paired retain-versus-resample interventions on 24 frozen giraffe reasoning sentences; and
3. matched J-lens/R-lens readouts used only as observational corroboration.

Primary behavioral and resampling outcomes use frozen, blind external measurement. The primary
judge receives only the neutral quantity question, trace, and answer—not arm, condition,
threshold, seed, or observed outcome as metadata. The protocol invokes both blind external final-outcome adjudicators for every
behavioral and resampling final. Only exact known status-and-value consensus enters primary
inference; unresolved disagreement is missing, never
replaced by the local numeric parser.

Sentence resampling has two fixed stages for every selected anchor and arm: 10 paired
continuations in stage one, followed unconditionally by 10 more paired continuations in stage
two. No observed effect, confidence interval, divergence rate, or measurement success controls
stage-two allocation. The base trace is the inference unit; continuations are not treated as
independent traces. Intervention eligibility is determined without using whether the resampled
final could be measured, and missing paired outcomes remain visible to the missingness analysis.

Other frozen integrity rules include a 10 percentage-point practical-equivalence region,
cluster-aware inference, an independently worded Chicago coffee-cup task, task-by-condition
measurement gates, and a rule that lens evidence cannot establish causality or motivation by
itself. The complete specification is in
[`config/preregistration.yaml`](config/preregistration.yaml).

## Local setup and no-network validation

Python 3.11 or newer is required.

```bash
make setup
make test
make smoke
```

`make smoke` is deterministic, uses no network or model download, and labels every generated row
`synthetic_smoke: true`. Every smoke analysis row is also individually content-hash authenticated;
an unmarked, mixed, or altered row is rejected before statistics. The smoke run exercises the
pipeline with synthetic inputs. It is not a substitute for the required 4B GPU compatibility gate
or the 122B primary run.

On the already reserved CUDA host, `make qwen4b-gpu-integration-smoke` runs the separately bounded
real Qwen3.5-4B gate: the exact tokenizer/chat template, one rollout, exact-token retain and
resample continuations, deterministic parser/trajectory and span fixtures, and the full 5×3
structural probe grid. It is always labeled synthetic/non-primary. Because no matched 4B J/R lens
artifacts exist and vLLM does not expose the required same-forward model-runtime contract, the gate
records zero transported or fabricated lens rows and an explicit transport boundary. Normal local
tests skip the identically scoped `integration` + `gpu` test unless
`RUN_QWEN4B_GPU_INTEGRATION=1` is set.

`make sample` and `make resample` are deliberately validation-only aliases. This validation-only mode
authenticates completed behavioral and resampling artifacts, respectively. It exposes no
provider route, price, approval, or model-construction surface and fails if the canonical artifact
is absent.

## Immutable paid-run contract

No Make target, configuration example, or paid CLI accepts an ad hoc judge model or provider
price. Immediately before paid execution, the following ignored inputs must be independently
frozen and content-addressed:

- `config/gpu_lock.yaml`: pinned container, vLLM and sentence-transformers wheels and hashes,
  exact semantic inference-stack versions, software commits, model/tokenizer revisions, and lens
  revisions;
- `.runpod/gpu_quote_lock.json`: exact eight-GPU provider identity, secure-cloud/data-center
  allowlist, CUDA compatibility, disk sizes, compute and running-storage rates, source, timestamp,
  and one runtime allocation for each canonical GPU phase;
- `.runpod/api_route_quote_lock.json`: the exact four OpenRouter roles, model slugs, current token
  prices, source, and timestamp;
- `.runpod/paid_run_approval.json`: the user's explicit approval of the exact hashes, routes,
  phase list, hardware, runtimes, and USD caps.

The approval document uses schema version 2. Its bindings include a canonical SHA-256 of the
entire parsed `config/gpu_lock.yaml` mapping, not only the image and wheel fields. Any change to a
repository commit, Hub revision, lens hash, image metadata, note, or other lock entry therefore
invalidates the approval even when the container digest and wheel hash are unchanged.

The preregistered quality-first route plan dual-judges every valid final with Claude Opus 5 and
Gemini 3.1 Pro Preview; Claude Opus 5 also performs primary trajectory measurement, and the two
frontier routes serve as distinct replacement classifiers. Entry-tier or economy models are not
used for primary measurement. Those names are a frozen design statement, not a current-price
claim. The fresh API quote lock is authoritative, and every response records its reported
model/provider, usage, and raw-response hash. The API hard stop remains USD 100 and the total hard
stop remains USD 325.

Every paid command first reconstructs the expected bindings from the run config,
preregistration, software lock, and both quote locks. It then validates the exact user approval
before a provider client, tokenizer download, or model backend can be constructed. A second
immutable receipt binds that phase to one content-addressed execution plan before the first paid
operation. Each phase-relevant quote has a six-hour freshness window; an expired quote or a
mismatched, malformed, predated, or future-dated approval fails closed.

Keep `HF_TOKEN`, `OPENROUTER_API_KEY`, and `RUNPOD_API_KEY` in RunPod Secrets or an ignored
`.env.local`. Never paste values into chat, a report, a Make variable, a command-line argument,
or a tracked file. `GPU_BUDGET_SESSION_ID` is a separate, one-phase opaque nonce: supply it only
through the environment, never store it in `.env.local`, a filename, or an artifact.

No command in this README authorizes paid execution. Before the first reservation or API call,
the exact live RunPod offer and OpenRouter catalog must be checked, the software lock must match,
both quote locks and the approval must be fresh, and the user must explicitly approve the
displayed costed plan.

### Freeze and review the private paid bundle

`scripts/freeze_paid_bundle.py` is an offline artifact helper; it does not query a provider,
reserve a GPU, start a Pod, construct a model backend, or call an API. All of its outputs are
exclusive, content-addressed files below a non-symlink `.runpod/` directory. It enforces mode
`0700` on private directories and `0600` on private files and refuses to overwrite an existing
lock or approval.

Prepare independently reviewed, unhashed JSON quote specifications at
`.runpod/specs/gpu_quote_spec.json` and `.runpod/specs/api_route_quote_spec.json`. They must contain
the complete strict quote-lock schemas but omit `content_hash`. Then freeze both locks and inspect
the non-authorizing, secret-free cost and binding preview:

For the currently reviewed RunPod storage schedule, the GPU specification records the combined
running storage rate as `(50 GB + 650 GB) × USD 0.10/GB-month ÷ 720 hours`, or approximately
`0.0972222222` USD/hour. The source URL and timestamp belong in the quote specification; do not
silently substitute a different billing divisor or omit storage from the projected GPU spend.

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/freeze_paid_bundle.py preview \
  --config config/run_122b.yaml \
  --preregistration config/preregistration.yaml \
  --gpu-lock config/gpu_lock.yaml \
  --gpu-quote-lock .runpod/gpu_quote_lock.json \
  --api-quote-lock .runpod/api_route_quote_lock.json \
  --gpu-quote-spec .runpod/specs/gpu_quote_spec.json \
  --api-quote-spec .runpod/specs/api_route_quote_spec.json
```

If authenticated quote locks already exist, rerun `preview` without the two `--*-quote-spec`
arguments; it reloads and validates them instead of rewriting them. `ready_for_explicit_user_approval`
is true only while both quotes are fresh. Review every displayed path, hash, route, hardware field,
phase runtime, and cap. Only after the user explicitly approves that exact preview, provide a
non-secret approval identifier, an explicit timezone-aware approval timestamp no earlier than the
applicable quotes, and each approved canonical phase:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/freeze_paid_bundle.py approve \
  --config config/run_122b.yaml \
  --preregistration config/preregistration.yaml \
  --gpu-lock config/gpu_lock.yaml \
  --gpu-quote-lock .runpod/gpu_quote_lock.json \
  --api-quote-lock .runpod/api_route_quote_lock.json \
  --output .runpod/paid_run_approval.json \
  --approval-id "$EXPLICIT_NONSECRET_APPROVAL_ID" \
  --approved-at "$EXPLICIT_APPROVAL_TIMESTAMP" \
  --allow-phase behavior_baseline_gpu \
  --allow-phase behavior_baseline_api
```

Repeat `--allow-phase` only for phases the user actually approved. The helper never infers approval
from a preview, environment variable, chat message, or existing file; `approve` requires every
approval field on its command line and emits only hashes and non-secret metadata.

## Split-phase command reference

GPU generation and external measurement are separate so a provider retry cannot silently reload
the 122B model, and a GPU resume cannot silently call an API.

| Target | Actual behavior |
|---|---|
| `make reproduce` | Fetch the pinned, unlicensed upstream repository into ignored `data/upstream/` and write compact derived hashes/statistics only. |
| `make behavior-baseline-generate` | Authenticate the `behavior_baseline_gpu` reservation and active watchdog, then generate only baseline BF16 rollouts into a durable GPU checkpoint. It creates no API client. |
| `make behavior-baseline-adjudicate` | Authenticate `behavior_baseline_api`, double-judge every baseline final, measure trajectories with the primary route, enforce gates, and freeze baseline-derived thresholds. It loads no model backend. |
| `make behavior-treatment-generate` | Authenticate `behavior_treatment_gpu`, require the frozen baseline thresholds, and generate only treatment rollouts. It creates no API client. |
| `make behavior-treatment-adjudicate` | Authenticate `behavior_treatment_api`, double-judge every treatment final, enforce cell-level gates, and publish the complete behavioral release and sampling manifest. |
| `make sample` | Validate the completed behavioral release. This is always free and validation-only. |
| `make anchors` | Validate an existing anchor manifest; otherwise use frozen classified candidates locally, or—only if candidates are absent—authenticate `anchors_api`, classify outcome-blind candidates on both frozen routes, and freeze 24 anchors. |
| `make positions` | Validate an existing complete position bundle; otherwise authenticate `positions_api`, adjudicate exact first-estimate spans, and freeze all five token positions. |
| `make resample-generate` | Authenticate `resample_gpu`, freeze both complete allocations, and generate all 960 exact-prefix retain/resample continuations. It creates no API client. |
| `make resample-adjudicate` | Authenticate `resample_api`, double-judge every generated final, classify eligible replacements on both frozen routes, enforce quality gates, and publish the canonical resampling artifact. It loads no model backend. |
| `make resample` | Validate the completed canonical resampling artifact. This is always free and validation-only. |
| `make lens` | Validate an existing lens JSONL for free; if absent, freeze the causal fixed-common probe design, authenticate `lens_gpu` and the active session, run the ordered 4B→122B compatibility protocol, and produce the full matched observational J/R grid. Two bounded 122B failures produce an authenticated behavior-only failure root. |
| `make analyze` | Run the frozen estimands, cluster inference, missingness bounds, hypothesis rules, and three core figures. |
| `make report` | Stage `result_context.json` and `result_context.md`; it does not create a DOCX or Google Doc. |
| `make reproduce-results` | Credential-free: authenticate the aggregate-only public result bundle, rebuild its sanitized JSONL tables, and regenerate the core figures. It never reads raw traces or creates a model/provider client. |
| `make stage-results-release` | Maintainer step: authenticate the private analysis hash inventory, select only approved aggregate fields, and stage the content-addressed public bundle. |

The split primary order is:

```text
baseline GPU → baseline API/threshold freeze
→ treatment GPU → treatment API → sample validation
→ anchors → positions
→ resample GPU → resample API → resample validation
→ lens GPU/validation → analyze → report
```

All paths shown in `Makefile` are non-secret and overridable for artifact relocation. Route
identities, token prices, GPU rate, GPU family, and approved runtimes are not Make variables; paid
commands read them only from the authenticated locks.

## Cumulative GPU lifecycle

The four canonical GPU phases are `behavior_baseline_gpu`, `behavior_treatment_gpu`,
`resample_gpu`, and `lens_gpu`. Each phase needs a new random nonce, its own reservation receipt,
an armed private watchdog, confirmed provider stop, and exact settlement before the next phase can
start. The commands derive hardware, rate, quote metadata, and approved phase runtime from
`.runpod/gpu_quote_lock.json`; there is no free-form price argument.

Before starting a phase locally, place a new nonce only in the current shell and reserve the full
approved phase maximum in the cumulative ledger:

```bash
GPU_PHASE='behavior_baseline_gpu'
GPU_BUDGET_SESSION_ID="$(openssl rand -hex 32)"
export GPU_BUDGET_SESSION_ID

make gpu-reserve GPU_PHASE="$GPU_PHASE"
```

Inject the same nonce as a RunPod Secret for that phase, sync the exact reservation receipt and
ledger without modifying either, start the explicitly approved Pod, and run from the repository
root on the Pod:

```bash
make gpu-bootstrap GPU_PHASE="$GPU_PHASE"
make gpu-active-verify GPU_PHASE="$GPU_PHASE"
make behavior-baseline-generate
```

The production target repeats the same active-session library gate immediately before backend
construction. Both the standalone verifier and CLI require the receipt, canonical cost ledger,
exact phase, `.runpod/sessions/<session-hash>/` directory, nonce hash, live watchdog PID, `armed`
watchdog state, and bound hardware preflight. The nonce itself never appears in argv or output.

After checksummed artifact sync, request the non-destructive stop described in
[`RUNPOD.md`](RUNPOD.md), wait for `stopped_confirmed`, verify billing has stopped in the provider
UI, and settle using the authoritative charge for that session:

```bash
make gpu-settle \
  GPU_PHASE="$GPU_PHASE" \
  PROVIDER_INCURRED_USD='AUTHORITATIVE_POST_STOP_PROVIDER_CHARGE'
```

`PROVIDER_INCURRED_USD` is reconciliation evidence after stop, not a rate or budget override.
An unsettled prior reservation, stale active session, mismatched receipt, missing watchdog, or
cumulative overrun blocks the next phase. The full Pod setup, re-arm, stop, and recovery contract
is in [`RUNPOD.md`](RUNPOD.md).

## Checkpoint and inference boundaries

Behavioral generation, behavioral adjudication, both resampling allocations, resampling
adjudication, and paid response bodies are independently checkpointed. A successful paid response
body is durably written before parsing, so a parser retry does not duplicate billing. Existing
content-addressed plans and outputs must match byte-for-byte; the pipeline does not silently
overwrite or reinterpret them.

For the sentence intervention, every one of the 24 anchors receives 20 retain and 20 resample
continuations: 960 total. Stage two is unconditional. Primary analysis preserves the full frozen
pair inventory, uses outcome-independent intervention eligibility, clusters by base trace, and
keeps unmeasurable outcomes for missingness bounds instead of selecting complete cases.

Lens rows are observational. They can be displayed as layer-by-position heatmaps, but they cannot
formally support a causal or motivational conclusion unless the preregistered J/R sign,
adjacent-band, temporal, cluster-uncertainty, and trace-level resampling-association conditions all
hold. Every concept uses one global 3+3 probe universe; a causal-prefix collision invalidates the
whole trace × position × concept cell and produces explicit null rows rather than changing probe
weights. The accuracy-anchor association uses paired resample-minus-retain outcomes, equal fixed
layer-band weights, direction-stratified Kendall tau-a, exact within-direction permutations, and
is exploratory rather than causal. The 27B lens is methodology support only after both limited
122B compatibility attempts fail; it is not a substitute for 122B internal-state evidence.
The compatibility attempts are bound to
`data/manifests/lens_compatibility_prefix_manifest.json`, and both success and terminal failure
must link `data/manifests/lens_release_authorization.json`, which authenticates the paid plan,
receipt, approval bindings, active GPU session, and probe design. Analysis repeats pinned-tokenizer
recomputation of lexical collisions, while report validation recomputes the exact eight-trace
association from the linked raw inputs. Verdict-facing lens evidence requires the exact 4+4/576
universe and strictly positive J and R tau-a; p-values and leave-one-out ranges are descriptors,
not additional gate thresholds.

## Cost and time boundaries

The hard ceilings are frozen in the run configuration, preregistration, approval, and append-only
ledger:

- GPU compute: USD 220 maximum;
- external judging/classification: USD 100 maximum;
- unallocated reserve: USD 5;
- total project spend: USD 325 maximum.

Every API request reserves budget before dispatch and must return auditable usage. Each GPU phase
reserves its entire approved maximum against prior incurred and unresolved reservations. The
watchdog stops at the earlier of the phase runtime limit or the 97% safe-budget deadline. A local
smoke run, completed artifact, or document build never authorizes a paid operation.

The five-hour investigation allowance uses actual, nonoverlapping start/stop timestamps. Start a
counted activity before doing it and stop immediately afterward. Replication, sentence-resampling
generation, model/lens download, and document production are logged as `excluded`, not silently
omitted.

```bash
make time-status

make time-start \
  TIME_CATEGORY='hypotheses_and_preregistration' \
  TIME_DESCRIPTION='Freeze competing hypotheses and decision rules' \
  TIME_STATUS='counted'

make time-stop
```

Only one timer session may be active. Do not hand-edit the ledger or replace measured elapsed time
with the category allocations in the preregistration.

## License and upstream boundary

The Value Leakage reproduction repository is referenced at commit
`16d129859e1f0e281363fb4f5910bcaeea316b10`. No license file was present at that pinned root when
inspected. This repository therefore does not redistribute its code, prompts, figures, or raw
model traces. `make reproduce` fetches it into the ignored `data/upstream/` cache and records
hashes and compact derived reference statistics only.

This repository's original code is MIT-licensed. External code, model weights, and lens weights
are downloaded separately and remain subject to their own terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Reproducibility layout and release gate

- `config/`: preregistration, primary and smoke profiles, software/GPU lock, and probe manifest.
- `data/manifests/`: content hashes plus cumulative cost and measured-time ledgers.
- `data/raw/`, `data/interim/`, `data/upstream/`: ignored local inputs and generated artifacts.
- `.runpod/`: ignored quote locks, approval, immutable paid receipts, reservations, and private Pod
  session evidence.
- `src/model_forensics/`: independent implementation.
- `tests/`: deterministic unit, integration-contract, and smoke tests.
- `reports/figures/`: curated figures; `reports/staging/` is ignored.
- `reports/results/`: exact-path, aggregate-only release evidence and table hashes used by
  `make reproduce-results`; raw reasoning, provider bodies, trace IDs, and infrastructure IDs are
  forbidden by both the schema and release audit.

Before any public commit or push, run:

```bash
make release-check
```

The release audit examines tracked and nonignored untracked files; rejects raw/upstream/cache
paths, credentials, private keys, model weights, archives, symlinks, oversized files, and paths
outside the explicit release allowlist; and fails closed if required notices are absent.
