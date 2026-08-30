# Private paid-bundle rotation

`scripts/rotate_paid_bundle.py` is the provider-free archival boundary between one expired
quote/approval window and the next fresh preview. It imports no provider client, performs no
network operation, and cannot reserve, start, stop, or modify a Pod. Rotation does not authorize a
paid command.

The command fails before creating an archive unless all of these local facts authenticate:

- the canonical `data/manifests/cost_ledger.yaml` has no `gpu` entry whose status is `estimated`;
- `.runpod/pod_lifecycle.json` is content-addressed and says exactly `operation=stopped` with
  `pod.status=EXITED`;
- every directory below `.runpod/sessions/` passes the existing completed-session validator and
  has an authenticated `cumulative-gpu-phase-settlement-v2` settlement (including the paired
  provider no-start receipt when that is the closure evidence);
- both canonical quote locks are valid, content-addressed, owned regular files with one link;
- any present approval and quote specs pass their strict offline schemas; and
- none of the source, manifest, archive, completion, or parent paths is a symlink or hardlink.

Rotation holds an exclusive, nonblocking `.runpod/paid_bundle.lock` for the entire transaction.
Every reservation, lifecycle re-arm, paid GPU/API command, preview, and approval consumer holds the
same lock in shared mode from before reading private controls until its work returns. A concurrent
consumer or rotation therefore fails closed instead of acting later on controls cached before the
archive transition.

The approval is optional because the normal expired state can be pre-approval. The lifecycle is
not optional: an absent lifecycle cannot prove that the existing research Pod is stopped, so this
tool deliberately has no `--allow-no-lifecycle` bypass.

For this study, first finish the source/release checks, commit the exact reviewed tree, verify the
clean commit, and push it to the canonical public origin. Then rotate the expired private bundle,
before writing new quote specs or running a fresh preview. Run with the content-derived archive
identity:

```bash
make paid-bundle-rotate
```

Or choose one explicit, non-secret, lowercase ID:

```bash
make paid-bundle-rotate PAID_BUNDLE_ID='quote-window-20260830-a'
```

The destination is
`.runpod/archive/paid-bundles/<bundle-id>/`. `manifest.json` records every canonical source path,
archive path, exact byte length, and SHA-256. All exact-byte copies are installed and fsynced before
any canonical source is removed. `rotation_complete.json` is installed only after removal succeeds.
Neither file is replaced. A process interruption therefore leaves an immutable incomplete
transaction: rerun with the same explicit ID, or rerun without an ID when it is the sole incomplete
content-derived transaction. Recovery authenticates existing bytes and continues; any different,
unmanifested, truncated, linked, or concurrently recreated file fails closed.

After successful rotation, independently refresh the two quote specs from official current
sources, run the non-authorizing `freeze_paid_bundle.py preview` with the exact intended
`--allow-phase` set, inspect its review payload, and obtain a new explicit approval for that fresh
window. Never copy an archived lock or approval back into a canonical path.
