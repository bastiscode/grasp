# Design: sampler + curriculum v2 (coverage-paced growth)

Status: PARTIALLY IMPLEMENTED (2026-07-22). Revises the **sampling** (B)
and **revisit/retire** (D) parts of `06-curriculum-sampling-redesign.md`;
keeps its data model (A: GroupRecord, EWMA competence/learnability) and
give-up reward fix (C) unchanged.

**Implemented so far (the regression fix's growth driver):** lever B's
*bottom-k* revisit selection. `select_revisits` now returns the visited
items ranked by learnability ascending, and `run_revisits` caps at
`max_revisits`, so the teacher revisits the `k = max_revisits`
least-learnable items every curriculum round — a fixed per-round growth
size that keeps hitting its budget (restores the baseline's growth without
resurrecting `success_f1`). The old `learnability < retire_threshold` gate
and the `retire_threshold` knob are **removed**. Still PROPOSED / not yet
built: the coverage-first sampling (A), the coverage gate on growth (B), the
competence-split retirement (C), and the bounded pool (D).

## Why

The `06` redesign shipped and was A/B-tested against the 0.501 baseline
(`grpo-curriculum-2x`), same setup, 12 rounds from base. Result:
**grpo-curriculum-v2 = 0.456 honest QALD-7 f1, ~0.045 WORSE**, with a
healthy (non-collapsed) trajectory. Two measured causes:

1. **Curriculum starvation.** `select_revisits` gates on `learnability <
   retire_threshold (0.05)`, which rarely fires when items have only 1–2
   groups of nonzero-variance history. The teacher revisited only 0–2 of 8
   items per round (vs. the old criterion's 4/8 cap every round), so the
   pool grew 39→51 (+12) instead of 39→87 (+48). The student trained on
   ~4× less task diversity → weaker generalization. (The old
   `success_rate ∈ {0,1}` criterion was a *broken* signal but doubled as
   the pool's diversity engine; removing the pathology removed an
   unattributed benefit.)
2. **Coverage lag.** In the v2 final pool, **19 of 51 items (37%) were
   never rolled out once.** With `items_per_round=8` and `new_fraction=0.25`
   only ~2 new items are drawn per round, while the teacher adds ~2/round —
   coverage never catches up, so the pool fills with never-attempted items
   that have no competence/learnability estimate and dilute the sampler.

These are orthogonal: (1) is too little total growth, (2) is growth
outrunning consumption. v2 had both. The fix is to make growth
**self-paced to student throughput** and **predictable**, and to keep the
pool's per-item history dense enough for the EWMA estimates to mean
something.

## The invariant: coverage before expansion

Grow the pool only once the current items have been covered (rolled out at
least once). Precedent: `take_notes_from_samples` already pre-generates
epochs so every sample is seen once before any is revisited. This bounds
the unvisited backlog and paces growth to how fast the student actually
consumes items.

"Visited once" is a **coverage** guarantee, not a **learning** one — a
single all-fail group teaches nothing — so it must not be read as "the pool
is well-estimated," only "nothing is unseen."

## Three levers

### A. Sample — coverage-first, then exploit (replaces `new_fraction`)

Each round of `n = items_per_round`:
1. **Coverage:** draw unvisited items (`n_groups == 0`) first, up to
   `round(coverage_max_fraction * n)` slots. Capped (not "all unvisited")
   so a round is never entirely never-seen items — that would be all
   zero-variance groups and no gradient. This replaces `new_fraction` with
   clearer semantics: it is the *max* share spent draining the unvisited
   backlog, and unvisited items are drawn preferentially within it.
2. **Exploit:** fill the remaining slots by competence bin + learnability
   weight, exactly as in `06` (skip zero-variance items, cap per bin at
   `per_bin_cap_fraction * n`).
3. If no unvisited items remain, all slots go to exploit.

### B. Grow — coverage-gated, bottom-k (predictable batch)

- **Gate:** the teacher grows the pool only when coverage ≥
  `coverage_threshold` (default not 1.0 but ~0.9, or "no unvisited item
  older than N rounds") — a soft gate, so one perpetually-skipped item
  cannot freeze the curriculum forever.
- **Batch:** when the gate is open, revisit the **bottom-k** items by
  learnability (`k = max_revisits`), regardless of an absolute threshold,
  so growth reliably hits its budget. Per growth event:
  `growth ≈ k × max_proposals × (accept-and-novel rate)` — a constant
  per-event budget → **linear, predictable growth** and a knob
  (`k × max_proposals × interval`), not the emergent artifact it was in
  `06`/baseline.
- Tradeoff to accept: bottom-k always revisits k items even if the k
  least-learnable are healthy, which can yield some lower-value proposals
  on an all-learnable pool. Acceptable for predictable diversity; dedup
  catches exact repeats.

The loop is epoch-style: drain unvisited → gate opens → teacher adds a
bottom-k batch → drain the new items → …

### C. Retire — competence-split (manages the tail)

Low learnability alone should not "retire" an item; branch on competence:

- **Hopeless** (low competence, ~0 learnability, even after easier
  variants): the real prune candidate. The teacher proposes an *easier*
  variant; if the lineage stays hopeless after sustained history (long
  EWMA, not a single round), prune the original.
- **Mastered** (high competence, ~0 learnability): keep for **rehearsal**
  (deleting it risks catastrophic forgetting), already down-weighted to
  ~0 by learnability sampling. Optionally propose a *harder* variant
  (progression).

Note: dead items cost ~nothing per round — the learnability-weighted
sampler already never draws them. So retirement is about (a) keeping
per-item history dense and (b) bounding the tail on long runs, not about
saving rollouts.

### D. Bounded pool (long-run only)

For truly long/indefinite runs, cap the pool at `pool_cap` with eviction:
evict sustained-hopeless items first, keep a rehearsal reservoir of
mastered items (replay-buffer style). Not needed at the current 12-round /
~90-item scale — **defer** until long runs. See `07-roadmap.md`.

## Config (proposed; defaults to tune on a healthy run)

| knob | where | default | role |
|---|---|---|---|
| `coverage_max_fraction` | RolloutConfig | 0.5 | max share/round draining unvisited items (replaces `new_fraction`) |
| `coverage_threshold` | RlConfig | 0.9 | coverage fraction that opens the growth gate |
| `max_revisits` (= bottom-k) | RlConfig | 4 | items revisited per growth event (fixed growth size) — IMPLEMENTED |
| ~~`retire_threshold`~~ | RlConfig/curriculum | REMOVED | old learnability gate; bottom-k has no threshold |
| `pool_cap` | RlConfig | None | long-run bounded-pool size (D); None = unbounded |
| `competence_half_life`, `learnability_half_life`, `sample_n_bins`, `per_bin_cap_fraction` | RolloutConfig | (from `06`) | unchanged |
| ~~`new_fraction`~~ | RolloutConfig | REMOVED | replaced by `coverage_max_fraction` + the coverage gate |

## Staging

- **Now (regression fix):** A (coverage-first sampling) + B
  (coverage-gated bottom-k growth). This directly addresses both measured
  causes and is the minimal change to test against 0.501.
- **Later:** C's pruning (needs sustained-history logic) and D (bounded
  pool). C's *variant* proposing (easier/harder by competence) can come
  with B if cheap.

## Open decisions

1. `coverage_threshold` exact value and whether to add the age-based
   escape ("no unvisited item older than N rounds").
2. `coverage_max_fraction` — 0.5 keeps half the round for gradient; may
   want higher during an initial catch-up phase, lower later.
3. Whether bottom-k should still respect a loose learnability floor (skip
   revisiting genuinely-healthy items when the pool is small).
4. Confounding: `06` changed sampler + reward together. A clean isolation
   of the give-up fix alone vs. the sampler is still not done.

## Relationship to other docs

- `06-curriculum-sampling-redesign.md`: data model + give-up fix kept;
  sampling/retire revised here.
- `03-implementation.md`: holdout validation (now reports the honest f1)
  is the yardstick for the re-run; KL-anchor TODO is the general drift
  guard.
- `07-roadmap.md`: bounded pool (D) and the fixed-dataset baseline (which
  needs none of the curriculum machinery) live there.
