# Design proposal: curriculum sampling from f1, and the give-up reward fix

Status: IMPLEMENTED (2026-07-22). Supersedes the perfect-rate binning
described in `03-implementation.md` (Phase 5) and addresses the two
abstention findings in its Open questions. `success_f1` and the per-rollout
`success` bool are removed entirely. Not yet trained/evaluated — the run
that validates it is the next step.

## Why

Two independent problems surfaced in the 24-round continuation run (peaked
QALD-7 0.512 at round 12, collapsed to 0.321 by round 16 with 25/50
invalid):

1. **Bin degeneration.** `success = f1 >= success_f1` with `success_f1 =
   1.0` makes the curriculum's `success_rate` a *perfect*-answer rate, so
   the `failing` bin means "never perfect," not "f1 = 0." Partial-credit
   items (steady f1 0.3–0.75) are mislabeled `failing`; the middle bins
   starve (measured: `new 42, failing 133, struggling 3, improving 5,
   mastered 0`); the `range` sampler collapses onto new+failing. 62 of the
   129 `failing` items actually carried learnable signal (nonzero
   within-group f1 variance) that the bool discarded.
2. **Give-up reward asymmetry.** Every losing outcome scores `-w_invalid -
   penalties`, so among failures a short no-error give-up is strictly less
   negative than a long exploring-then-failing attempt. In zero-variance
   (all-fail) groups — which #1 manufactures — trajectory length is the
   only signal, so the policy learns to quit fast.

The two fixes are orthogonal: A/B (below) repopulate the difficulty ladder
and sample the learnable frontier; C removes the give-up gradient directly.
Ship C even though B should make all-fail groups rarer — it is cheap and
independently correct.

Simulation evidence (on the saved post-collapse pool, `n=8` draws):
learnable items drawn per round 3.36 → 6.0; dead-0 draws 1.53 → 0.0;
previously-hidden `failing`-but-learnable items rescued 1.36 → 3.8/draw.

---

## A. Data model — group records + EWMA stats (`pool.py`, `reward.py`)

Replace the per-rollout success bool with per-round group records, and
derive competence/learnability as exponentially-weighted moving averages
over those groups. The **group** is the natural unit: the GRPO signal is
the within-group f1 spread, and storing it at scoring time avoids
reconstructing groups from a flat attempt list (fragile when rollouts skip
and a round yields <group_size attempts).

New/changed types:

```
class GroupRecord(BaseModel):
    f1s: list[float]        # per-rollout execution f1 this round (raw, nothing lost)
    mean_f1: float          # sum(f1s)/len(f1s)
    std_f1: float           # population std of f1s -- the within-group GRPO signal
    n_invalid: int = 0      # diagnostics
    round: int | None = None

class PoolItemInfo(BaseModel):
    ...                     # unchanged fields
    groups: list[GroupRecord] = []      # REPLACES `attempts: list[Attempt]`
    # `competence`/`learnability` are NOT stored: they are recomputed from
    # `groups` + the current decay params (so decay is re-tunable), see below.
```

Remove `Attempt`, `PoolItem.success_rate`, `PoolItem.learnability` (bool
form) and `RewardConfig.success_f1` (`success` no longer exists). Keep
per-rollout f1 inside `GroupRecord.f1s`, so no raw data is lost and the
EWMAs can be recomputed at any decay setting.

New item-level stats, computed from `groups` with a half-life in groups
(`alpha = 1 - 0.5 ** (1/half_life)`, folded oldest→newest):

```
def ewma(values: list[float], half_life: float) -> float | None:
    # geometric-recency mean; None if values is empty

class PoolItem:
    def competence(self, half_life: float) -> float | None:
        # ewma of per-group mean_f1  -> which difficulty bin (current ability)
    def learnability(self, half_life: float) -> float | None:
        # ewma of per-group std_f1   -> sampling weight (is there gradient)
    def n_groups(self) -> int:
        # history depth; items with 0 groups are `new` (exploration only)
```

Recording (in `reward_episodes`, which already receives the whole round's
episodes so the group is in hand):

```
def build_group_record(episodes, rewards, round: int | None) -> GroupRecord
    # group the round's episodes/rewards for ONE item, collect f1s
    # (invalid/failed -> f1 0.0), compute mean/std, count invalids

class TaskPool:
    def add_group(self, id: str, group: GroupRecord) -> None   # REPLACES add_attempt
```

`reward_episodes` groups its episodes by `item_id`, builds one
`GroupRecord` per item, and calls `add_group` once per item per round
(instead of `add_attempt` per episode).

Decay: two half-lives (in groups). Competence tracks *current* ability →
short; learnability wants a *stable* estimate → longer. Note (measured):
on the current thin-history pool these barely move results because most
items have 1–2 groups; set by principle now, retune once a healthy run
gives items real history.

---

## B. Sampling — competence bins + variance-weighted, capped, `new`-reserved (`pool.py`)

Replace the five perfect-rate bins and equal-quota logic. The item stays
the sampling unit; its bin comes from `competence`, its weight from
`learnability`.

```
def competence_bin(item, half_life, n_bins) -> int | None
    # floor(competence / (1/n_bins)); None for `new` (no groups) -> exploration bucket
```

Draw of `n` items:
1. Reserve `k_new = round(new_fraction * n)` slots for `new` items
   (no learnability estimate yet) — pure exploration so the teacher's
   proposals actually get attempted.
2. Bin the rest by `competence_bin` (width `1/n` by default → up to `n`
   reachable rungs).
3. Per-bin allocation ∝ count of items with `learnability > 0` in the bin,
   **capped at `round(per_bin_cap_fraction * n)`** (= n/2) so no single
   competence band dominates. Empty/short bins redistribute like today's
   `range_quotas` shortfall loop.
4. Within a bin, draw distinct items weighted by `learnability` (sends the
   slot to a high-signal item, never a dead-0 one).

New signatures (old `range_bin`/`range_quotas`/`sample` retired or
rewritten):

```
def learnability_quotas(n, bin_learnable_counts: dict[int,int],
                        cap: int, k_new: int) -> dict[int,int]
def sample(self, n, *, competence_half_life, learnability_half_life,
           n_bins, new_fraction, per_bin_cap_fraction, seed=None) -> list[PoolItem]
```

`STRATEGIES` collapses: this replaces `range`/`range+learnability`. Keep
`uniform` for eval. (No standalone `learnability` strategy — it is now the
default behavior.)

---

## C. Give-up reward fix — floor invalid outcomes (`reward.py`, ORTHOGONAL)

> Superseded (2026-07-23): the flat-`-w_invalid` invalid floor described here
> shipped, but the reward has since been resymmetrized (`w_invalid = 1`,
> range `[-1,+1]`), the `w_err` error penalty removed entirely, and the
> `w_step` penalty replaced by a `[0,1]`-factor length ramp
> (`max_step_penalty * w_invalid`, ramping `step_grace`→`max_steps`). See the
> reward shape in `03-implementation.md` (Phase 2) and `09-dapo-techniques.md`
> for the current version. The rest of this section is kept as the original
> proposal.

Make every failing outcome score the **same**, so failing-long is never
worse than failing-short and the give-up gradient disappears. Penalties
(`w_err`, `w_step`) remain a tie-breaker among *valid* outcomes only.

Change in `compute_reward` — the two invalid branches:

```
# now:      reward.reward = -config.w_invalid - penalties
# proposed: reward.reward = -config.w_invalid            # flat, no penalties
```

The valid branch is unchanged: `reward = w_f1 * f1 - penalties`.

Resulting ordering (with defaults w_f1=1, w_invalid=0.5, penalties ≲0.02):
- valid, any f1 ≥ 0  → ≈ f1 − small  (≥ −0.02)
- invalid (broken query, no query, early give-up, cancel-without-query) → −0.5, all equal

So a working-but-wrong query (≈ −0.02) still beats any give-up (−0.5) —
strong anti-abstention pressure — while all give-ups are equal, so
GRPO gets no "quit faster" signal inside an all-fail group. `fn_errors`
and `steps` are still recorded for diagnostics, just not priced into
invalid outcomes. No new knob. `w_invalid`/`w_err`/`w_step` keep their
defaults.

---

## D. `select_revisits` — continuous analog (`curriculum.py`)

Today: revisit items with `success_rate in {0.0, 1.0}`. New: revisit the
low-signal extremes — `learnability < retire_threshold` AND competence near
0 (→ teacher proposes an *easier* variant) or near 1 (→ mastered, retire /
propose *harder*). The learnable middle (learnability above threshold) keeps
training and is not revisited. Threshold on learnability magnitude, not
`>0` (the EWMA tail never zeroes).

---

## Config knobs (all with defaults; no new CLI subcommands)

| knob | where | default | role |
|---|---|---|---|
| `competence_half_life` | RolloutConfig | 1.5 | EWMA groups for competence (bin) |
| `learnability_half_life` | RolloutConfig | 3.0 | EWMA groups for learnability (weight) |
| `n_bins` | RolloutConfig | = group draw `n` | competence bins, width `1/n_bins` |
| `new_fraction` | RolloutConfig | 0.25 | exploration slots for `new` items |
| `per_bin_cap_fraction` | RolloutConfig | 0.5 | cap per competence bin (n/2) |
| `retire_threshold` | CqdConfig/curriculum | 0.05 | learnability below which an extreme is revisited |
| ~~`success_f1`~~ | RewardConfig | REMOVED | (success bool gone) |
| ~~`sample_last_k`, `sample_push_fraction`, `sample_strategy`~~ | RolloutConfig | REPLACED | by the above |

## Migration / back-compat

- Pools on disk store `attempts`; the new field is `groups`. Provide a
  one-shot `attempts → groups` shim (chunk by group_size) OR regenerate
  pools (cheap). Old pools without groups: all items read as `new` until
  re-attempted — acceptable.
- `uniform` sampling (eval path) unchanged.

## Open decisions (not blocking sign-off)

1. Exact half-lives — can't be pinned on the current thin pool; principle
   values above, retune on a healthy run.
2. `retire_threshold` and `per_bin_cap_fraction` exact values.
3. Curriculum pacing: bound how fast the teacher grows the pool vs. how
   fast items accumulate group history (the binding constraint — EWMA
   makes estimates well-behaved but does not create signal). Possibly a
   separate change.
4. Whether to keep `groups` as full audit trail (raw f1s) or trim to
   summary stats for very long histories.
