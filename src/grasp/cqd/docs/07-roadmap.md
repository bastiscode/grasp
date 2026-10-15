# Roadmap: GRASP-RL / CQD future directions

Forward-looking directions beyond the current CQD prototype. Recorded from
the 2026-07-22 planning notes; the sequencing is a recommendation, not yet
committed. The KL-to-reference anchor is tracked separately as an
engineering TODO in `03-implementation.md` (Open questions), since it is an
enabler for these rather than a direction of its own.

A cross-cutting principle governs three of the items below: **any prompt
scaffold added at training time that is absent at eval creates a train/eval
distribution shift** — the policy learns to lean on a crutch it won't have.
The unifying discipline is to (a) make the scaffold part of the deployed
system so it is present at eval too, and/or (b) inject it *stochastically*
(dropout) and fade it out, so the policy stays robust with and without it.

## 1. Fixed-dataset RL baseline (READY TO RUN 2026-07-28, freebase/WQSP)

Train on a fixed, given dataset with **no teacher / make-harder-easier
logic** — just the existing success-rate-aware sampling over a frozen pool.
Freebase WQSP is one candidate, not the only target: any dataset with a
usable reward signal (reference queries or gold answer sets) fits, and we
will likely want several across knowledge graphs.

Why this is the highest-leverage next step:
- It **isolates GRPO-on-execution-f1 from the teacher/curriculum**, so we
  can attribute how much of the current results come from RL vs. the
  auto-curriculum — which we cannot claim today.
- It is nearly free to build: it is CQD with `curriculum_interval = 0` plus
  a dataset→pool loader. The redesigned sampler (EWMA competence +
  learnability, see `06-curriculum-sampling-redesign.md`) already *is*
  "sample by prior success rate" on a frozen pool.
- It pairs directly with the new periodic holdout validation
  (`val_files` / `val_interval` on `RlConfig`): a real train/test split
  gives an honest held-out curve.

Caveats, as resolved on 2026-07-28:
- **Scoring — no change needed.** The worry was that WQSP ships gold *answer
  sets* rather than queries, which would have turned the reward into
  "predicted result vs. static gold set". It does not: both splits carry
  reference SPARQL in the `SparqlQaSample` schema, so `compute_reward` runs
  unmodified.
- **Freebase — endpoint is local.** The qlever `freebase` server runs on
  **tagus:7002**, not amur (which serves wikidata on :7001). WQSP's
  machine-generated references, CVT joins and
  `FILTER(!isLiteral … langMatches)` boilerplate included, execute on it;
  **3097 of 3453** train references verify non-empty (89%), and train/test
  share no content-derived ids. The `grasp-2/main/freebase` index has
  `entities/fuzzy` + `properties/embedding`, matching the `run.yaml`
  defaults, and is symlinked into `data/kg-index/freebase`.

What was built: `datasets.build_pool` (split → verified pool file, doubling
as a `val_file`) plus scratchpad `build_wqsp_pools.py` / the no-teacher
`run_rl_wqsp_freebase.py`.

**Sampler coverage is the live design question.** `train_rl` rescales
`new_fraction` by `items_per_round / cap` (the dynamic-sampling cap), so the
stock 0.25 admits only ~2 never-attempted items per round: 20 rounds touch
**~65 items regardless of pool size** — a result about 65 questions, not
about WQSP. Simulated coverage over 20 rounds at `items_per_round=8`:
0.25 → ~65 items, 0.5 → ~95, 0.75 → ~130, 1.0 → ~174. The frozen-pool runs
therefore want a **very high new_fraction** (0.75 chosen), with 1.0 as the
degenerate end: every item gets exactly one group, so the competence and
learnability EWMAs never accumulate and the ZPD weighting — the mechanism
this baseline exists to isolate — goes inert.

## 2. In-context hints (IMPLEMENTED 2026-07-23, teacher-prompt only)

Give the student **in-context hints for things it cannot know or keeps
failing to find** — a specific IRI, a related example, a special command /
SERVICE, a KG idiosyncrasy. A weak student cannot discover these from a
scalar reward efficiently.

**No new field or plumbing.** The student's prompt *is* the question
(`run_episode` passes only `item.question`), so the teacher embeds the hint
inline in the question text, phrased as an informed user naturally would.
Implemented as a `cq-distillation` prompt change only (`rules()` +
`system_information` step 1 in `tasks/cq_distillation/__init__.py`): the
teacher may now scaffold a hard task with hints **instead of** simplifying
the reference query, keeping the reward target at full complexity (a better
lever than `difficulty="easier"`, which lowers the target itself).

**Why this is not the distribution-shift crutch it first looks like.** The
cross-cutting rule above worried that a training-only scaffold teaches the
policy to lean on something absent at eval. But the target is *handling
user inputs well*, and real user questions genuinely do carry hints
("who directed X — I think the property is P57", "use the wikidata
SERVICE"). Hint-bearing questions are therefore **in-distribution**, not a
shift away from it — the student learning "sometimes a question carries a
hint, sometimes not, solve either" is exactly right. The hint is woven into
the NL text, not a structured slot the policy can key on. Benchmark-specific
tuning to a hint-free style (e.g. QALD-7) is instead the job of the
fixed-dataset variant (item 1).

The held-out QALD-7 curve stops being "the thing hints must not move" and
becomes a narrower **coverage diagnostic**: can the student still solve
plain, hint-free questions? (a guard against hints dominating, not an
objective they betray). Naturally diluted, since only would-be-"easier"
proposals carry hints while "similar"/"harder" stay plain. Staged off for
the in-flight DAPO + sampler-regression validation run (one variable at a
time); on for curriculum rounds after. The learned notes of item 3 are the
same mechanism with a second, trace-driven author.

## 3. Note-taking-from-traces variant

A CQD variant that runs note-taking over the student's agent traces (the
`notes_from_traces` plumbing already exists) and injects the notes into the
prompt **sometimes**. This is a second learning channel — in-context
memory alongside weight updates.

Design points:
- "Sometimes" = **stochastic dropout**, which both realizes the
  distribution-shift discipline and stops the policy becoming
  note-dependent.
- **GRPO group consistency:** all rollouts *within one item's group* must
  see the *same* notes, or the within-group advantage compares different
  prompts. Vary notes across items/rounds, freeze them within a group.
- Note growth must be bounded (`max_notes` / `max_note_length` already
  exist); a note that does not improve reward should be droppable.
- A/B against a validated plain-CQD baseline — it adds a confound, so it
  should not be folded in from the start.

## 4. Reward for other tasks — entity linking

Generalize reward computation beyond `sparql-qa`. The `entity_linking.py`
task already exists, but `compute_reward`, `Episode`, and `run_episode`
(which calls `generate("sparql-qa", ...)`) are currently hard-wired to
sparql-qa: they execute `reference_sparql` and key on `answer` / `cancel`.
The change is to factor a per-task scorer, with sparql-qa as the first
implementation.

Why entity linking is attractive as the second task:
- Reward is precision / recall / F1 over the linked entity set — **no
  endpoint execution**, so no timeouts and no oversized-result blowups
  (e.g. the ">10,000,000 rows" failures seen on sparql-qa rollouts). A
  faster, far less flaky RL testbed.

Scope discipline: **EL reward + fixed-dataset EL RL is achievable**; an
**EL auto-curriculum is a separate project**, because the proposer /
distillation stack is sparql-centric and would need its own EL proposer.

## 5. Dense reward shaping — IRI-Jaccard (REMOVED 2026-08-04, measured worthless)

**Do not rebuild this.** The premise — that execution-f1 is too sparse, so GRPO
groups carry too little variance to learn from — was measured on the real
training distribution and is **false**. In the 9B run's pool
(`runs/curriculum_qwen35_9b_zpd_pool.jsonl`, whose `info.groups` hold per-group
`f1s`/`rewards`), only **18 of 263 recorded groups (7%) have flat reward** and
get dropped by `trainable_items`. 93% already carry a gradient: the curriculum
plus DAPO dynamic sampling supplies the variance this item was meant to create,
so a dense shaping term has a ceiling of ~7% more trainable groups.

A potential-based per-turn variant (a grounding potential, provably
objective-preserving) was also built and measured: within the mixed-outcome
groups GRPO actually trains on, the better rollout grounds only +0.033 +/- 0.038
more of the reference (base — 12/23 items, chance). Both terms
(`RewardConfig.w_iri` and `RlConfig.w_grounding`) and their machinery are
deleted; `docs/11-process-supervision.md` S3b records the full measurement,
including the difficulty-artifact trap that made this look promising twice.

**Consequence for this roadmap: gradient sparsity is not the bottleneck.** What
remains is where the gradient points (credit assignment: `advantage_estimator`
outcome vs process, docs/11) and the failure modes the QALD-7 traces named —
step limit on 30% of rollouts, `SERVICE wikibase:label` in 28% of episodes and
unlearned across 12 rounds, 14-16% of executes failing to parse.

## 6. Quieter validation signal (POTENTIAL — greedy, not multi-sample)

`train.rl.run_validation` rolls out each holdout item **once** at the training
temperature, so each item's val-f1 is a single stochastic draw. Boundary
questions (2-hop joins) flip right/wrong between draws, so the aggregate val-f1
carries ±0.05–0.08 sampling noise that swamps real progress (the 0.53↔0.33
oscillation of the KL run).

**Multi-sample (K rollouts per item, average per item) is REJECTED:** it scales
the wrong way — K× cost on a val set that will only grow, and validation is
already expensive even at val_interval=2. Use **greedy (temperature 0)**
validation instead: same one-rollout-per-item cost, but deterministic, so
round-to-round movement reflects the policy rather than sampling luck. It
measures the policy's mode rather than its expected sampled f1 — a slightly
different quantity, but the right trade for *tracking a trend* cheaply.

## Deprecated

- **Hard-first / top-down curriculum seeding** — "run hard questions first,
  gradually remove complexity until the small model can solve it, then ramp
  back up." An earlier idea for how to approach CQD, explicitly retired
  (2026-07-22). It is also technically fragile: starting RL on all-hard
  items yields zero-variance all-fail GRPO groups → no gradient, which is
  the exact abstention-collapse surface.
