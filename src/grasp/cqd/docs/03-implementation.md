# Implementation plan for `grasp.cqd`

Phased so that each phase produces something testable on its own. Phases 1-3
require no RL at all and already yield a useful artifact (a synthetic
training/eval dataset + an SFT distillation baseline).

## Proposed module layout

```
src/grasp/cqd/
  docs/                  # these files
  __init__.py
  configs.py             # CqdConfig, TeacherConfig, StudentConfig, RewardConfig, ...
  seeds.py               # load/validate/verify a given seed JSONL (see 05-seed-data.md)
  pool.py                # TaskPool: items, lineage, difficulty, coverage, sampling
  proposer.py            # teacher-side generation: propose/verify/adapt/fan-out
  rollout.py             # student rollouts: parallel core.generate, trace -> Episode
  reward.py              # reward assembly from Episode + reference SPARQL
  curriculum.py          # feedback summarization + difficulty/fan-out policy
  train/
    __init__.py
    data.py              # Episode -> token-level training items (masking!)
    ppo.py               # PPO trainer integration (external RL framework)
    sft.py               # SFT distillation baseline from teacher traces
tasks/
  cq_distillation/       # new GraspTask for the teacher (lives with other tasks)
```

New task(s) go under `src/grasp/tasks/` like all others; everything else
under `src/grasp/cqd/`.

> **CLI policy:** everything is implemented as importable modules/functions
> first. Any `grasp cqd ...` CLI subcommand is only a proposal — CLI entry
> points are added only after explicit sign-off (name + arguments agreed
> beforehand).

## Phase 1 — seeds + proposer task (no training)

**Goal:** from a given CQ seed file, produce a pool of verified `(question,
reference SPARQL, metadata)` items.

1. `seeds.py`: load a user-provided seed JSONL (`sparql` required, `question`
   optional; format in 05-seed-data.md), validate the schema, and verify each
   seed by execution (`evaluate.get_result_or_error`): parseable, executes,
   non-empty and non-degenerate result. Report failures, don't silently drop.
   Seeds without a `question` are fine — the cq-distillation task phrases
   one while proposing.
2. New task `cq-distillation` (`tasks/cq_distillation/`), the single teacher
   task, modeled on `QuestionGenerationTask` (recent-list mechanism) with
   the execute-and-verify submission pattern of `SparqlToQuestionTask`:
   - **Input:** the CQ (seed SPARQL + optional question), plus optionally a
     previously distilled (question, SPARQL) pair derived from it, plus
     optionally the student's agent trace(s) on that pair (rendered via
     `notes_from_traces.format_output`; absent at the start and for fresh
     proposals). The *agent* decides from the trace whether to reformulate
     simpler or harder; without a trace it proposes fresh variations.
   - **Tools:** full KG fn set (the teacher explores to ground new
     entities/properties) + `submit_proposal(kg, question, sparql, intent,
     difficulty?)` + `show_proposals` + `stop`.
   - **Diversity:** every submission carries a short *intent* description;
     `show_proposals` lists recent proposals + intents (last K globally +
     the current CQ lineage) so the agent avoids duplicates and
     diversifies. If soft dedup proves too weak, add an embedding
     near-duplicate check in `submit_proposal` (reuse `ExampleIndex`
     machinery) later.
   - **Verification inside `submit_proposal`:** execute the SPARQL
     (`get_result_or_error`), reject on error/empty/oversized result or
     unparseable query (`load_sparql_parser`), and return the rejection
     reason to the teacher so it can fix and resubmit — same pattern as
     `sparql-qa`'s `answer_rejected`/know-before-answer check.
   - **Output:** list of accepted pool items, each with `cq_id` backlink.
3. `pool.py`: `TaskItem` extends `Sample` semantics
   (`{id, question, sparql, info}`) with `info = {cq_id, parent_id, intent,
   difficulty, tags, kg, teacher_model, verified_result_size, attempts:
   [{success, f1, steps, errors}]}`. `cq_id` (root competency question) is
   mandatory on every proposal — the proposer's anchor and the unit of
   coverage tracking; `parent_id` is the immediate predecessor proposal
   (None for first-generation proposals). Declared `difficulty` is coarse
   metadata only — the measured success rate in `attempts` is the real
   difficulty signal. Persist as JSONL so `grasp evaluate f1` and
   `grasp examples` work on pools unchanged. Conceptually the pool is
   *task replay* in the spirit of Prioritized Level Replay / ACCEL (see
   04-related-work.md): tasks are sampled by learning potential and mutated
   (by the teacher) at the ability frontier — but items trigger fresh
   on-policy rollouts, no trajectories are replayed (PPO).
4. Reference quality: we trust the teacher — no standalone audit (an
   independent agentic solve per item roughly doubles teacher cost).
   Instead, label noise is monitored via free signals:
   - **Implicit confirmation from training rollouts**: every student attempt
     is an independent solve; an item ever solved with F1 = 1 has its
     reference confirmed. Items with persistent F1 = 0 across many attempts
     *while sibling items succeed* are flagged as suspicious from pool
     `attempts` stats (no extra runs) for occasional manual inspection.
   - **Held-out benchmark as backstop**: synthetic reward rising while
     benchmark F1 stagnates indicates reference noise (or distribution
     mismatch).
   - **One-time manual skim** of ~20 pairs after the first real generation
     run to calibrate baseline trust in the teacher.

**Deliverable:** a `generate_pool(config)` entry point (CLI exposure TBD) →
pool JSONL + coverage report. Testable by inspection.

## Phase 2 — student rollouts + reward (no training)

**Goal:** measure the untrained student on pool items; validate the reward.

1. `rollout.py`: `run_episode(item, student_config, managers) -> Episode`
   wrapping `core.generate("sparql-qa", ...)`; a thread/async pool for
   batched rollouts against a vLLM server (the loop is sync + I/O-bound, so
   a `ThreadPoolExecutor` is fine; follow `bash/run.sh` batching patterns).
   `Episode = {item, output_sparql, messages, token_ids/logprobs per
   assistant turn, elapsed}`.
   - Config: `model_provider: openai/completions`, `model_kwargs:
     {logprobs: true, extra_body: {return_token_ids: true}}`. Verify
     `Response.token_ids` is actually populated end-to-end with our vLLM
     version — first thing to smoke-test in this phase.
2. `reward.py` (current shape):
   ```
   valid episode:      R = w_f1 * f1_score(exec(pred), exec(ref))     # dominant term
                         - max_step_penalty * w_invalid * ramp(steps) # soft length penalty
   invalid / give-up:  R = -w_invalid                                 # flat, no penalty
   ramp(steps) = clamp((steps - step_grace) / (max_steps - step_grace), 0, 1)
   ```
   The length penalty ramps linearly from 0 at `step_grace` to its max
   (`max_step_penalty * w_invalid`) at the rollout step cap `max_steps`,
   which is passed into `compute_reward` from `rollout_config` — the single
   source of truth (no duplicated reward-side cap). Defaults: `w_f1 = 1`,
   `w_invalid = 1` (reward symmetric `[-1, +1]`: a give-up is as far below a
   valid-but-wrong answer ≈0 as a perfect answer +1 is above it — strong
   anti-abstention gradient); `max_step_penalty = 0.15` is a FACTOR in
   `[0, 1]` (< 1 guarantees a max-length valid-but-wrong answer still beats a
   give-up); `step_grace = 20`. There is **no function-error penalty**:
   repeated malformed queries already cost steps (this ramp) or end invalid,
   and taxing them directly would penalize healthy trial-and-error
   exploration — `fn_errors` is a diagnostic only. No exact-match bonus (see
   below).

   No exact-match bonus: questions admit multiple valid interpretations,
   and `assignment_f1_score` deliberately forgives presentation differences
   (extra columns, ordering — see its perfect-match-with-extras test), so a
   strict-equality bonus would reward matching the teacher's arbitrary
   presentation rather than being right; near the top of the reward range,
   false negatives from interpretation variance dominate over over-broad
   reward hacking (whose precision loss plain F1 already caps). Revisit
   only if over-broad plateau behavior is actually observed. The main
   ambiguity defense is generation-side: the cq-distillation task requires
   questions whose expected result is uniquely determined (name the
   information the answer should contain).
   Reference results are executed once and cached on the pool item
   (endpoint load: rollouts already hammer the endpoint; add a small
   result cache keyed by normalized query).

   **Error attribution (for later, keep global scalar for now):** not every
   error is the policy's fault and only policy-attributable ones may be
   penalized:
   - Episode-level `error.reason` of `api`/`timeout`/`empty` = LLM API or
     infrastructure failure → **drop the episode from the training batch**
     (no reward at all, and no pool `attempt` record — a zero reward or a
     penalty would be noise and corrupt learnability estimates). `loop` is
     model behavior and may be penalized.
   - Per-call `tool_call.error`: endpoint/server failures (see
     `utils.py:is_server_error`) are not the model's fault → count only
     client-attributable errors (malformed queries, invalid arguments,
     unknown functions) in the `fn_errors` **diagnostic** (no longer priced
     into reward — see the reward shape above; kept for monitoring only).

   **Penalty placement:** all penalties are *global* — folded into the
   single episode-level scalar, placed terminally. Under PPO, GAE + the
   value model distribute credit to tokens; under GRPO the (group-relative)
   advantage applies uniformly to all trajectory tokens. We deliberately do
   NOT localize penalties to the tokens of the offending call at first:
   (a) semantically failing SPARQL probes are often productive exploration
   and must not be punished locally (only malformed/schema-invalid calls
   would ever qualify), (b) dense shaping changes the optimum and invites
   degenerate call-avoidance. Episodes store per-call `error` flags and
   per-turn token ids, so a per-token reward tensor (verl supports this)
   for malformed calls only remains a later ablation at zero extra
   infrastructure cost; a turn-level middle ground (penalty on the erroring
   turn's last token) composes with PPO but not plain GRPO.
3. Per-item stats recorded back into `pool.py` (`attempts`) → success rates
   drive learnability sampling: prefer items with `0 < success < 1`
   (GRPO-style groups of k rollouts per item give this for free).

   The learnability weight (implemented in `pool.py`) is the smoothed
   Bernoulli variance `p̂(1-p̂)` with `p̂ = (s+1)/(n+2)` over the item's
   (last k) attempts. Rationale: policy-gradient signal comes from outcome
   *contrast* within a batch — all-fail (p=0) and all-solve (p=1) items
   contribute zero advantage, and expected contrast is the outcome variance
   p(1-p), maximal at p=0.5. The Laplace smoothing gives unattempted items
   the maximum weight 0.25 (built-in exploration bonus), lets confidence
   grow with evidence (2/2 solved → 0.188, 20/20 → 0.043, never exactly 0,
   so mastered items keep getting rare regression checks), and `last_k`
   keeps the estimate tracking the improving policy instead of stale
   attempts. Same idea as PLR's learning potential / AZR's learnability
   reward, estimated from success variance.

   **Frontier-only sampling can erode mastered behavior.** Learnability
   weighting concentrates *all* rollouts near p=0.5 and structurally
   excludes mastered (p=1) and hopeless (p=0) items from ever entering a
   training batch. Observed consequence: a student SFT-distilled and
   GRPO-trained purely on a hard, frontier-style pool got *worse* on an
   easy out-of-distribution benchmark (QALD-7) — it unlearned the
   fast-confident-answer behavior because nothing in training ever
   exercised or rewarded it; conditional on producing an answer at all its
   F1 was actually higher than the base model's, but it answered on far
   fewer questions (non-conclusion: 3/50 base → 29/50 SFT → 6/50 after
   GRPO partially repaired it).

   Two new `TaskPool.sample` strategies address this — sample the *whole*
   competence range, occasionally push beyond it, rather than
   frontier-only:
   - `"range"`: stratify items into five success-rate bins (`_range_bin`:
     new, failing, struggling, improving, mastered) and give every bin an
     **equal quota** regardless of how many pool items happen to fall in
     it — this is what guarantees flat coverage rather than re-deriving
     something close to learnability weighting on a lopsided pool. A
     `push_fraction` (default 0.2) adds extra quota on top of the
     `failing` bin's equal share, to deliberately probe just beyond
     current ability. Quota shortfalls (a bin with fewer items than its
     quota) redistribute round-robin to bins with spare capacity, so the
     batch always fills to `n` when the pool is large enough. Within a
     bin, items are drawn uniformly — the point of stratifying is flat
     range coverage, not further biasing inside a stratum.
   - `"range+learnability"`: identical stratification and quotas, but
     within-bin draws are weighted by `learnability` instead of uniform,
     so already-settled items within a stratum yield to still-informative
     ones. Kept as a separate strategy (not the default) so `"range"` vs
     `"range+learnability"` vs plain `"learnability"` is a clean ablation.
   - `learnability` itself is kept as-is (not replaced) for exactly that
     ablation comparison, and because pure frontier-seeking is still the
     right choice when the pool doesn't span multiple difficulty tiers.

**Deliverable:** a `collect_rollouts(config)` entry point → episodes +
reward distribution report
(mean/variance per difficulty tag). Reward validated against manual trace
inspection (`grasp show`).

## Phase 3 — SFT distillation baseline

**Goal:** the ablation baseline and likely the RL initialization.

Implemented in `train/data.py` + `train/sft.py`:

1. Teacher solves pool questions via `sparql-qa` (`rollout.py` against the
   teacher endpoint, `token_data=False`); episodes are scored with
   `reward_episode_file(save_pool=False)` (teacher attempts must not enter
   the student's pool statistics) and filtered to answers with
   `f1 >= min_f1` (default 1.0).
2. `train/data.py:episode_sft_sample`: episode messages → OpenAI chat
   format → student chat template with the episode's recorded tool schema
   (`Episode.functions`) → tokens + assistant-only loss mask. The
   conversion folds the teacher's *reasoning* into the assistant content:
   thinking teachers put nearly all text there, and distilling without it
   taught the student to emit bare tool calls, which degenerated into
   call loops (observed: f1 0.0, 11/39 loop aborts). The non-thinking
   student instead learns to reason in plain content. Masks are
   located via character-level incremental prefix rendering mapped through
   the offset mapping (tokenization itself is not incremental — tokens
   merge across message boundaries). Verified token-exact against vLLM's
   per-turn completion lengths. The re-rendering is template-canonical,
   not byte-identical to the rollout (GRASP strips response content on
   parsing — a `\n` of drift per turn); fine for SFT, and the reason the
   RL path never re-renders.
3. `train/sft.py:train_sft`: plain transformers `Trainer` + peft LoRA
   (all-linear), assistant-token loss, bf16, gradient checkpointing,
   wandb reporting. Saves the adapter for vLLM serving and RL init.
4. Evaluate checkpoint as a GRASP agent on the pool (rollout + reward with
   `save_pool=False`) and later on a held-out benchmark
   (`grasp evaluate f1`).

**Deliverable:** small model + SFT beats small model zero-shot. If this
already fails, fix data before touching RL.

## Phase 4 — PPO training

**Goal:** RL on top of Phase 2 rollouts.

Implemented in `train/rl.py` as a compact **GRPO** loop over external
GRASP rollouts (integration style (b) below), the simple variant suited
to a single machine (vLLM student server on one GPU, training on
another):

- One round = sample pool items (learnability) → `collect_rollouts` with
  `group_size` rollouts per item through the vLLM server → `reward.py`
  (attempts recorded) → group-normalized advantages (drop skipped
  episodes; degenerate all-equal groups carry no signal) → clipped
  policy-gradient pass → save LoRA adapter → hot-load it into vLLM
  (`/v1/load_lora_adapter`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`) → next
  round. wandb logs reward/f1/loss/kl/clip-fraction per round.
- Every assistant turn is its own sample: its prompt is the exact
  server-side context (`prompt_token_ids`), its completion the exact
  sampled tokens — no re-tokenization, no logprob drift (verified: HF
  bf16 logprobs match vLLM's sampling logprobs to mean |Δ| ≈ 0.01).
  Turns share their episode's advantage. `logits_to_keep` materializes
  logits only at completion positions, keeping 12k-token prompts
  trainable on 24GB.
- No value model (that's the G in GRPO), no KL penalty term (β = 0) to
  start; the clipped ratio against the rollout logprobs bounds updates.
  Init from the Phase 3 SFT adapter (or base). Two DAPO stabilizers are
  now on (see `09-dapo-techniques.md`): **clip-higher** (asymmetric clip
  `clip_eps_low = 0.2`, `clip_eps_high = 0.28`, guards entropy collapse)
  and **token-level loss** (each optimizer window normalizes by its total
  completion-token count, so long turns aren't down-weighted). KL and
  dynamic sampling remain candidate levers.
- **Loss masking** is structural: only assistant completion tokens enter
  the loss (system / user / tool-result tokens live in `prompt_ids` only).
- **Episode length:** cap `max_steps` and skip over-long turns
  (`max_seq_len`), logged.

The original framework plan (verl, PPO with a value model and GAE,
GRASP-as-environment integration style (a)) remains the scale-up path —
the ablation PPO-vs-GRPO is on the paper checklist — but is deliberately
not the first implementation:

- **(a) GRASP as environment inside verl**: rollout inference inside
  verl's vLLM workers, weight sync solved for us; heavier setup.
- **(b) External rollouts** (Phase 2 machinery) + importing token
  ids/logprobs into a custom step — what `train/rl.py` does; feasibility
  of the token path is proven.

**Deliverable:** `train_rl(config, rollout_config)` entry point (CLI
exposure TBD). Metrics: reward curve, pool/held-out F1, valid-call rate,
mean steps.

## Phase 5 — closing the loop (auto-curriculum)

**Goal:** interleave teacher and training.

The simple version is implemented in `curriculum.py`:

1. `select_revisits(pool, item_ids, last_k)`: rule-based controller — it
   only decides *what gets revisited*; the teacher agent decides
   simpler/harder itself from the trace. Simple rule: items whose (recent)
   attempts the student never or always solves (success rate exactly 0 or
   1, i.e. no learning signal); restricted to the current round's items so
   traces are available.
2. `render_trace(episode, reward)`: student episode rendered via
   `notes_from_traces.format_output`, prefixed with the scored execution
   F1 so the teacher sees the outcome, not just the behavior.
3. `run_revisits(...)`: runs `cq-distillation` on each selected item with
   the CQ (from the seeds file via `cq_id`), the item as the previously
   distilled pair, and the rendered traces; new proposals enter the pool.
4. `run_curriculum_round(cqd_config, rollout_config, reward_config)`: one
   loop iteration — rollout → reward/attempts → revisit — over an
   existing pool (the teacher-only version, no student training).

**Wired into RL training (the actual two-agent loop).** `train_rl` takes
a `curriculum_interval` (0 = off) and a `teacher_config`; every
`curriculum_interval` GRPO rounds, after rewards are scored and attempts
recorded, it calls `run_revisits` on that round's signal-less items using
the episodes already in hand and the shared rollout KG managers/notes
(no second model load). Ingested proposals land in the same pool the next
round samples — so the item set co-evolves with the student instead of
staying frozen, which is the whole point of the teacher/student setup and
the direct fix for the frozen-difficulty half of the QALD regression
(the sampling half is addressed by the `range` strategy). `pool/size` and
`pool/added_proposals` are logged per round.

**Multi-hop lineages.** A revisit target can itself be a previously
proposed item, not just a first-generation one. `cq_id` (the root
competency question) is stamped once at ingestion and copied unchanged to
every descendant, while `parent_id` advances by one hop each time; so
revisiting the pool's own output extends an existing chain
(seed CQ → p1 → p2 → …) rather than restarting from the seed, and
`TaskPool.lineage` walks the whole path back (it is already cycle-safe).
This lets a finite seed list grow into arbitrarily deep,
teacher-and-student-co-evolved difficulty ladders.

Deliberately not in the simple version (add when the need is observed):
success-rate bands instead of exact 0/1, retiring items that stay at 0
across reformulations, coverage-gap detection (CQs with few proposals →
fresh proposals without traces), targeting mean pool success ~0.3–0.7,
representative-trace subsampling, and the async variant (proposer runs
while the trainer trains).

## Ablations / experiments (paper checklist)

- CQ-seeded vs. unseeded generation (question-generation task as the
  unseeded baseline).
- Curriculum vs. fixed difficulty mix vs. random pool sampling.
- PPO vs. GRPO vs. SFT-only vs. zero-shot.
- Reward ablations (F1 only vs. + error/step shaping).
- Transfer: train on Wikidata pool, test on other KGs' benchmarks
  (does tool-use skill transfer even when KG specifics don't?).
- Teacher label-noise estimate (from student-rollout agreement stats) vs.
  final student quality.

## Dependencies

Add `[project.optional-dependencies] cqd = [...]`: `verl` (or `trl`),
`torch`, `transformers`, `peft`, `wandb`, `vllm` (training env only). Core
grasp package stays dependency-light; `grasp.cqd` imports RL deps lazily.

## Open questions

- ~~Does `Response.token_ids`/`token_logprobs` survive the whole
  vLLM → OpenAI-client → GRASP path for tool-calling responses?~~
  Resolved: yes, verified end-to-end against vLLM (token ids and logprobs
  align with `completion_tokens` for tool-calling responses).
- Reward for `cancel`: whatever final query the episode returns — an
  answer's query or a cancel's `best_attempt` — is the prediction and is
  scored with plain execution f1, matching `evaluate_f1`'s uniform
  treatment of `output["sparql"]`. A cancel with **no working query** is
  scored exactly like a broken/absent answer (`-w_invalid`), *not* a
  positive floor. Since every pool reference is answerable (see the
  unanswerable-exclusion note below), giving up is never the correct
  action, so it must not pay more than a failed attempt. An earlier
  `cancel_reward = 0.1` floor sat *above* the achievable expected f1 on
  hard items and, under GRPO on the weak 4B student, drove an abstention
  collapse: cancel (~-0.03) strictly beat a risky attempt that often came
  out invalid (-0.55), so the policy learned to cancel everything (a large
  curriculum+range run degraded to QALD-7 f1 0.06, mean f1 → 0 by round 3,
  most groups degenerate). Neither range sampling nor the growing pool
  fixed it — the bottleneck was reward shaping, not the task distribution.
  `RewardConfig.cancel_reward` is now reserved (unused) for the future
  correct-abstention reward once genuinely-unanswerable items exist. The
  cancel=invalid fix is confirmed: curriculum runs stopped collapsing
  (grpo-curriculum-2x, 12 rounds from base, hit QALD-7 0.501 with 0
  invalid at eval).
- **Second abstention channel — give-up bias from bin degeneration
  (TODO).** With the cancel floor closed, a *longer* 24-round continuation
  from the 0.501 model still degraded: it peaked at round 12 (QALD-7 0.512,
  healthy) then collapsed to 0.321 by round 16 with 25/50 invalid. The
  invalids are **not** broken queries — 22/25 are early terminations with
  no final query at all (`type=None`, ~5 steps vs ~8 healthy). Training
  signature: episode length (trainable-turns proxy) collapsed ~350→~180
  around rounds 15–17 and training reward went negative (round 17
  −0.246), matching the OOD degradation window.
  **How it expressed (pre-fix reward):** every losing outcome scored
  `-w_invalid - penalties`, with `penalties = (w_err·fn_errors + w_step·max(0,
  steps-grace)) / max_steps`, so among failures a short no-error give-up
  was strictly less negative than a long exploring-then-failing attempt. In
  a zero-variance (all-fail) GRPO group, trajectory *length* is the only
  thing distinguishing rollouts → the policy learns to quit fast.
  **Why so many all-fail groups (the root cause):** *not* curriculum
  flooding — the `range` sampler spans five difficulty bins by design
  (only ~3/8 failing) and the pool is not unsolvable (71/141 attempted
  items get nonzero f1). The cause is the success threshold:
  `success = f1 >= success_f1` with `success_f1 = 1.0`, so `success_rate`
  is a *perfect*-answer rate and the "failing" bin (`rate <= 0`) means
  "never perfect," not "f1 = 0." Partial-credit items (steady f1 0.3–0.75)
  are mislabeled "failing"; the middle bins (`struggling`/`improving`/
  `mastered`, which need success_rate strictly in (0,1)) starve — final
  pool bins were `new 42, failing 133, struggling 3, improving 5,
  mastered 0`. With four of five bins near-empty, `range_quotas`
  redistributes the unfillable middle quota into new+failing (~6/8 of each
  round), many of them true-0 → zero-variance groups → the give-up bias
  bites. The curriculum only feeds the `new` bin; even a frozen pool would
  starve the middle bins under `success_f1 = 1.0`. Candidate fixes to
  weigh: **(1)** lower/soften `success_f1` (e.g. 0.5 or graded success) so
  partial-credit items graduate into the middle bins and the sampler
  actually spans difficulty — upstream, most surgical; **(2)** floor
  invalid episodes at exactly `-w_invalid` (no step/error penalty on
  already-invalid outcomes) so failing-long is no worse than
  failing-short, killing the give-up gradient directly; **(3)** a
  KL-to-reference anchor; **(4)** holdout-based early stop (the training
  reward completely masked this collapse — see the
  periodic-holdout-validation item above). (1) and (2) are complementary.
  **Implemented (2026-07-23):** (1) `success_f1` and the perfect-rate bins
  are gone — competence/learnability are now EWMAs over rollout groups and
  the teacher grows the pool by bottom-k least-learnable revisits (see
  `06-` and `08-`). (2) invalid/give-up now scores a flat `-w_invalid` with
  **no** length/error penalty, and the reward was resymmetrized to
  `w_invalid = 1` (range `[-1,+1]`) with the length penalty a `[0,1]` factor
  of `w_invalid` and no function-error term (see the reward shape in Phase 2
  and `09-`). Holdout validation (4) is implemented. KL (3) remains a TODO
  (next bullet). Still open: whether these fully stop the collapse on a weak
  student — the grpo-qwen25-3b run still collapsed with an earlier subset of
  these fixes, motivating dynamic sampling + the step-limit filter.
- **KL-to-reference anchor (TODO).** `train_step` runs GRPO *without* a KL
  penalty: the objective is only the clipped surrogate
  `-min(unclipped, clipped)`. The `kl` we log is a diagnostic against the
  *rollout* policy (π_old, refreshed every round for the importance
  ratio), **not** a penalty and **not** against a frozen reference — so it
  provides no leash on slow multi-round drift, which is exactly the
  abstention-collapse surface. Left out originally by simple-first (KL is
  the first thing minimal GRPO reproductions cut) and because a frozen
  reference costs scarce 24GB memory (we already OOM on 7B training and 4B
  only fits with `logits_to_keep` + gradient checkpointing). The two
  abstention fixes above are *surgical*; a KL anchor is the *general*
  guard that makes long / aggressive-curriculum runs safe without
  babysitting for the next novel collapse. Cheap implementation path: the
  frozen reference's per-token logprobs can be **cached at rollout time**
  (vLLM already returns them), so no second resident model or
  train-time forward pass — add `+ beta * KL(pi_new || pi_ref)` to the
  loss with a small `beta` and watch the KL curve. Decision gate: if the
  fixed-dataset baseline and current runs stay stable without it, we may
  not need it; add it the moment drift reappears on longer runs. See
  `07-roadmap.md`.
- Unanswerable / empty-answer questions are currently excluded from the
  pool by construction (seed verification and `submit_proposal` reject
  empty results), because `f1_score(empty, empty) = 1.0` makes
  empty-reference items trivially hackable ("emit any query returning
  nothing"). This also excludes legitimate questions whose true answer is
  empty. Incorporating them (honest abstention / correct-empty training)
  would need teacher-verified unanswerability — proving absence is much
  harder than presence — plus anti-hacking safeguards. Out of scope for
  now.
- **ASK queries are different**: their answer space is binary, so false is
  a legitimate answer and false-ASK references stay allowed — excluding
  them would make every pool ASK item's answer "true" and hand full reward
  to the degenerate policy "always emit a trivially-true ASK". The
  safeguard for ASK items is rough true/false *balance* (a guessing policy
  then caps at ~50%), which is a pool/curriculum monitoring concern, not a
  verification gate.
- One KG first (Wikidata) — multi-KG training later; per-KG pools vs. one
  mixed pool.
- Teacher budget: proposals are expensive (exploration + verification).
  Track cost per accepted item; let one teacher run submit several
  proposals per CQ to amortize exploration cost.
- **Periodic holdout validation (IMPLEMENTED 2026-07-22).** The per-round
  `reward/f1` on the training pool is a poor progress signal: sampling
  draws a different item mix each round, so f1 is a sawtooth driven by
  sample composition, not learning. Curriculum-injected items are
  never-seen (and "easier" only relative to a 0.0-success item), so they
  routinely land below the running average and depress the number
  regardless of true progress. So `train_rl` now takes `val_files:
  list[ValFile]` (each `{kg, file}`, mirroring the note-taking `samples`
  config) and `val_interval` (0 disables); every `val_interval` rounds,
  after the round's adapter is hot-loaded, `run_validation` rolls out the
  current student once per holdout sample (no training, throwaway pool),
  scores with the training reward, and logs per-file
  `val/<name>/{f1,invalid,answered}` plus aggregate `val/{f1,invalid}`
  into the round's `run.log`. This is the honest curve to watch and gives
  a principled best-checkpoint / early-stop signal, replacing the stopgap
  of eval-ing several late `round_N` checkpoints at the end. Phase 4's
  deliverable already lists held-out F1 as an intended metric; this
  implements it. Val KGs must be among the rollout `knowledge_graphs`
  (managers reused); pin the KG endpoint to amur:7001 via the reward
  config as usual.
