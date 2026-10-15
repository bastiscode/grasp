# Design: process supervision and reward shaping in the CQD trainer

Status: IMPLEMENTED in part (2026-08-03). Covers `RlConfig.advantage_estimator`,
`turn_returns`, `process_advantages` in `train/rl.py`, and the shaping-safety
analysis that decides which reward terms we are allowed to move around.

Motivation: 12 rounds of outcome-supervised GRPO on Qwen3.5-9B produced a
*measured null* on QALD-7 (paired delta +0.021 +/- 0.044 on the clean items).
The trace diff showed the model was making mechanically-checkable mistakes it
never unlearned -- 28% of episodes emitted `SERVICE wikibase:label`, which
QLever cannot execute, at an unchanged rate across all 12 rounds. With one
scalar reward per episode, every turn of an episode gets the same advantage,
so the policy learns "episodes with bad calls score lower" and never learns
*which call* was bad. That is a credit-assignment problem, not a reward
problem.

## 1. Outcome vs process supervision (GRPO, as specified)

Source: **DeepSeekMath** (Shao et al., arXiv 2402.03300), the paper that
introduces GRPO. It defines *both* modes.

Outcome supervision (S4.1.2) -- one reward per output, normalized within the
group of G outputs for one question, assigned to every token:

```
Â_{i,t} = r̃_i = (r_i − mean(r)) / std(r)
```

Process supervision (S4.1.3) -- a reward at the end of each step. Given step
rewards `R = {{r_1^index(1) ... r_1^index(K_1)}, ..., {r_G^index(1) ...}}`,
where `index(j)` is the end token index of the j-th step:

```
r̃_i^index(j) = (r_i^index(j) − mean(R)) / std(R)      # R = ALL steps, ALL outputs
Â_{i,t}      = Σ_{index(j) ≥ t} r̃_i^index(j)          # sum over FOLLOWING steps
```

Two things follow directly and are worth stating because they justify our
implementation shape:

- The advantage is **constant within a step**. For any token `t` inside step
  `j`, the set `{index(j') ≥ t}` is the same, so `Â_{i,t}` does not vary
  across the step. Our per-turn scalar broadcast over the turn's completion
  tokens is therefore exactly faithful, not an approximation.
- Neither mode is discounted; the process advantage is a plain suffix sum.

**Neither TRL nor verl implements process supervision.** verl has 14 advantage
estimators and all are outcome-based (`compute_grpo_outcome_advantage`'s own
docstring: "operating only on Outcome reward (with only one scalar reward for
each response)"). TRL computes `advantages = rewards - mean_grouped_rewards`,
one scalar per completion, broadcast with `unsqueeze(1)`. (Careful when reading
TRL: its `all_process_advantages` refers to distributed *processes*/ranks, not
process supervision.) Turn-level credit lives in the agentic forks --
verl-agent (official GiGPO code), VAGEN, AgentGym-RL, MT-GRPO.

## 2. Why we do not use the paper's normalization order

The paper normalizes each step reward against the group pool and *then* sums
the following steps. Summing K normalized rewards produces a quantity whose
magnitude grows like `sqrt(K)`, so **early turns of long episodes get
systematically larger advantages purely because more steps remain**.

For 3-6 similar-length math steps that is negligible. Our episodes run 7 to 30
turns: the early turns of a 30-turn flail would carry ~`sqrt(30/7)` ≈ 2x the
advantage magnitude of the early turns of a 7-turn solve -- upweighting exactly
the episodes the step budget exists to discourage.

So we use the **REINFORCE++ order**: return-to-go first, normalize after.
verl's `compute_reinforce_plus_plus_outcome_advantage` is the reference:

```python
for t in reversed(range(token_level_rewards.shape[1])):
    running_return = token_level_rewards[:, t] + gamma * running_return
    returns[:, t] = running_return
advantages = verl_F.masked_whiten(returns, response_mask)
```

Same credit semantics as the paper (an action is credited with everything that
follows it), without the length-dependent inflation.

`process_advantages` normalizes over every turn return in the *item's* group,
and counts the "needs at least two rollouts" rule in EPISODES, not turns --
otherwise a single 20-turn rollout looks like a group of 20.

## 3. Reward shaping safety: the three categories

This is the part that is easy to get wrong, and the reason this file exists.

**Ng, Harada & Russell (ICML 1999), "Policy Invariance Under Reward
Transformations".** A shaping term `F(s,a,s')` leaves the optimal policy
unchanged for every MDP with the given transition structure **iff** it is
potential-based:

```
F(s,a,s') = γ·Φ(s') − Φ(s)
```

for some `Φ` over states. It telescopes along any trajectory to
`γ^T·Φ(s_T) − Φ(s_0)`, so the path drops out -- nothing done in the middle can
change the total. Episodic tasks additionally require **`Φ(terminal) = 0`**,
which makes the sum the constant `−Φ(s_0)`, identical for every policy. What
shaping *does* change is *when* credit arrives, which is the entire point.

Classic failures from ignoring this: a cyclist rewarded for progress rides in
circles; a soccer agent rewarded for touching the ball vibrates against it.
Both shaping terms are path-dependent, so looping farms them.

Sorting our terms:

### (a) Exact redistribution -- STRONGEST guarantee, prefer this

Take a term the terminal reward *already contains* and pay it out in
increments. Total episode return is **identical**, not merely policy-invariant;
only the timing of the charge moves. The theorem is not needed at all, and the
`Φ(terminal) = 0` condition is irrelevant because nothing is being added.

The length ramp qualifies. With `L(t)` the ramp as a function of steps used,
`Φ(s_t) = −L(t)` gives increments `−(L(t+1) − L(t))` summing to `−L(T)` --
precisely the terminal penalty `compute_reward` already applies. Under
return-to-go, turn `t` is then charged `−(L(T) − L(t))`: the cost of the length
*it still has ahead of it*. Early turns bear the cost of the episode being
long, and the turn that finally answers bears almost none. That is better
credit than the uniform charge, at zero objective cost.

**IMPLEMENTED**, see S4.

### (b) Potential-based shaping proper -- MEASURED, THEN REMOVED

A genuinely new intermediate signal with `Φ(terminal) ≡ 0` by construction, so
accumulated bonuses are clawed back at termination and the total is the constant
`−Φ(s_0)`. Provably cannot change the optimum.

We built one (a grounding potential over reference entity/property ids), measured
it, and **deleted it along with the older `RewardConfig.w_iri` dense term**. The
code is gone; the finding is why, and is worth not re-deriving:

- **As a baseline it is a coin flip.** Under return-to-go the increments
  telescope, `Σ_{k≥t} F_k = −Φ(s_t)`, so the term is exactly a state-dependent
  baseline -- useful only if `Φ` predicts return. Within the mixed-outcome groups
  GRPO trains on, the better rollout grounded only **+0.033 +/- 0.038** more of
  the reference (base; 12/23 items -- chance). +0.146 +/- 0.057 for the best
  checkpoint, on 11 items. And a baseline uncorrelated with outcome ADDS
  estimator variance rather than removing it.
- **As a reward term its ceiling is ~7%.** A dense grounding reward earns its
  keep only in groups where every rollout scores f1 = 0, which have no reward
  variance and are dropped by `trainable_items`. Measured on the real training
  distribution (263 recorded groups in `runs/curriculum_qwen35_9b_zpd_pool.jsonl`,
  whose `info.groups` store per-group `f1s`/`rewards`): **18/263 = 7% are flat.**
  93% already carry a gradient.
- **The GOTCHA that made this look promising twice.** On the QALD-7 holdout,
  grounding separates outcomes beautifully (f1=1.0 -> 0.69/0.76 vs f1=0.0 ->
  0.46/0.45) and 23/38 groups look flat. Both are artifacts. The separation is
  question DIFFICULTY: solved items' references have 2.50 ids vs 4.28/4.01 for
  partial/failed, while the agent's absolute overlap is flat at 2.3-2.7
  everywhere -- and difficulty cancels exactly in within-group normalization.
  The holdout is also harder and not curriculum-selected, hence 60% flat there
  vs 7% in training. **Measure within item, on the training pool's groups.**
- A monotone recall variant (`|D ∩ ref| / |ref|`) is worse still: corr(f1)
  +0.33/+0.36 vs jaccard's +0.40/+0.51, within-item gap negative for base.

**The consequence is bigger than the verdict on these two terms: gradient
sparsity is NOT the bottleneck.** GRPO has plenty to learn from. What is left is
where the gradient points -- the credit-assignment question in this document --
and the concrete failure modes the traces named (step limit 30% of rollouts,
`SERVICE wikibase:label` 28% of episodes and unlearned across 12 rounds, 14-16%
of executes failing to parse), which `max_steps=30`, `know_before_use` and
`turn_error_penalty` address.

### (c) Objective change -- allowed, but be explicit about it

`turn_error_penalty` is here, and it is **not** compliant. The functional form
passes: with `n(s)` the cumulative wrong-call count, `Φ(s) = −λ·n(s)` gives
`Φ(s_{t+1}) − Φ(s_t) = −λ·e_t`. The terminal condition fails:
`Φ(terminal) = −λ·n_T` varies across trajectories, so

```
Σ_t F_t = Φ(s_T) − Φ(s_0) = −λ·n_T
```

is path-dependent rather than the required constant. Writing a path-dependent
penalty as the difference of a running counter does not launder it; it
relocates the path-dependence into the endpoint.

**This cannot be repaired.** Any compliant term telescopes to a constant, hence
assigns every trajectory the same total shaping, hence cannot prefer 3 wrong
calls over 30. "Penalize wrong calls beyond their step cost" is a change of
objective by construction -- the theorem working, not an implementation gap.

We accept it because a wrong call burns a step and the step budget is our
binding constraint (30% of rollouts die at the cap). Keep `λ` small: it is a
direct tax on exploratory probing, and a failing probe is often the cheapest
way to discover the schema. The diagnostic is therefore NOT just
`train/penalized_turns` falling -- it must fall while `reward/f1` holds and
executions-per-episode stay healthy (4.2-4.4 in the base traces). All three
moving down together means `λ` bought silence instead of skill.

### Caveat: compliance is a reward-level property only

Ng's theorem constrains the reward function and the policy ordering it induces.
Our estimator then does several things the theorem says nothing about:

- dividing by the group std (the Dr.GRPO critique: reweights groups by variance)
- DAPO token-level normalization, so long turns contribute more gradient
- clipping, which truncates updates asymmetrically
- `trainable_items` dropping zero-variance groups entirely

Any of these can shift what a run converges to. "PBRS-compliant" buys
objective-level cleanliness, not an end-to-end guarantee -- a reason to prefer
(a) over (c) when both are available, not a proof of safety.

## 4. What is implemented

`RlConfig.advantage_estimator: Literal["outcome", "process"] = "outcome"`.
Default is `outcome` so pulling this does not silently change a run, and so the
arms can be ablated with the outcome arm **bit-identical** to every previous
run (`group_advantages` is called verbatim; with `turn_error_penalty=0.0` the
advantages are exactly the old ones).

`turn_returns` keeps the reward channels separate, because they must propagate
differently:

```
return(t) = base + Σ_{k ≥ t} length_increment(k)
                 + Σ_{k ≥ t} turn_error_discount^(k−t) · penalty(k)
```

- `base` -- the episode reward, undiscounted to every turn. In KGQA the early
  grounding turns causally determine whether the final query is right, so
  decaying their outcome credit would starve exactly the turns that decide the
  episode. This is why there is no global `gamma`.
- `length_increment` -- category (a) redistribution, undiscounted (it is an
  outcome-channel term). Active only when `decompose_length_penalty` is set,
  in which case `base` excludes the terminal ramp (`EpisodeReward.step_penalty`
  records it so the two cannot double-count).
- `penalty` -- category (c), decaying backwards with `turn_error_discount`.

`turn_error_discount` is restricted by a validator to **0.0 or 1.0**:

- `1.0` full return-to-go: a wrong call is charged to it and every turn before
  it. Correct RL credit, but a turn-19 error also lands on turn 3.
- `0.0` the penalty stays on the turn that made the call.

Intermediates are rejected on purpose -- we have no evidence yet about how far
error credit should travel, and the endpoints bracket it. Widen only once a run
shows the answer lies between them.

Return-to-go has one structural limit worth knowing: it can only distinguish
turns *before* an error from turns *after* it. An error on the final turn is
charged to every turn equally and the group normalization absorbs the rest, so
there is no localization. Mild for us -- the errors we care about happen
mid-episode during exploration, not on the answer turn.

`trainable_items` stays keyed on OUTCOME variance under both estimators, even
though process supervision could extract a gradient from a group whose rollouts
score identical f1 but differ in wrong calls. Holding the refill criterion
fixed keeps batch composition identical across arms, so the comparison isolates
credit assignment instead of also changing which items get trained on.

## 5. Where the advantage lands on tokens

`RlSample` = one assistant turn. `prompt_ids` is the whole context as vLLM
rendered it; `completion_ids` is only what the policy generated that turn. The
turn's single scalar advantage multiplies every completion token:

```python
ratio = (new_logprobs - old_logprobs).exp()          # per completion token
pg = -torch.min(ratio * advantage, ratio.clamp(1-eps_lo, 1+eps_hi) * advantage)
```

- **Tool-call RESULT tokens get nothing.** vLLM's `token_ids` are only the
  generated tokens; GRASP executes the call and appends the result afterwards,
  so it lands in the *next* turn's `prompt_ids`. Environment output is context,
  never completion -- no loss, no advantage. It does consume the context that
  `max_seq_len` guards.
- Turn length still weights the gradient: DAPO sums per-token losses and
  divides by the window's token total, so a 400-token turn moves the policy
  ~10x more than a 40-token one at the same advantage.

## 6. Worked example

One item, group of 2, `turn_error_penalty = 0.5`, no length penalty (3 turns,
well under `step_grace`). Rollout A: 3 turns, turn 2 fires
`SERVICE wikibase:label` (rejected under `know_before_use`), still answers
correctly, f1 1.0. Rollout B: 3 clean turns, wrong answer, f1 0.0.

```
per-turn returns   A: [0.5, 0.5, 1.0]     B: [0.0, 0.0, 0.0]
pooled            [0.5, 0.5, 1.0, 0.0, 0.0, 0.0]  mean 0.3333  pstdev 0.3727

advantage          turn 1     turn 2     turn 3
A process          0.4472     0.4472     1.7889
A outcome          1.0000     0.5000     1.0000
B process         -0.8944    -0.8944    -0.8944
B outcome         -1.0000    -1.0000    -1.0000
```

Process supervision makes turn 3 the most reinforced block in the batch: it
produced the correct query and carries none of the error's cost. The outcome
estimator instead reinforces turns 1 and 3 equally and only dents turn 2.

## 7. Open items

1. The ablation. All arms are runnable today on the same pool and seed, and
   `outcome` is bit-identical to the runs we already have:
   - `outcome` (baseline)
   - `process`, `turn_error_discount` 0.0 vs 1.0 -- differ ONLY in how far
     error credit propagates, which isolates that question directly
   - `process` + `decompose_length_penalty`
2. Whether `turn_error_penalty` earns its keep at all -- if the ablation is
   flat, drop it and keep the objective clean (see S3c). It is the only term
   that changes the objective, so it carries the only real risk.

### If category (b) shaping is ever revisited

Two traps, recorded because both cost time once:

- **A useless check.** "Turn the term on and verify `reward/f1` does not move"
  proves nothing: a shaping term that never enters `compute_reward` leaves
  `reward/f1` structurally untouched, so it cannot move for implementation
  reasons -- and if training behaviour changes it SHOULD move. Not a diagnostic
  in either direction. Verify the shaping sums to zero per episode numerically
  instead (it did: `max |Σ_t F_t| = 1.4e-17` over 300 recorded episodes).
- **The off-by-one at the terminal transition.** The last turn's shaping is
  `0 − Φ(s_{T−1})`, not `0 − Φ(s_{T−2})`; with the wrong index the sum comes out
  at `Φ(s_{T−1}) − Φ(s_{T−2})` instead of 0, silently breaking the guarantee.
- **`iri_jaccard`-style "both empty -> 1.0" conventions.** A reference with no
  extractable ids would set `Φ(s_0) = w` instead of 0, so the shaping sums to
  `−w` and the term stops being objective-preserving.
