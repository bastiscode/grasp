# Design: DAPO techniques for the CQD GRPO trainer

Status: PROPOSED (2026-07-23). Candidate stabilizers for `train_rl` /
`group_advantages` / `train_step`, motivated by the abstention collapse of
the grpo-qwen25-3b run (val f1 peaked 0.245 @ round 4, decayed to 0.02 by
round 16; trainable turns cratered 224 → ~35 — the training batch ran out
of gradient). KL-to-reference is tracked separately and is NOT in scope
here (deferred by request; see `03-implementation.md`).

Source: **DAPO** ("DAPO: An Open-Source LLM RL System at Scale", Yu et al.,
arXiv 2503.14476, ByteDance/Tsinghua, 2025). PDF saved at
`scratchpad/DAPO_2503.14476.pdf`. Base model Qwen2.5-32B, benchmark AIME
2024 (50 pts avg@32, verl framework). DAPO's objective is GRPO with these
four changes and, notably, **no KL term** (they argue long-CoT policies
*should* diverge far from init — the opposite of our weak-model collapse
problem, which is why KL is still on our table but tracked elsewhere).

Context difference to keep in mind: DAPO is a 32B model doing single-turn
long-CoT math (responses up to 20k tokens). We are a 3–4B model doing
*multi-turn tool-use* SPARQL (`max_steps=15`, `max_completion_tokens=1536`
per turn, `RlTurn` = one assistant message). So the techniques port, but
the length-related ones need re-scoping.

DAPO objective (Eq. 8), for reference:
```
J(θ) = E[ (1/Σ|o_i|) ΣᵢΣₜ min( r_{i,t}Â_{i,t},
                                clip(r_{i,t}, 1-ε_low, 1+ε_high) Â_{i,t} ) ]
Â_{i,t} = (R_i - mean{R}) / std{R}      # group-relative, same as ours
```

---

## 1. Dynamic Sampling  (IMPLEMENTED 2026-07-23)

**DAPO:** over-sample and **filter out groups whose rollouts all get the
same reward** (accuracy 0 or 1 → zero advantage, no gradient), keeping the
constraint `0 < #correct < G`. "Keep sampling until the batch is fully
filled with samples whose accuracy is neither 0 nor 1." Constant effective
batch size, no dead gradient.

**Our code today (`group_advantages`, rl.py):** a group with `pstdev <
1e-6` returns `advantage = None`, and `train_rl` does `if advantage is
None: continue`. So zero-variance groups are **dropped and not replaced** —
the batch shrinks. This is exactly the collapse mechanism: as more sampled
items went all-fail, trainable turns fell 224 → ~35 and the gradient came
from a handful of surviving groups.

**Done (`train_rl` round loop + `trainable_items`, rl.py):** each round
draws up to `cap = items_per_round * dynamic_sample_max_factor` (default 4×)
candidates from `pool.sample`, then rolls them out in `items_per_round`-sized
chunks, scoring each chunk and **stopping as soon as `items_per_round` items
have a non-None group-relative advantage** (reward variance). Zero-variance
groups and step-limit-filtered episodes (see #4) simply don't count toward
the target, so the loop refills past them; the trainable batch no longer
shrinks. `new_fraction` is scaled by `items_per_round / cap` so the `new`
exploration budget stays at its per-round size despite the larger draw.
All rolled-out items still get their `GroupRecord` (pool competence/
learnability is honest); only *training* uses the variance filter. Logged:
`train/rolled_out_items`, `train/trainable_items`. `dynamic_sample_max_factor
= 1` disables it (single chunk = old behavior).

**Cost / caveats:** up to `cap`× the rollouts in the worst case (a very hard
pool for a weak student); bounded by the cap, and the round log surfaces the
actual `rolled_out` vs `trainable` counts so silent truncation is visible.
Composes with the `08-sampler-curriculum-v2.md` coverage/growth work — both
decide which items get rolled out (the chunked draw already front-loads the
sampler's high-priority items, backfill last).

---

## 2. Clip-Higher  (IMPLEMENTED 2026-07-23)

**DAPO:** decouple PPO's symmetric clip into an asymmetric range,
`clip(r, 1-ε_low, 1+ε_high)` with **ε_low = 0.2, ε_high = 0.28**. A tight
upper clip caps how fast a low-probability token's probability can rise,
which suppresses exploration and drives **entropy collapse**; raising
ε_high alone leaves room to reinforce rare-but-good tokens.

**Was:** symmetric — `ratio.clamp(1 - clip_eps, 1 + clip_eps)`, single
`clip_eps = 0.2`.

**Done:** `RlConfig.clip_eps` replaced by `clip_eps_low` (0.2) /
`clip_eps_high` (0.28); `train_step` clamps to `[1-clip_eps_low,
1+clip_eps_high]`; the `clip_frac` metric now counts ratios outside that
asymmetric band. No other call sites; the wandb config log picks the new
fields up automatically.

---

## 3. Token-Level Policy Gradient Loss  (IMPLEMENTED 2026-07-23)

**DAPO:** normalize the loss by the **total token count across the batch**
(Eq. 12: `1/Σ|o_i| · ΣΣ ...`), not by averaging each sample's mean loss and
then averaging across samples. Sample-level averaging under-weights tokens
in long responses; token-level makes every token count equally.

**Was:** loss per `RlTurn` was `-torch.min(...).mean()` (mean over that
turn's completion tokens), each turn backwarded with weight
`1/grad_accumulation`. So it was **turn-level**: a 5-token turn and a
500-token turn contributed equally, down-weighting tokens in long turns.

**Done:** `train_step` now iterates in `grad_accumulation`-sized windows;
per turn the loss is `.sum()` over its tokens and backwarded as
`loss / window_tokens`, where `window_tokens = Σ len(completion_ids)` over
the window (known up front, no extra forward pass). So one optimizer step =
mean per-token loss over its window → token-level. Turns are still
backwarded individually (unchanged memory profile). Metrics (`loss`, `kl`,
`clip_frac`) are now reported as per-token means for consistency;
`trained_turns` still counts turns.

---

## 4. Overlong Reward Shaping  (lowest priority / needs re-scoping)

**DAPO:** truncated (length-capped) responses otherwise get a
wrong-answer reward, which is noise. Two parts:
- **Overlong Filtering:** *mask the loss* of truncated samples (don't train
  on them at all).
- **Soft Overlong Punishment** (Eq. 13): a length-dependent soft penalty
  ramping 0 → −1 over the last `L_cache` tokens before `L_max`, and −1
  beyond. DAPO: `L_max = 20480`, `L_cache = 4096` (soft zone 16 384–20 480).

**Our code today:** episodes hitting `max_steps` / `max_completion_tokens`
end with no final query → scored invalid → flat `-w_invalid`. So a
truncation is priced as a failed answer — the exact noise DAPO removes. We
also already skip turns longer than `max_seq_len` (`skipped_long`) at the
loss stage.

**Note — the constructive half already exists.** DAPO's *Soft Overlong
Punishment* (graded "wrap it up" pressure) is, for us, the valid-episode
**length ramp** already in the reward: `max_step_penalty · w_invalid ·
clamp((steps − step_grace)/(max_steps − step_grace), 0, 1)` — 0 up to
`step_grace = 20`, rising to `0.15·w_invalid` at the step cap. So only the
*filtering* half is outstanding.

**Filtering half — REVISED 2026-07-23 after the format-collapse finding.**
The first version detected a truncation as `type is None and error is None`,
on the premise that the agent loop only exits via answer/cancel, an error, or
the step cap. **That premise was wrong**: there is a fourth exit — an
assistant message with **no parseable tool call** (nothing to run → loop
ends), which also has `type=None, error=None` but `steps < max_steps`. On the
grpo-qwen25-3b-dapo run this was the *dominant* failure (46/50 at r16): RL
degraded tool-call formatting until the model dumped the call as raw text plus
a stray `<|im_start|>`. The broad filter caught these early degenerate exits
and **excluded them from training every round**, so the policy never got a
gradient away from breaking format — the filter likely *caused* the collapse,
and separately hid it from `val/f1` (see `cqd-dapo-format-collapse` memory).

Corrected rule (`compute_reward`): filter only a **genuine budget truncation**
(`type is None and error is None and steps >= max_steps`); an **early
degenerate** exit (`steps < max_steps`) falls through to `-w_invalid` and is
trained on, pushing the model back onto the template. A `filter_step_limit`
flag makes even this training-only: `run_validation` and eval pass `False`, so
every no-answer episode counts as a failure (honest f1 0) and is never
dropped. The DAPO soft-length numbers (20 480 / 4 096) still don't transfer;
the soft-penalty half remains the valid-episode length ramp (above).

---

## Status

All four DAPO techniques were implemented (2026-07-23): **dynamic sampling**
(#1), **clip-higher** (#2), **token-level loss** (#3), and the **overlong
step-limit filter** (#4). The grpo-qwen25-3b-dapo run with all of them on
completed and is analyzed in the `cqd-dapo-format-collapse` memory: it fixed
the cancel-based abstention collapse but hit a **tool-call format collapse**
(honest QALD-7 0.28 base → 0.08 r16), traced to the mis-scoped #4 filter
(now corrected, see above) with **clip-higher (#2) a suspected accelerant**
(raising eps_high amplifies off-template tokens on a small model). Next run:
the corrected filter first; consider rolling eps_high back toward 0.2 and/or
adding a format/KL anchor if drift persists.

Not in scope: **KL-to-reference** (deferred). Note DAPO *removes* KL for the
opposite reason to ours (they want divergence; we are collapsing *into* a
degenerate policy), so KL remains a separate, still-open lever — these four
are the DAPO-native alternatives that don't require a frozen reference model
in memory.

## Relationship to other docs

- `08-sampler-curriculum-v2.md`: dynamic sampling (#1) overlaps the
  sampling/coverage design — reconcile which items get rolled out.
- `03-implementation.md`: give-up/abstention analysis (overlong shaping #4)
  and the deferred KL-anchor TODO.
- Motivating run: grpo-qwen25-3b (see memory `cqd-v2-regression` thread).
