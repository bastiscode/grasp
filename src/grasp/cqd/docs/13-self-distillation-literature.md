# Literature: self-distillation policy optimization (SDPO) and agentic follow-ups

Status: LITERATURE REVIEW (2026-08-26). Nothing implemented, nothing decided.
Written up because the cluster attacks the same problem CQD attacks, by a
different mechanism, and one paper in it runs on Freebase/WebQSP with SPARQL.

Verification note: the original paper was read in full from the PDF. Entries
marked `[fetched]` in section 7 were read from their arXiv abstract or HTML
page and their mechanisms are reported here. Entries marked `[snippet]`
surfaced only in search results and were not read -- they are listed for
completeness; do not cite their numbers without reading them first. Numbers
below came through a summarising model and should be spot-checked against the
PDFs before they go into a paper.

## 1. The original: SDPO (arXiv 2601.20802)

Huebotter et al., ETH Zurich / MPI-IS / MIT / Stanford. Code at
`github.com/lasgroup/SDPO`.

One model, two roles. Sample rollouts from the student `pi(.|x)`, obtain
environment feedback `f`, then re-prompt the *same* model with `f` and
re-score the student's own tokens under `pi(.|x, f, y<t)`. Distil that back:

```
L_SDPO = sum_t KL( pi(.|x, y<t) || stopgrad( pi(.|x, f, y<t) ) )
```

Written as an advantage this is a drop-in swap in any GRPO implementation:

```
A_GRPO[i,t] = r_i - mean(r)                          (constant over t)
A_SDPO[i,t] = log pi(y|x, f_i, y<t) / pi(y|x, y<t)   (per token)
```

`f` has two slots, filled by a fixed template: the **environment output**
(runtime errors, execution diagnostics) and a **correct solution**, which is
just a successful rollout from the same group when one exists. Neither slot
requires an external model. The student's own attempt is passed as the
assistant turn to be re-scored, but is deliberately *not* repeated in the
teacher's prompt -- their Table 6 shows that doing so biases the teacher
toward the attempt (same-output 10% -> 30%, entropy 0.38 -> 0.23) and hurts.

Engineering details worth keeping: top-K distillation with K=100 instead of
full logits (avoids holding two full-vocab logit tensors); EMA-regularised or
base-interpolated self-teacher; symmetric Jensen-Shannon instead of plain KL.
Overhead over GRPO is 6-17% per step -- one extra forward, no extra
generation. All their SDPO results are strictly on-policy, one gradient step
per generation batch; off-policy SDPO is left as future work.

Headline results:

- SciKnowEval (chemistry/physics/biology/materials) + ToolAlpaca, Qwen3-8B
  and Olmo3-7B-Instruct: beats a well-tuned GRPO (asymmetric clipping,
  unbiased normalisation, IS correction) on almost every cell. Olmo3 on
  chemistry reaches GRPO's 5h accuracy in 50 minutes -- a 6x speedup -- and
  ends >10 points higher.
- LCBv6, Qwen3-8B: GRPO's final accuracy in 4x fewer generations.
- Response length drops 3-11x *while* accuracy rises. Qualitatively, GRPO's
  extra tokens are filler and circular reasoning ("Wait", "Hmm", repeated
  identical calculations); dense per-token credit removes them.

Stated limitations: performance tracks the model's in-context learning
ability, so SDPO "is primarily applicable for RL-training stronger base
models, while it can underperform GRPO on weaker models" -- confirmed in
their Figure 11, where SDPO loses to GRPO at Qwen3-0.6B and needs the GRPO
term mixed back in (`lambda = 0.9`) at 1.7B. Performance also depends on
feedback quality; uninformative or misleading feedback is not learnable.

### 1.1 The initiation result (their section 5)

This is the part that bears directly on our stuck question -- whether RL can
bootstrap a behaviour the policy never exhibits.

They define `discovery@k = P(discovery time <= k)`, the probability that at
least one of `k` sequential attempts succeeds. It generalises `pass@k` to
methods whose policy changes between attempts. Setting: single hard LCBv6
questions, binary reward, no curriculum, batch size 16, pure test-time
self-distillation on that one question.

- Very hard (pass@64 < 0.03, 9 questions): best-of-k 41.5%, multi-turn 35.6%,
  SDPO **53.2%** discovery@2750. ~3x fewer generations to reach 22%.
- Hard (pass@64 < 0.5, 19 questions): SDPO 78% discovery@2750, ~2.4x fewer
  generations to 67%. SDPO solves every question the baselines solve, plus
  one (Q3) that neither solves within 2750 attempts.

The load-bearing detail: **the initial self-teacher's accuracy is <1% on
almost all of these questions and exactly 0% on 78% of them.** So this is not
"the teacher can already do it, just distil it." Retrospective credit
assignment is useful even when the retrospective model still fails. Their
framing: RLVR cannot start before the first success because a binary reward
carries no information until then; rich feedback removes that barrier.

## 2. Why this matters for CQD -- and why it is also competition

**It is aimed at our problem.** Their footnote 9 is explicit: RLVR on a
binary reward "provides no signal until the first solution has already been
found", and "for this reason, several works consider explicitly constructing
curricula of solvable questions ... which self-distillation avoids." That
footnote describes CQD. If dense retrospective credit makes hard items
learnable without a ladder, the auto-curriculum's marginal value drops. The
two are not mutually exclusive -- a teacher that constructs items and a
mechanism that extracts dense signal per item compose fine -- but "the
curriculum is the contribution" gets weaker, and that should be decided
consciously rather than discovered in review.

**Our environment is unusually rich in exactly the feedback it needs, and we
currently throw all of it away.** A CQD episode produces SPARQL parse errors
(14-16% of executes fail to parse), empty result sets, endpoint timeouts,
actual result rows, `search_property` / `list` tool outputs, and ~3000 chars
of reasoning per turn -- across 12-15 turns. All of that is compressed into
one scalar f1 at the end. That is the credit-assignment bottleneck the paper
names, in an extreme form. And because we hold reference SPARQL, the "correct
solution" slot is available even when no rollout in the group succeeds.

**It reframes the reference-access objection from doc 12 section 2.** We
rejected giving the teacher the reference on the grounds that its exploration
becomes theatre. SDPO puts the correct solution straight into the privileged
context and it is fine, because the teacher never emits a trajectory to
imitate -- it only re-scores tokens the student already produced. So there
are three distinct categories, not two:

- reference-conditioned **item construction** -- safe (what the curriculum does)
- reference-conditioned **credit assignment** -- safe (what SDPO does)
- reference-conditioned **trajectory demonstration** -- leaky (doc 12's concern)

Their Table 6 marks the boundary empirically: the leak appears exactly when
the student's attempt enters the teacher's *prompt* rather than only being
scored.

**It removes the failure class that has eaten this project.** Every collapse
we have had -- the cancel floor, `ASK { ?s ?p ?o }`, empty == empty,
valid-but-wrong queries -- is the policy finding a degenerate maximiser of a
scalar. A logit-level advantage has no scalar to push. Not immune, though:
f1 still decides which rollout becomes the "correct solution", so the metric
leniencies still leak in through that door.

## 3. The agentic follow-ups: three convergent findings

Six papers, nearly all on ALFWorld / WebShop / Search-based QA, at
Qwen2.5-3B, Qwen2.5-7B and Qwen3-1.7B: SDAR, AgentOPSD, GRSD, ADRS, Skill-SD,
PBSD (section 7 for URLs).

### 3.1 Naive SDPO does not survive multi-turn

Three independent reports, and this is the single most important finding for
us given our collapse history:

- **Skill-SD**: "naively combining OPSD with RL often leads to training
  collapse."
- **SDAR** diagnoses the mechanism. Once the student drifts off
  teacher-supported trajectories, per-turn KL surges and task performance
  degrades catastrophically (their Observation-1). And **>50% of tokens carry
  a negative teacher-student gap** because of imperfect retrieval and
  teacher-student drift (Observation-2) -- so uniform distillation spends most
  of its gradient pushing tokens *down*.
- **SRPO** reports that SDPO "frequently collapses during prolonged
  training" and is eventually **surpassed by GRPO**. Attributed to two
  causes: self-distilling already-correct samples creates optimisation
  ambiguity, and self-teacher reliability decays as training proceeds. Their
  fix routes correct samples to GRPO and failed samples to SDPO, with
  entropy-aware weighting: +3.4% over GRPO and +6.3% over SDPO on a
  five-benchmark average with Qwen3-8B, at up to 17.2% lower per-step cost.

Adopting vanilla SDPO would mean adopting a method with a documented collapse
mode, into a project whose entire failure history is collapse.

### 3.2 Nobody replaces the policy gradient; everyone reshapes it

This is the field's answer to the design question the original paper leaves
open -- where does `f` attach in a multi-turn trajectory. The shared recipe:
aggregate token-level teacher-student log-prob gaps into **turn-level**
evidence, then use that as a **bounded, sign-preserving modulation of the
GRPO advantage you already trust**.

GRSD states it most cleanly:

```
A_hat[i,t] = A[i] * (1 + lambda * q_hat[i,t] * m(x))     lambda = 0.5
```

with `q_hat` the bounded turn-level gap clipped to `[-G, G]`, `m(x)` a mask
that fires only on groups containing **both** successes and failures, and the
constraint `lambda * G <= 1` guaranteeing that modulation **cannot flip the
advantage's sign**. The others are variations on the same conservatism:

- **SDAR**: gap-based sigmoid gate, `g_t = sigmoid(beta * Delta_t)` with
  `beta = 5.0`, distillation kept as an auxiliary term (`lambda = 0.01`) with
  RL as the primary backbone. Explicitly motivated by the >50%-negative-gap
  problem -- the gate attenuates teacher rejections.
- **ADRS**: within-step centering and normalisation of the raw teacher
  log-probs (they drift across interaction steps), then a "Teacher Value
  Advantage" gate estimating teacher reliability from the association between
  privileged confidence and realised return, then the gated signal is added
  to token **rewards before advantage construction** -- keeping it on the
  native reward -> advantage -> policy path rather than as a side objective.
- **AgentOPSD**: goes furthest. Sums per-token gaps into turn evidence
  `e_k = sum_t delta_k,t`, then runs a **recursive Bayesian belief update in
  log-odds space** over the turns, and identifies *pivotal* turns by marginal
  belief revision between consecutive states. Critic-free, no extra rollouts.
  Ablating the recursion costs 89.1 -> 82.8 on ALFWorld; using magnitude-only
  credit without outcome-sign alignment costs 80.5.
- **SERL-SQL**: bounded reweighting applied **only to executable SQL and
  tool-action tokens** (reasoning and formatting tokens keep plain GRPO
  weights), sign-aligned to the trajectory advantage, with a coefficient that
  **decays** from hindsight-dependent to purely reward-driven over training.

Sample results at our model scale (ALFWorld success %, Qwen2.5-3B): vanilla
21.9, GRPO 75.0, SDAR 84.4, AgentOPSD 84.4, ADRS 94.5. Search-QA accuracy
(3B): GRPO 36.4, SDAR 43.4, AgentOPSD 46.7, ADRS 45.0.

### 3.3 The privileged context is usually *not* the gold answer

In SDAR, AgentOPSD, Skill-SD and ADRS the teacher's privileged channel `c+`
is a **compact natural-language "skill" summarised from the agent's own past
trajectories** -- successful behaviours, recurring mistakes, workflows --
retrieved per task and injected into the teacher context only. Skill-SD gives
the reason: fixed ground-truth privileged information "cannot capture the
diverse valid strategies in agent tasks."

This is roadmap item 3 (`notes_from_traces`), whose plumbing already exists
in `KgConfig.notes_file` / `core.load_notes`. Four independent papers use it
as the teacher's privileged channel. That convergence is worth more than any
single number in this document.

GRSD is the variant that needs no retrieval infrastructure at all: it
generates the guidance on the fly by having the policy reflect on its own
verified rollouts within the current on-policy group, and contrasts
reflections from successful vs. failed rollouts under a stop-gradient
snapshot.

### 3.4 The model-scale objection is weaker than the original paper implies

The original limitation reads "needs stronger base models." But the agentic
follow-ups get their *largest* gains at exactly our scale: AgentOPSD at
Qwen2.5-3B takes ALFWorld 75.0 -> 84.4 and Search-QA 36.4 -> 46.7; ADRS at 3B
reports ALFWorld 75.0 -> 94.5; SDAR's single biggest gain is WebShop-Acc
+20.3% at Qwen3-**1.7B**.

Plausible reason: they never ask the model to diagnose its own failure from
scratch. They hand it a skill note or an aligned gold action, and "score
these tokens given this hint" is a far easier in-context task than "work out
what you did wrong."

Consequence for us: the precondition to test is **not** "can our student
recognise an empty result as a failure" (the judgment probe answered that:
0/18). It is "does conditioning on the reference query, or on a skill note,
shift the model's log-probs over its own tokens in a way that correlates with
correctness." Different question, easier, and still cheap -- single-turn, no
training.

## 4. Two papers in our task family

### 4.1 GAPD -- agentic KBQA on Freebase (arXiv 2605.29584)

The closest existing work. GrailQA / WebQSP / GraphQuestions, S-expressions
translated to SPARQL and executed against the KB, Llama-3.1-8B student. The
privileged context is the **gold logical form**, parsed into an action
sequence and executed to collect intermediate states and entity sets.

The contribution is solving the alignment problem: when the student's
trajectory diverges from the reference, which turn does which gold action
attach to? Their answer, *Entity-Anchor Matching*, is Jaccard similarity over
the intermediate **entity sets** reached by student vs. gold execution:

```
sim(s_t, u_m) = |E(s_t) ^ E(u_m)| / |E(s_t) v E(u_m)|
```

Pick the max-similarity gold state, condition the policy on that action via a
special prompt, `stopgrad` the teacher. Their ablation shows the alignment is
the load-bearing part: replacing Entity-Anchor Matching with naive
turn-index alignment drops WebQSP from 83.4 to 79.3 -- *below* the baseline.
Supervising only the first turn recovers most but not all of the gain.

Results vs. their KBQA-R1 baseline: GrailQA EM 83.9 -> 87.8, F1 86.1 -> 89.5;
WebQSP F1 83.4 -> 86.1; GraphQuestions F1 53.8 -> 55.1. Training ~1.5x slower
than outcome-reward-only RL; inference unaffected and *more* efficient (2.68
vs 2.91 interactions per query). No code release mentioned.

Note on our no-KG-specific-optimisations rule: entity-set Jaccard between a
partial student query and a reference prefix is completely KG-agnostic, and
we already compute result-set overlap for f1. This mechanism can be adopted
without importing any Freebase-shaped assumption.

Note on their absolute numbers: an 83.4 F1 WebQSP baseline is far above ours
and reflects a different setup (gold S-expressions available, fixed dataset,
different entity-linking assumptions). The mechanism transfers; the numbers
are not comparable to ours.

### 4.2 SERL-SQL -- multi-turn text-to-SQL (arXiv 2608.00485)

The closest analogue to our *reward structure*, and it makes the opposite
choice on privileged information: the teacher sees **execution hindsight
only** -- "the returned result, runtime or parser error, empty-result
feedback, and other execution diagnostics" -- and explicitly **not** the gold
SQL. A synchronised teacher re-scores student tokens under that hindsight;
the gap becomes a bounded reweighting factor.

Its three mechanisms are listed in 3.2 above (action masking, sign-aware
reweighting, training decay). Benchmarks BIRD / Spider / Spider-DK /
Spider2.0-SQLite on Qwen2.5-Coder 7B/14B/32B; BIRD-Dev 76.56 (32B) against a
77.84 oracle best-of-N upper bound. Code at `github.com/Ffunkytao/SERL-SQL`.

Two things to take from it. First, they feed **empty-result feedback** as
teacher context -- precisely the signal our judgment probe showed the model
failing to interpret unaided. Second, their residual failure modes are ours:
"join construction, semantic misunderstanding, value filtering, and schema
linking" cause the majority of failures, i.e. execution feedback alone does
not fix the underlying reasoning gap.

## 5. Reference implementation

TRL ships an official experimental trainer, `trl.experimental.sdpo`
(v1.10.0), documented at `huggingface.co/docs/trl/sdpo_trainer`. Our trainer
is custom so we would not adopt it, but the templates and defaults are
directly liftable:

- `distillation_weight` -- convex blend, `loss = (1-w)*policy + w*distill`.
  `1.0` = pure SDPO (default), `0.0` = plain GRPO. This is SDPO+GRPO from the
  paper's section 4.5, exposed as one knob.
- `distillation_mode` in `{sampled_token, full_logits, topk_logits}` with
  `distillation_topk`; `distillation_is_clip = 2.0` for IS clipping on the
  distillation term; `use_liger_kernel` for a fused JSD.
- `teacher_model_kind` in `{ema, live, base}`, `teacher_update_rate = 0.05`,
  `teacher_sync_steps = 1`.
- `reprompt_template` / `solution_template` / `feedback_template` as config
  strings, matching the paper's Table 2 verbatim; `privileged_context` as a
  dataset column plus `include_environment_feedback`;
  `environment_feedback_only_without_solution`;
  `dont_reprompt_on_self_success = True` (note: this is SRPO's fix, already
  the default here).
- `use_teacher_server` with `teacher_model_kind="live"` scores teacher
  log-probs on the running vLLM server, removing the teacher forward from
  the training step entirely.
- PEFT/LoRA supported; `loss_type` defaults to `dapo`.

Incidentally its generation defaults are `top_k = 0` and `top_p = 1.0` --
independent agreement with the rollout-sampling change we had already decided
on for our own runs.

## 6. What we would build, if we build it

Not vanilla SDPO. The modulation form, which is what every agentic paper
converged on and is the only version whose failure mode is bounded:

1. Aggregate per-token teacher-student log-prob gaps into turn-level
   evidence, as in AgentOPSD's `e_k`.
2. Apply it as a bounded, sign-preserving multiplier on the existing GRPO
   advantage (GRSD's form), gated to mixed-outcome groups. By construction it
   cannot invert a reward we trust -- which matters given our history.
3. Mask it to the tokens that matter -- SPARQL and tool-call tokens, not
   reasoning prose (SERL-SQL's action masking).
4. Decay it toward pure reward over training (SERL-SQL), and use an EMA
   teacher (original paper, and the TRL default).

Ingredients we already have: reference SPARQL (GAPD's gold action) and
execution diagnostics (parse errors, empty results, result rows, tool
outputs -- SERL-SQL's hindsight). The extra forward is the same shape as the
KL-to-reference forward `train_step` already pays for (same model, different
prompt instead of different weights).

The missing ingredient is **alignment**: which turn gets which gold action.
GAPD shows naive turn-index alignment is worse than no supervision at all,
and that result-set Jaccard fixes it. We already compute result-set overlap.

Cheapest first step is still a probe, not a run: measure the privileged
log-prob gap (section 3.4) rather than unaided self-diagnosis. SERL-SQL's
diagnostics-only variant is the cleaner starting point since it never touches
the reference.

And the standing blocker is unchanged: none of this is detectable at +/-0.09
on a 38-item holdout. One thing worth stealing regardless of whether we build
any of the above is the **`discovery@k` metric** -- for us, "P(at least one
of k rollouts reaches f1 >= 0.9)". It targets the tail we actually care about,
it is far better powered than mean f1 over 38 items, and it is what a
curriculum needs to know anyway.

## 7. References

Original and core:

- Reinforcement Learning via Self-Distillation (SDPO) -- Huebotter et al.,
  ETH Zurich. `[read in full]`
  https://arxiv.org/abs/2601.20802
- SRPO: Unifying Group-Relative and Self-Distillation Policy Optimization via
  Sample Routing -- reports SDPO collapse under prolonged training.
  `[fetched]` https://arxiv.org/abs/2604.02288
- Denser != Better: Limits of On-Policy Self-Distillation for Continual
  Post-Training `[snippet]` https://arxiv.org/abs/2607.01763
- Restoring the Sweet Spot: Pass-Rate Weighted Self-Distillation for LLM
  Reasoning `[snippet]` https://arxiv.org/abs/2605.27765
- Physics-Guided Policy Optimization with Self-Distillation -- addresses
  per-batch trust in the self-teacher. `[snippet]`
  https://arxiv.org/abs/2606.03620
- Self-Distilled RLVR `[snippet]` https://arxiv.org/abs/2604.03128

Agentic / multi-turn:

- SDAR: Self-Distilled Agentic Reinforcement Learning -- gated auxiliary
  objective; diagnoses multi-turn instability. `[fetched]`
  https://arxiv.org/abs/2605.15155
- AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning
  -- turn-level Bayesian credit. `[fetched]`
  https://arxiv.org/abs/2608.05987
- GRSD: Group-Reflective Self-Distillation for Agentic Reinforcement Learning
  -- advantage modulation, no retrieval needed. `[fetched]`
  https://arxiv.org/abs/2607.28076
- ADRS: Agentic Reinforcement Learning with Self-Distilled Reward Shaping
  -- shaping before advantage construction. `[fetched]`
  https://arxiv.org/abs/2608.03223
- Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents
  -- AppWorld / Sokoban; importance-weighted reverse KL. `[fetched]`
  https://arxiv.org/abs/2604.10674
- PBSD: Privileged Bayesian Self-Distillation for Long-Horizon Credit
  Assignment `[fetched]` https://arxiv.org/abs/2606.09348
- OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
  `[snippet]` https://arxiv.org/abs/2606.26790
- SIRI: Self-Internalizing Reinforcement Learning with Intrinsic Skills for
  LLM Agent Training `[snippet]` https://arxiv.org/abs/2606.02355
- Agent Reinforcement Learning via Pivotal-Aware Self-Feedback Retry
  `[snippet]` https://arxiv.org/abs/2607.03702
- Self-Distilled Reinforcement Learning for Co-Evolving Agentic Recommender
  Systems (CoARS) `[snippet]` https://arxiv.org/abs/2604.10029

Our task family:

- GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in
  Knowledge Base Question Answering -- Freebase, GrailQA/WebQSP/GraphQ,
  S-expression -> SPARQL. `[fetched]` https://arxiv.org/abs/2605.29584
- SERL-SQL: Selective Hindsight Distillation for Text-to-SQL Reinforcement
  Agentic Learning `[fetched]` https://arxiv.org/abs/2608.00485

Other self-distillation variants (not read):

- Contrastive Reinforced Policy Optimization via Privileged Self-Distillation
  `[snippet]` https://arxiv.org/abs/2607.28026
- DemoPSD: Disagreement-Modulated Policy Self-Distillation `[snippet]`
  https://arxiv.org/abs/2607.02502
- Skill-Conditioned Gated Self-Distillation for LLM Reasoning `[snippet]`
  https://arxiv.org/abs/2605.28791
- Self-Supervised On-Policy Distillation for Reasoning Language Models
  `[snippet]` https://arxiv.org/abs/2605.17497
- AVSD: Adaptive-View Self-Distillation by Balancing Consensus and
  Teacher-Specific Privileged Signals `[snippet]`
  https://arxiv.org/abs/2605.20643
- MIRA: Memory-Integrated Reinforcement Learning Agent with Limited LLM
  Guidance `[snippet]` https://arxiv.org/abs/2602.17930
- Policy Improvement Reinforcement Learning `[snippet]`
  https://arxiv.org/abs/2604.00860

Non-arXiv:

- On-Policy Self-Distillation for Reinforcement Learning in LLM
  Post-Training: A Unified Framework and Survey (July 2026) -- no arXiv id
  captured; found via ResearchGate. `[snippet]`
- TRL SDPO trainer docs -- https://huggingface.co/docs/trl/sdpo_trainer
- awesome-on-policy-distillation --
  https://github.com/chrisliu298/awesome-on-policy-distillation

Related in-repo docs: `07-roadmap.md` (item 3, notes-from-traces),
`11-process-supervision.md` (turn-level credit assignment, the same problem
attacked with process rewards), `12-ask-for-help-sft.md` (section 2, the
reference-access argument this literature revises).
