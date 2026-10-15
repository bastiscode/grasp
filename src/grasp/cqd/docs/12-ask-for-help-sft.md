# Design: hinted teacher traces (`ask_for_help`) as an SFT corpus

Status: PROPOSAL, NOT DECIDED (2026-08-07). Nothing implemented. Written up so
the reasoning survives; revisit before building.

Motivation: GRPO cannot reinforce a behaviour it never samples. Measured on the
WQSP arms, the student policies contain no productive exploration to reinforce:
`execute` collapsed from 22/60 to 2/60 episodes over training, `list` from 15/60
to 1/60 (gemma4) and 8 to 0 (qwen), and the trained checkpoint issues **0/60**
episodes with more than one distinct executed query. Raising rollout temperature
recovers the *tool call* in the base model (`list` 1/8 -> 7/8 at temp 1.3) but
not in the trained one (0/8 at every temperature), and in neither case does it
recover a *correct answer*. So the behaviour has to be put into the policy by
supervision before RL can refine it.

## 1. Why not plain rejection sampling

The obvious recipe is STaR-style: run the teacher unaided, keep only traces that
verify against the reference, discard the rest. That works when the teacher
solves most items.

It does not fit CQD. The curriculum deliberately pushes toward items the teacher
itself struggles with, so "discard the failures" discards exactly the
distribution we care about and yield collapses where the signal is. Rejection
sampling is the right answer for a fixed benchmark whose items the teacher can
already do; it is the wrong answer for the method.

## 2. Why not let the teacher see the reference

A teacher holding the reference query does not search badly, notice, and
recover -- and that recovery is the behaviour we are trying to install. Its
exploration becomes theatre, and the trace teaches the student to go straight to
a confident property choice. That is the blind-answering pathology already
measured elsewhere in these runs, installed deliberately.

## 3. The proposal

Give the teacher an `ask_for_help` function during generation only. When called,
a helper *that does* hold the reference replies with a hint. The teacher
continues, and the help is removed from the trace afterwards.

The hint must be **strategy, never content**. A hint naming the relation leaks
the answer and yields a trace the student cannot reproduce; a hint about how to
look yields a trace where every action is one the model could plausibly have
chosen unaided, so removing the hint leaves nothing unexplained.

Enforce this with a **closed hint vocabulary** rather than free-form help --
auditable, and strategy-level by construction. Free-form help with the reference
in context will leak identifiers eventually. Candidate hints, all KG-agnostic:

- enumerate the properties of an entity you already hold
- the relation you need may not be the top search hit, scan the whole page
- try the inverse direction
- if an object carries no label of its own, inspect it before using it
- constrain the search with an entity you have already found

## 4. Rewriting the trace

Verified against the code (`train/data.py:68` renders `episode.functions` into
`tools=` for `apply_chat_template`; the system prompt does *not* list functions,
they reach the model only via the API `tools` parameter):

1. drop the `ask_for_help` tool call and its result from the assistant message
2. drop the turn entirely if that was its only call, or merge its thought forward
3. rewrite the following thought so the next action is self-motivated -- done by
   the teacher itself, given a fixed and already-verified action sequence, so it
   can only make the prose worse, never the actions wrong
4. remove `ask_for_help` from `episode.functions`; miss this and every SFT
   example renders a tool the student is never offered
5. recompute `steps` and `fn_errors`, both of which feed the reward
6. re-verify

## 5. Verification, mechanically

**Answer correctness** comes free: `SftConfig.min_f1` defaults to 1.0, so only
traces reproducing the reference train.

**Non-leakage is checkable, not a matter of discipline.** Require that every IRI
appearing in a rewritten thought or in the final query was already present in
some earlier tool result -- the `know_before_use` invariant applied to the whole
trace. A trace naming an identifier it had not yet seen got it from the helper,
and is dropped.

**Stale token data must be stripped.** Assistant messages carry per-token
logprobs and ids from the rollout. Once the text is edited these no longer
correspond to it. SFT is safe (it re-renders from text) but the same episode
files feed the RL trainer, where stale logprobs would silently corrupt the
importance ratio. Drop `logprobs`/`token_ids` on any edited message, and have
the RL data path refuse episodes missing them, so a rewritten trace can never
reach GRPO by accident.

## 6. Contrast, or it becomes a ritual

The corpus must contain items solved with no help, and items where one hint
sufficed. A corpus of uniformly hinted traces teaches "always ask, then always
enumerate" -- the same failure as the execute-before-answer gate, which produced
compliance during training and reverted the moment the constraint was lifted.

## 7. Relationship to the auto-curriculum

Orthogonal in mechanism, but not independent, and the dependency runs one way.

The curriculum decides *which items* to practise; this decides *which behaviours
are in the policy*. Different axes. But both exist to solve the same underlying
problem -- GRPO needs within-group reward variance or there is no gradient. The
curriculum manufactures variance by moving the item distribution toward the
student; an SFT stage manufactures it by moving the student toward the items.

**The curriculum is capped by the repertoire.** `TaskPool.learnability` is an
EWMA of the within-group reward *standard deviation*, so an item the student
never solves has stdev 0 in every group and its learnability decays to 0. The
ZPD bump floors the weight at ~0.14 rather than zero -- deliberately, so "a
stuck item stays reachable and can recover" -- but recovery needs an occasional
success, and a success needs the behaviour to already exist. Any region of the
item space requiring an absent behaviour therefore parks at the floor and stays
there, with the sampler behaving exactly as designed. Measured example: the
policies execute more than one distinct query in 0/60 episodes, and 38% of the
WQSP pool needs a hop through an unprojected join variable; that whole block is
unreachable by curriculum scheduling alone.

So this is **not a replacement and not merely orthogonal -- it is enabling**. It
raises the ceiling the curriculum operates under. The curriculum is still needed
afterwards: SFT alone reproduces the execute-before-answer result, a behaviour
present under the training condition that disappears once the condition is
lifted. RL under a curriculum is what makes it stick.

**Keep it offline, and resist the online variant.** Leaving `ask_for_help`
available *during* RL and fading it out turns it into a scaffolding curriculum,
fully entangled with the sampler. Two reasons not to:

1. It corrupts the curriculum's own measurements. A helped item looks learnable
   because it was helped, not because the student improved, so competence and
   learnability stop meaning what they are defined to mean. Keeping the sampler
   honest would require tracking help-free competence separately.
2. It destroys attribution. The fixed-dataset baseline exists to measure what
   the auto-curriculum contributes by difference. Fold help into the RL loop and
   a gain can no longer be attributed to the curriculum rather than the
   scaffold.

## 8. Open questions

- **Teacher success rate unaided**, per difficulty band. Decides how much of the
  corpus needs help at all. Not yet measured.
- **Supervised timing.** Even a strategy hint supervises *when* to act: the
  teacher asks exactly when stuck, and the student learns the action at that
  point without having experienced being stuck. Milder than content leakage, and
  arguably what the subsequent RL stage is for, but it is the reason to keep the
  SFT stage small and follow it with RL rather than treat it as the endpoint.
- **Format portability.** The teacher is Qwen3.5-397B; students are Qwen2.5-7B
  (hermes parser) and gemma4-12B (gemma4 parser). Traces need re-serialising per
  student, or SFT teaches tool-call syntax the student's parser cannot emit.
- **Teacher budget.** The UFR endpoint returned `budget_exceeded` after 5
  rollouts of a single question. A few hundred items needs rate-limiting or
  spreading over days.
- **Does it survive RL?** The gate showed a behaviour can be present during
  training and absent once the pressure is removed. Whatever SFT installs must
  be re-measured after the RL stage, not just after SFT.

## 9. Evidence this is worth doing

One teacher rollout on a question every student failed (`who plays edward
scissorhands?`) scored f1 1.0 in 11 steps, and did three things no student did:
filtered `list` by property instead of dumping it (`page 1 of 2` rather than
`page 1 of 78`), scanned past a wrong top search hit to find the right relation
further down the same page, and used a reverse lookup to resolve the
intermediate node. All three are general tool-use behaviours, not
knowledge-graph-specific tricks, which is what makes them worth distilling.
