# Competency Question Distillation (CQD)

## One-paragraph summary

CQD is a learning process for SPARQL-based QA agents. A **large teacher agent**
(a GRASP agent backed by a strong LLM) is seeded with a small set of
high-quality **competency questions (CQs)** for a particular knowledge graph.
The CQs span the peculiarities and specialties one has to know when dealing
with that KG (modeling quirks, qualifier/statement patterns, unit handling,
class hierarchies, naming conventions, ...). From these seeds, the teacher
proposes natural-language questions together with **reference SPARQL queries**.
A **small student agent** (a GRASP agent backed by a small, trainable LLM)
tries to solve the proposed questions. Based on the student's traces, the
teacher **adapts the difficulty** of subsequent questions (an auto-curriculum)
and **fans out** across the KG (new domains, new query shapes) to increase
coverage. The student is trained with **RL** (PPO as a starting point) using a
reward computed from execution F1 against the reference SPARQL, function-call
errors, number of steps taken, and similar trace-level signals.

## Why this should work

- **Verifiable reward without human labels.** Execution F1 against a
  teacher-produced reference query is a checkable signal: the teacher only has
  to produce a *correct* query once (and can verify it by executing it), while
  the student has to *find* it. Generating + verifying is easier than solving
  from scratch, which is exactly the asymmetry distillation needs.
- **CQs as an inductive bias.** Random query sampling from a KG produces
  mostly unnatural, uninformative queries. CQs written by KG experts encode
  what actually matters about a KG and what real users ask. Seeding generation
  with them biases the synthetic distribution toward useful skills.
- **Auto-curriculum keeps the learning signal alive.** If questions are too
  hard, execution F1 is ~0 everywhere and PPO gets no gradient signal; too
  easy and the student stops improving. The teacher watching student traces
  and steering difficulty keeps the student in the zone where reward variance
  is highest (success rate strictly between 0 and 1).
- **Trace-level shaping.** Beyond final F1, traces expose dense signals:
  malformed function calls, wasted steps, empty search results, syntactically
  invalid SPARQL. These shape the reward so the student learns *efficient*
  tool use, not just correct final answers.

## The two agents

### Teacher (large, frozen)

Runs as a normal GRASP agent (API model, e.g. GPT-5.x / Claude / large Qwen).
Responsibilities:

1. **Question proposal**: given a CQ (or a previously generated question) and
   KG access via the GRASP tools, produce `(question, reference SPARQL)`
   pairs. The reference query must execute successfully and return a
   non-degenerate result (non-empty, not absurdly large), and the question
   must be answerable *only* via the KG (self-contained, unambiguous).
2. **Verification**: execute the reference query, sanity-check the result,
   optionally solve its own question independently and check consistency.
3. **Curriculum control**: consume student trace summaries + rewards for
   recent episodes and decide, per seed/domain: make questions harder, easier,
   or move on (fan out to a new domain / query feature).
4. **Fan-out**: mutate seeds along explicit axes: other domains/classes of the
   KG, other query shapes (aggregation, filters, qualifiers, paths, ordering,
   negation, unions), other entities of the same type, compositions of seeds.

### Student (small, trainable)

Runs as a GRASP agent against a trainable model served with an
OpenAI-compatible API (vLLM). Starting points: Qwen3-4B/8B-class models. The
student sees only the natural-language question (never the reference query)
and interacts with the KG through the standard GRASP function-calling loop
until it commits to a final SPARQL query (or answer).

## Task interplay

Only three components are actual GRASP tasks (agent runs via
`core.generate`); pool, reward, curriculum controller, and trainer are plain
code around them.

| Task | Status | Run by | Input → Output | Role |
|---|---|---|---|---|
| `sparql-qa` | existing, unchanged | student (every rollout) | question → final SPARQL + trace | The thing being trained; its rollouts double as implicit reference checks (an item ever solved with F1 = 1 confirms its reference) |
| `cq-distillation` | new | teacher | CQ SPARQL + optional (question, SPARQL) pair + optional student trace → verified (question, SPARQL) proposals | The single teacher task: phrases NL questions, reformulates pairs simpler/harder based on the trace, diversifies via a recent-proposals list |

`cq-distillation` covers everything on the teacher side — there is no
separate SPARQL→question step (the teacher phrases the question while
proposing; `sparql-to-question` only serves as a code pattern). Every
proposal stores a backlink `cq_id` to the root competency question it
derives from (plus `parent_id` to its immediate predecessor proposal), and
each submission carries a short *intent* description; the agent can inspect
the list of recent proposals + intents to avoid duplicates and diversify
(same mechanism as `show_questions` in the question-generation task).

`question-generation` (existing) is deliberately not in the loop (no
reference SPARQL → no reward); it serves as the unseeded ablation baseline.

```
                 TEACHER (large, frozen model)
        ┌───────────────────────────────────────────────┐
seeds ──┼──>  [cq-distillation] (NEW)                   │
JSONL   │     explore KG; phrase/reformulate            │
        │     (question, sparql) simpler or harder;     │
        │     submit_proposal verifies by execution;    │
        │     inspects recent proposals + intents       │
        └──────────────┬───────────────────▲────────────┘
                       │ verified proposals│ items to revisit
                       ▼ (cq_id, parent_id)│ + student traces
                 ┌───────────┐      ┌──────┴──────────┐
                 │ TASK POOL │      │ CURRICULUM CTRL │  plain code
                 └─────┬─────┘      └──────▲──────────┘
                       │ sample items      │ success rates per item/lineage
                       ▼                   │
        ┌──────────────────────────────┐   │
        │  STUDENT (small, trainable)  ├───┘
        │  [sparql-qa] — unchanged     │
        └──────────────┬───────────────┘
                       │ trace + final SPARQL
                       ▼
              REWARD (plain code: execution F1 vs. reference,
              tool-error / step penalties)
                       ▼
              PPO TRAINER (plain code / verl) updates student
```

One cycle: controller selects which pool items/lineages to revisit and
attaches student traces → teacher reformulates or proposes fresh (verified)
pairs → pool → student rollouts → reward → PPO step → controller reads
success rates → next selection. Note the difficulty decision lives *in the
teacher agent* (it sees the trace and decides simpler vs. harder); the
controller only decides what gets revisited. Phases 1-3 run the halves
separately; the feedback arrows close in phase 5.

## The training loop (high level)

```
seeds = load_competency_questions(kg)
pool  = TaskPool(seeds)                        # (question, reference sparql, difficulty, lineage)

repeat:
  # 1. teacher generates / adapts tasks
  new_tasks = teacher.propose(pool, feedback)  # generation + verification
  pool.add(new_tasks)

  # 2. student rollouts
  batch = pool.sample(strategy="learnability") # prefer tasks with 0 < success rate < 1
  episodes = [student.run(task.question) for task in batch]

  # 3. reward computation
  for ep in episodes:
      ep.reward = combine(
          execution_f1(ep.final_sparql, task.reference_sparql),
          fn_call_error_penalty(ep.trace),
          step_penalty(ep.trace),
          format/validity terms,
      )

  # 4. PPO update on student policy (token-level, over the whole trace)
  ppo.step(episodes)

  # 5. feedback to teacher
  feedback = summarize(episodes)               # per-task success rates, failure modes
```

## Design decisions to make early

| Decision | Options | Leaning |
|---|---|---|
| RL algorithm | PPO, GRPO, REINFORCE++ | Start PPO per idea; GRPO is the cheaper fallback (no value model) and fits execution-F1 group rewards well |
| RL framework | verl, TRL, OpenRLHF | verl has the best multi-turn/agentic + vLLM rollout support |
| Reward granularity | episode-level only vs. per-step shaping | Episode-level F1 + small per-step penalties; avoid over-shaping |
| Teacher online or offline | generate-then-train (static dataset) vs. interleaved | Phase 1: offline generation (validate data quality + student training separately). Phase 2: interleave for the curriculum |
| Student final output | SPARQL query (sparql-qa task) | Reuse the existing `sparql-qa` task/prompt unchanged so the trained model stays a drop-in GRASP model |
| Reference validity | teacher self-check vs. independent re-solve | Execute + heuristics always; trust the teacher otherwise (independent re-solves are too expensive) and monitor noise via free signals from student rollouts |

## Failure modes to watch

- **Reward hacking on F1**: student learns degenerate queries that
  accidentally overlap results (e.g. `SELECT *` over a small class). Mitigate
  with exact-match bonuses, penalties for over-broad queries, and result-size
  sanity checks in the reward.
- **Teacher label noise**: wrong reference queries poison the reward. Verify
  by execution, filter empty/trivial results; monitor via student-rollout
  agreement (persistent F1 = 0 outliers) and the held-out benchmark, plus a
  one-time manual skim of the first generated pool.
- **Curriculum collapse**: teacher drifts to one domain or one query shape.
  Track coverage metrics (domains, SPARQL features used) and make fan-out
  explicit in the teacher prompt/controller, not emergent.
- **Distribution mismatch**: synthetic questions sound unlike benchmark
  questions. Keep a held-out real benchmark (e.g. QALD) as the actual metric;
  synthetic reward going up while QALD stays flat = distribution problem.
- **PPO instability with tool-use traces**: long multi-turn episodes with
  interleaved environment tokens need correct loss masking (only train on
  assistant tokens). This is a known sharp edge; get the masking right first.

## Success criteria

1. Student (small model) after CQD training beats the same small model
   zero-shot as a GRASP agent on held-out benchmarks for the target KG
   (execution F1, valid-call rate, steps per episode).
2. Student approaches the teacher's performance at a fraction of the cost.
3. Ablations: CQ-seeded generation vs. unseeded generation; curriculum vs.
   fixed difficulty; RL vs. SFT on teacher traces (distillation baseline).
