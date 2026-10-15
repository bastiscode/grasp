# Related work

Result of a literature scan (2026-07-17). Grouped by which component of CQD
the work relates to. **Bold** = position against these first.

## RL for SPARQL / KGQA (the student, in isolation)

- **Learning to Refine: An Agentic RL Approach for Iterative SPARQL Query
  Construction** (arXiv:2511.11770, Nov 2025). 3B model, GRPO from scratch
  (no SFT), think-act-observe loop executing SPARQL against the KG and
  refining. This is essentially CQD's student — but trained on *fixed
  existing data*: no teacher, no generated curriculum. The closest single
  paper; the CQD contribution must be framed relative to it.
- Text-to-SPARQL with GRPO on DBLP (arXiv:2605.20066, May 2026): GRPO on
  Qwen3-1.7B, execution/structural/answer rewards; single-turn, benchmark
  data. Confirms execution-based rewards carry most of the gain.
- KG-R1 (arXiv:2509.26383): RL-trained agentic KG-RAG (retrieval actions,
  not SPARQL generation), transferable across KGs.
- SQL analogues of the reward design: Reasoning-SQL (arXiv:2503.23157),
  Arctic-Text2SQL-R1 (arXiv:2505.20315), CogniSQL-R1-Zero
  (arXiv:2507.06013) — GRPO with execution + partial rewards.
- RLEF (arXiv:2410.02089, ICML 2025): the general grounding-in-execution-
  feedback RL recipe (code domain).

## Proposer-solver / auto-curriculum (the teacher-student loop)

- **Absolute Zero (AZR)** (arXiv:2505.03335): one model self-plays proposer +
  solver on code tasks verified by a Python executor, learnability reward
  for the proposer. CQD = AZR with the executor replaced by a SPARQL
  endpoint, and the proposer/solver split into asymmetric large/small
  models, seeded by CQs instead of zero data.
- **R-Zero** (arXiv:2508.05004, ICLR 2026): Challenger/Solver co-evolution,
  challenger rewarded for tasks at the solver's ability edge. Same-size
  models, no executor ground truth (majority-vote pseudo-labels) — CQD's
  execution-verified references are a stronger label source.
- **Self-Challenging Language Model Agents** (arXiv:2506.01716): agent
  explores tools to generate tasks-with-verifiers ("Code-as-Task"), then RL
  as executor. Closest *agentic* task-self-generation work; not KG/SPARQL,
  self- rather than teacher-generated.
- Classic RL precedents: asymmetric self-play (arXiv:2101.04882), PAIRED /
  unsupervised environment design (arXiv:2012.02096) — regret/difficulty-
  adaptive task generation.
- **Prioritized Level Replay** (PLR, Jiang et al., arXiv:2010.03934, ICML
  2021): a buffer of environment levels sampled by estimated learning
  potential — the direct precedent for the CQD task pool with
  learnability-based sampling (task replay, not experience replay: PPO
  needs fresh on-policy rollouts per sampled item). **ACCEL**
  (Parker-Holder et al., arXiv:2203.01302, 2022) extends PLR by *mutating*
  levels from the buffer to track the frontier of agent ability —
  structurally the same move as the teacher reformulating pool pairs
  simpler/harder; CQD replaces random mutations with an informed LLM
  teacher and a grounded KG environment.
- Montessori-Instruct (arXiv:2410.14208, ICLR 2025): teacher optimized (via
  data influence on a specific student) to generate SFT data — the
  "teacher adapts to student learning signal" idea, but SFT-only.
- SwS (arXiv:2506.08989, NeurIPS 2025), QueST (arXiv:2510.17715): synthesize
  problems targeting the solver's observed weaknesses during RLVR (math).

## Query-to-question backtranslation / synthetic KGQA data

- **FlexKBQA** (arXiv:2308.12060, AAAI 2024): sample SPARQL from the KB →
  LLM backtranslation to questions → train a small model + execution-guided
  self-training. The strongest prior for the fan-out+distill pipeline;
  static sampling, no difficulty adaptation, no RL.
- OmniSQL (arXiv:2503.02240), SING-SQL (arXiv:2509.25672): scaled SQL
  synthesis with *static* complexity knobs (vs. CQD's adaptive difficulty).
- SENSE (arXiv:2408.03256, ACL 2024): strong/weak-LLM synthetic SQL data +
  preference learning.
- PGDA-KGQA (arXiv:2506.09414): prompt-guided (question, SPARQL) pair
  augmentation for KGQA, SFT-only.
- Q-NL Verifier (arXiv:2503.01385) + Wikidata Workshop 2025 (synthetic pairs
  + verifier filtering, 43%→89%): verification of query→question
  translations matters a lot — supports CQD's verification-heavy proposer.
- Self-Instruct (arXiv:2212.10560): the canonical seed-then-expand pattern.

## Competency questions (the seeds)

- RevOnt (Journal of Web Semantics 2024): reverse-engineer CQs from KG
  triples via LLMs (Wikidata). CQ generation exists — but as an ontology-
  engineering artifact, never as QA-agent training scaffold.
- Bench4KE (arXiv:2505.24554): benchmark for automated CQ generation.
- RAG-based CQ generation (arXiv:2409.08820); "Can LLMs Generate Competency
  Questions?" (Springer 2024/25); VSPO (arXiv:2511.07991).
- **Gap: no found work connects CQs to KGQA training data generation or
  agent training.** This is CQD's most clearly unclaimed angle.

## Agent distillation / tool-use RL (training the small agent)

- Agent Distillation (arXiv:2505.17612): distill agentic
  thought-action-observation behavior into small models — via SFT on teacher
  *traces*. Key contrast: CQD's teacher makes *tasks*, not traces (student
  learns on-policy). Also: From Correction to Mastery (arXiv:2509.14257),
  SOD step-wise on-policy distillation (arXiv:2605.07725), AgentTuning
  (arXiv:2310.12823).
- Search-R1 (arXiv:2503.09516): PPO/GRPO for a multi-turn tool-calling agent
  with retrieved-token loss masking — the training-infrastructure template
  (masking!) for CQD's student.
- ToRL (arXiv:2503.23383): tool-integrated RL, emergent self-regulation of
  ineffective tool calls — relevant to the error/step reward terms.

## Curriculum RL for LLM reasoning

Online difficulty filtering (arXiv:2504.03380), competence-difficulty
alignment (arXiv:2505.17652), CLPO (arXiv:2509.25004), AdaCuRL
(arXiv:2511.09478), ADCL (arXiv:2505.08364): all *select/filter from a fixed
pool* by observed success rate; CQD additionally *generates* new items at the
target difficulty. Use their sampling strategies for the pool-sampling part.

## Novelty assessment

Each ingredient exists separately:
- RL + execution rewards for (agentic) SPARQL: taken (2511.11770,
  2605.20066).
- Backtranslation + small-model distillation for KBQA: taken (FlexKBQA).
- Difficulty-adaptive proposer-solver loops: taken in math/code (AZR,
  R-Zero, Self-Challenging Agents).

The combination that appears unclaimed:
1. **Competency questions as seed + coverage scaffold** for KGQA training
   data (CQ literature is disconnected from QA training entirely).
2. **Teacher-generated, student-adapted curriculum feeding an RL-trained
   (not SFT-distilled) tool-using SPARQL agent** — SPARQL-RL work uses fixed
   benchmarks; challenger-solver work is same-model self-play, not grounded
   in a KG.
3. **KG endpoint as the verifiable environment** in an AZR/R-Zero-style
   loop, with execution F1 against a teacher-verified reference query as
   the verifier.

Framing advice: don't sell "RL for SPARQL agents" (occupied). Sell
CQ-grounded coverage + asymmetric teacher-student curriculum + KG-as-
verifiable-environment, with 2511.11770-style training as the baseline the
curriculum improves upon. The space moves fast — re-run this scan before
writing the paper.
