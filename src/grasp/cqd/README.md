# CQD — Competency Question Distillation

Training process for small SPARQL QA agents: a large teacher GRASP agent,
seeded with competency questions for a KG, proposes verified
(question, reference SPARQL) tasks with student-adaptive difficulty
(auto-curriculum) and KG-wide fan-out; a small student GRASP agent solves
them and is trained with RL (PPO) on execution F1 against the reference,
function-call errors, and step count.

Status: design phase — no implementation yet.

## Docs

| File | Content |
|---|---|
| [docs/01-idea.md](docs/01-idea.md) | The idea: teacher/student roles, training loop, design decisions, failure modes, success criteria |
| [docs/02-code-reuse.md](docs/02-code-reuse.md) | What existing GRASP code is reused (agent loop, tasks, F1 metrics, model layer, managers) and the gaps |
| [docs/03-implementation.md](docs/03-implementation.md) | Module layout and phased implementation plan (seeds → proposer → rollouts/reward → SFT baseline → PPO → closed loop) |
| [docs/04-related-work.md](docs/04-related-work.md) | Literature scan: closest prior work and novelty assessment |
| [docs/05-seed-data.md](docs/05-seed-data.md) | Competency question seed sources (QLever examples API) and filtering |
| [docs/06-curriculum-sampling-redesign.md](docs/06-curriculum-sampling-redesign.md) | Curriculum sampling from f1 and the give-up reward fix (EWMA competence/learnability data model) |
| [docs/07-roadmap.md](docs/07-roadmap.md) | Future directions: fixed-dataset baseline, in-context hints, notes from traces, entity-linking reward |
| [docs/08-sampler-curriculum-v2.md](docs/08-sampler-curriculum-v2.md) | Sampler + curriculum v2: coverage-paced growth, revisit/retire |
| [docs/09-dapo-techniques.md](docs/09-dapo-techniques.md) | DAPO stabilizers for the GRPO trainer (token-level loss, clip-higher, dynamic sampling) |
| [docs/10-training-setup.md](docs/10-training-setup.md) | **How to set up and start training on a new machine / SLURM cluster** — services, env, vLLM flags, sbatch template, GPU sizing, gotchas |
