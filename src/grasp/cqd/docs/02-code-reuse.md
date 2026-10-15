# What GRASP already provides for CQD

Map of existing code that CQD reuses, with pointers. Paths relative to
`src/grasp/`.

## The agent loop — reuse as-is for both agents

`core.generate(task_name, input, config, managers, ..., custom_model=None)`
(`core.py:132`) is the single agent loop used by every GRASP entry point.
Both CQD agents are just `generate` calls with different configs:

- **Teacher** = `generate(...)` with a strong API model and a (new)
  proposer task.
- **Student** = `generate("sparql-qa", question, student_config, ...,
  custom_model=student_model)` — unchanged `sparql-qa` task, so the trained
  model remains a drop-in GRASP model.

Useful properties of the loop for CQD:

- Function-call errors never crash a run: they are caught and stored as
  `tool_call.error` + fed back to the model as text (`core.py:363`), so the
  trace contains exactly the error events the reward needs.
- The returned output dict `{type, task, output, elapsed, error, messages[],
  known[]}` (`core.py:464`) is a fully serialized episode: `messages` is the
  whole conversation with per-call `elapsed`, `error`, `args`, `result`.
- `max_steps` / `max_loops` bound episode length (`configs.py:GraspConfig`).

## Tasks (`tasks/`)

| Existing task | Role in CQD |
|---|---|
| `sparql-qa` (`tasks/sparql_qa/`) | The student's task, unchanged. `output()` executes the final query and returns `{sparql, kg, answer, result, ...}` — the `sparql` field is what the reward compares against the reference. |
| `sparql-to-question` (`tasks/sparql_to_question.py`) | **Pattern reference only** (not run in the loop): `setup()` executes the input SPARQL, `answer(sparql, questions[])` returns cleaned SPARQL + NL questions — the execute-and-verify submission pattern the cq-distillation task copies. Question phrasing itself happens inside cq-distillation. |
| `question-generation` (`tasks/question_generation/`) | Teacher primitive for fan-out: generates diverse questions across KGs via `submit_question` / `find_frequent` tools. Currently produces questions *without* reference SPARQL — CQD's proposer must close that gap. |
| `notes_from_traces` (`tasks/notes_from_traces/`) | `format_output` (line 65) renders a serialized trace into a readable step-by-step transcript — reuse it to show the teacher what the student did (curriculum feedback). |
| `general-qa`, `cea`, ... | Not needed. |

Task plumbing to imitate: registry in `tasks/__init__.py` (`_REGISTRY`,
`get_task`), base class `tasks/base.py:GraspTask` (implement
`system_information`, `rules`, `function_definitions`, `call_function`,
`done`, `output`).

## Reward ingredients — all already implemented

- **Execution F1**: `sparql/metrics.py:f1_score(pred, target, exact=1024)`
  (line 55). ASK → boolean equality; both-empty → 1.0; one-empty → 0.0;
  small results scored with Hungarian-assignment F1 (order-insensitive,
  `assignment_f1_score`), large with multiset F1 (`exact_f1_score`). This is
  the same metric as `grasp evaluate f1`.
- **Query execution**: `evaluate.py:get_result_or_error(sparql, endpoint,
  timeout, sparql_result_max_rows)` (line 45) → `(result, err)`; lower-level
  `sparql/utils.py:execute` (line 949) with streaming + timeouts, raising
  `SPARQLExecuteException` with client/server error distinction.
- **Function-call errors**: count `tool_call.error is not None` over the
  trace messages.
- **Step count / latency**: `len(messages)` and per-call/episode `elapsed`,
  both in the trace dict.
- **Validity checks**: `sparql/utils.py` grammar utilities
  (`load_sparql_parser`, `fix_prefixes`, `query_type`, `ask_to_select`,
  `prettify`) for syntactic reward terms and query normalization.
- Retry/validity heuristics: `utils.py:is_server_error`,
  `is_retryable_evaluation`, `is_invalid_output`.

## Data structures

- **`Sample` / `SparqlQaSample`** (`examples/__init__.py:22`,
  `tasks/sparql_qa/examples.py:19`): `{id, question, sparql, paraphrases[],
  info{}}` — the canonical (question, reference-SPARQL) pair. CQD tasks
  extend this (difficulty, lineage, tags go into `info` or a subclass).
  Loaded by `evaluate.load_inputs`, indexed by `ExampleIndex` (`grasp
  examples`), so CQD-generated pools are automatically compatible with
  evaluation and few-shot example indices.
- **`SelectResult` / `AskResult` / `Binding`** (`sparql/types.py`): execution
  results consumed by `f1_score`.
- **`Message` / `Response` / `ToolCall`** (`model/base.py`): the trace.
  `Response` already carries `prompt_token_ids`, `token_ids`,
  `token_logprobs`, `token_texts` (base.py:49-52) — the hook for PPO.

## Model layer (`model/`)

- `get_model(config)` dispatches to `openai/responses`, `openai/completions`,
  `anthropic`. **Student**: `openai/completions` against a vLLM server
  (`ModelConfig.model_endpoint`); `OpenAICompletionsModel.call`
  (`model/openai.py:181-215`) already extracts logprobs and vLLM's
  `token_ids`/`prompt_token_ids` from `provider_specific_fields`. Caveat:
  logprobs/token-ids must be requested via `model_kwargs`
  (`logprobs: true`, vLLM `return_token_ids`) — not on by default.
- `coerce_tool_call_args` (openai.py) papers over small-model tool-call
  quirks — relevant since the student is small.
- `custom_model=` parameter of `core.generate` lets the trainer inject a
  model object (e.g. one pinned to the latest policy checkpoint / a specific
  vLLM deployment) without touching config files.

## KG access (`manager/`, `functions.py`)

- `KgManager` (`manager/__init__.py:86`): endpoint, prefixes, search indices
  (entities/properties/literals/shapes), `execute_sparql`, result formatting.
  Loaded via `load_kg_manager(KgConfig)`; multi-KG fan-out is native
  (`GraspConfig.knowledge_graphs` is a list, `managers` everywhere a list).
- Agent tools built by `functions.py:kg_functions` (fn sets `base` ...
  `search_extended`); dispatch via `call_function`. CQD adds no new KG tools.

## Config & CLI

- Pydantic configs in `configs.py`; YAML with `env(VAR:default)`
  interpolation. CQD adds a `CqdConfig` following `NotesConfig` /
  `JudgeConfig` patterns (`configs.py:167-216`). CLI exposure only after
  explicit sign-off (see CLI policy in 03-implementation.md).
- Existing eval CLI (`grasp evaluate f1`) works unchanged on student
  checkpoints for held-out benchmarks.

## Training infra precedent

- **No RL code exists in the repo.** Closest precedent:
  `baselines/grisp/train.py` — HF `Trainer` subclass with custom loss, LoRA
  (`peft`), bf16, W&B, chat templates, resumable sampling. Useful as the
  template for repo conventions (config style, checkpointing, W&B), but CQD's
  RL trainer will live on top of an external RL framework rather than extend
  GRISP's supervised trainer (see 03-implementation).
- Optional-dependency pattern to copy: `pyproject.toml` `[project.optional-dependencies]`
  (`grisp = [transformers, peft, wandb]`) → add a `cqd = [...]` extra so the
  core package stays lightweight.

## Gaps CQD must fill (preview of 03-implementation)

1. The **cq-distillation task** producing verified `(question, reference
   SPARQL)` pairs with `cq_id` backlinks — phrasing, reformulation
   (simpler/harder from student traces), diversity via a recent-proposals
   list, and execute-verified submission in one task.
2. A **task pool** with lineage/difficulty/coverage bookkeeping and
   learnability-based sampling.
3. **Reward assembly** from trace + F1 (thin new code over existing pieces).
4. **Rollout collection** at RL batch sizes (parallel `generate` calls
   against vLLM) and conversion of traces into token-level PPO training
   items with correct loss masking.
5. The **PPO trainer** itself (external framework) and the
   **teacher-student orchestration loop**.
