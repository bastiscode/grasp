# Training setup: moving CQD to a SLURM cluster

How to bring up a CQD training run from nothing on a new machine, in
particular a SLURM cluster. Written after the runs on `tagus` (2× RTX 4090),
so every command and number here comes from a run that actually happened.

Read this top to bottom the first time; the biggest cost is not the code, it
is the **two data services** the training loop talks to (§2).

## 1. What the training loop actually needs

`train_rl` (`cqd/train/rl.py`) is a synchronous loop. Each round it:

1. samples items from the pool, runs the student agent on them
   (`collect_rollouts`) — this drives **a vLLM server** and issues **SPARQL
   queries** and **entity/property search** per agent step;
2. scores the episodes by executing the reference SPARQL — **SPARQL endpoint
   again**;
3. computes GRPO advantages and does a LoRA optimizer step **locally on the
   trainer GPU**;
4. saves the round's adapter and **hot-loads it into the running vLLM
   server** so the next round rolls out with the updated policy;
5. every `curriculum_interval` rounds (if > 0) calls the **teacher LLM**.

So four dependencies: a vLLM server, a SPARQL endpoint per KG, a search
index per KG, and — only for curriculum runs — a teacher endpoint.

## 2. The hard part: the KG services

The code assumes these exist and are reachable. On the Freiburg network they
are already running; **on a cluster they are not**, and nothing in the repo
brings them up for you.

| Service | Where it runs today | Size | Needed for |
|---|---|---|---|
| QLever wikidata | `amur:7001` | — (not on our disk) | wikidata runs |
| QLever freebase | `tagus:7002` (docker `qlever.server.freebase`) | **91 GB** index at `/local/data-ssd/qlever/qlever-indices/indices-sebastian/freebase` | WQSP runs |
| search index wikidata | `data/kg-index/wikidata` → grasp-dev | **55 GB** | wikidata runs |
| search index freebase | `data/kg-index/freebase` → grasp-dev | **17 GB** | WQSP runs |

Three options, best first:

1. **Reach back over the network.** If the cluster can open outbound TCP to
   the institute, just point at `http://amur:7001` / `http://tagus:7002` and
   change nothing else. Test with a real query before committing a job:
   ```
   curl -s -m 15 "http://tagus:7002" \
     --data-urlencode "query=SELECT ?s WHERE { ?s ?p ?o } LIMIT 1" \
     -H "Accept: application/sparql-results+json"
   ```
   Latency matters less than you would think — the agent issues on the order
   of tens of queries per episode, not thousands.
2. **SSH tunnel from the login node**, if compute nodes have no route out
   but the login node does. Fragile across job restarts; use only for smoke
   tests.
3. **Stand QLever up on the cluster.** Copy the index directory and run the
   QLever server there. This is the only option that survives a cluster with
   no outbound network, and it is a ~91 GB transfer for freebase alone.

The **search index** (`GRASP_INDEX_DIR`) is different: it is read directly
off the filesystem by the agent's `search_entities` / `search_properties`
tools, so it must be on a disk the compute node can see. 17 GB (freebase) or
55 GB (wikidata) on fast local scratch if you can; the property index is an
embedding index and gets memory-mapped.

Start with freebase/WQSP if you have a choice: it is 17 GB + 91 GB instead of
55 GB + a wikidata dump, and the fixed-dataset run needs no teacher.

## 3. Code and environment

```bash
git clone <grasp-cqd> && cd grasp-cqd
python -m venv .venv && source .venv/bin/activate   # python >= 3.12
pip install -e ".[cqd]"        # torch, transformers>=5, peft, wandb
pip install vllm               # the student server; not a package dependency
```

Pre-stage the model weights if compute nodes have no internet:

```bash
export HF_HOME=/path/to/shared/hf-cache
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
```

Run the test suite once on the cluster — it is CPU-only, takes ~5 s, and
catches a broken install immediately:

```bash
python -m pytest src/grasp/cqd/test_cqd.py -q     # 61 passed
```

## 4. Environment variables

| Variable | Value | Notes |
|---|---|---|
| `GRASP_INDEX_DIR` | `<repo>/data/kg-index` | entity/property search indices; defaults to `~/.grasp/index` if unset |
| `HF_HOME` | shared cache path | so every job does not re-download |
| `WANDB_API_KEY` | your key | or `WANDB_MODE=offline` and `wandb sync` later |
| `UFR_API_KEY` | **secret**, from your shell env | teacher only; see below |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `1` | **required**, else the per-round adapter hot-load 404s |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | the trainer's logits peak fragments the allocator |

`UFR_API_KEY` is a credential for the university LLM endpoint. Pass it as
`os.environ["UFR_API_KEY"]` in a run script or `env(UFR_API_KEY)` in YAML.
**Never write it into a file, a config, or a SLURM script** — SLURM scripts
end up in job logs and in the scheduler database.

KG endpoints are set per run script (`KgInfo(endpoint=...)` for the agent and
`RewardConfig.endpoints={kg: ...}` for scoring), not via env — see the
scripts in §6. The wikidata scripts read `KG_ENDPOINT` from
`configs/run.yaml`; **if it is unset the manager silently falls back to the
public qlever.dev**, which rejects our token and turns every rollout into an
error. That failure mode cost a whole run once; check the endpoint in the
first rollout's log.

## 5. The vLLM student server

This exact invocation is what the 4B runs used:

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8437 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-lora --max-loras 2 --max-lora-rank 32 \
  --gpu-memory-utilization 0.45          # only when sharing a card, see §7
```

Every flag earns its place:

- `--enable-auto-tool-choice --tool-call-parser hermes` — the student is a
  tool-using agent; without these it emits tool calls as prose and every
  episode scores zero.
- `--enable-lora` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` — the loop POSTs to
  `/v1/load_lora_adapter` after each round (`load_adapter_into_vllm`).
- `--max-lora-rank 32` must be **≥ `RlConfig.lora_r`** (default 32).
- `--max-model-len 32768` must exceed `RlConfig.max_seq_len` (24576) with
  room for the longest single turn. Too small and rollouts die with
  "prompt contains at least N input tokens" 400s.

Wait for readiness before starting the trainer:

```bash
until curl -sf http://localhost:8437/v1/models > /dev/null; do sleep 10; done
```

## 6. Launching a run

Two ready-made scripts live in `data/cqd/` (outside the package, since they
carry absolute paths and run-specific knobs):

- **`run_rl_wqsp_freebase.py`** — fixed-dataset baseline on freebase/WQSP.
  No teacher, `curriculum_interval=0`, frozen 3097-item pool. **Start here**:
  fewest moving parts, no teacher credential, no seed data.
- the curriculum runs (warm-start + teacher + auto-curriculum) follow the
  shape recorded in `07-roadmap.md` and `08-sampler-curriculum-v2.md`; they
  additionally need `UFR_API_KEY`, a seeds file and a wikidata endpoint.

Both are plain Python — run them from the repo root, since they load
`configs/run.yaml` by relative path:

```bash
cd $REPO && python data/cqd/run_rl_wqsp_freebase.py
```

A minimal SLURM script, single GPU, vLLM and trainer sharing the card:

```bash
#!/bin/bash
#SBATCH --job-name=cqd-wqsp
#SBATCH --gres=gpu:1              # an 80 GB+ card; see §7
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$REPO"
source .venv/bin/activate
export GRASP_INDEX_DIR="$REPO/data/kg-index"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8437 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-lora --max-loras 2 --max-lora-rank 32 \
  --gpu-memory-utilization 0.45 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

until curl -sf http://localhost:8437/v1/models > /dev/null; do sleep 10; done

python data/cqd/run_rl_wqsp_freebase.py
```

Notes for the SLURM version specifically:

- **`--time` must cover the whole run.** 20 rounds of the 4B curriculum run
  took **~15 h** on 2× 4090. A fixed-dataset run is comparable. If your
  partition caps below that, lower `num_rounds` and chain jobs by passing the
  previous `output_dir/adapter` as `RlConfig.adapter` — the loop resumes from
  an adapter cleanly.
- **The trap matters.** A vLLM server orphaned by a cancelled job holds the
  GPU until the node is drained.
- If vLLM and the trainer get **separate GPUs** (`--gres=gpu:2`), drop
  `--gpu-memory-utilization` and give vLLM `CUDA_VISIBLE_DEVICES=0`, the
  trainer `CUDA_VISIBLE_DEVICES=1`.
- Set `VLLM_URL` / `FREEBASE_ENDPOINT` / `CQD_OUT` to override the script's
  defaults without editing it.

## 7. GPU sizing

LoRA training of Qwen3-4B, `max_seq_len` 24576, measured or derived from the
tagus runs:

| | |
|---|---|
| vLLM weights (bf16) + CUDA graphs | ~11 GB |
| vLLM KV cache | 144 KB/token — 12.5 GB gave **91k tokens, 2.78× concurrency** at 32k/request |
| trainer: base weights bf16 | 8 GB |
| trainer: LoRA r32 params + grads + AdamW | < 1 GB |
| trainer: activations @ 24k, grad-checkpointed | ~6 GB |
| trainer: logits @ 24k × 151,936 vocab | 7–15 GB, transient peak |

- **2× 24 GB (what we used)** works: one card each. The catch is that 12.5 GB
  of KV caps concurrency at 2.78 full-length episodes, so `parallelism=8`
  spends most of its time queueing.
- **1× 80 GB (H100/A100)** fits both comfortably: ~11 GB vLLM + ~52 GB KV at
  `--gpu-memory-utilization 0.45` + ~30 GB trainer peak.
- **1× 141 GB (H200)** is the pleasant option — same split with ~50 GB spare,
  and KV for ~11 concurrent full-length episodes.
- **Full fine-tuning is a different machine class**: AdamW states alone are
  32–48 GB and the KL reference stops being free (today `disable_adapter()`
  gives it at zero cost), so budget ~63 GB with fused cross-entropy, ~95 GB
  naively. Stick to LoRA.

**vLLM preallocates `gpu_memory_utilization × total` at startup and never
releases it.** When sharing a card, set it explicitly — the default 0.9 will
claim the whole GPU and the trainer OOMs on its first backward.

Nothing is lost by co-locating: the loop is strictly sequential (roll out all
episodes, *then* train), so the two never contend for compute.

## 8. Smoke test before burning a long job

In order, each catching a different failure:

1. `python -m pytest src/grasp/cqd/test_cqd.py -q` — install sanity, no GPU.
2. A raw `curl` against the SPARQL endpoint (§2) — network/reachability.
3. Build a tiny pool and check the manager loads:
   ```python
   from grasp.core import setup
   managers, _ = setup(rollout_config)   # loads GRASP_INDEX_DIR indices
   ```
   This is where a missing or unreadable index directory shows up.
4. **One episode end-to-end** with `num_rounds=1`, `items_per_round=1`,
   `group_size=2`. If the student produces a parseable tool call and a
   non-error reward, the plumbing is correct. This ~2 minute check is worth
   it: freebase entity search must map names onto `m.xxx` mids, and a student
   trained only on wikidata Q-ids has never done that.

## 9. Gotchas that have actually bitten us

- **`KG_ENDPOINT` unset** → silent fallback to qlever.dev → "Access token was
  provided but it was invalid" → teacher proposes nothing, rounds are no-ops.
- **`MODEL_*` unset** for the student → rollouts hit the OpenAI API with no
  key. Set `MODEL_PROVIDER=openai/completions`,
  `MODEL_ENDPOINT=http://localhost:8437/v1`, `MODEL_API_KEY=EMPTY`, or set
  them on the config object as the run scripts do.
- **`step_grace` ≥ `max_steps`** silently disables the length penalty: the
  ramp is `max(1, max_steps - step_grace)`. Keep `step_grace` below
  `max_steps` (we use 13 against 20).
- **`--max-model-len` too small** → 400s mid-run, and the round's episodes are
  dropped as over-long, biasing the batch toward short/easy episodes.
- **Adapter name collisions** on a re-run: `unload_run_adapters` clears
  `<run_name>-r*` at startup, so reuse of a `run_name` is safe, but two
  concurrent runs sharing a vLLM server and a `run_name` will fight.
- **wandb on an offline node**: set `WANDB_MODE=offline`, then `wandb sync
  wandb/offline-run-*` from the login node.

## 10. Where things live

| Path | Content |
|---|---|
| `data/cqd/wqsp_train_pool.jsonl` | 3097 verified freebase/WQSP train items |
| `data/cqd/wqsp_val_pool.jsonl` | 50-item frozen holdout |
| `data/cqd/build_wqsp_pools.py` | rebuilds both from `grasp-dev/data/benchmark` |
| `data/cqd/run_rl_wqsp_freebase.py` | the fixed-dataset run |
| `data/cqd/zpd_best_r16/` | best 4B curriculum adapter, QALD-7 val f1 **0.5333** |
| `data/cqd/archive/kl_best_r10/` | the KL-baseline adapter, val f1 0.5322 |
| `data/kg-index/<kg>` | search indices (symlinks into grasp-dev today) |

`~/phd` is NFS and follows you between institute machines; `/local/data-ssd`
and `/tmp` do not. On a cluster, treat everything under `data/cqd` as the
thing to copy.
