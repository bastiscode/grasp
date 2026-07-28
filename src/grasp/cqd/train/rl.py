# GRPO training of the student against the task pool. Each round samples
# pool items, collects a group of rollouts per item via vLLM, scores them
# with execution-f1 rewards, normalizes to within-group advantages (no
# value model), and takes clipped policy-gradient steps on the exact tokens
# vLLM generated; the updated adapter is hot-loaded for the next round.
# Every assistant turn is its own sample sharing the episode's advantage,
# and only completion-position logits are materialized (logits_to_keep) so
# long prompts stay trainable on a single 24GB GPU.

import json
import os
import random
import statistics

import requests
import torch
from pydantic import BaseModel
from universal_ml_utils.io import load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.core import load_notes, setup
from grasp.cqd.configs import CqdConfig, RolloutConfig
from grasp.cqd.curriculum import descend
from grasp.cqd.pool import PoolItem, PoolItemInfo, TaskPool, item_id
from grasp.cqd.reward import RewardConfig, reward_episodes, reward_stats
from grasp.cqd.rollout import collect_rollouts, enable_token_data
from grasp.cqd.seeds import load_seeds
from grasp.cqd.train.data import RlTurn, episode_rl_turns


# a holdout validation file paired with its kg (mirrors the note-taking
# samples config, grasp.configs.NotesFromSamplesInput)
class ValFile(BaseModel):
    kg: str
    file: str


class RlConfig(BaseModel):
    # student base model (HF id or path); must be what the vLLM server runs
    model: str
    # optional adapter to start from (e.g. the SFT adapter)
    adapter: str | None = None
    output_dir: str
    # base url of the student vLLM server for LoRA hot-loading; the server
    # must run with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING
    vllm_url: str = "http://localhost:8437"
    # loop shape
    num_rounds: int = 10
    items_per_round: int = 8
    # rollouts per item (the GRPO group size)
    group_size: int = 4
    # optimization
    learning_rate: float = 2e-5
    # PPO clip bounds. We ran DAPO "clip-higher" (0.2/0.28) and the Qwen3-4B
    # diverged off-distribution late in training (policy blew past the base,
    # clip_frac -> 0.44, gibberish generation). Tightened to a symmetric-ish
    # 0.15/0.2 to slow low-probability-token growth, paired with the KL leash
    # below.
    clip_eps_low: float = 0.15
    clip_eps_high: float = 0.2
    # KL-to-reference penalty (k3 estimator) that anchors the policy to the
    # frozen base model (the LoRA adapter disabled). 0 disables it. This is
    # the leash the DAPO recipe removed; without it the 4B student drifted
    # off-distribution over 20 rounds. Costs one extra forward per turn.
    kl_coef: float = 0.04
    grad_accumulation: int = 8
    max_grad_norm: float = 1.0
    # skip turns longer than this many tokens (prompt + completion)
    max_seq_len: int = 16_384
    # lora, used only when starting fresh (no adapter given)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    seed: int = 22
    # auto-curriculum: every curriculum_interval rounds (0 disables), the
    # teacher revisits the max_revisits least-learnable of this round's
    # rolled-out items (see curriculum.select_revisits) and distills
    # easier/harder variants into the pool from the student's traces, so the
    # item set co-evolves with the student instead of staying frozen.
    # max_revisits is the fixed per-round growth size. Requires a seeds_file
    # (the CqdConfig teacher side).
    curriculum_interval: int = 0
    max_revisits: int | None = None
    # the curriculum step probes every fresh proposal with the current student
    # and drops those whose competence (mean f1) does not exceed
    # min_competence. At the default 0 that is exactly the proposals no rollout
    # got any credit on: their reward spread comes only from executable-vs-not,
    # so they teach validity rather than correctness. Items with even partial
    # credit are kept and left to the sampler to weight (pool.draw_weight).
    # curriculum_max_rounds bounds the propose -> probe -> ease again descent
    # within one curriculum step.
    min_competence: float = 0.0
    curriculum_max_rounds: int = 2
    # coverage-paced growth: on each curriculum round, also probe this many
    # not-yet-covered seed CQs with the current student and fold them into the
    # same reformulation step (see curriculum.descend). Broadening
    # (new CQs) and deepening (revisiting pool items) are the one bottom-k step;
    # a hard freshly-probed seed becomes an easier variant. 0 disables coverage
    # (pure revisit of the existing pool). Warm-start seeds the rest.
    coverage_batch: int = 0
    # dynamic sampling (DAPO): roll out items in chunks until items_per_round
    # of them have reward variance (a real GRPO gradient), refilling past
    # zero-variance and step-limit-filtered groups so the batch never starves
    # -- up to items_per_round * dynamic_sample_max_factor items rolled out.
    # 1 disables it (single chunk, no refill).
    dynamic_sample_max_factor: int = 4
    # holdout validation: every val_interval rounds (0 disables), roll out
    # the current student once per sample on these frozen files (no training)
    # and log execution f1 to wandb under val/, so real progress is visible
    # apart from the sawtooth per-round train f1. Each file's kg must be
    # among the rollout knowledge_graphs (managers are reused).
    val_files: list[ValFile] = []
    val_interval: int = 0
    # logging
    wandb_project: str = "grasp-cqd"
    run_name: str = "grpo"


class RlSample(BaseModel):
    turn: RlTurn
    advantage: float


# group-normalized advantages, grouped by pool item; None for skipped
# episodes (no reward) and degenerate groups (all rewards equal, no signal)
def group_advantages(rewards: list) -> list[float | None]:
    by_item: dict[str, list[float]] = {}
    for reward in rewards:
        if reward.reward is not None:
            by_item.setdefault(reward.item_id, []).append(reward.reward)

    advantages: list[float | None] = []
    for reward in rewards:
        group = by_item.get(reward.item_id, [])
        if reward.reward is None or len(group) < 2:
            advantages.append(None)
            continue
        std = statistics.pstdev(group)
        if std < 1e-6:
            advantages.append(None)
            continue
        mean = statistics.mean(group)
        advantages.append((reward.reward - mean) / std)

    return advantages


# item ids whose group carries a real gradient (non-None group-relative
# advantage); used by dynamic sampling to size the batch by trainable groups
def trainable_items(rewards: list) -> set[str]:
    advantages = group_advantages(rewards)
    return {
        reward.item_id
        for reward, advantage in zip(rewards, advantages)
        if advantage is not None
    }


# start the experiment's wandb run, or attach to one the caller already
# started. Warm-start (curriculum.warmstart_pool) runs before train_rl but is
# part of the same experiment, so a script that calls this first gets both
# phases in one run instead of the warm-start hours going unlogged.
def init_wandb(project: str, name: str, config: dict | None = None):
    import wandb

    if wandb.run is not None:
        if config is not None:
            wandb.run.config.update(config, allow_val_change=True)
        return wandb.run
    return wandb.init(project=project, name=name, config=config or {})


def load_student_model(config: RlConfig):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if config.adapter is not None:
        model = PeftModel.from_pretrained(
            model, config.adapter, is_trainable=True
        )
    else:
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

    if config.gradient_checkpointing:
        # required for gradients to flow into LoRA weights through
        # checkpointed blocks whose inputs don't require grad
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model


# new per-token logprobs of the turn's completion under the current policy,
# materializing logits only at completion positions
def turn_logprobs(model, turn: RlTurn) -> torch.Tensor:
    ids = torch.tensor(
        [turn.prompt_ids + turn.completion_ids], device=model.device
    )
    k = len(turn.completion_ids)
    # logits at position i predict token i + 1, so the predictions for the
    # k completion tokens are the last k + 1 logits without the final one
    out = model(input_ids=ids, logits_to_keep=k + 1)
    logits = out.logits[0, :-1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    completion = torch.tensor(turn.completion_ids, device=model.device)
    return logprobs.gather(-1, completion[:, None]).squeeze(-1)


# one pass over the round's samples with clipped policy-gradient updates
# plus a KL-to-reference penalty (GRPO). DAPO token-level loss: within each
# gradient-accumulation window the per-token losses are summed and
# normalized by the window's total completion-token count, so every token
# contributes equally instead of every turn (tokens in long turns are no
# longer down-weighted). Turns are still backwarded one at a time to cap
# memory; the window token total is known up front from completion_ids.
def train_step(
    model,
    optimizer,
    samples: list[RlSample],
    config: RlConfig,
) -> dict:
    model.train()
    rng = random.Random(config.seed)
    rng.shuffle(samples)

    total = {
        "loss": 0.0,
        "kl": 0.0,
        "kl_ref": 0.0,
        "clip": 0.0,
        "tokens": 0,
        "turns": 0,
    }
    optimizer.zero_grad()

    for start in range(0, len(samples), config.grad_accumulation):
        window = samples[start : start + config.grad_accumulation]
        window_tokens = sum(len(s.turn.completion_ids) for s in window)
        if window_tokens == 0:
            continue

        for sample in window:
            turn = sample.turn
            new_logprobs = turn_logprobs(model, turn)
            old_logprobs = torch.tensor(
                turn.logprobs, device=new_logprobs.device, dtype=new_logprobs.dtype
            )

            ratio = (new_logprobs - old_logprobs).exp()
            advantage = sample.advantage
            unclipped = ratio * advantage
            clipped = (
                ratio.clamp(1 - config.clip_eps_low, 1 + config.clip_eps_high)
                * advantage
            )
            pg = -torch.min(unclipped, clipped)

            # KL-to-reference (k3 estimator): reference = base model with the
            # LoRA adapter disabled. Detached ref logprobs; gradient flows
            # through new_logprobs, pulling the policy back toward base.
            if config.kl_coef > 0.0:
                with torch.no_grad(), model.disable_adapter():
                    ref_logprobs = turn_logprobs(model, turn)
                log_r = ref_logprobs - new_logprobs
                kl_ref = log_r.exp() - log_r - 1.0
                per_token = pg + config.kl_coef * kl_ref
            else:
                kl_ref = None
                per_token = pg

            # sum over the turn's tokens; the window normalizer makes the
            # accumulated update token-level (mean per token over the window)
            loss = per_token.sum()
            (loss / window_tokens).backward()

            with torch.no_grad():
                total["loss"] += loss.item()
                total["kl"] += (old_logprobs - new_logprobs).sum().item()
                if kl_ref is not None:
                    total["kl_ref"] += kl_ref.sum().item()
                total["clip"] += (
                    (
                        (ratio < 1 - config.clip_eps_low)
                        | (ratio > 1 + config.clip_eps_high)
                    )
                    .float()
                    .sum()
                    .item()
                )
                total["tokens"] += len(turn.completion_ids)
                total["turns"] += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    # report per-token means (token-level), consistent with the loss
    tokens = max(1, total["tokens"])
    return {
        "loss": total["loss"] / tokens,
        "kl": total["kl"] / tokens,
        "kl_ref": total["kl_ref"] / tokens,
        "clip_frac": total["clip"] / tokens,
        "trained_turns": total["turns"],
    }


# never reuse a previously loaded name or path with changed weights: vLLM
# caches adapter tensors and unload+reload was observed to serve stale
# weights, so the loop uses a fresh directory and name per round
def load_adapter_into_vllm(
    url: str,
    name: str,
    path: str,
    unload: str | None = None,
) -> None:
    response = requests.post(
        f"{url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": path},
        timeout=120,
    )
    assert response.ok, f"Loading adapter {name} failed: {response.text}"

    if unload is not None:
        requests.post(
            f"{url}/v1/unload_lora_adapter",
            json={"lora_name": unload},
            timeout=120,
        )


# unload adapters left on the server from a previous run of the same
# run_name (`{run_name}-r*`) so a rerun doesn't collide on a loaded name;
# each round still loads a fresh round_N dir, so no stale path is reused
def unload_run_adapters(url: str, run_name: str) -> int:
    try:
        response = requests.get(f"{url}/v1/models", timeout=30)
        response.raise_for_status()
        names = [m["id"] for m in response.json().get("data", [])]
    except requests.RequestException:
        return 0

    prefix = f"{run_name}-r"
    unloaded = 0
    for name in names:
        if name.startswith(prefix):
            requests.post(
                f"{url}/v1/unload_lora_adapter",
                json={"lora_name": name},
                timeout=120,
            )
            unloaded += 1
    return unloaded


# roll out the current student (served as rollout_config.model) once per
# holdout sample, score with the training reward, and return wandb metrics:
# per-file val/<name>/{f1,invalid,answered} plus aggregate val/{f1,invalid}
# over all samples. No training, no pool mutation (a throwaway pool just
# carries the references for scoring).
def run_validation(
    val_files: list[ValFile],
    rollout_config: RolloutConfig,
    reward_config: RewardConfig,
    managers,
    notes: list[str] | None,
    kg_notes: dict[str, list[str]] | None,
    logger,
) -> dict:
    metrics: dict = {}
    f1s: list[float] = []
    total_invalid = 0
    for vf in val_files:
        items = []
        for row in load_jsonl(vf.file):
            id = item_id(vf.kg, row["question"], row["sparql"])
            items.append(
                PoolItem(
                    id=id,
                    question=row["question"],
                    sparql=row["sparql"],
                    info=PoolItemInfo(kg=vf.kg),
                )
            )
        # dedup within the file by content id
        pool = TaskPool(list({it.id: it for it in items}.values()))

        episodes = collect_rollouts(
            list(pool.items.values()),
            rollout_config,
            managers,
            num_rollouts=1,
            parallelism=rollout_config.parallelism,
            kg_notes=kg_notes,
            notes=notes,
            progress=True,
            logger=logger,
        )
        rewards = reward_episodes(
            episodes,
            pool,
            reward_config,
            max_steps=rollout_config.max_steps,
            logger=logger,
            # a held-out metric must count a truncated / degenerate episode as
            # a failure (0), not silently drop it the way training does; else
            # the denominator shrinks to the easy items and val/f1 inflates
            filter_step_limit=False,
        )
        stats = reward_stats(rewards)

        name = os.path.splitext(os.path.basename(vf.file))[0]
        metrics[f"val/{name}/f1"] = stats["mean_f1"]
        metrics[f"val/{name}/f1_answered"] = stats["mean_f1_answered"]
        metrics[f"val/{name}/invalid"] = stats["invalid"]
        metrics[f"val/{name}/answered"] = stats["answered"]

        for r in rewards:
            if r.reward is None:
                continue
            # honest aggregate: give-ups (f1 None) count as 0, not excluded
            f1s.append(r.f1 if r.f1 is not None else 0.0)
            total_invalid += r.invalid

    metrics["val/f1"] = sum(f1s) / len(f1s) if f1s else None
    metrics["val/invalid"] = total_invalid
    return metrics


# Phase 4 entry point: GRPO rounds against the pool, returns the final
# adapter dir. With curriculum_interval > 0, teacher_config is required and
# every curriculum_interval rounds the teacher revisits the round's
# signal-less items and grows the pool (auto-curriculum). The teacher
# reuses the rollout managers/notes, so its KG settings must match
# rollout_config's.
def train_rl(
    config: RlConfig,
    rollout_config: RolloutConfig,
    reward_config: RewardConfig | None = None,
    teacher_config: CqdConfig | None = None,
    covered_seeds: set[str] | None = None,
    log_level: str | int | None = None,
) -> str:
    import wandb

    logger = get_logger("CQD RL", log_level)
    reward_config = reward_config or RewardConfig()

    curriculum = config.curriculum_interval > 0
    if curriculum:
        assert teacher_config is not None, (
            "curriculum_interval > 0 requires a teacher_config"
        )
        assert teacher_config.pool_file == rollout_config.pool_file, (
            "teacher_config and rollout_config must share the pool_file"
        )

    # coverage-paced growth reserve: seed CQs not yet covered (by warm-start or
    # earlier rounds), drawn a batch at a time on each curriculum round. Shuffled
    # once (deterministically) so coverage does not front-load one domain.
    reserve: list = []
    if curriculum and config.coverage_batch > 0:
        assert teacher_config is not None
        done = covered_seeds or set()
        reserve = [
            s for s in load_seeds(teacher_config.seeds_file)
            if s.question and s.id not in done
        ]
        random.Random(config.seed).shuffle(reserve)
        logger.info(f"Coverage reserve: {len(reserve)} uncovered seed CQs")

    rollout_config.num_rollouts = config.group_size
    enable_token_data(rollout_config)

    managers, _ = setup(rollout_config)
    notes, kg_notes = load_notes(rollout_config)

    model = load_student_model(config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    run = init_wandb(
        config.wandb_project,
        config.run_name,
        {
            "rl": config.model_dump(),
            "reward": reward_config.model_dump(),
            "rollout": {
                "pool_file": rollout_config.pool_file,
                "competence_half_life": rollout_config.competence_half_life,
                "learnability_half_life": rollout_config.learnability_half_life,
                "max_steps": rollout_config.max_steps,
            },
        },
    )

    # clear any adapters left on the server by a previous run of this
    # run_name, so loading this run's per-round adapters doesn't collide
    stale = unload_run_adapters(config.vllm_url, config.run_name)
    if stale:
        logger.info(f"Unloaded {stale} stale '{config.run_name}-r*' adapters")

    # serve the starting adapter, if any
    adapter_name = config.model
    if config.adapter is not None:
        adapter_name = f"{config.run_name}-r0"
        load_adapter_into_vllm(config.vllm_url, adapter_name, config.adapter)
    rollout_config.model = adapter_name

    # track the best holdout checkpoint so a late collapse (see the DAPO
    # format-collapse finding) does not cost us the peak: whenever the honest
    # val/f1 improves, copy the current adapter to output_dir/best.
    best_val_f1: float | None = None
    best_round: int | None = None

    for round_num in range(1, config.num_rounds + 1):
        pool = TaskPool.load(rollout_config.pool_file)
        # dynamic sampling: draw up to `cap` candidates, roll them out in
        # items_per_round-sized chunks, and stop once items_per_round of them
        # carry a gradient (nonzero reward variance) -- refilling past
        # zero-variance and step-limit-filtered groups so the trainable batch
        # stays full. Keep the `new` exploration budget at ~its per-round size
        # regardless of the cap by scaling new_fraction down.
        cap = config.items_per_round * max(1, config.dynamic_sample_max_factor)
        new_fraction = rollout_config.new_fraction * config.items_per_round / cap
        candidates = pool.sample(
            cap,
            competence_half_life=rollout_config.competence_half_life,
            learnability_half_life=rollout_config.learnability_half_life,
            target_competence=rollout_config.sample_target_competence,
            new_fraction=new_fraction,
        )

        episodes: list = []
        rewards: list = []
        drawn = 0
        while drawn < len(candidates):
            chunk = candidates[drawn : drawn + config.items_per_round]
            drawn += len(chunk)
            chunk_episodes = collect_rollouts(
                chunk,
                rollout_config,
                managers,
                num_rollouts=config.group_size,
                parallelism=rollout_config.parallelism,
                kg_notes=kg_notes,
                notes=notes,
                progress=True,
                logger=logger,
            )
            episodes.extend(chunk_episodes)
            rewards.extend(
                reward_episodes(
                    chunk_episodes,
                    pool,
                    reward_config,
                    max_steps=rollout_config.max_steps,
                    logger=logger,
                    round=round_num,
                )
            )
            if len(trainable_items(rewards)) >= config.items_per_round:
                break
        pool.save(rollout_config.pool_file)

        n_trainable = len(trainable_items(rewards))
        logger.info(
            f"Round {round_num}: rolled out {drawn} items to reach "
            f"{n_trainable} trainable groups (cap {cap}, "
            f"target {config.items_per_round})"
        )

        # auto-curriculum: let the teacher grow the pool from this round's
        # signal-less items before the next round samples it. Uses the
        # scored episodes already in hand and the rollout KG managers/notes.
        # Ingested proposals get parent_id = the revisited item and inherit
        # its root cq_id, so revisiting the pool's own multi-hop output
        # extends existing lineages rather than restarting from the seed.
        added_proposals = 0
        covered_this_round = 0
        if curriculum and round_num % config.curriculum_interval == 0:
            assert teacher_config is not None
            # unified curriculum step: revisit this round's bottom-k items AND
            # broaden by folding in a batch of not-yet-covered seed CQs, probed
            # with the CURRENT student. The seed probes feed only the teacher,
            # not training (samples below use `episodes` only).
            cover = reserve[: config.coverage_batch]
            del reserve[: len(cover)]
            covered_this_round = len(cover)
            # same descent as warm-start: the teacher proposes, the student
            # ATTEMPTS the fresh variants, and only those it can already solve
            # sometimes are kept -- so the teacher sees how its own proposal
            # fared and nothing enters the pool unprobed.
            accepted, rejected = descend(
                teacher_config,
                pool,
                rollout_config,
                reward_config,
                episodes,
                rewards,
                cover,
                num_rollouts=config.group_size,
                max_rounds=config.curriculum_max_rounds,
                min_competence=config.min_competence,
                max_revisits=config.max_revisits,
                managers=managers,
                notes=notes,
                kg_notes=kg_notes,
                round=round_num,
                logger=logger,
            )
            for item in rejected:
                assert item.id is not None
                pool.remove(item.id)
            added_proposals = len(accepted)
            # run_revisits saved the pool with the transient seeds; cover_and_revisit
            # dropped them from memory, so re-save the seed-free pool (the next
            # round reloads it from disk and must not train on raw seeds).
            pool.save(rollout_config.pool_file)
            logger.info(
                f"Round {round_num}: teacher added {added_proposals} solvable "
                f"proposals ({len(rejected)} too hard, dropped; covered "
                f"{covered_this_round} new seeds, {len(reserve)} left); "
                f"pool now {len(pool)} items"
            )

        advantages = group_advantages(rewards)
        samples = []
        skipped_long = 0
        for episode, advantage in zip(episodes, advantages):
            if advantage is None:
                continue
            for turn in episode_rl_turns(episode):
                if len(turn.prompt_ids) + len(turn.completion_ids) > config.max_seq_len:
                    skipped_long += 1
                    continue
                samples.append(RlSample(turn=turn, advantage=advantage))

        stats = reward_stats(rewards)
        logger.info(
            f"Round {round_num}: mean reward {stats['mean_reward']}, "
            f"mean f1 {stats['mean_f1']}, {len(samples)} trainable turns "
            f"({skipped_long} skipped as too long)"
        )

        train_metrics = {}
        if samples:
            train_metrics = train_step(model, optimizer, samples, config)

        # save the updated adapter and hot-load it for the next round
        adapter_dir = os.path.join(config.output_dir, f"round_{round_num}")
        model.save_pretrained(adapter_dir)
        new_name = f"{config.run_name}-r{round_num}"
        load_adapter_into_vllm(
            config.vllm_url,
            new_name,
            adapter_dir,
            unload=adapter_name if adapter_name != config.model else None,
        )
        adapter_name = new_name
        rollout_config.model = adapter_name

        # periodic holdout validation of the just-loaded (post-update) policy
        val_metrics = {}
        if (
            config.val_files
            and config.val_interval
            and round_num % config.val_interval == 0
        ):
            val_metrics = run_validation(
                config.val_files,
                rollout_config,
                reward_config,
                managers,
                notes,
                kg_notes,
                logger,
            )
            logger.info(f"Round {round_num} validation: {val_metrics}")

            val_f1 = val_metrics.get("val/f1")
            if val_f1 is not None and (best_val_f1 is None or val_f1 > best_val_f1):
                best_val_f1 = val_f1
                best_round = round_num
                best_dir = os.path.join(config.output_dir, "best")
                model.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "best_info.json"), "w") as bf:
                    bf.write(json.dumps({"round": round_num, "val_f1": val_f1}, indent=2))
                logger.info(
                    f"New best val/f1 {val_f1:.4f} at round {round_num} -> {best_dir}"
                )
            if best_val_f1 is not None:
                val_metrics["val/best_f1"] = best_val_f1

        run.log(
            {
                "round": round_num,
                "reward/mean": stats["mean_reward"],
                "reward/f1": stats["mean_f1"],
                "reward/f1_answered": stats["mean_f1_answered"],
                "reward/answered": stats["answered"],
                "reward/cancelled": stats["cancelled"],
                "reward/invalid": stats["invalid"],
                "reward/skipped": sum(stats["skipped"].values()),
                "train/samples": len(samples),
                "train/skipped_long": skipped_long,
                "train/rolled_out_items": drawn,
                "train/trainable_items": n_trainable,
                "pool/size": len(pool),
                "pool/added_proposals": added_proposals,
                "pool/covered_seeds": covered_this_round,
                "pool/reserve": len(reserve),
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **val_metrics,
            }
        )

    final_dir = os.path.join(config.output_dir, "adapter")
    model.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "rl_config.json"), "w") as f:
        f.write(json.dumps(config.model_dump(), indent=2))
    logger.info(f"Saved final adapter to {final_dir}")
    if best_round is not None:
        logger.info(
            f"Best holdout checkpoint: round {best_round} (val/f1 "
            f"{best_val_f1:.4f}) at {os.path.join(config.output_dir, 'best')}"
        )
    run.finish()

    return final_dir
