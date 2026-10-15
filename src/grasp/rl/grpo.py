# Plain GRPO for GRASP agents. Each step takes the next batch_size samples of
# a shuffled pass over the training data, rolls out group_size episodes per
# sample against a vLLM server, scores them with the task reward, normalizes
# rewards within each group (no value model) and takes clipped policy-gradient
# steps with a KL penalty to the base model on the exact tokens vLLM produced.
# The updated LoRA adapter is then hot-loaded into the server for the next
# step. Every assistant turn is its own sample sharing its episode's advantage.
#
# The vLLM server must serve the base model with LoRA enabled and runtime
# adapter loading allowed, e.g.
#   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve <model> --enable-lora \
#     --max-lora-rank <lora_r> --enable-auto-tool-choice --tool-call-parser ...
#
# Usage: python -m grasp.rl.grpo <config.yaml> <output_dir>

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics

import requests
import torch
from pydantic import BaseModel
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.configs import GraspConfig
from grasp.core import load_notes, setup
from grasp.rl.data import RlTurn, episode_turns
from grasp.rl.reward import EpisodeReward, RewardConfig, reward_episodes
from grasp.rl.rollout import (
    Episode,
    RlItem,
    collect_rollouts,
    enable_token_data,
    tool_calls,
)
from grasp.tasks import get_task


# a data file in the task's sample format; kg is the kg the sparql-qa
# references belong to (falls back to the row's info.kg or the only
# configured kg), unused for cea
class DataFile(BaseModel):
    file: str
    kg: str | None = None


class GrpoConfig(GraspConfig):
    task: str = "sparql-qa"
    train_files: list[DataFile]
    val_files: list[DataFile] = []
    reward: RewardConfig = RewardConfig()

    # vLLM server serving config.model (base url, without /v1)
    vllm_url: str = "http://localhost:8000"
    # initial adapter, e.g. from SFT; None starts a fresh LoRA
    adapter: str | None = None

    num_epochs: int = 1
    # samples per step, each rolled out group_size times
    batch_size: int = 8
    group_size: int = 8
    # max concurrent episodes, bounds vLLM and KG endpoint load
    parallelism: int = 16

    learning_rate: float = 2e-5
    # linear warmup (optimizer steps), then constant
    lr_warmup_steps: int = 50
    clip_eps_low: float = 0.15
    clip_eps_high: float = 0.2
    # KL to the base model (k3 estimator); without it policies drifted off
    # distribution into gibberish after ~15 steps in earlier runs
    kl_coef: float = 0.04
    # cap of the truncated importance weight pi_old / pi_vllm per token
    tis_cap: float = 2.0
    # turns per optimizer step, loss normalized per token within the window
    grad_accumulation: int = 8
    max_grad_norm: float = 1.0
    # skip turns longer than this many tokens (prompt + completion)
    max_seq_len: int = 16_384

    lora_r: int = 32
    lora_alpha: int = 64
    # no dropout: it would make pi_old and the updated policy stochastic and
    # put noise into the PPO ratio
    lora_dropout: float = 0.0
    # a module name, list of names, or a regex (needed for multimodal
    # checkpoints where "all-linear" would also adapt the vision tower)
    lora_target_modules: str | list[str] = "all-linear"
    gradient_checkpointing: bool = True

    # validate every val_interval steps (and once before training), 0 = never
    val_interval: int = 0
    val_rollouts: int = 1
    # keep the adapter of every save_interval-th step, 0 = only the latest
    save_interval: int = 0

    # None disables wandb logging
    wandb_project: str | None = None
    run_name: str = "grpo"


class RlSample(BaseModel):
    turn: RlTurn
    advantage: float


def load_items(files: list[DataFile], config: GrpoConfig, managers) -> list[RlItem]:
    sample_cls = get_task(config.task, managers, config).sample_cls()
    assert sample_cls is not None, f"Task {config.task} has no sample class"
    single_kg = (
        config.knowledge_graphs[0].kg if len(config.knowledge_graphs) == 1 else None
    )

    items = []
    for data_file in files:
        name = os.path.splitext(os.path.basename(data_file.file))[0]
        for i, row in enumerate(load_jsonl(data_file.file)):
            sample = sample_cls(**row)
            info = getattr(sample, "info", None) or {}
            items.append(
                RlItem(
                    id=f"{name}:{sample.id or i}",
                    sample=sample,
                    kg=data_file.kg or info.get("kg") or single_kg,
                )
            )

    ids = [item.id for item in items]
    assert len(set(ids)) == len(ids), "Duplicate sample ids in data files"
    return items


# group-normalized advantages, None for skipped episodes and groups without
# reward variance (no signal)
def group_advantages(rewards: list[EpisodeReward]) -> list[float | None]:
    groups: dict[str, list[float]] = {}
    for reward in rewards:
        if reward.reward is not None:
            groups.setdefault(reward.item_id, []).append(reward.reward)

    advantages: list[float | None] = []
    for reward in rewards:
        group = groups.get(reward.item_id, [])
        std = statistics.pstdev(group) if len(group) > 1 else 0.0
        if reward.reward is None or std < 1e-6:
            advantages.append(None)
        else:
            advantages.append((reward.reward - statistics.mean(group)) / std)
    return advantages


def mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


# reward and environment-usage statistics: scores count give-ups as 0, and
# env/fn/<name> is the fraction of episodes calling <name> at least once
def episode_stats(episodes: list[Episode], rewards: list[EpisodeReward]) -> dict:
    scored = [(e, r) for e, r in zip(episodes, rewards) if not r.skipped]
    stats: dict = {
        "reward": mean([r.reward for _, r in scored]),
        "score": mean([r.score or 0.0 for _, r in scored]),
        "score_valid": mean([r.score for _, r in scored if r.score is not None]),
        "invalid": mean([r.invalid for _, r in scored]),
        "truncated": mean([r.truncated for _, r in scored]),
        "skipped": len(rewards) - len(scored),
        "env/steps": mean([e.steps for e, _ in scored]),
        "env/fn_errors": mean([e.fn_errors for e, _ in scored]),
        "env/done": mean([e.done for e, _ in scored]),
    }
    fn_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for episode, _ in scored:
        for name in {call["name"] for call in tool_calls(episode.messages)}:
            fn_counts[name] = fn_counts.get(name, 0) + 1
        output_type = (episode.output or {}).get("type")
        if output_type is not None:
            type_counts[output_type] = type_counts.get(output_type, 0) + 1
    for name, count in fn_counts.items():
        stats[f"env/fn/{name}"] = count / len(scored)
    for name, count in type_counts.items():
        stats[f"env/type/{name}"] = count / len(scored)
    return stats


# sampling kwargs for training rollouts: temperature 1 without truncation or
# penalties, so tokens come from exactly the policy the trainer scores (vLLM
# returns raw logprobs, i.e. before temperature, top-k/p and penalties).
# Other kwargs, e.g. chat_template_kwargs, are kept.
SAMPLING_KWARGS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
}


def on_policy_kwargs(model_kwargs: dict) -> dict:
    kwargs = {k: v for k, v in model_kwargs.items() if k not in SAMPLING_KWARGS}
    extra_body = {
        k: v
        for k, v in (model_kwargs.get("extra_body") or {}).items()
        if k not in SAMPLING_KWARGS
    }
    return {
        **kwargs,
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {**extra_body, "top_k": -1, "min_p": 0.0},
    }


def load_student_model(config: GrpoConfig, adapter: str | None):
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

    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

    if config.gradient_checkpointing:
        # lets gradients reach the LoRA weights through checkpointed blocks
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model


# per-token logprobs of the turn's completion under the current policy,
# materializing logits only at completion positions
def turn_logprobs(model, turn: RlTurn) -> torch.Tensor:
    ids = torch.tensor([turn.prompt_ids + turn.completion_ids], device=model.device)
    k = len(turn.completion_ids)
    # logits at position i predict token i + 1
    logits = model(input_ids=ids, logits_to_keep=k + 1).logits[0, :-1].float()
    completion = torch.tensor(turn.completion_ids, device=model.device)
    return torch.log_softmax(logits, dim=-1).gather(-1, completion[:, None]).squeeze(-1)


# per-token logprobs of every turn before any update: the current policy
# (pi_old) and, with the KL penalty on, the base model (adapter disabled).
# Recomputing pi_old here instead of taking vLLM's logprobs keeps the PPO
# ratio free of kernel and numerics differences between vLLM and PyTorch.
@torch.no_grad()
def pre_update_logprobs(model, samples: list[RlSample], kl: bool) -> list[tuple]:
    model.eval()
    out = []
    for sample in samples:
        old = turn_logprobs(model, sample.turn).cpu()
        ref = None
        if kl:
            with model.disable_adapter():
                ref = turn_logprobs(model, sample.turn).cpu()
        out.append((old, ref))
    model.train()
    return out


# one pass over the step's turns with clipped policy-gradient updates plus
# the KL penalty. The tokens were sampled by vLLM, not by the PyTorch policy,
# so each token's loss is weighted by the truncated importance ratio
# min(pi_old / pi_vllm, tis_cap) (Yao et al. 2025). Token-level loss:
# per-token losses are summed within each gradient-accumulation window and
# normalized by its completion token count. Turns are backwarded one at a
# time to cap memory.
def train_step(
    model,
    optimizer,
    scheduler,
    samples: list[RlSample],
    config: GrpoConfig,
    seed: int,
) -> dict:
    random.Random(seed).shuffle(samples)
    pre = pre_update_logprobs(model, samples, config.kl_coef > 0.0)
    total = {
        "loss": 0.0,
        "kl_old": 0.0,
        "kl_ref": 0.0,
        "clip": 0.0,
        "tokens": 0,
        "mismatch_abs": 0.0,
        "mismatch_kl": 0.0,
        "mismatch_extreme": 0.0,
        "tis_capped": 0.0,
    }
    optimizer.zero_grad()

    for start in range(0, len(samples), config.grad_accumulation):
        window = samples[start : start + config.grad_accumulation]
        window_pre = pre[start : start + config.grad_accumulation]
        window_tokens = max(1, sum(len(s.turn.completion_ids) for s in window))

        for sample, (old, ref) in zip(window, window_pre):
            turn = sample.turn
            new = turn_logprobs(model, turn)
            old = old.to(new.device)
            vllm = torch.tensor(turn.logprobs, device=new.device, dtype=new.dtype)

            # pi_old / pi_vllm, the sampler-trainer mismatch
            log_mismatch = old - vllm
            tis = log_mismatch.exp().clamp(max=config.tis_cap)

            ratio = (new - old).exp()
            clipped = ratio.clamp(1 - config.clip_eps_low, 1 + config.clip_eps_high)
            per_token = -tis * torch.min(
                ratio * sample.advantage, clipped * sample.advantage
            )

            kl_ref = None
            if ref is not None:
                log_r = ref.to(new.device) - new
                kl_ref = log_r.exp() - log_r - 1.0
                per_token = per_token + config.kl_coef * kl_ref

            loss = per_token.sum()
            (loss / window_tokens).backward()

            with torch.no_grad():
                total["loss"] += loss.item()
                total["kl_old"] += (old - new).sum().item()
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
                total["mismatch_abs"] += log_mismatch.abs().sum().item()
                # k3 estimate of KL(pi_vllm || pi_old) on vLLM's samples
                total["mismatch_kl"] += (
                    (log_mismatch.exp() - log_mismatch - 1.0).sum().item()
                )
                total["mismatch_extreme"] += (
                    (log_mismatch.abs() > math.log(2)).float().sum().item()
                )
                total["tis_capped"] += (
                    (log_mismatch.exp() > config.tis_cap).float().sum().item()
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    tokens = max(1, total["tokens"])
    return {
        "loss": total["loss"] / tokens,
        "kl_old": total["kl_old"] / tokens,
        "kl_ref": total["kl_ref"] / tokens,
        "clip_frac": total["clip"] / tokens,
        "turns": len(samples),
        "tokens": total["tokens"],
        # sampler-trainer mismatch: mean |log pi_old - log pi_vllm|, its KL,
        # the share of tokens off by more than 2x, and the share TIS capped
        "mismatch/abs_logprob": total["mismatch_abs"] / tokens,
        "mismatch/kl": total["mismatch_kl"] / tokens,
        "mismatch/extreme_frac": total["mismatch_extreme"] / tokens,
        "mismatch/tis_capped_frac": total["tis_capped"] / tokens,
    }


# never reuse a loaded name or path with changed weights: vLLM caches adapter
# tensors and was observed to serve stale weights after unload + reload, so
# every step gets a fresh name and directory
def load_adapter_into_vllm(url: str, name: str, path: str, unload: str | None) -> None:
    response = requests.post(
        f"{url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": os.path.abspath(path)},
        timeout=120,
    )
    assert response.ok, f"Loading adapter {name} failed: {response.text}"
    if unload is not None:
        requests.post(
            f"{url}/v1/unload_lora_adapter", json={"lora_name": unload}, timeout=120
        )


# unload adapters a previous run with the same run_name left on the server
def unload_run_adapters(url: str, run_name: str) -> None:
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    for model in response.json().get("data", []):
        if re.fullmatch(rf"{re.escape(run_name)}-s\d+", model["id"]):
            requests.post(
                f"{url}/v1/unload_lora_adapter",
                json={"lora_name": model["id"]},
                timeout=120,
            )


def adapter_dir(output_dir: str, step: int) -> str:
    return os.path.join(output_dir, "adapters", f"step_{step}")


def run_validation(
    items_by_file: dict[str, list[RlItem]],
    rollout_config: GraspConfig,
    config: GrpoConfig,
    managers,
    notes,
    kg_notes,
    dump_file: str,
    logger,
) -> dict:
    metrics: dict = {}
    rows = []
    scores = []
    for name, items in items_by_file.items():
        # GRASP's output fallback stays on to measure the agent as deployed
        episodes = collect_rollouts(
            config.task,
            items,
            rollout_config,
            managers,
            config.val_rollouts,
            config.parallelism,
            require_done_call=False,
            kg_notes=kg_notes,
            notes=notes,
            logger=logger,
        )
        rewards = reward_episodes(
            config.task,
            episodes,
            {item.id: item for item in items},
            config.reward,
            rollout_config.max_steps,
            logger,
        )
        for key, value in episode_stats(episodes, rewards).items():
            metrics[f"val/{name}/{key}"] = value
        for episode, reward in zip(episodes, rewards):
            rows.append({**reward.model_dump(), "file": name, "output": episode.output})
            if not reward.skipped:
                scores.append(reward.score or 0.0)

    metrics["val/score"] = mean(scores)
    if len(scores) > 1:
        metrics["val/score_sem"] = statistics.stdev(scores) / math.sqrt(len(scores))
    # per-rollout rows allow paired comparisons between checkpoints
    os.makedirs(os.path.dirname(dump_file), exist_ok=True)
    dump_jsonl(rows, dump_file)
    return metrics


def train(
    config: GrpoConfig, output_dir: str, log_level: str | int | None = None
) -> None:
    logger = get_logger("GRPO", log_level)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(config.model_dump_json(indent=2))

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    items = load_items(config.train_files, config, managers)
    items_by_id = {item.id: item for item in items}
    val_items = {
        os.path.splitext(os.path.basename(f.file))[0]: load_items([f], config, managers)
        for f in config.val_files
    }
    steps_per_epoch = math.ceil(len(items) / config.batch_size)
    total_steps = steps_per_epoch * config.num_epochs
    logger.info(
        f"{len(items)} training samples, {steps_per_epoch} steps per epoch, "
        f"{total_steps} steps total"
    )

    # rollouts go to the vLLM server, the model name selects the adapter
    rollout_config = config.model_copy(deep=True)
    rollout_config.model_provider = "openai/completions"
    rollout_config.model_endpoint = f"{config.vllm_url}/v1"
    rollout_config.model_api_key = rollout_config.model_api_key or "EMPTY"
    # validation samples with the configured recipe, training on-policy
    train_rollout_config = rollout_config.model_copy(deep=True)
    train_rollout_config.model_kwargs = on_policy_kwargs(config.model_kwargs)
    enable_token_data(train_rollout_config)

    # resume from the last completed step if there is one
    state_file = os.path.join(output_dir, "trainer_state.pt")
    state = None
    if os.path.exists(state_file):
        state = torch.load(state_file, map_location="cpu", weights_only=False)
    start_step = state["step"] if state else 0
    best_score = state["best_score"] if state else None

    model = load_student_model(
        config, adapter_dir(output_dir, start_step) if state else config.adapter
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    # constant after warmup: steps vary in optimizer updates and runs are
    # early-stopped on validation, so a decay has no meaningful endpoint
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, config.lr_warmup_steps))
    )
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        logger.info(f"Resuming after step {start_step}")

    run = None
    if config.wandb_project is not None:
        import wandb

        # stable id per output dir, so resumed jobs continue the same run
        run = wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            id=hashlib.sha1(os.path.abspath(output_dir).encode()).hexdigest()[:16],
            resume="allow",
            config=config.model_dump(),
        )

    unload_run_adapters(config.vllm_url, config.run_name)
    served = None
    if state or config.adapter:
        served = f"{config.run_name}-s{start_step}"
        load_adapter_into_vllm(
            config.vllm_url,
            served,
            adapter_dir(output_dir, start_step) if state else config.adapter,  # type: ignore[arg-type]
            unload=None,
        )
    rollout_config.model = served or config.model
    train_rollout_config.model = rollout_config.model

    def validate(step: int) -> dict:
        nonlocal best_score
        metrics = run_validation(
            val_items,
            rollout_config,
            config,
            managers,
            notes,
            kg_notes,
            os.path.join(output_dir, "val", f"step_{step}.jsonl"),
            logger,
        )
        score = metrics["val/score"]
        logger.info(f"Step {step} validation score {score}")
        if (
            step > 0
            and score is not None
            and (best_score is None or score > best_score)
        ):
            best_score = score
            best_dir = os.path.join(output_dir, "best")
            model.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "best_info.json"), "w") as f:
                json.dump({"step": step, "score": score}, f)
            logger.info(f"New best validation score at step {step}")
        return metrics

    if start_step == 0 and val_items and config.val_interval > 0:
        metrics = validate(0)
        if run is not None:
            run.log({"step": 0, **metrics})

    for step in range(start_step + 1, total_steps + 1):
        epoch, index = divmod(step - 1, steps_per_epoch)
        order = list(items)
        random.Random((config.seed or 0) + epoch).shuffle(order)
        batch = order[index * config.batch_size : (index + 1) * config.batch_size]

        episodes = collect_rollouts(
            config.task,
            batch,
            train_rollout_config,
            managers,
            config.group_size,
            config.parallelism,
            require_done_call=True,
            kg_notes=kg_notes,
            notes=notes,
            logger=logger,
        )
        rewards = reward_episodes(
            config.task, episodes, items_by_id, config.reward, config.max_steps, logger
        )

        samples = []
        skipped_long = 0
        trainable = set()
        for episode, reward, advantage in zip(
            episodes, rewards, group_advantages(rewards)
        ):
            if advantage is None:
                continue
            trainable.add(episode.item_id)
            for turn in episode_turns(episode):
                if len(turn.prompt_ids) + len(turn.completion_ids) > config.max_seq_len:
                    skipped_long += 1
                    continue
                samples.append(RlSample(turn=turn, advantage=advantage))

        stats = episode_stats(episodes, rewards)
        logger.info(
            f"Step {step}/{total_steps} (epoch {epoch + 1}): reward {stats['reward']}, "
            f"score {stats['score']}, {len(trainable)}/{len(batch)} groups with "
            f"signal, {len(samples)} turns ({skipped_long} too long)"
        )

        train_metrics = {}
        if samples:
            train_metrics = train_step(
                model, optimizer, scheduler, samples, config, (config.seed or 0) + step
            )

        # save and serve the new adapter, then drop the previous one unless
        # it is a kept checkpoint
        path = adapter_dir(output_dir, step)
        model.save_pretrained(path)
        name = f"{config.run_name}-s{step}"
        load_adapter_into_vllm(config.vllm_url, name, path, unload=served)
        served = name
        rollout_config.model = name
        train_rollout_config.model = name
        torch.save(
            {
                "step": step,
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            state_file,
        )
        previous = step - 1
        if previous > 0 and not (
            config.save_interval and previous % config.save_interval == 0
        ):
            shutil.rmtree(adapter_dir(output_dir, previous), ignore_errors=True)

        val_metrics = {}
        if val_items and config.val_interval > 0 and step % config.val_interval == 0:
            val_metrics = validate(step)

        if run is not None:
            run.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    **{f"rollout/{k}": v for k, v in stats.items()},
                    "train/trainable_groups": len(trainable),
                    "train/skipped_long": skipped_long,
                    "train/lr": scheduler.get_last_lr()[0],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **val_metrics,
                }
            )

    if run is not None:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for GRASP agents")
    parser.add_argument("config", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(GrpoConfig(**load_config(args.config)), args.output_dir, args.log_level)
