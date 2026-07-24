# SFT distillation baseline: train the student on teacher episodes. Teacher
# episodes are sparql-qa rollouts of a strong model on pool items, filtered
# by scored execution f1 (with save_pool=False so they don't enter the
# student's pool stats), re-rendered with the student's chat template, and
# trained with LoRA, loss on assistant tokens only.

import json
import os
from logging import Logger

from pydantic import BaseModel
from universal_ml_utils.io import load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.cqd.reward import EpisodeReward
from grasp.cqd.rollout import Episode
from grasp.cqd.train.data import IGNORE_INDEX, SftSample, episode_sft_sample


class SftConfig(BaseModel):
    # student model to train (HF id or path)
    model: str
    # teacher episodes and the corresponding rewards (aligned by index,
    # from reward_episode_file on the same episodes file)
    episodes_file: str
    rewards_file: str
    output_dir: str
    # keep only answer episodes with at least this f1
    min_f1: float = 1.0
    # skip samples longer than this many tokens
    max_seq_len: int = 16_384
    # lora
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # optimization
    learning_rate: float = 1e-4
    epochs: float = 2.0
    batch_size: int = 1
    grad_accumulation: int = 8
    warmup_ratio: float = 0.05
    logging_steps: int = 1
    seed: int = 22
    gradient_checkpointing: bool = True
    # logging
    wandb_project: str = "grasp-cqd"
    run_name: str = "sft"


def load_sft_episodes(
    config: SftConfig,
    logger: Logger,
) -> list[Episode]:
    episodes = [Episode(**e) for e in load_jsonl(config.episodes_file)]
    rewards = [EpisodeReward(**r) for r in load_jsonl(config.rewards_file)]
    assert len(episodes) == len(rewards), (
        "Episodes and rewards are misaligned; rewards_file must come from "
        "reward_episode_file on the same episodes file"
    )

    kept = [
        episode
        for episode, reward in zip(episodes, rewards)
        if episode.type == "answer"
        and reward.f1 is not None
        and reward.f1 >= config.min_f1
    ]
    logger.info(
        f"Keeping {len(kept)} of {len(episodes)} teacher episodes "
        f"(answered with f1 >= {config.min_f1})"
    )
    return kept


def build_samples(
    episodes: list[Episode],
    tokenizer,
    config: SftConfig,
    logger: Logger,
) -> list[SftSample]:
    samples = []
    skipped = 0
    for episode in episodes:
        sample = episode_sft_sample(episode, tokenizer)
        if len(sample.input_ids) > config.max_seq_len:
            skipped += 1
            continue
        samples.append(sample)

    if skipped:
        logger.warning(
            f"Skipped {skipped} samples longer than {config.max_seq_len} tokens"
        )
    loss_tokens = sum(
        sum(label != IGNORE_INDEX for label in s.labels) for s in samples
    )
    logger.info(
        f"Built {len(samples)} samples, "
        f"{sum(len(s.input_ids) for s in samples):,} tokens total, "
        f"{loss_tokens:,} with loss"
    )
    return samples


# Phase 3 entry point: LoRA SFT on teacher episodes. Returns the adapter dir.
def train_sft(
    config: SftConfig,
    log_level: str | int | None = None,
) -> str:
    # imports here to keep grasp importable without training dependencies
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    logger = get_logger("CQD SFT", log_level)
    os.environ["WANDB_PROJECT"] = config.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    episodes = load_sft_episodes(config, logger)
    samples = build_samples(episodes, tokenizer, config, logger)
    assert samples, "No training samples left after filtering"

    def collate(batch: list[SftSample]) -> dict:
        max_len = max(len(s.input_ids) for s in batch)
        pad_id = tokenizer.pad_token_id
        input_ids = []
        labels = []
        attention_mask = []
        for s in batch:
            pad = max_len - len(s.input_ids)
            input_ids.append(s.input_ids + [pad_id] * pad)
            labels.append(s.labels + [IGNORE_INDEX] * pad)
            attention_mask.append([1] * len(s.input_ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    # lm head + cross-entropy only at label positions, in checkpointed
    # chunks so only one chunk's full-vocab logits exist at a time. The
    # default loss materializes float32 logits for the whole sequence, which
    # does not fit next to the model on a 24GB GPU for long traces.
    class SelectiveLossTrainer(Trainer):
        LOSS_CHUNK = 1024

        def compute_loss(
            self, model, inputs, return_outputs=False, **kwargs
        ):
            base = (
                model.get_base_model()
                if hasattr(model, "get_base_model")
                else model
            )
            hidden = base.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state

            # logits at position i predict token i + 1
            shifted = inputs["labels"][:, 1:]
            select = shifted != IGNORE_INDEX
            hidden_select = hidden[:, :-1][select]
            labels_select = shifted[select]

            def chunk_loss_sum(h, labels):
                logits = base.lm_head(h).float()
                return torch.nn.functional.cross_entropy(
                    logits, labels, reduction="sum"
                )

            n = labels_select.numel()
            loss = hidden.new_zeros((), dtype=torch.float32)
            for i in range(0, n, self.LOSS_CHUNK):
                loss = loss + torch.utils.checkpoint.checkpoint(
                    chunk_loss_sum,
                    hidden_select[i : i + self.LOSS_CHUNK],
                    labels_select[i : i + self.LOSS_CHUNK],
                    use_reentrant=False,
                )
            loss = loss / max(1, n)

            assert not return_outputs, "Outputs not materialized"
            return loss

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if config.gradient_checkpointing:
        model.config.use_cache = False

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        report_to=["wandb"],
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        save_strategy="no",
        seed=config.seed,
        remove_unused_columns=False,
    )

    trainer = SelectiveLossTrainer(
        model=model,
        args=args,
        train_dataset=samples,  # type: ignore[arg-type]
        data_collator=collate,
    )
    result = trainer.train()
    logger.info(f"Training finished: {result.metrics}")

    adapter_dir = os.path.join(config.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "sft_config.json"), "w") as f:
        f.write(json.dumps(config.model_dump(), indent=2))
    logger.info(f"Saved adapter to {adapter_dir}")

    return adapter_dir
