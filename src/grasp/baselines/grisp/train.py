import argparse
import os
import random
from logging import Logger

import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.integrations.integration_utils import NeptuneCallback
from universal_ml_utils.configuration import load_config
from universal_ml_utils.logging import get_logger

from grasp.baselines.grisp.data import (
    GRISPCollator,
    GRISPSelectionDataset,
    GRISPSkeletonDataset,
    load_samples,
)
from grasp.baselines.grisp.utils import set_chat_template
from grasp.configs import KgConfig
from grasp.manager import load_kg_manager


class Lora(BaseModel):
    r: int = 32
    lora_alpha: int = 32
    target_modules: list[str] | str = "all-linear"
    save_modules: list[str] | None = None
    dropout: float = 0.05


class GRISPTrainConfig(BaseModel):
    # model
    model: str
    do_compile: bool = False
    lora: Lora | None = None

    # data
    type: str
    train_files: list[str]
    val: list[str] | float
    max_length: int = 8192
    mask_inputs: bool = True
    num_workers: int = 4
    knowledge_graph: KgConfig | None = None

    # data augmentation
    skeleton_p: float = 0.2
    selection_p: float = 0.2

    # training hyperparameters
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    batch_size: int = 16
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    seed: int = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRISP model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the training artifacts",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    config: GRISPTrainConfig,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(config.model, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer = set_chat_template(tokenizer)

    if config.lora is not None:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            modules_to_save=config.lora.save_modules,
        )
        model = get_peft_model(model, peft_config)
        assert isinstance(model, PeftModel)

    return model, tokenizer


def get_datasets(
    cfg: GRISPTrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    logger: Logger,
) -> tuple[Dataset, Dataset]:
    if cfg.type == "both":
        cfg.type = "skeleton"
        train_skel, val_skel = get_datasets(
            cfg,
            tokenizer,
            logger,
        )
        cfg.type = "selection"
        train_sel, val_sel = get_datasets(
            cfg,
            tokenizer,
            logger,
        )
        cfg.type = "both"
        train_data = ConcatDataset([train_skel, train_sel])
        val_data = ConcatDataset([val_skel, val_sel])
        return train_data, val_data

    samples = load_samples(cfg.train_files)
    dataset_kwargs = {
        "samples": samples,
        "tokenizer": tokenizer,
        "mask_inputs": cfg.mask_inputs,
        "log_level": logger.level,
    }
    if cfg.type == "skeleton":
        dataset_cls = GRISPSkeletonDataset
        dataset_kwargs["p"] = cfg.skeleton_p

    elif cfg.type == "selection":
        dataset_cls = GRISPSelectionDataset
        assert cfg.knowledge_graph is not None, (
            "KG config must be provided for selection dataset"
        )
        manager = load_kg_manager(cfg.knowledge_graph)
        dataset_kwargs["manager"] = manager
        dataset_kwargs["skeleton_p"] = cfg.skeleton_p
        dataset_kwargs["selection_p"] = cfg.selection_p
        logger.warning("Setting num workers to 0 for selection type training")
        cfg.num_workers = 0
    else:
        raise ValueError(f"Unknown train type: {cfg.type}")

    train_data = dataset_cls(**dataset_kwargs)

    if isinstance(cfg.val, list):
        dataset_kwargs["samples"] = load_samples(cfg.val)
        dataset_kwargs["is_val"] = True
        val_data = dataset_cls(**dataset_kwargs)
        return train_data, val_data

    assert cfg.val > 0 and cfg.val < 1.0, "Val split size must be a float in (0, 1)"
    indices = list(range(len(train_data)))
    random.seed(cfg.seed)
    random.shuffle(indices)
    num_val_samples = round(len(train_data) * cfg.val)
    num_val_samples = max(1, min(num_val_samples, len(train_data) - 1))
    val_indices = indices[:num_val_samples]
    train_indices = indices[num_val_samples:]
    logger.info(
        f"Splitting data into {len(train_indices):,} train "
        f"and {len(val_indices):,} val samples"
    )

    val_samples = [train_data.samples[i] for i in val_indices]
    train_samples = [train_data.samples[i] for i in train_indices]

    dataset_kwargs["samples"] = train_samples
    train_data = dataset_cls(**dataset_kwargs)

    dataset_kwargs["samples"] = val_samples
    dataset_kwargs["is_val"] = True
    val_data = dataset_cls(**dataset_kwargs)
    return train_data, val_data


def main(args: argparse.Namespace) -> None:
    assert "NEPTUNE_PROJECT" in os.environ, "NEPTUNE_PROJECT env var not set"
    assert "NEPTUNE_API_TOKEN" in os.environ, "NEPTUNE_API_TOKEN env var not set"

    logger = get_logger("GRISP TRAIN", args.log_level)

    config = GRISPTrainConfig(**load_config(args.config))

    model, tokenizer = load_model_and_tokenizer(config)

    if config.gradient_checkpointing:
        # get rid of incompatibility warning
        model.config.use_cache = False  # type: ignore

    logger.info(f"Using model:\n{model}")
    total = model.num_parameters()  # type: ignore
    logger.info(f"Total parameters: {total / 1e9:.1f}B")
    trainable = model.num_parameters(only_trainable=True)  # type: ignore
    logger.info(
        f"Trainable parameters: {trainable / 1e6:.2f}M ({trainable / total:.2%})"
    )

    train_data, val_data = get_datasets(config, tokenizer, logger)
    collator = GRISPCollator(
        tokenizer.pad_token_id,  # type: ignore
        config.max_length,
        args.log_level,
    )

    run_name = os.path.basename(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    # save config
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = len(train_data) // effective_batch_size  # type: ignore
    logging_steps = max(1, steps_per_epoch // 100)  # log 100 times per epoch

    # eval once per epoch, but at least 10 times during training
    total_steps = steps_per_epoch * config.num_epochs
    eval_steps = max(1, min(steps_per_epoch, total_steps // 10))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="best",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=logging_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        seed=config.seed,
        bf16=True,
        report_to="none",
        run_name=run_name,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=config.gradient_checkpointing,
        torch_compile=config.do_compile,
        dataloader_num_workers=config.num_workers,
        dataloader_prefetch_factor=4 if config.num_workers > 0 else None,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        callbacks=[
            NeptuneCallback(
                name=run_name,
                base_namespace="grisp",
                tags=["grisp", "training"],
            ),
            EarlyStoppingCallback(
                early_stopping_patience=max(10, config.num_epochs // 10),
            ),
        ],
    )

    trainer.train()


if __name__ == "__main__":
    main(parse_args())
