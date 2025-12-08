import argparse
import glob
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger

import torch
from grammar_utils.parse import LR1Parser
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_json, dump_jsonl, load_json, load_jsonl
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import extract_field

from grasp.baselines.grisp.data import (
    ALT_LABELS,
    Skeleton,
    format_alternatives,
    get_selection_prompt_and_options,
    get_skeleton_prompt,
)
from grasp.baselines.grisp.train import GRISPTrainConfig
from grasp.baselines.grisp.utils import load_sparql_parser, set_chat_template
from grasp.configs import KgConfig
from grasp.manager import KgManager, load_kg_manager
from grasp.sparql.types import (
    Alternative,
    ObjType,
    Position,
    Selection,
)
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRISP model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to GRISP run configuration",
    )
    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the training run directory",
    )
    parser.add_argument(
        "--selection-run",
        type=str,
        default=None,
        help="Path to the training run directory for the selection model, "
        "if different from the main model",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, etc.)",
    )

    # add two subparsers: run and file for running grisp
    # on a single question and on a benchmark file
    input_parsers = parser.add_subparsers(dest="command", required=True)
    run_parser = input_parsers.add_parser(
        "run",
        help="Run GRISP on a single question",
    )
    run_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Question to run GRISTP on, if not given, read from stdin",
    )
    run_parser.add_argument(
        "-if",
        "--input-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Format of the input (raw text or JSON)",
    )
    run_parser.add_argument(
        "--input-field",
        type=str,
        default="question",
        help="Field to extract input from",
    )

    file_parser = input_parsers.add_parser(
        "file",
        help="Run GRISP on a benchmark file",
    )
    file_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar",
    )
    file_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to file in JSONL format to run GRISP on, if not given, read JSONL from stdin",
    )
    file_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the inputs",
    )
    file_parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N inputs",
    )
    file_parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Limit number of inputs (after skipping) to N",
    )
    file_parser.add_argument(
        "--input-field",
        type=str,
        default="question",
        help="Field to extract input from",
    )
    file_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="File to write the output to",
    )
    file_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed inputs (only used with --output-file)",
    )
    file_parser.add_argument(
        "--none-output-invalid",
        action="store_true",
        help="Consider outputs with None as invalid (only used with --retry-failed)",
    )
    file_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    return parser.parse_args()


MAX_IRIS = 131_072


class GRISPRunConfig(BaseModel):
    kg: str
    endpoint: str | None = None

    temperature: float | None = 0.4
    min_p: float | None = None
    top_k: int | None = None
    top_p: float | None = 0.9
    repeat_penalty: float | None = None
    do_sample: bool = True

    skeleton_n: int = 8
    skeleton_top_k: int = 3

    selection_top_k: int = 3
    autocomplete: bool = True
    backtrack: bool = True
    rerank: bool = True
    check_empty: bool = True


def generate_skeletons(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    parser: LR1Parser,
    logger: Logger,
) -> list[Skeleton]:
    input = get_skeleton_prompt(manager.kg, question)

    device = next(model.parameters()).device
    enc = tokenizer.apply_chat_template(
        input,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)  # type: ignore
    prompt_length = enc["input_ids"].shape[1]  # type: ignore

    fmt = tokenizer.decode(enc["input_ids"][0])  # type: ignore
    logger.debug(f"Generating skeletons:\n{fmt}")

    outputs = model.generate(  # type: ignore
        **enc,
        generation_config=GenerationConfig(
            num_beams=cfg.skeleton_n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            repetition_penalty=cfg.repeat_penalty,
            do_sample=cfg.do_sample,
            max_new_tokens=1024,
            renormalize_logits=True,
            num_return_sequences=cfg.skeleton_n,
            return_dict_in_generate=True,
            output_scores=True,
        ),
    )

    skeletons = []
    seen = set()
    for i in range(len(outputs["sequences"])):
        token_ids = outputs["sequences"][i]
        decoded_token_ids = token_ids[prompt_length:]
        decoded = tokenizer.decode(decoded_token_ids, skip_special_tokens=True)

        if cfg.skeleton_n > 1:
            score = outputs["sequences_scores"][i].item()
            logger.debug(f"Generated skeleton with score={score:.5f}:\n{decoded}")
        else:
            logger.debug(f"Generated skeleton:\n{decoded}")

        if decoded in seen:
            logger.debug("Already seen skeleton, skipping")
            continue

        try:
            skeleton = Skeleton.parse(decoded, parser)
        except Exception as e:
            logger.warning(f"Failed to parse skeleton, skipping: {e}")
            continue

        seen.add(decoded)
        skeletons.append(skeleton)

    # only take top k skeletons, others are just for logging
    logger.debug(
        f"Generated {len(skeletons)} valid unique skeletons, "
        f"taking top {cfg.skeleton_top_k}"
    )
    return skeletons[: cfg.skeleton_top_k]


def reorder_alternatives(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    manager: KgManager,
    question: str,
    sparql: str,
    selections: list[Selection],
    alternatives: list[Alternative],
    logger: Logger,
) -> list[Alternative]:
    prompt, options = get_selection_prompt_and_options(
        manager,
        question,
        sparql,
        selections,
        alternatives,
    )

    input_ids = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )  # type: ignore

    fmt = tokenizer.decode(input_ids)  # type: ignore
    logger.debug(f"Reranking alternatives:\n{fmt}")

    device = next(model.parameters()).device
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    option_ids = []
    for option in options:
        option_token_ids = tokenizer.encode(
            option,
            add_special_tokens=False,
        )  # type: ignore
        assert len(option_token_ids) == 1, "Option must be a single token"
        option_ids.append(option_token_ids[0])

    option_ids = torch.tensor(option_ids, dtype=torch.long, device=device)

    # shape [1, S, V]
    with torch.inference_mode():
        logits = model(input_ids.unsqueeze(0)).logits

    # get last logits [V]
    logits = logits[0, -1]
    # get option logits, [|O|]
    logits = logits[option_ids]
    # normalize options
    scores = torch.softmax(logits, dim=-1)
    # sort
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    sorted_scores = sorted_scores.tolist()
    sorted_indices = sorted_indices.tolist()
    logger.debug(
        "Reranked alternatives:\n"
        + "\n".join(
            f"Rank {i + 1}: "
            f"{ALT_LABELS[index] if index < len(alternatives) else 'None'} "
            f"(score={score:.2%})"
            for i, (score, index) in enumerate(zip(sorted_scores, sorted_indices))
        )
    )

    reranked = []
    for index in sorted_indices:
        if index == len(alternatives):
            break

        option = ALT_LABELS[index]
        alt = alternatives[index]
        reranked.append(alt)
    logger.debug(
        f"Taking the top {len(reranked)} reranked alternatives "
        "before 'None' for selection in order"
    )

    return reranked


@dataclass
class Alternatives:
    alternatives: deque[Alternative]
    obj_type: ObjType

    @property
    def is_empty(self) -> bool:
        return len(self.alternatives) == 0

    def pop(self) -> Alternative:
        assert not self.is_empty, "No alternatives left"
        return self.alternatives.popleft()


def find_alternatives(
    manager: KgManager,
    cfg: GRISPRunConfig,
    prefix: str,
    query: str,
    logger: Logger,
) -> Alternatives:
    try:
        logger.debug(f"Autocompleting SPARQL query prefix:\n{prefix}")
        autocomp_sparql, query_type, position = manager.autocomplete_prefix(
            prefix,
            limit=MAX_IRIS + 1,
        )
        logger.debug(
            f"Determined query type and position from prefix: "
            f"'{query_type}', '{position.value}'"
        )
    except Exception as e:
        logger.warning(f"Error autocompleting SPARQL prefix: {e}")
        # select all triples as fallback
        autocomp_sparql = "SELECT * WHERE {{ ?s ?p ?o }} LIMIT 0"
        query_type = "select"
        # if autocompletion fails, we are typically at a property
        position = Position.PROPERTY

    search_items = manager.get_default_search_items(position)
    if cfg.autocomplete and query_type != "ask":
        try:
            logger.debug(
                f"Searching for fitting IRIs at position {position.value} "
                f"with autocompletion SPARQL:\n{autocomp_sparql}"
            )
            search_items = manager.get_search_items(autocomp_sparql, position, MAX_IRIS)

            total_items = sum(len(v) for v in search_items.values())
            logger.debug(
                f"Got {total_items} fitting IRIs for position {position.value}"
            )
        except Exception as e:
            logger.warning(f"Error getting fitting IRIs: {e}")
    else:
        logger.debug("Skipping autocompletion of fitting IRIs, all used")

    logger.debug(
        f"Searching with query '{query}' from natural-language IRI in fitting IRIs"
    )
    alternatives = manager.get_selection_alternatives(
        query,
        search_items,
        cfg.selection_top_k,
    )

    if position == Position.PROPERTY:
        obj_type = ObjType.PROPERTY
    else:
        obj_type = ObjType.ENTITY

    alternatives = alternatives.get(obj_type, [])

    logger.debug(
        f"Found {len(alternatives)} alternatives:\n{format_alternatives(alternatives)}"
    )

    return Alternatives(deque(alternatives), obj_type)


def is_api_failure(exception: Exception) -> bool:
    exc_msg = str(exception).lower()
    return "read timeout" in exc_msg or "503" in exc_msg or "504" in exc_msg


def replace_iris_left_to_right(
    skeleton: Skeleton,
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    logger: Logger,
) -> str | None:
    # init empty memo
    memo: dict[str, Alternatives] = {}

    while True:
        if skeleton.done:
            if not cfg.check_empty:
                break

            try:
                # reject empty queries
                sparql = skeleton.materialize()
                logger.debug(f"Checking result of final SPARQL query:\n{sparql}")
                result = manager.execute_sparql(skeleton.materialize())
                logger.debug(f"Result:\n{manager.format_sparql_result(result)}")
                reject = result.is_empty
            except Exception as e:
                logger.warning(f"Error executing final SPARQL to check emptiness: {e}")
                reject = not is_api_failure(e)

            if not reject:
                break

            elif skeleton.replaced == 0 or not cfg.backtrack:
                logger.debug("Final SPARQL query is empty, abandoning skeleton")
                return None

            logger.debug(
                "Final SPARQL query is empty, backtracking to previous placeholder"
            )
            skeleton.pop_selection()
            continue

        prefix, sparql, query, variant = skeleton.prepare_for_selection()

        # set alternatives for current placeholder
        if prefix not in memo:
            alternatives = find_alternatives(manager, cfg, prefix, query, logger)
            if cfg.rerank:
                # use model to rerank alternatives before selecting
                reranked = reorder_alternatives(
                    model,
                    tokenizer,
                    manager,
                    question,
                    sparql,
                    skeleton.selections,
                    list(alternatives.alternatives),
                    logger,
                )

                alternatives.alternatives = deque(reranked)

            memo[prefix] = alternatives

        alternatives = memo[prefix]
        if alternatives.is_empty:
            if skeleton.replaced == 0:
                logger.debug(
                    "No valid alternatives left for the first placeholder, abandoning skeleton"
                )
                return None

            elif not cfg.backtrack:
                logger.debug(
                    "No valid alternatives left for the current placeholder, "
                    "abandoning skeleton due to backtracking disabled"
                )
                return None

            logger.debug(
                "No valid alternatives left for the current placeholder, backtracking"
            )
            skeleton.pop_selection()
            continue

        # just try out next alternative in order
        alternative = alternatives.pop()
        if not alternative.variants:
            # just to be sure to have no parsing errors
            variant = None

        if variant is not None:
            assert alternative.variants, "Expected variants for alternative"
            if variant not in alternative.variants:
                logger.debug(
                    f"Variant '{variant}' not found in alternative variants, "
                    f"trying next alternative"
                )
                continue

            logger.debug(
                f"Variant '{variant}' found in alternative "
                f"variants ({alternative.variants})"
            )

        show_variants = [variant] if variant is not None else None
        logger.debug(
            f"Adding the following alternative at placholder {skeleton.replaced}/{skeleton.total}:\n"
            f"{alternative.get_selection_string(include_variants=show_variants)} "
        )
        selection = Selection(
            alternative=alternative,
            variant=variant,
            obj_type=alternatives.obj_type,
        )
        skeleton.add_selection(selection, manager)

    # convert back to sparql, fix prefixes, and prettify
    sparql = skeleton.materialize()
    sparql = manager.fix_prefixes(sparql)
    return manager.prettify(sparql)


def run(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    parser: LR1Parser,
    logger: Logger,
    select_model: PreTrainedModel | PeftModel | None = None,
    select_tokenizer: PreTrainedTokenizerBase | None = None,
) -> str | None:
    skeletons = generate_skeletons(
        model,
        tokenizer,
        cfg,
        question,
        manager,
        parser,
        logger,
    )

    for skeleton in skeletons:
        sparql = replace_iris_left_to_right(
            skeleton,
            select_model or model,
            select_tokenizer or tokenizer,
            cfg,
            question,
            manager,
            logger,
        )

        # take first fully assigned skeleton as final answer
        if sparql is not None:
            return sparql

    return None


def is_invalid_output(output: dict | None, none_output_invalid: bool = False) -> bool:
    return (
        output is None
        or output.get("error") is not None
        or (output["output"] is None and none_output_invalid)
    )


def find_best_checkpoint(run_directory: str) -> str:
    # all subdir starting with checkpoint-*
    checkpoints = glob.glob(os.path.join(run_directory, "checkpoint-*"))
    assert len(checkpoints) > 0, "No checkpoints found"

    def best_ckpt_key(checkpoint_dir: str) -> int | float:
        path = os.path.join(checkpoint_dir, "trainer_state.json")
        state = load_json(path)
        global_step = state["global_step"]

        log_entry = next(
            (
                entry
                for entry in state["log_history"]
                if entry["step"] == global_step and entry.get("eval_loss") is not None
            ),
        )
        # sort by eval loss
        return log_entry["eval_loss"]

    checkpoints.sort(key=best_ckpt_key)
    return checkpoints[0]


def load_model_and_tokenizer(
    directory: str,
    device: str,
    logger: Logger,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
    checkpoint = find_best_checkpoint(directory)
    logger.info(f"Best checkpoint found at {checkpoint}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype="auto",
        device_map=device,
    )
    model.config.use_cache = True
    model.eval()
    logger.info(f"Loaded model {model.config.name_or_path}:\n{model}")

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    tokenizer = set_chat_template(tokenizer)

    return model, tokenizer


def main(args: argparse.Namespace) -> None:
    logger = get_logger("GRISP", args.log_level)

    train_cfg_path = os.path.join(args.run_directory, "config.yaml")
    train_cfg = GRISPTrainConfig(**load_config(train_cfg_path))
    assert train_cfg.type in ["skeleton", "both"], (
        "Main run should either be of type 'skeleton' or 'both'"
    )

    run_cfg = GRISPRunConfig(**load_config(args.config))
    logger.debug(f"Using run configuration:\n{run_cfg.model_dump_json(indent=2)}")

    logger.info(f"Loading model from {args.run_directory}")
    model, tokenizer = load_model_and_tokenizer(args.run_directory, args.device, logger)

    skeleton_model, skeleton_tokenizer = model, tokenizer
    selection_model, selection_tokenizer = None, None

    if train_cfg.type == "skeleton" and args.selection_run is None:
        logger.warning(
            "Main model is skeleton only, selection quality may be suboptimal"
        )
    elif train_cfg.type == "skeleton":
        logger.info(f"Loading selection model from {args.selection_run}")
        selection_model, selection_tokenizer = load_model_and_tokenizer(
            args.selection_run,
            args.device,
            logger,
        )

    logger.info(
        f"Using model {skeleton_model.config.name_or_path} for skeleton generation"  # type: ignore
        + (" and selection" if selection_model is None else "")
    )
    if selection_model is not None:
        logger.info(
            f"Using separate model {selection_model.config.name_or_path} for selection"  # type: ignore
        )

    manager = load_kg_manager(KgConfig(kg=run_cfg.kg, endpoint=run_cfg.endpoint))
    parser = load_sparql_parser()

    # adapted from GRASP cli
    run_on_file = args.command == "file"
    outputs = []
    if run_on_file:
        if args.input_file is None:
            inputs = [json.loads(line) for line in sys.stdin]
        else:
            inputs = load_jsonl(args.input_file)

        # id fallback in case of missing ids
        # before shuffling/skipping/taking
        for i, ipt in enumerate(inputs):
            id = extract_field(ipt, "id")
            if id is None:
                ipt["id"] = str(i)

        if args.shuffle:
            assert train_cfg.seed is not None, (
                "Seed must be set for deterministic shuffling"
            )
            random.seed(train_cfg.seed)
            random.shuffle(inputs)

        skip = max(0, args.skip)
        take = args.take or len(inputs)
        inputs = inputs[skip : skip + take]

        if args.output_file:
            if os.path.exists(args.output_file) and not args.overwrite:
                outputs = load_jsonl(args.output_file)

            # save info in config file next to output file
            output_stem, _ = os.path.splitext(args.output_file)
            config_file = output_stem + ".config.json"

            dump_json(train_cfg.model_dump(), config_file, indent=2)

        if args.progress:
            # wrap with tqdm
            inputs = tqdm(inputs, desc="GRISP")

    else:
        if args.input is None:
            ipt = sys.stdin.read()
        else:
            ipt = args.input

        if args.input_format == "json":
            inputs = [json.loads(ipt)]
        else:
            inputs = [{"question": ipt}]
            args.input_field = "question"  # overwrite

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id") or "unknown"

        ipt = extract_field(ipt, args.input_field)
        assert ipt is not None, f"Question not found for input {i:,}"

        if i < len(outputs):
            # overwrite id
            output = outputs[i]
            output["id"] = id
            if not args.retry_failed or not is_invalid_output(
                output,
                args.none_output_invalid,
            ):
                continue

        output = {
            "id": id,
            "type": "output",
            "error": None,
            "output": None,
            "config": run_cfg.model_dump(),
        }

        start = time.perf_counter()

        try:
            sparql = run(
                skeleton_model,
                skeleton_tokenizer,
                run_cfg,
                ipt,
                manager,
                parser,
                logger,
                selection_model,
                selection_tokenizer,
            )

            out = {
                "sparql": None,
                "kg": manager.kg,
                "selections": None,
                "result": None,
                "endpoint": manager.endpoint,
                "formatted": "No SPARQL generated",
            }
            if sparql is not None:
                result, selections = prepare_sparql_result(
                    sparql,
                    manager.kg,
                    [manager],
                    max_rows=10,
                    max_columns=10,
                )
                out["sparql"] = result.sparql
                out["selections"] = manager.format_selections(selections)
                out["result"] = result.formatted
                out["formatted"] = format_sparql_result(manager, result, selections)

            output["output"] = out

        except Exception as e:
            logger.error(f"Error processing input {i:,} (id={id}): {e}")
            output["error"] = {
                "reason": "failure",
                "content": str(e),
            }

        end = time.perf_counter()
        output["elapsed"] = end - start

        if not run_on_file:
            print(json.dumps(output))
            break

        elif args.output_file is None:
            print(json.dumps(output))
            continue

        if i < len(outputs):
            outputs[i] = output
        else:
            outputs.append(output)

        dump_jsonl(outputs, args.output_file)

    if run_on_file and args.output_file is not None:
        # final dump, necessary if no new outputs were added
        # but some outputs were updated with ids
        dump_jsonl(outputs, args.output_file)


if __name__ == "__main__":
    main(parse_args())
