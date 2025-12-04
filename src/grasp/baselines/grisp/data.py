import argparse
import collections
import os
import random
import re
import string

import torch
from grammar_utils.parse import LR1Parser
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.configs import KgConfig
from grasp.manager import KgManager, load_kg_manager
from grasp.sparql.item import Item, get_sparql_items
from grasp.sparql.types import Alternative, Selection
from grasp.sparql.utils import find_all
from grasp.tasks import SparqlQaSample
from grasp.utils import get_available_knowledge_graphs

BOI = "<iri>"
EOI = "</iri>"

BOR = "<rep>"
EOR = "</rep>"

ALT_LABELS = string.ascii_uppercase

IGNORE_INDEX = -100


class IRI(BaseModel):
    identifier: str
    label: str
    aliases: list[str]

    @staticmethod
    def from_item(item: Item, manager: KgManager) -> "IRI":
        label = item.alternative.label or item.alternative.get_identifier()
        aliases = item.alternative.aliases or []

        if item.variant is not None:
            # add variant to label and aliases in brackets
            #  e.g., "population (wdt)"
            label = f"{label} ({item.variant})"
            aliases = [f"{alias} ({item.variant})" for alias in aliases]

        identifier = manager.denormalize(
            item.alternative.identifier,
            item.obj_type,
            item.variant,
        )
        assert identifier is not None, "Failed to denormalize identifier"
        identifier = manager.format_iri(identifier)

        return IRI(
            identifier=identifier,
            label=label,
            aliases=aliases,
        )


class GRISPSample(BaseModel):
    kg: str
    questions: list[str]
    sparql: list[str | IRI]

    @property
    def has_placeholders(self) -> bool:
        return any(isinstance(part, IRI) for part in self.sparql)


def extract_value_from_nl_iri(nl_iri: dict) -> str:
    return nl_iri["value"][len(BOI) : -len(EOI)].strip()


def extract_query_and_variant_from_nl_iri(nl_iri: dict) -> tuple[str, str | None]:
    query = extract_value_from_nl_iri(nl_iri)
    variant: str | None = None

    m = re.search(r" \((.*)\)$", query)
    if m is not None:
        # remove variant in parentheses at the end
        query = query[: m.start()].strip()
        variant = m.group(1).strip()

    return query, variant


class Skeleton:
    @staticmethod
    def parse(sparql: str, parser: LR1Parser) -> "Skeleton":
        parse = parser.parse(sparql)
        return Skeleton(sparql, parse)

    def __init__(self, sparql: str, parse: dict) -> None:
        self.parse = parse
        self.sparql_encoded = sparql.encode()
        self.nl_iris = list(find_all(self.parse, "NL_IRI"))
        self.selections: list[Selection] = []
        self.identifiers: list[str] = []

    @property
    def replaced(self) -> int:
        return len(self.selections)

    @property
    def total(self) -> int:
        return len(self.nl_iris)

    @property
    def done(self) -> bool:
        return len(self.selections) >= len(self.nl_iris)

    def materialize(self) -> str:
        assert self.done, "Not all NL IRIs have been replaced"

        sparql = ""
        start = 0
        for nl_iri, identifier in zip(
            self.nl_iris,
            self.identifiers,
        ):
            byte_start, byte_end = nl_iri["byte_span"]
            sparql += self.sparql_encoded[start:byte_start].decode()
            sparql += identifier
            start = byte_end

        sparql += self.sparql_encoded[start:].decode()
        return sparql

    def prepare_for_selection(self) -> tuple[str, str, str, str | None]:
        assert not self.done, "All NL IRIs have already been replaced"
        idx = len(self.selections)

        prefix = ""
        start = 0
        for i in range(idx):
            nl_iri = self.nl_iris[i]
            byte_start, byte_end = nl_iri["byte_span"]

            identifier = self.identifiers[i]

            prefix += self.sparql_encoded[start:byte_start].decode()
            prefix += identifier
            start = byte_end

        nl_iri = self.nl_iris[idx]
        byte_start, byte_end = nl_iri["byte_span"]
        prefix += self.sparql_encoded[start:byte_start].decode()

        query, variant = extract_query_and_variant_from_nl_iri(nl_iri)
        value = extract_value_from_nl_iri(nl_iri)
        sparql = prefix + f"{BOR}{value}{EOR}" + self.sparql_encoded[byte_end:].decode()
        return prefix, sparql, query, variant

    def add_selection(self, selection: Selection, manager: KgManager) -> None:
        assert not self.done, "All NL IRIs have already been replaced"
        identifier = manager.denormalize(
            selection.alternative.identifier,
            selection.obj_type,
            selection.variant,
        )
        assert identifier is not None, "Failed to denormalize identifier"
        identifier = manager.format_iri(identifier)
        label = selection.alternative.get_label()
        if label is not None:
            label += f" ({identifier})"
        else:
            label = identifier

        self.selections.append(selection)
        self.identifiers.append(identifier)

    def pop_selection(self) -> Selection:
        assert len(self.selections) > 0, "No selections to pop"
        selection = self.selections.pop()
        self.identifiers.pop()
        return selection


def get_skeleton_prompt(
    kg: str,
    question: str,
    sparql: str | None = None,
) -> list[dict]:
    system = f"""\
You are an expert SPARQL query generator. \
Your task is to generate SPARQL query skeletons over \
the {kg} knowledge graph for answering user questions.
Instead of actual IRIs, you should generate natural language \
placeholders surrounded by {BOI} and {EOI} tags. \
The placeholders may contain optional additional information \
helpful for disambiguation in brackets, e.g., "population (wdt)" \
for wikidata properties."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    if sparql is not None:
        messages.append({"role": "assistant", "content": sparql})

    return messages


def format_alternatives(alternatives: list[Alternative]) -> str:
    if len(alternatives) == 0:
        return "No alternatives found"

    assert len(alternatives) < len(ALT_LABELS), (
        f"Number of alternatives must be less than {len(ALT_LABELS)}"
    )

    top_k_string = "\n".join(
        # dont show variants in the listing
        f"{lab}. {alt.get_selection_string(show_matched_alias=False, include_variants=[])}"
        for lab, alt in zip(ALT_LABELS, alternatives)
    )
    none_lab = ALT_LABELS[len(alternatives)]
    top_k_string += f"\n{none_lab}. None of the above"
    # in case only the none alternative is shown
    top_k_string = top_k_string.strip()

    return f"Alternatives:\n{top_k_string}"


def get_selection_prompt_and_options(
    manager: KgManager,
    question: str,
    sparql: str,
    selections: list[Selection],
    alternatives: list[Alternative],
) -> tuple[list[dict], list[str]]:
    system = f"""\
You are a SPARQL expert. Your task is to select the best fitting \
{manager.kg} item for replacing a natural-language placeholder \
in a SPARQL skeleton. The placeholder to be replaced is marked \
{BOR}...{EOR} in the skeleton. There may also be other unresolved \
placeholders coming afterwards, marked {BOI}...{EOI}.

You are given the user question, the SPARQL skeleton, \
info about already resolved placeholders, \
and a list of alternatives for the current placeholder. \
You should output the letter corresponding to the best \
fitting alternative."""

    messages = [
        {"role": "system", "content": system},
    ]

    user = f"Question:\n{question}\n\nSPARQL skeleton:\n{sparql}"
    if selections:
        user += f"\n\n{manager.format_selections(selections)}"

    user += f"\n\n{format_alternatives(alternatives)}"
    messages.append({"role": "user", "content": user})

    options = ALT_LABELS[: len(alternatives) + 1]  # including none option
    return messages, list(options)


def materialize_skeleton(parts: list[str | IRI], is_val: bool) -> str:
    formatted_parts = []
    for part in parts:
        if isinstance(part, str):
            formatted_parts.append(part)
            continue

        # choose main label 80% of the time,
        # otherwise choose a random alias if available
        if not is_val and part.aliases and random.random() < 0.2:
            iri = random.choice(part.aliases)
        else:
            iri = part.label

        formatted_parts.append(f"{BOI}{iri}{EOI}")

    return "".join(formatted_parts)


def materialize_sparql(parts: list[str | IRI]) -> str:
    formatted_parts = []
    for part in parts:
        if isinstance(part, str):
            formatted_parts.append(part)
            continue

        formatted_parts.append(part.identifier)

    return "".join(formatted_parts)


def materialize_sample(sample: GRISPSample, is_val: bool = False) -> tuple[str, str]:
    if is_val:
        question = sample.questions[0]
    else:
        question = random.choice(sample.questions)

    return question, materialize_skeleton(sample.sparql, is_val)


def get_ranges(
    messages: list[dict[str, str]],
    text: str,
    role: str,
) -> list[tuple[int, int]]:
    ranges = []
    start = 0

    for i, message in enumerate(messages):
        if message["role"] != role:
            continue

        start = text.find(message["content"], start)
        assert start != -1, "Message content not found in conversation transcript"

        next_message = messages[i + 1] if i + 1 < len(messages) else None
        if next_message is None:
            ranges.append((start, len(text)))
            break

        next_content = next_message["content"]
        end = text.find(next_content, start + len(message["content"]))
        assert end != -1, "Next message content not found in conversation transcript"
        ranges.append((start, end))
        start = end

    return ranges


def get_masked_labels(
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    ranges: list[tuple[int, int]],
) -> list[int]:
    labels = []

    for token_id, (off_start, off_end) in zip(token_ids, offsets):
        if any(off_start >= start and off_end <= end for start, end in ranges):
            labels.append(token_id)
        else:
            labels.append(IGNORE_INDEX)

    return labels


def tokenize_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    mask_inputs: bool,
) -> dict:
    if not mask_inputs:
        enc: dict = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            enable_thinking=False,
        )  # type: ignore
        enc["labels"] = enc["input_ids"]  # type: ignore
        return enc  # type: ignore

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
    )  # type: ignore
    assert isinstance(text, str)
    enc = tokenizer(text, return_offsets_mapping=True)  # type: ignore
    ranges = get_ranges(messages, text, "assistant")
    labels = get_masked_labels(
        enc["input_ids"],  # type: ignore
        enc["offset_mapping"],  # type: ignore
        ranges,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def format_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def load_samples(file_paths: list[str]) -> list[GRISPSample]:
    samples = []
    for path in file_paths:
        loaded_samples = load_jsonl(path)
        samples.extend((GRISPSample(**sample) for sample in loaded_samples))
    return samples


class GRISPSkeletonDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        is_val: bool = False,
        log_level: str | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs
        self.is_val = is_val

        self.logger = get_logger(f"GRISP SKELETON DATASET ({is_val=})", log_level)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        question, skeleton = materialize_sample(sample, self.is_val)
        prompt = get_skeleton_prompt(sample.kg, question, skeleton)
        self.logger.debug(f"Sample {idx}:\n{format_messages(prompt)}")
        return tokenize_messages(prompt, self.tokenizer, self.mask_inputs)


class GRISPSelectionDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPSample],
        manager: KgManager,
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        is_val: bool = False,
        log_level: str | None = None,
    ) -> None:
        self.parser = load_sparql_parser()
        self.manager = manager
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs
        self.is_val = is_val

        self.logger = get_logger(f"GRISP SELECTION DATASET ({is_val=})", log_level)

        self.samples = [sample for sample in samples if sample.has_placeholders]
        self.logger.info(
            f"Filtered {len(samples):,} samples to "
            f"{len(self.samples):,} samples with placeholders"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        question, skeleton = materialize_sample(sample, self.is_val)
        sparql = materialize_sparql(sample.sparql)

        _, items = get_sparql_items(sparql, self.manager)
        items = [item for item in items if not item.is_other_or_literal]
        assert len(items) > 0, "No valid item to replace found in sample"

        skeleton = Skeleton.parse(skeleton, self.parser)

        upper = random.randint(0, len(items) - 1)
        for item in items[:upper]:
            skeleton.add_selection(item.selection, self.manager)

        item = items[upper]
        target_alt = item.selection.alternative
        self.logger.debug(f"Target alternative: {target_alt.get_selection_string()}")

        # prefix and variant not used during training
        # because we dont autocomplete for efficiency
        # and only check for variant after selection
        _, sparql, query, _ = skeleton.prepare_for_selection()

        k = 10 if self.is_val else random.randint(2, 15)

        alternatives = self.manager.get_selection_alternatives(
            query,
            # None means search in the full index corresponding to the obj_type
            {item.obj_type: None},
            k,
        )
        alternatives = alternatives.get(item.obj_type, [])

        drop_infos = not self.is_val and random.random() < 0.2
        drop_target = not self.is_val and random.random() < 0.2
        shuffle_alts = not self.is_val and random.random() < 0.2
        self.logger.debug(
            f"Augmentations: {drop_infos=}, {drop_target=}, {shuffle_alts=}"
        )

        if shuffle_alts:
            # shuffle all alternatives to counter position bias
            # only the None alternative should always be last
            none_alt = alternatives.pop()
            random.shuffle(alternatives)
            alternatives.append(none_alt)

        target_option: int | None = None
        for i, alt in enumerate(alternatives):
            if drop_infos and alt.infos:
                alt.infos.clear()

            if alt != target_alt:
                continue

            # drop target alternative 20% of the time during training
            if drop_target:
                alternatives.pop(i)
            else:
                target_option = i

            break

        prompt, options = get_selection_prompt_and_options(
            self.manager,
            question,
            sparql,
            skeleton.selections,
            alternatives,
        )

        # if target option is None, we need to select the last
        # option, which is the "None of the above" option
        option = options[-1] if target_option is None else options[target_option]
        prompt.append({"role": "assistant", "content": option})

        self.logger.debug(f"Sample {idx}:\n{format_messages(prompt)}")
        return tokenize_messages(prompt, self.tokenizer, self.mask_inputs)


def pad(values: list[list[int]], pad_value: int, max_length: int) -> torch.Tensor:
    padded = []

    max_length = min(max(len(seq) for seq in values), max_length)
    # pad max length to multiple of 8 for better efficiency
    if max_length % 8 != 0:
        max_length += 8 - (max_length % 8)

    for seq in values:
        if len(seq) <= max_length:
            seq = seq + [pad_value] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        padded.append(seq)

    return torch.tensor(padded, dtype=torch.long)


class GRISPCollator:
    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.max_length = max_length
        self.pad_values = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "labels": IGNORE_INDEX,
        }

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        assert len(batch) > 0, "Batch must not be empty"
        return {
            key: pad(
                [sample[key] for sample in batch],
                pad_value=self.pad_values[key],
                max_length=self.max_length,
            )
            for key in batch[0]
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data for GRISP model training"
    )
    parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=get_available_knowledge_graphs(),
        help="Knowledge graph to prepare data for",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="SPARQL endpoint for the knowledge graph",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file containing query-SPARQL pairs",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output JSONL file to save the processed data",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the samples into usable inputs for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    manager = load_kg_manager(KgConfig(kg=args.knowledge_graph, endpoint=args.endpoint))
    logger = get_logger("GRISP DATA", level="INFO")

    if os.path.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output_file} already exists. "
            f"Use --overwrite to overwrite it."
        )

    samples = load_jsonl(args.input_file)
    logger.info(f"Loaded {len(samples):,} samples from {args.input_file}")
    random.seed(args.seed)

    invalid = 0
    error = 0
    outputs = []
    for sample in tqdm(samples, desc="Preparing samples"):
        sample = SparqlQaSample(**sample)

        try:
            sparql = manager.fix_prefixes(
                sample.sparql,
                remove_known=True,
            )
            sparql = manager.prettify(sparql)
            sparql, items = get_sparql_items(
                sparql,
                manager,
            )

            if any(item.invalid for item in items):
                invalid += 1
                logger.debug(f"Invalid sample {sample.id}:\n{sparql}")
                continue

            parts = []
            start = 0
            for item in items:
                # others can only be invalid, which is already checked above
                # literals should be predicted directly
                if item.is_other_or_literal:
                    continue

                item_start, item_end = item.item_span

                parts.append(sparql[start:item_start])
                parts.append(IRI.from_item(item, manager))

                start = item_end

            if start < len(sparql):
                parts.append(sparql[start:])

            grisp_sample = GRISPSample(
                kg=args.knowledge_graph,
                questions=[sample.question] + sample.paraphrases,
                sparql=parts,
            )

            outputs.append(grisp_sample.model_dump())

        except Exception as e:
            error += 1
            logger.debug(
                f"Error processing sample {sample.id}:\n{sample.sparql}\n\n{e}"
            )
            continue

    dump_jsonl(outputs, args.output_file)

    logger.info(f"Total samples processed: {len(samples):,}")
    inv_frac = invalid / len(samples)
    logger.info(f"Total invalid samples skipped: {invalid:,} ({inv_frac:.2%})")
    err_frac = error / len(samples)
    logger.info(f"Total errors encountered: {error:,} ({err_frac:.2%})")


if __name__ == "__main__":
    main(parse_args())
