import argparse
import os
import random
import re
import string
from dataclasses import dataclass
from logging import Logger

import torch
from grammar_utils.parse import LR1Parser  # type: ignore
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging

from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.configs import KgConfig, KgInfo
from grasp.manager import KgManager, load_kg_manager
from grasp.sparql.item import Item, extract_sparql_items
from grasp.sparql.types import Alternative, ObjType, Position, Selection
from grasp.sparql.utils import find_all, infer_position_from_prefix, query_type
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.utils import format_list, get_available_knowledge_graphs

BOI = "<iri>"
EOI = "</iri>"

BOR = "<rep>"
EOR = "</rep>"

ALT_LABELS = string.ascii_uppercase + string.digits

IGNORE_INDEX = -100

Messages = list[dict[str, str]]
AlternativeGroups = dict[ObjType, list[Alternative]]
OrderedAlternatives = list[tuple[Alternative, ObjType, str | None]]


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
            item.obj_type.index_name,
            item.variant,
        )
        assert identifier is not None, (
            f"Failed to denormalize identifier {item.alternative.identifier}"
        )
        identifier = manager.format_iri(identifier)

        return IRI(identifier=identifier, label=label, aliases=aliases)


class GRISPSample(BaseModel):
    kg: str
    questions: list[str]
    sparql: list[str | IRI]

    @property
    def has_placeholders(self) -> bool:
        return any(isinstance(part, IRI) for part in self.sparql)


class GRISPMaterializedSample(BaseModel):
    skeletons: list[Messages]
    selections: list[Messages]

    @property
    def has_skeletons(self) -> bool:
        return len(self.skeletons) > 0

    @property
    def has_selections(self) -> bool:
        return len(self.selections) > 0


def extract_value_from_nl_iri(nl_iri: dict) -> str:
    return nl_iri["value"][len(BOI) : -len(EOI)].strip()


def extract_query_and_variant_from_nl_iri(nl_iri: dict) -> tuple[str, str | None]:
    query = extract_value_from_nl_iri(nl_iri)
    variant: str | None = None

    m = re.search(r" \(([^)]*)\)$", query)
    if m is not None:
        # remove variant in parentheses at the end
        query = query[: m.start()].strip()
        variant = m.group(1).strip()

    return query, variant


@dataclass
class Info:
    prefix: str
    sparql: str
    query: str
    variant: str | None
    value: str

    def build_queries(
        self,
        obj_types: dict[ObjType, bool],
    ) -> dict[ObjType, tuple[str, str | None]]:
        queries = {}
        for obj_type, supports_variants in obj_types.items():
            if supports_variants and self.variant is not None:
                queries[obj_type] = (self.query, self.variant)
            else:
                queries[obj_type] = (self.value, None)
        return queries


class Skeleton:
    @staticmethod
    def parse(sparql: str, parser: LR1Parser) -> "Skeleton":
        sparql_parse = parser.parse(sparql)
        return Skeleton(sparql, sparql_parse)

    def __init__(self, sparql: str, sparql_parse: dict) -> None:
        self.sparql_parse = sparql_parse
        self.sparql_encoded = sparql.encode()
        self.nl_iris = list(find_all(self.sparql_parse, "NL_IRI"))
        self.selections: list[Selection] = []
        self.identifiers: list[str] = []

    @property
    def nl_sparql(self) -> str:
        return self.sparql_encoded.decode()

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

    def prepare_for_selection(self) -> Info:
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
        return Info(
            prefix=prefix,
            sparql=sparql,
            query=query,
            variant=variant,
            value=value,
        )

    def add_selection(self, selection: Selection, manager: KgManager) -> None:
        assert not self.done, "All NL IRIs have already been replaced"
        identifier = manager.denormalize(
            selection.alternative.identifier,
            selection.obj_type.index_name,
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


def ordered_alternatives(
    alternative_groups: AlternativeGroups,
    queries: dict[ObjType, tuple[str, str | None]],
) -> OrderedAlternatives:
    return ordered_alternatives_with_interleave(
        alternative_groups,
        queries,
        interleave=False,
    )


def ordered_alternatives_with_interleave(
    alternative_groups: AlternativeGroups,
    queries: dict[ObjType, tuple[str, str | None]],
    interleave: bool = False,
) -> OrderedAlternatives:
    variants = {obj_type: variant for obj_type, (_, variant) in queries.items()}

    entities = [
        (alternative, ObjType.ENTITY, variants.get(ObjType.ENTITY))
        for alternative in alternative_groups.get(ObjType.ENTITY, [])
    ]
    properties = [
        (alternative, ObjType.PROPERTY, variants.get(ObjType.PROPERTY))
        for alternative in alternative_groups.get(ObjType.PROPERTY, [])
    ]

    if not interleave or not entities or not properties:
        return entities + properties

    ordered = []
    max_len = max(len(entities), len(properties))
    for i in range(max_len):
        if i < len(entities):
            ordered.append(entities[i])
        if i < len(properties):
            ordered.append(properties[i])
    return ordered


def count_alternatives(alternatives: OrderedAlternatives) -> int:
    return len(alternatives)


def format_alternatives(alternatives: OrderedAlternatives) -> str:
    if len(alternatives) == 0:
        return "No alternatives found"

    assert len(alternatives) < len(ALT_LABELS) - 1, (
        f"Number of alternatives must be less than {len(ALT_LABELS) - 1}"
    )

    grouped = {}

    for label, (alternative, obj_type, variant) in zip(ALT_LABELS, alternatives):
        alt = alternative.get_selection_string(
            show_matched_label=False,
            include_variants=[variant] if variant else None,
        )
        if obj_type not in grouped:
            grouped[obj_type] = []
        grouped[obj_type].append(f"{label}. {alt}")

    alt_groups = []
    for obj_type, alts in grouped.items():
        alt_group = f"{obj_type.value.capitalize()} alternatives:\n"
        alt_group += "\n".join(alts)
        alt_groups.append(alt_group)

    # always add none alternative
    none_lab = ALT_LABELS[len(alternatives)]
    alt_groups.append(f"{none_lab}. None of the above")
    top_k_string = "\n\n".join(alt_groups)

    return top_k_string


def find_alternative_groups(
    manager: KgManager,
    prefix: str,
    queries: dict[ObjType, tuple[str, str | None]],
    top_k: int,
    logger: Logger,
    skip_constraint: bool = False,
    max_candidates: int | None = None,
) -> AlternativeGroups:
    sparql_query_type = query_type(prefix, manager.sparql_parser, is_prefix=True)
    constraint_sparql = None

    try:
        if not skip_constraint and sparql_query_type == "select":
            logger.debug(f"Deriving constraint query from SPARQL prefix:\n{prefix}")
            constraint_sparql, position = manager.derive_constraint_query_from_prefix(
                prefix,
                limit=None if max_candidates is None else max_candidates + 1,
            )
            logger.debug(f"Derived constraint SPARQL:\n{constraint_sparql}")
        else:
            position = infer_position_from_prefix(prefix, manager.sparql_parser)

        logger.debug(
            f"Determined query type and position: "
            f"'{sparql_query_type}', '{position.value}'"
        )
    except Exception as e:
        logger.debug(f"Error analyzing SPARQL prefix: {e}")
        # full search across both indices as fallback
        position = None

    if position is None:
        obj_types = [ObjType.ENTITY, ObjType.PROPERTY]
    elif position == Position.PROPERTY:
        obj_types = [ObjType.PROPERTY]
    else:
        obj_types = [ObjType.ENTITY]

    identifier_maps: dict[ObjType, dict[str, list[str]] | None] = {
        obj_type: None for obj_type in obj_types
    }
    if not skip_constraint and position is not None and constraint_sparql is not None:
        # can only be one
        obj_type = obj_types[0]
        try:
            logger.debug(
                f"Searching for candidate IRIs at position {position.value} "
                f"with constraint SPARQL:\n{constraint_sparql}"
            )
            identifier_map = manager.get_candidate_ids(
                obj_type.index_name,
                constraint_sparql,
                max_candidates,
                # 6 seconds to execute query, 3 to read result
                request_timeout=(3.5, 6.0),
                read_timeout=3.0,
            )
            logger.debug(
                f"Got {len(identifier_map)} candidate IRIs for position {position.value}"
            )
            identifier_maps[obj_type] = identifier_map
        except Exception as e:
            logger.warning(f"Error getting candidate IRIs: {e}")
    else:
        logger.debug(
            "Skipping constraint-based IRI filtering, full search will be performed"
        )

    alternative_groups = {}
    for obj_type in obj_types:
        assert obj_type in queries, f"Missing query for object type {obj_type}"
        query, _ = queries[obj_type]
        logger.debug(f"Searching with query '{query}' in '{obj_type.index_name}' index")
        alternatives = manager.search_index(
            obj_type.index_name,
            query,
            top_k,
            identifier_maps[obj_type],
        )
        if alternatives:
            alternative_groups[obj_type] = alternatives

    logger.debug(
        "Found "
        f"{count_alternatives(ordered_alternatives(alternative_groups, queries))} alternatives:\n"
        + format_alternatives(ordered_alternatives(alternative_groups, queries))
    )
    return alternative_groups


def get_selection_prompt_and_options(
    manager: KgManager,
    question: str,
    sparql: str,
    selections: list[Selection],
    alternatives: OrderedAlternatives,
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

    options = ALT_LABELS[: count_alternatives(alternatives) + 1]
    return messages, list(options)


def materialize_skeleton(
    parts: list[str | IRI],
    is_val: bool = False,
    p: float = 0.2,
) -> str:
    formatted_parts = []
    for part in parts:
        if isinstance(part, str):
            formatted_parts.append(part)
            continue

        # choose main label 80% of the time,
        # otherwise choose a random alias if available
        if not is_val and part.aliases and random.random() < p:
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


def materialize_sample(
    sample: GRISPSample,
    is_val: bool = False,
    p: float = 0.2,
) -> tuple[str, str]:
    if is_val:
        question = sample.questions[0]
    else:
        question = random.choice(sample.questions)

    return question, materialize_skeleton(sample.sparql, is_val, p)


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

    enc = tokenizer.apply_chat_template(
        messages,
        return_assistant_tokens_mask=True,
        return_dict=True,
        enable_thinking=False,
    )  # type: ignore

    mask = enc["assistant_masks"]
    chat_temp = tokenizer.chat_template
    assert chat_temp is not None, "Expected chat template to be set in tokenizer"
    if "{% generation %}" not in chat_temp or all(m == 0 for m in mask):
        # invalid assitant tokens mask, fallback to computing labels
        # manually
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        )
        prompt_len = len(prompt_ids)
        non_prompt_ids = enc["input_ids"][prompt_len:]
        labels = [IGNORE_INDEX] * prompt_len + non_prompt_ids
    else:
        labels = [
            id if mask == 1 else IGNORE_INDEX
            for id, mask in zip(enc["input_ids"], mask, strict=True)
        ]

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def load_samples(
    file_paths: list[str],
    materialized: bool = False,
) -> list[GRISPSample] | list[GRISPMaterializedSample]:
    samples = []
    for path in file_paths:
        loaded_samples = load_jsonl(path)
        samples.extend(
            (
                GRISPSample(**sample)
                if not materialized
                else GRISPMaterializedSample(**sample)
                for sample in loaded_samples
            )
        )
    return samples


def tokenize_and_log(
    messages: Messages,
    tokenizer: PreTrainedTokenizerBase,
    mask_inputs: bool,
    logger: Logger,
) -> dict[str, torch.Tensor]:
    output = tokenize_messages(messages, tokenizer, mask_inputs)
    logger.debug(f"Sample:\n{tokenizer.decode(output['input_ids'])}")
    logger.debug(f"Length: {len(output['input_ids']):,}")
    label_ids = [label for label in output["labels"] if label != IGNORE_INDEX]
    logger.debug(
        f"Last 10 Input IDS: {output['input_ids'][-10 - len(label_ids) : -len(label_ids)]}"
    )
    logger.debug(f"Target IDs: {label_ids}")
    target = tokenizer.decode(label_ids)
    logger.debug(f"Target:\n{target}")
    return output


class GRISPMaterializedSkeletonDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPMaterializedSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        log_level: str | None = None,
    ) -> None:
        self.samples = [sample for sample in samples if sample.has_skeletons]
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

        self.logger = get_logger(
            "GRISP MATERIALIZED SKELETON DATASET",
            log_level,
        )

        self.counter = [0] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        count = self.counter[idx]
        messages = sample.skeletons[count % len(sample.skeletons)]
        self.counter[idx] += 1
        self.logger.debug(
            f"({type(self).__name__}) Accessing sample {idx} count {count}"
        )

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


def prepare_skeleton(
    sample: GRISPSample,
    is_val: bool = False,
    p: float = 0.2,
) -> Messages:
    question, skeleton = materialize_sample(sample, is_val, p)
    return get_skeleton_prompt(sample.kg, question, skeleton)


class GRISPSkeletonDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        is_val: bool = False,
        p: float = 0.2,
        log_level: str | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs
        self.is_val = is_val
        self.p = p

        self.logger = get_logger(f"GRISP SKELETON DATASET ({is_val=})", log_level)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        messages = prepare_skeleton(sample, self.is_val, self.p)

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


class GRISPMaterializedSelectionDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPMaterializedSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        log_level: str | None = None,
    ) -> None:
        self.samples = [sample for sample in samples if sample.has_selections]
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

        self.logger = get_logger("GRISP MATERIALIZED SELECTION DATASET", log_level)

        self.counter = [0] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        count = self.counter[idx]
        messages = sample.selections[count % len(sample.selections)]
        self.counter[idx] += 1
        self.logger.debug(
            f"({type(self).__name__}) Accessing sample {idx} count {count}"
        )

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


def prepare_selection(
    sample: GRISPSample,
    manager: KgManager,
    is_val: bool = False,
    skeleton_p: float = 0.2,
    selection_p: float = 0.2,
) -> Messages:
    question, skeleton = materialize_sample(sample, is_val, skeleton_p)
    sparql = materialize_sparql(sample.sparql)

    supports_variants = {
        obj_type: manager.get_normalizer(obj_type.index_name).supports_variants
        for obj_type in [ObjType.ENTITY, ObjType.PROPERTY]
    }

    _, items = extract_sparql_items(sparql, manager)
    items = [
        item
        for item in items
        if item.is_entity_or_property or (item.is_unindexed and item.has_label)
    ]
    assert len(items) > 0, "No valid item to replace found in sample"

    parser = load_sparql_parser()
    skeleton = Skeleton.parse(skeleton, parser)

    upper = random.randint(0, len(items) - 1)
    for item in items[:upper]:
        skeleton.add_selection(item.selection, manager)

    item = items[upper]
    target_alt = item.selection.alternative

    info = skeleton.prepare_for_selection()
    queries = info.build_queries(supports_variants)

    # TODO: fix hardcoded k values
    k = 10 if is_val else random.randint(2, 10)

    if item.is_entity_or_property:
        selection_logger = get_logger("GRISP SELECTION PREP")
        alternative_groups = find_alternative_groups(
            manager,
            info.prefix,
            queries,
            k,
            logger=selection_logger,
            skip_constraint=True,
        )
    else:
        alternative_groups = {}

    drop_infos = not is_val and random.random() < selection_p
    drop_target = not is_val and random.random() < selection_p
    shuffle_alts = not is_val and random.random() < selection_p

    if shuffle_alts:
        # shuffle within each obj-type bucket while preserving grouped layout
        for current in alternative_groups.values():
            random.shuffle(current)

    alternatives = ordered_alternatives(alternative_groups, queries)

    target_option: int | None = None
    for i, (alt, obj_type, _) in enumerate(alternatives):
        if drop_infos and alt.info:
            alt.info.clear()

        if alt != target_alt or obj_type != item.obj_type:
            continue

        # drop target alternative 20% of the time during training
        if drop_target:
            alternatives.pop(i)
        else:
            target_option = i

        break

    prompt, options = get_selection_prompt_and_options(
        manager,
        question,
        sparql,
        skeleton.selections,
        alternatives,
    )

    # if target option is None, we need to select the last
    # option, which is the "None of the above" option
    option = options[-1] if target_option is None else options[target_option]
    prompt.append({"role": "assistant", "content": option})
    return prompt


class GRISPSelectionDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPSample],
        manager: KgManager,
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        is_val: bool = False,
        skeleton_p: float = 0.2,
        selection_p: float = 0.2,
        log_level: str | None = None,
    ) -> None:
        self.parser = load_sparql_parser()
        self.manager = manager
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs
        self.is_val = is_val

        self.skeleton_p = skeleton_p
        self.selection_p = selection_p

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

        messages = prepare_selection(
            sample,
            self.manager,
            self.is_val,
            self.skeleton_p,
            self.selection_p,
        )

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


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
    def __init__(
        self,
        pad_token_id: int,
        max_length: int,
        log_level: str | int | None = None,
    ) -> None:
        self.max_length = max_length
        self.pad_values = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "labels": IGNORE_INDEX,
        }

        self.logger = get_logger("GRISP COLLATOR", log_level)
        self.logger.info(
            f"Collating batch items to max length {self.max_length} with the "
            f"following pad values: {self.pad_values}"
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        assert len(batch) > 0, "Batch must not be empty"
        output = {
            key: pad(
                [sample[key] for sample in batch],
                pad_value=self.pad_values[key],
                max_length=self.max_length,
            )
            for key in batch[0]
        }
        # ensure at least one label is not IGNORE_INDEX
        # to avoid nan issues during training
        labels = output["labels"]
        if torch.all(labels == IGNORE_INDEX):
            self.logger.warning(
                "No labels for this batch, setting one to "
                "avoid nan issues during training"
            )
            last_dim = labels.shape[1] - 1
            labels[0, last_dim] = output["input_ids"][0, last_dim]

        return output


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
        "--allow-unknown",
        action="store_true",
        help="Allow (directly predict) unkown items instead of skipping the sample",
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    logger = get_logger("GRISP DATA", args.log_level)

    cfg = KgConfig(kg=args.knowledge_graph, info=KgInfo(endpoint=args.endpoint))
    manager = load_kg_manager(cfg)
    manager.set_info_retrieval(enable=False)

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
            sparql = manager.fix_prefixes(sample.sparql, remove_known=True)
            sparql = manager.prettify(sparql)
            sparql, items = extract_sparql_items(sparql, manager)

            invalid_items = [
                item
                for item in items
                if (item.is_unindexed and not item.has_label)
                or (item.is_unknown and not args.allow_unknown)
            ]
            if invalid_items:
                invalid += 1
                invalid_str = format_list(
                    item.alternative.get_selection_string(
                        include_variants=[item.variant] if item.variant else None
                    )
                    for item in invalid_items
                )

                logger.debug(
                    f"Invalid sample {sample.id}:\n\n{sample.question}\n\n"
                    f"{sparql}\n\n{invalid_str}"
                )
                continue

            parts = []
            start = 0
            for item in items:
                # literals, common, and unknown (if allowed) should be predicted directly
                if item.is_literal or item.is_common or item.is_unknown:
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
                f"Error processing sample {sample.id}:\n"
                f"{sample.model_dump_json(indent=2)}\n\n{e}"
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
