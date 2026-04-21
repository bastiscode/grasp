import json
import os
from logging import Logger
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote_plus

import ijson
import requests
from grammar_utils.parse import LR1Parser  # type: ignore
from search_rdf import Data
from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl, dump_text, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.functions import parse_iri_or_literal
from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_index_sparql,
    load_kg_info,
    merge_prefixes,
)
from grasp.sparql.types import Binding
from grasp.sparql.utils import (
    find_longest_prefix,
    get_qlever_endpoint,
    has_scheme,
    load_entity_index_sparql,
    load_iri_and_literal_parser,
    load_property_index_sparql,
)
from grasp.utils import get_index_dir, ordered_unique


def download_data(
    out_dir: str,
    sparql: str,
    prefixes: dict[str, str],
    parser: LR1Parser,
    logger: Logger,
    endpoint: str | None = None,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    add_id_as_label: None | str = None,
    result_file: str | None = None,
    overwrite: bool = False,
) -> None:
    data_file = Path(out_dir, "data.jsonl")
    if data_file.exists() and not overwrite:
        logger.info(f"Data already exists at {data_file}, skipping download")
        return

    if result_file is not None:
        logger.info(f"Loading data to {data_file} from file {result_file}")
        bindings = stream_json_file(result_file)
    else:
        assert endpoint is not None, (
            "Endpoint must be provided if no result file is given"
        )
        logger.info(
            f"Downloading data to {data_file} from {endpoint} "
            f"with parameters {params or {}}, headers {headers or {}}, "
            f"and SPARQL:\n{sparql}"
        )
        bindings = stream_json(endpoint, sparql, params, headers)

    dump_jsonl(
        prepare_items(bindings, prefixes, parser, add_id_as_label, logger),
        data_file.as_posix(),
    )


def build_data_and_mapping(
    index_dir: str,
    logger: Logger,
    overwrite: bool = False,
) -> None:
    data_file = Path(index_dir, "data.jsonl")
    data_dir = Path(index_dir, "data")
    if not data_dir.exists() or overwrite:
        # build index data
        logger.info(f"Building data at {data_dir}")
        Data.build_from_jsonl(data_file.as_posix(), data_dir.as_posix())
    else:
        logger.info(f"Data already exists at {data_dir}, skipping build")


def get_data(
    kg: str,
    endpoint: str | None = None,
    entity_sparql: str | None = None,
    property_sparql: str | None = None,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    add_id_as_label: str | None = None,
    entity_file: str | None = None,
    property_file: str | None = None,
    log_level: str | int | None = None,
    overwrite: bool = False,
) -> None:
    logger = get_logger("GRASP DATA", log_level)
    parser = load_iri_and_literal_parser()

    info = load_kg_info(kg)

    params = params or {}
    if info.params:
        params = {**info.params, **params}

    headers = headers or {}
    if info.headers:
        headers = {**info.headers, **headers}

    endpoint = endpoint or info.endpoint or get_qlever_endpoint(kg)

    prefixes = get_common_sparql_prefixes()
    prefixes, _, _ = merge_prefixes(prefixes, info.prefixes or {}, logger)
    logger.info(f"Using prefixes:\n{json.dumps(prefixes, indent=2)}")

    kg_dir = get_index_dir(kg)

    # entities
    entity_dir = os.path.join(kg_dir, "entities")
    os.makedirs(entity_dir, exist_ok=True)
    if entity_sparql is None:
        entity_sparql = load_index_sparql(entity_dir, logger)
    if entity_sparql is None:
        entity_sparql = load_entity_index_sparql()

    download_data(
        entity_dir,
        entity_sparql,
        prefixes,
        parser,
        logger,
        endpoint,
        params,
        headers,
        add_id_as_label,
        entity_file,
        overwrite,
    )
    dump_text(entity_sparql, os.path.join(entity_dir, "index.sparql"))
    build_data_and_mapping(entity_dir, logger, overwrite)

    # properties
    property_dir = os.path.join(kg_dir, "properties")
    os.makedirs(property_dir, exist_ok=True)
    if property_sparql is None:
        property_sparql = load_index_sparql(property_dir, logger)
    if property_sparql is None:
        property_sparql = load_property_index_sparql()

    download_data(
        property_dir,
        property_sparql,
        prefixes,
        parser,
        logger,
        endpoint,
        params,
        headers,
        add_id_as_label="always",  # for properties we also want to search via id
        result_file=property_file,
        overwrite=overwrite,
    )
    dump_text(property_sparql, os.path.join(property_dir, "index.sparql"))
    build_data_and_mapping(property_dir, logger, overwrite)


def stream_json(
    endpoint: str,
    sparql: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> Iterator[dict]:
    try:
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
            **(headers or {}),
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=params,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to stream SPARQL results as JSON: {e}") from e

    class _StreamReader:
        def __init__(self, response: requests.Response) -> None:
            self.stream = response.iter_content(chunk_size=None)

        def read(self, n: int) -> bytes:
            if n == 0:
                return b""
            return next(self.stream, b"")

    yield from ijson.items(_StreamReader(response), "results.bindings.item")


def stream_json_file(path: str) -> Iterator[dict]:
    with open(path, "r") as f:
        yield from ijson.items(f, "results.bindings.item")


def split_iri(iri: str) -> tuple[str, str]:
    if "://" not in iri:
        return "", iri

    # split iri into prefix and last part after final / or #
    last_hashtag = iri.rfind("#")
    last_slash = iri.rfind("/")
    last = max(last_hashtag, last_slash)
    if last == -1:
        return "", iri
    else:
        return iri[:last], iri[last + 1 :]


def camel_case_split(s: str) -> str:
    # split camelCase into words
    # find uppercase letters
    words = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper() and i > 0 and s[i - 1].islower():
            words.append(s[last:i])
            last = i

    if last < len(s):
        words.append(s[last:])

    return " ".join(words)


def get_object_name_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
    pfx = find_longest_prefix(obj_id, prefixes)
    if pfx is None:
        # no known prefix, split after final / or # to get objet name
        _, obj_name = split_iri(obj_id)
    else:
        _, long = pfx
        obj_name = obj_id[len(long) :]

    # url decode the object name
    return unquote_plus(obj_name)


def get_value_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
    obj_name = get_object_name_from_id(obj_id, prefixes)
    label = " ".join(camel_case_split(part) for part in split_at_punctuation(obj_name))
    return label.strip()


# we consider _, -, and . as url punctuation
PUNCTUATION = {"_", "-", "."}


def split_at_punctuation(s: str) -> Iterator[str]:
    start = 0
    for i, c in enumerate(s):
        if c not in PUNCTUATION:
            continue

        yield s[start:i]
        start = i + 1

    if start < len(s):
        yield s[start:]


def parse_binding(
    binding: dict,
    parser: LR1Parser,
) -> tuple[str, Binding | None, list[str]]:
    assert binding["id"]["type"] == "uri", "Expected id to be a URI"
    id = binding["id"]["value"]

    tag_binding = binding.get("tag", binding.get("tags", None))
    if tag_binding is not None:
        assert tag_binding["type"] == "literal", "Expected tags to be a literal"
        tags = tag_binding["value"].lower().split(",")
    else:
        tags = []

    if "value" not in binding:
        return id, None, tags

    value_binding = Binding.from_dict(binding["value"])
    if value_binding.typ == "literal" and has_scheme(value_binding.value):
        # may be an uri converted to literal, double check
        value_binding = parse_iri_or_literal(value_binding.value, parser)

    return id, value_binding, tags


def prepare_items(
    bindings: Iterator[dict],
    prefixes: dict[str, str],
    parser: LR1Parser,
    add_id_as_label: None | str = None,
    logger: Logger | None = None,
) -> Iterator[dict]:
    # collect all labels for an id (which are consecutive in the stream)
    last_id = None
    fields = []
    for num, binding in enumerate(bindings, start=1):
        id, value_binding, tags = parse_binding(binding, parser)

        if logger and num % 1_000_000 == 0:
            logger.info(f"Processed {num:,} bindings so far")

        if logger:
            logger.debug(
                f"Processing binding #{num:,}: id={id}, value={value_binding}, tags={tags}"
            )

        if last_id is not None and id != last_id:
            # yield previous item
            if add_id_as_label == "always" or (
                add_id_as_label == "empty" and not fields
            ):
                fields.append(
                    {
                        "type": "text",
                        "value": get_value_from_id(last_id, prefixes),
                        "tags": [],
                    }
                )

            yield {
                "identifier": last_id,
                "fields": ordered_unique(fields, key=lambda f: f["value"]),
            }

            fields = []

        last_id = id
        if value_binding is None:
            continue
        elif value_binding.typ == "uri":
            value = get_value_from_id(value_binding.value, prefixes)
        else:
            value = value_binding.value

        fields.append({"type": "text", "value": value, "tags": tags})

    if last_id is None:
        return

    # dont forget final item
    if add_id_as_label == "always" or (add_id_as_label == "empty" and not fields):
        fields.append(
            {
                "type": "text",
                "value": get_value_from_id(last_id, prefixes),
                "tags": [],
            }
        )

    yield {
        "identifier": last_id,
        "fields": ordered_unique(fields, key=lambda f: f["value"]),
    }


def merge_data(
    kgs: list[str],
    sub_dir: str,
    out_dir: str,
    logger: Logger,
    overwrite: bool = False,
):
    out_dir = os.path.join(out_dir, sub_dir)
    data_file = os.path.join(out_dir, "data.jsonl")
    kg_info = ", ".join(kgs)
    if os.path.exists(data_file) and not overwrite:
        logger.info(
            f"Merged data for {sub_dir} of knowledge graphs {kg_info} "
            f"already exists at {data_file}, skipping merge"
        )
        return

    logger.info(
        f"Merging data for {sub_dir} of knowledge graphs {kg_info} into {data_file}"
    )

    os.makedirs(out_dir, exist_ok=True)

    others = []

    for kg in tqdm(kgs[1:], desc="Building mappings for data to merge"):
        kg_data_file = os.path.join(get_index_dir(kg), sub_dir, "data.jsonl")

        items = load_jsonl(kg_data_file)
        others.append({item["identifier"]: item for item in items})

    # first kg is the main one, to which we add data from the others
    kg = kgs[0]
    kg_data_file = os.path.join(get_index_dir(kg), sub_dir, "data.jsonl")

    def merge() -> Iterator[str]:
        with open(kg_data_file, "r") as f:
            for line in tqdm(f, desc="Merging data"):
                item = json.loads(line)

                identifier = item["identifier"]

                # collect fields from other kgs and add them
                # to the current item
                seen = set(field["value"] for field in item["fields"])
                for mapping in others:
                    if identifier not in mapping:
                        continue

                    other_item = mapping[identifier]
                    for field in other_item["fields"]:
                        if field["value"] in seen:
                            continue

                        item["fields"].append(field)
                        seen.add(field["value"])

                yield item

    dump_jsonl(merge(), data_file)


def merge_kgs(
    kgs: list[str],
    out_kg: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
):
    assert len(kgs) >= 2, "At least two knowledge graphs are required to merge"

    logger = get_logger("GRASP MERGE", log_level)

    out_dir = get_index_dir(out_kg)

    merge_data(kgs, "entities", out_dir, logger, overwrite)

    ent_dir = os.path.join(out_dir, "entities")
    build_data_and_mapping(ent_dir, logger, overwrite)

    merge_data(kgs, "properties", out_dir, logger, overwrite)

    prop_dir = os.path.join(out_dir, "properties")
    build_data_and_mapping(prop_dir, logger, overwrite)
