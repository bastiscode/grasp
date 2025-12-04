import csv
import os
from logging import Logger
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote_plus

import requests
from search_index import IndexData, Mapping
from tqdm import tqdm
from universal_ml_utils.io import dump_lines, dump_text
from universal_ml_utils.logging import get_logger

from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_data_and_mapping,
    load_kg_prefixes,
)
from grasp.sparql.utils import (
    find_longest_prefix,
    get_endpoint,
    is_iri,
    load_entity_index_sparql,
    load_property_index_sparql,
)
from grasp.utils import get_index_dir, ordered_unique


def download_data(
    out_dir: str,
    endpoint: str,
    sparql: str,
    logger: Logger,
    prefixes: dict[str, str],
    params: dict[str, str] | None = None,
    add_id_as_label: bool = False,
    overwrite: bool = False,
    disable_id_fallback: bool = False,
) -> None:
    data_file = Path(out_dir, "data.tsv")
    if data_file.exists() and not overwrite:
        logger.info(f"Data already exists at {data_file}, skipping download")
        return

    logger.info(
        f"Downloading data to {data_file} from {endpoint} "
        f"with parameters {params or {}} and SPARQL:\n{sparql}"
    )

    stream = stream_csv(endpoint, sparql, params)
    dump_lines(
        prepare_csv(stream, prefixes, logger, add_id_as_label, disable_id_fallback),
        data_file.as_posix(),
    )


def build_data_and_mapping(
    index_dir: str,
    logger: Logger,
    overwrite: bool = False,
) -> None:
    data_file = Path(index_dir, "data.tsv")
    offsets_file = data_file.with_name("offsets.bin")
    mapping_file = data_file.with_name("mapping.bin")
    if not offsets_file.exists() or overwrite:
        # build index data
        logger.info(f"Building offsets file at {offsets_file}")
        IndexData.build(data_file.as_posix(), offsets_file.as_posix())
    else:
        logger.info(f"Offsets file already exists at {offsets_file}, skipping build")

    data = IndexData.load(data_file.as_posix(), offsets_file.as_posix())

    if not mapping_file.exists() or overwrite:
        # build mapping
        logger.info(f"Building mapping file at {mapping_file}")
        Mapping.build(data, mapping_file.as_posix())  # type: ignore
    else:
        logger.info(f"Mapping file already exists at {mapping_file}, skipping build")


def get_data(
    kg: str,
    endpoint: str | None = None,
    entity_query: str | None = None,
    property_query: str | None = None,
    query_params: dict[str, str] | None = None,
    overwrite: bool = False,
    disable_id_fallback: bool = False,
    log_level: str | int | None = None,
) -> None:
    logger = get_logger("GRASP DATA", log_level)

    if endpoint is None:
        endpoint = get_endpoint(kg)
        logger.info(
            f"Using endpoint {endpoint} for {kg} because "
            "no endpoint is set in the config"
        )

    prefixes = get_common_sparql_prefixes()
    prefixes.update(load_kg_prefixes(kg))

    kg_dir = get_index_dir(kg)

    # entities
    ent_dir = os.path.join(kg_dir, "entities")
    os.makedirs(ent_dir, exist_ok=True)
    ent_sparql = entity_query or load_entity_index_sparql()
    download_data(
        ent_dir,
        endpoint,
        ent_sparql,
        logger,
        prefixes,
        query_params,
        overwrite=overwrite,
        disable_id_fallback=disable_id_fallback,
    )
    dump_text(ent_sparql, os.path.join(ent_dir, "index.sparql"))
    build_data_and_mapping(ent_dir, logger, overwrite)

    # properties
    prop_dir = os.path.join(kg_dir, "properties")
    os.makedirs(prop_dir, exist_ok=True)
    prop_sparql = property_query or load_property_index_sparql()
    download_data(
        prop_dir,
        endpoint,
        prop_sparql,
        logger,
        prefixes,
        query_params,
        add_id_as_label=True,  # for properties we also want to search via id
        overwrite=overwrite,
        disable_id_fallback=disable_id_fallback,
    )
    dump_text(prop_sparql, os.path.join(prop_dir, "index.sparql"))
    build_data_and_mapping(prop_dir, logger, overwrite)


def stream_csv(
    endpoint: str,
    sparql: str,
    query_params: dict[str, str] | None = None,
) -> Iterator[list[str]]:
    try:
        headers = {
            "Accept": "text/csv",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=query_params,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()

        lines = (line.decode("utf-8") for line in response.iter_lines())
        for row in csv.reader(lines):
            # pad to 3 columns
            while len(row) < 3:
                row.append("")
            yield row

    except Exception as e:
        raise ValueError(f"Failed to stream csv: {e}") from e


def split_iri(iri: str) -> tuple[str, str]:
    if not is_iri(iri):
        return "", iri

    # split iri into prefix and last part after final / or #
    last_hashtag = iri.rfind("#")
    last_slash = iri.rfind("/")
    last = max(last_hashtag, last_slash)
    if last == -1:
        return "", iri[1:-1]
    else:
        return iri[1:last], iri[last + 1 : -1]


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
        obj_name = obj_id[len(long) : -1]

    # url decode the object name
    return unquote_plus(obj_name)


def get_label_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
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


def prepare_csv(
    lines: Iterator[list[str]],
    prefixes: dict[str, str],
    logger: Logger,
    add_id_as_label: bool = False,
    disable_id_fallback: bool = False,
) -> Iterator[str]:
    num = 0

    # skip original header
    next(lines)

    # yield own header
    yield "id\tlabels"

    for line in lines:
        assert len(line) == 3, f"Expected 3 columns, got {len(line)}: {line}"

        id, label, synonyms = line

        # wrap id with brackets
        id = f"<{id}>"

        # filter out empty label and synonyms
        labels = []
        if label:
            labels.append(label)

        elif not disable_id_fallback:
            # main label is empty, set it from the object id
            labels.append(get_label_from_id(id, prefixes))

        for syn in synonyms.split(";;;"):
            if syn:
                labels.append(syn)

        if not disable_id_fallback:
            # add label from id
            labels.append(get_label_from_id(id, prefixes))

            if add_id_as_label:
                # also add object name from id as label
                labels.append(get_object_name_from_id(id, prefixes))

        # make sure no duplicates are in the labels
        labels = ordered_unique(labels)
        yield "\t".join([id] + labels)

        num += 1
        if num % 1_000_000 == 0:
            logger.info(f"Processed {num:,} items so far")


def merge_data(
    kgs: list[str],
    sub_dir: str,
    out_dir: str,
    prefixes: dict[str, str],
    logger: Logger,
    overwrite: bool = False,
    add_id_as_label: bool = False,
):
    out_dir = os.path.join(out_dir, sub_dir)
    data_file = os.path.join(out_dir, "data.tsv")
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

    index_data = {}
    index_mappings = {}
    for kg in kgs:
        index_dir = os.path.join(get_index_dir(kg), sub_dir)

        data, mapping = load_data_and_mapping(index_dir)
        index_data[kg] = data
        index_mappings[kg] = mapping

    # first kg is the main one, to which we add data from the others
    kg = kgs[0]

    def merge() -> Iterator[str]:
        yield "id\tlabels"  # header

        for row in tqdm(index_data[kg], f"Merging data for {sub_dir}"):
            id, *labels = row

            for i in range(1, len(kgs)):
                data = index_data[kgs[i]]
                mapping = index_mappings[kgs[i]]
                index = mapping.get(id)
                if index is None:
                    continue

                _, *other_labels = data.get_row(index)
                labels.extend(other_labels)

            if not labels:
                labels.append(get_label_from_id(id, prefixes))

            if add_id_as_label:
                object_name = get_object_name_from_id(id, prefixes)
                labels.append(object_name)

            labels = ordered_unique(labels)
            yield "\t".join([id] + labels)

    dump_lines(merge(), data_file)


def merge_kgs(
    kgs: list[str],
    out_kg: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
):
    assert len(kgs) >= 2, "At least two knowledge graphs are required to merge"

    logger = get_logger("GRASP MERGE", log_level)

    prefixes = get_common_sparql_prefixes()
    for kg in kgs:
        prefixes.update(load_kg_prefixes(kg))

    out_dir = get_index_dir(out_kg)

    merge_data(kgs, "entities", out_dir, prefixes, logger, overwrite)

    ent_dir = os.path.join(out_dir, "entities")
    build_data_and_mapping(ent_dir, logger, overwrite)

    merge_data(
        kgs,
        "properties",
        out_dir,
        prefixes,
        logger,
        overwrite,
        add_id_as_label=True,
    )

    prop_dir = os.path.join(out_dir, "properties")
    build_data_and_mapping(prop_dir, logger, overwrite)
