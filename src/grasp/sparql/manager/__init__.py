import json
import os
import time
from typing import Any, Type

from search_index import IndexData, PrefixIndex, SearchIndex, SimilarityIndex
from universal_ml_utils.logging import get_logger

from grasp.sparql.manager.base import KgManager
from grasp.sparql.mapping import Mapping
from grasp.sparql.sparql import find_longest_prefix, get_index_dir


WIKIDATA_PROPERTY_VARIANTS = {
    "wdt": "<http://www.wikidata.org/prop/direct/",
    "wdtn": "<http://www.wikidata.org/prop/direct-normalized/",
    "p": "<http://www.wikidata.org/prop/",
    "pq": "<http://www.wikidata.org/prop/qualifier/",
    "pqn": "<http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pqv": "<http://www.wikidata.org/prop/qualifier/value/",
    "pr": "<http://www.wikidata.org/prop/reference/",
    "prn": "<http://www.wikidata.org/prop/reference/value-normalized/",
    "prv": "<http://www.wikidata.org/prop/reference/value/",
    "ps": "<http://www.wikidata.org/prop/statement/",
    "psn": "<http://www.wikidata.org/prop/statement/value-normalized/",
    "psv": "<http://www.wikidata.org/prop/statement/value/",
}


class WikidataPropertyMapping(Mapping):
    NORM_PREFIX = "<http://www.wikidata.org/entity/"

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        longest = find_longest_prefix(iri, WIKIDATA_PROPERTY_VARIANTS)
        if longest is None:
            return None

        short, long = longest
        iri = self.NORM_PREFIX + iri[len(long) :]
        return iri, short

    def denormalize(self, iri: str, variant: str | None) -> str | None:
        if variant is None:
            return iri
        elif variant not in WIKIDATA_PROPERTY_VARIANTS:
            return None
        elif not iri.startswith(self.NORM_PREFIX):
            return None
        pfx = WIKIDATA_PROPERTY_VARIANTS[variant]
        return pfx + iri[len(self.NORM_PREFIX) :]

    def default_variants(self) -> set[str] | None:
        return set(WIKIDATA_PROPERTY_VARIANTS.keys())


def load_data_and_mapping(
    index_dir: str,
    mapping_cls: Type[Mapping] | None = None,
) -> tuple[IndexData, Mapping]:
    try:
        data = IndexData.load(
            os.path.join(index_dir, "data.tsv"),
            os.path.join(index_dir, "offsets.bin"),
        )
    except Exception as e:
        raise ValueError(f"Failed to load index data from {index_dir}") from e

    if mapping_cls is None:
        mapping_cls = Mapping

    try:
        mapping = mapping_cls.load(
            data,
            os.path.join(index_dir, "mapping.bin"),
        )
    except Exception as e:
        raise ValueError(f"Failed to load mapping from {index_dir}") from e

    return data, mapping


def load_index_and_mapping(
    index_dir: str,
    index_type: str,
    mapping_cls: Type[Mapping] | None = None,
    **kwargs: Any,
) -> tuple[SearchIndex, Mapping]:
    logger = get_logger("KG INDEX LOADING")
    start = time.perf_counter()

    if index_type == "prefix":
        index_cls = PrefixIndex
    elif index_type == "similarity":
        index_cls = SimilarityIndex
    else:
        raise ValueError(f"Unknown index type {index_type}")

    data, mapping = load_data_and_mapping(index_dir, mapping_cls)

    try:
        index = index_cls.load(
            data,
            os.path.join(index_dir, index_type),
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Failed to load {index_type} index from {index_dir}") from e

    end = time.perf_counter()

    logger.debug(f"Loading {index_type} index from {index_dir} took {end - start:.2f}s")

    return index, mapping


def load_entity_index_and_mapping(
    name: str,
    index_dir: str | None = None,
    index_type: str | None = None,
    **kwargs: Any,
) -> tuple[SearchIndex, Mapping]:
    if index_dir is None:
        default_dir = get_index_dir()
        assert default_dir is not None, "KG_INDEX_DIR environment variable not set"
        index_dir = os.path.join(default_dir, name, "entities")

    return load_index_and_mapping(
        index_dir,
        # for entities use prefix index by default
        index_type or "prefix",
        **kwargs,
    )


def load_property_index_and_mapping(
    name: str,
    index_dir: str | None = None,
    index_type: str | None = None,
    **kwargs: Any,
) -> tuple[SearchIndex, Mapping]:
    if index_dir is None:
        default_dir = get_index_dir()
        assert default_dir is not None, "KG_INDEX_DIR environment variable not set"
        index_dir = os.path.join(default_dir, name, "properties")

    mapping_cls = WikidataPropertyMapping if name == "wikidata" else None

    return load_index_and_mapping(
        index_dir,
        # for properties use similarity index by default
        index_type or "similarity",
        mapping_cls,
        **kwargs,
    )


def load_example_index(dir: str, **kwargs: Any) -> SimilarityIndex:
    data = IndexData.load(
        os.path.join(dir, "data.tsv"),
        os.path.join(dir, "offsets.bin"),
    )

    index = SimilarityIndex.load(
        data,
        os.path.join(dir, "index"),
        **kwargs,
    )
    return index


def load_kg_prefixes(
    name: str,
    prefix_file: str | None = None,
) -> dict[str, str]:
    if prefix_file is None:
        index_dir = get_index_dir()
        assert index_dir is not None, "KG_INDEX_DIR environment variable not set"
        prefix_file = os.path.join(index_dir, name, "prefixes.json")

    with open(prefix_file) as f:
        return json.load(f)


def load_kg_indices(
    name: str,
    entities_dir: str | None = None,
    entities_type: str | None = None,
    entities_kwargs: dict[str, Any] | None = None,
    properties_dir: str | None = None,
    properties_type: str | None = None,
    properties_kwargs: dict[str, Any] | None = None,
) -> tuple[SearchIndex, SearchIndex, Mapping, Mapping]:
    ent_index, ent_mapping = load_entity_index_and_mapping(
        name,
        entities_dir,
        entities_type,
        **(entities_kwargs or {}),
    )

    prop_index, prop_mapping = load_property_index_and_mapping(
        name,
        properties_dir,
        properties_type,
        **(properties_kwargs or {}),
    )

    return ent_index, prop_index, ent_mapping, prop_mapping


def load_kg_manager(
    name: str,
    entities_dir: str | None = None,
    entities_type: str | None = None,
    entities_kwargs: dict[str, Any] | None = None,
    properties_dir: str | None = None,
    properties_type: str | None = None,
    properties_kwargs: dict[str, Any] | None = None,
    prefix_file: str | None = None,
    endpoint: str | None = None,
) -> KgManager:
    indices = load_kg_indices(
        name,
        entities_dir,
        entities_type,
        entities_kwargs,
        properties_dir,
        properties_type,
        properties_kwargs,
    )
    prefixes = load_kg_prefixes(name, prefix_file)
    return KgManager(name, *indices, prefixes, endpoint)
