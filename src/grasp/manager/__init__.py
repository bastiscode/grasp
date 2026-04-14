import logging
import os
import sys
from typing import Any, Iterable

from search_rdf import Data, EmbeddingIndex
from search_rdf.model import (
    HuggingFaceImageModel,
    OpenClipModel,
    SentenceTransformerModel,
)
from universal_ml_utils.logging import get_logger
from universal_ml_utils.table import generate_table

from grasp.configs import KgConfig
from grasp.manager.normalizer import Normalizer, WikidataPropertyNormalizer
from grasp.manager.utils import (
    EmbeddingModel,
    Index,
    SearchIndex,
    format_index_meta,
    get_common_sparql_prefixes,
    get_embedding_model_key,
    load_embedding_model,
    load_image_from_url,
    load_index_description,
    load_info_sparql,
    load_kg_info,
    load_other_indices,
    merge_prefixes,
    try_load_search_index,
)
from grasp.sparql.types import (
    Alternative,
    AskResult,
    ObjType,
    Position,
    Selection,
    SelectResult,
    SelectRow,
    group_selections,
)
from grasp.sparql.utils import (
    READ_TIMEOUT,
    REQUEST_TIMEOUT,
    SPARQLException,
    ask_to_select,
    autocomplete_prefix,
    execute,
    find_longest_prefix,
    fix_prefixes,
    format_iri,
    get_endpoint,
    load_iri_and_literal_parser,
    load_sparql_parser,
    prettify,
    query_type,
    wrap_iri,
)
from grasp.utils import (
    clip,
    format_list,
    format_prefixes,
    get_index_dir,
    ordered_unique,
)


class KgManager:
    def __init__(
        self,
        kg: str,
        prefixes: dict[str, str] | None = None,
        indices: dict[str, Index] | None = None,
        endpoint: str | None = None,
        description: str | None = None,
    ):
        self.kg = kg

        self.logger = get_logger(f"{self.kg.upper()} KG MANAGER")

        self.sparql_parser = load_sparql_parser()
        self.iri_literal_parser = load_iri_and_literal_parser()

        self.prefixes, _, self.kg_prefixes = merge_prefixes(
            get_common_sparql_prefixes(),
            prefixes or {},
            self.logger,
        )

        self.disable_info_retrieval = False
        self.endpoint = endpoint or get_endpoint(self.kg)
        self.indices = indices or {}
        self.description = description

        self.embedding_models: dict[str, EmbeddingModel] = {}

    def load_models(
        self,
        models: dict[str, EmbeddingModel] | None = None,
    ) -> dict[str, EmbeddingModel]:
        if models is None:
            models = {}

        for idx in self.indices.values():
            models = load_embedding_model(idx.index, models)

        self.embedding_models = models
        return models

    def set_info_retrieval(self, enable: bool) -> None:
        self.disable_info_retrieval = not enable

    def prettify(
        self,
        sparql: str,
        indent: int = 2,
        is_prefix: bool = False,
    ) -> str:
        return prettify(sparql, self.sparql_parser, indent, is_prefix)

    def execute_sparql(
        self,
        sparql: str,
        request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        read_timeout: float | None = READ_TIMEOUT,
        max_retries: int = 0,
        force_select_result: bool = False,
    ) -> SelectResult | AskResult:
        if force_select_result:
            # ask_to_select returns None if sparql is not an ask query
            sparql = ask_to_select(sparql, self.sparql_parser) or sparql

        sparql = self.fix_prefixes(sparql)

        self.logger.debug(f"Executing SPARQL query against {self.endpoint}:\n{sparql}")
        return execute(
            sparql,
            self.endpoint,
            request_timeout,
            read_timeout,
            max_retries,
        )

    def format_sparql_result(
        self,
        result: SelectResult | AskResult,
        show_top_rows: int = 5,
        show_bottom_rows: int = 5,
        show_left_columns: int = 5,
        show_right_columns: int = 5,
        column_names: list[str] | None = None,
        clip_literals: bool = True,
        time: float | None = None,
    ) -> str:
        tmf = ""
        if time is not None:
            tmf = f" in {time:.2f}s"

        # run sparql against endpoint, format result as string
        if isinstance(result, AskResult):
            return f"Got result{tmf}:\n{result.boolean}"

        if result.num_rows == 0:
            return f"Got no rows and {result.num_columns:,} columns" + tmf

        assert column_names is None or len(column_names) == result.num_columns, (
            f"Expected {result.num_columns:,} column names"
        )
        assert show_top_rows or show_bottom_rows, "At least one row must be shown"
        assert show_left_columns or show_right_columns, (
            "At least one column must be shown"
        )

        left_end = min(show_left_columns, result.num_columns)
        right_start = result.num_columns - show_right_columns
        if right_start > left_end:
            column_indices = list(range(left_end))
            column_indices.append(-1)
            column_indices.extend(range(right_start, result.num_columns))
        else:
            column_indices = list(range(result.num_columns))

        def format_row(row: SelectRow) -> list[str]:
            formatted_row = []
            for c in column_indices:
                if c < 0:
                    formatted_row.append("...")
                    continue

                var = result.variables[c]
                val = row.get(var, None)
                if val is None:
                    formatted_row.append("")

                elif val.typ == "bnode":
                    formatted_row.append(val.identifier())

                elif val.typ == "literal":
                    formatted = clip(val.value) if clip_literals else val.value
                    if val.lang is not None:
                        formatted += f" (lang:{val.lang})"
                    elif val.datatype is not None:
                        datatype = self.format_iri(val.datatype)
                        formatted += f" ({datatype})"

                    formatted_row.append(formatted)

                else:
                    assert val.typ == "uri"
                    identifier = val.identifier()
                    formatted = self.format_iri(identifier)

                    # for uri check whether it is in one of the datasets
                    index = "entities"
                    norm = self.normalize(identifier, index)

                    if norm is None or self.get_label(norm[0], index) is None:
                        index = "properties"
                        norm = self.normalize(identifier, index)

                    # still not found, just output the formatted iri
                    if norm is None or self.get_label(norm[0], index) is None:
                        formatted_row.append(formatted)
                        continue

                    label = self.get_label(norm[0], index)
                    assert label is not None, "should not happen"
                    formatted = f"{clip(label)} ({formatted})"
                    formatted_row.append(formatted)

            return formatted_row

        # generate a nicely formatted table
        column_names = column_names or result.variables
        header = [column_names[c] if c >= 0 else "..." for c in column_indices]
        top_end = min(show_top_rows, result.num_rows)
        bottom_start = max(result.num_rows - show_bottom_rows, top_end)

        data = [format_row(row) for row in result.rows(end=top_end)]

        if bottom_start > top_end:
            data.append(["..."] * len(header))

        data.extend(
            format_row(row) for row in result.rows(bottom_start, result.num_rows)
        )

        table = generate_table(
            data,
            [header],
            alignments=["left"] * len(header),
            max_column_width=sys.maxsize,
        )

        comp = "" if result.complete else "more than "
        formatted = (
            f"Got {comp}{result.num_rows:,} row{'s' * (result.num_rows != 1)} and "
            f"{result.num_columns:,} column{'s' * (result.num_columns != 1)}{tmf}"
        )

        showing = []

        if right_start > left_end:
            # columns restricted
            show_columns = []
            if show_left_columns:
                show_columns.append(f"first {show_left_columns}")
            if show_right_columns:
                show_columns.append(f"last {show_right_columns}")

            showing.append(f"the {' and '.join(show_columns)} columns")

        if bottom_start > top_end:
            # rows restricted
            show_rows = []
            if show_top_rows:
                show_rows.append(f"first {show_top_rows}")
            if show_bottom_rows:
                show_rows.append(f"last {show_bottom_rows}")

            showing.append(f"the {' and '.join(show_rows)} rows")

        if showing:
            formatted += ", showing " + " and ".join(showing) + " below"

        formatted += f":\n{table}"
        return formatted

    def find_longest_prefix(self, iri: str) -> tuple[str, str] | None:
        return find_longest_prefix(iri, self.prefixes)

    def format_iri(
        self,
        iri: str,
        base_uri: str | None = None,
        wrap: bool = False,
    ) -> str:
        return format_iri(iri, self.iri_literal_parser, self.prefixes, base_uri, wrap)

    def fix_prefixes(
        self,
        sparql: str,
        remove_known: bool = False,
        sort: bool = False,
    ) -> str:
        return fix_prefixes(
            sparql,
            self.sparql_parser,
            self.iri_literal_parser,
            self.prefixes,
            remove_known,
            sort,
        )

    def get(self, name: str) -> Index:
        if name not in self.indices:
            raise ValueError(f"Index '{name}' not found")
        return self.indices[name]

    def get_index(self, name: str) -> SearchIndex:
        return self.get(name).index

    def get_data(self, name: str) -> Data:
        return self.get(name).data

    def try_get(self, name: str) -> Index | None:
        return self.indices.get(name)

    def get_normalizer(self, name: str) -> Normalizer:
        index = self.try_get(name)
        if index is None or index.normalizer is None:
            return Normalizer()
        else:
            return index.normalizer

    def try_get_data(self, name: str) -> Data | None:
        index = self.try_get(name)
        if index is None:
            return None
        return index.data

    def get_info_sparql(self, name: str) -> str | None:
        index = self.try_get(name)
        if index is None:
            return None
        return index.info_sparql

    @property
    def index_names(self) -> list[str]:
        return sorted(self.indices)

    def normalize(
        self,
        identifier: str,
        index_name: str,
    ) -> tuple[str, str | None] | None:
        return self.get_normalizer(index_name).normalize(identifier)

    def denormalize(
        self,
        identifier: str,
        index_name: str,
        variant: str | None = None,
    ) -> str | None:
        return self.get_normalizer(index_name).denormalize(identifier, variant)

    def check_identifier(
        self,
        identifier: str,
        index_name: str,
    ) -> bool:
        data = self.try_get_data(index_name)
        if data is None:
            return False
        return data.id_from_identifier(identifier) is not None

    def get_label(
        self,
        identifier: str,
        index_name: str,
    ) -> str | None:
        data = self.try_get_data(index_name)
        if data is None:
            return None

        id = data.id_from_identifier(identifier)
        if id is None:
            return None

        return data.main_field(id) or data.field(id, 0)

    def build_alternative_with_info(
        self,
        identifier: str,
        info: dict | None = None,
        variants: list[str] | None = None,
        matched_via: str | None = None,
    ) -> Alternative:
        if info is None:
            info = {}

        # extract needed data from info dict
        label = info.get("label")
        aliases = info.get("alias", [])
        other = info.get("other", [])

        return self.build_alternative(
            identifier,
            label,
            aliases,
            other,
            variants,
            matched_via,
        )

    def build_alternative(
        self,
        identifier: str,
        label: str | None = None,
        aliases: list[str] | None = None,
        other: list[str] | None = None,
        variants: list[str] | None = None,
        matched_via: str | None = None,
    ) -> Alternative:
        # preprocess some fields
        if variants is not None:
            variants = ordered_unique(variants)

        if aliases is not None:
            aliases = ordered_unique(aliases, filter=lambda alias: alias != label)

        if other is not None:
            other = ordered_unique(other)

        return Alternative(
            identifier=identifier,
            short_identifier=self.format_iri(identifier),
            label=label,
            variants=variants,
            aliases=aliases,
            info=other,
            matched_label=matched_via,
        )

    def _embed_query(
        self,
        index: EmbeddingIndex,
        query: str,
        query_type: str = "text",
    ) -> list[float]:
        model_key = get_embedding_model_key(index)
        model = self.embedding_models[model_key]

        if query_type == "text":
            if isinstance(model, SentenceTransformerModel):
                return model.embed([query])[0].tolist()
            elif isinstance(model, OpenClipModel):
                return model.embed_text([query])[0].tolist()
            elif isinstance(model, HuggingFaceImageModel):
                raise ValueError("Image embedding model does not support text queries")
            else:
                raise ValueError(f"Unsupported embedding model type: {type(model)}")

        elif query_type == "image":
            image = load_image_from_url(query)
            if isinstance(model, OpenClipModel):
                return model.embed_image([image])[0].tolist()
            elif isinstance(model, HuggingFaceImageModel):
                return model.embed([image])[0].tolist()
            elif isinstance(model, SentenceTransformerModel):
                raise ValueError(
                    "SentenceTransformer model does not support image queries"
                )
            else:
                raise ValueError(f"Unsupported embedding model type: {type(model)}")

        else:
            raise ValueError(
                f"Unsupported query_type '{query_type}', expected 'text' or 'image'"
            )

    def search_index(
        self,
        index_name: str,
        query: str | None = None,
        k: int = 10,
        identifier_map: dict[str, list[str]] | None = None,
        query_type: str = "text",
        **search_kwargs: Any,
    ) -> list[Alternative]:
        index = self.get_index(index_name)
        data = self.get_data(index_name)
        normalizer = self.get_normalizer(index_name)

        field_map = {}

        if query is None:
            if identifier_map is None:
                identifiers = [data.identifier(id) or "" for id in range(k)]
            else:
                identifiers = sorted(
                    identifier_map,
                    key=lambda ident: data.id_from_identifier(ident) or len(data),
                )[:k]
        else:
            kwargs = {}
            if index.index_type == "embedding":
                kwargs["min_score"] = search_kwargs.get("min_score")
                assert isinstance(index, EmbeddingIndex)
                embedding = self._embed_query(index, query, query_type)
                kwargs["embedding"] = embedding
                # always perform exact search and a bit of re-ranking
                # to improve quality
                kwargs["exact"] = True
                # factor of oversampling for re-ranking
                kwargs["rerank"] = 2.0
            else:
                kwargs["query"] = query

            if identifier_map is None:
                allow_ids = None
            else:
                allow_ids = set()
                for identifier in identifier_map:
                    id = data.id_from_identifier(identifier)
                    if id is not None:
                        allow_ids.add(id)

            identifiers = []
            for id, field, _ in index.search(k=k, allow_ids=allow_ids, **kwargs):
                identifier = data.identifier(id)
                assert identifier is not None, "should not happen"
                identifiers.append(identifier)
                field_map[identifier] = data.field(id, field)

        info_sparql = self.get_info_sparql(index_name)
        infos = self.get_info_for_identifiers(identifiers, info_sparql, data)

        alternatives = []
        for identifier in identifiers:
            if identifier_map is not None:
                variants = identifier_map.get(identifier)
            else:
                variants = normalizer.default_variants()

            matched_via = field_map.get(identifier)

            alternative = self.build_alternative_with_info(
                identifier,
                infos.get(identifier, {}),
                variants,
                matched_via,
            )
            alternatives.append(alternative)

        return alternatives

    def get_candidate_ids(
        self,
        index_name: str,
        sparql: str,
        max_candidates: int | None = None,
        request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        read_timeout: float | None = READ_TIMEOUT,
        max_retries: int = 0,
    ) -> dict[str, list[str]]:
        typ = query_type(sparql, self.sparql_parser)
        if typ != "select":
            raise SPARQLException("SPARQL query is not a SELECT query")

        self.logger.debug(
            f"Getting candidate IDs for index '{index_name}' with {sparql}"
        )
        result = self.execute_sparql(sparql, request_timeout, read_timeout, max_retries)

        if not isinstance(result, SelectResult):
            raise SPARQLException("SPARQL query is not a SELECT query")
        if result.num_columns != 1:
            raise SPARQLException("SPARQL query must return a single column")
        if max_candidates is not None and len(result) > max_candidates:
            raise SPARQLException(
                f"Got more than the maximum supported number of "
                f"candidates ({max_candidates:,})"
            )

        self.logger.debug(
            f"Got {len(result):,} candidate items for index '{index_name}'"
        )

        normalizer = self.get_normalizer(index_name)
        data = self.get_data(index_name)

        identifier_map: dict[str, list[str]] = {}
        for bindings in result.bindings():
            binding = next(iter(bindings), None)
            if binding is None or binding.typ != "uri":
                continue

            iri = binding.identifier()

            norm = normalizer.normalize(iri)
            if norm is not None:
                normalized_iri, variant = norm
                if data.id_from_identifier(normalized_iri) is not None:
                    if normalized_iri not in identifier_map:
                        identifier_map[normalized_iri] = []
                    if variant is not None:
                        identifier_map[normalized_iri].append(variant)
                    continue

            # direct match fallback
            if data.id_from_identifier(iri) is not None:
                if iri not in identifier_map:
                    identifier_map[iri] = []

        return identifier_map

    def retrieve_info_for_identifiers(
        self,
        identifiers: Iterable[str],
        info_sparql: str,
    ) -> dict[str, dict]:
        info = {}

        try:
            assert "{IDS}" in info_sparql, (
                "SPARQL must contain {IDS} placeholder for identifiers"
            )
            info_sparql = info_sparql.replace(
                "{IDS}", " ".join(wrap_iri(id) for id in identifiers)
            )
            self.logger.debug(f"Retrieving infos with SPARQL:\n{info_sparql}")
            result = self.execute_sparql(
                info_sparql,
                # set info timeouts to something shorter than usual
                request_timeout=(4.0, 6.0),
                read_timeout=6.0,
            )
            assert isinstance(result, SelectResult) and result.num_columns == 3, (
                "Expected a SELECT query with three columns for info SPARQL"
            )
            id_var = result.variables[0]
            text_var = result.variables[1]
            type_var = result.variables[2]
            for row in result.rows():
                assert id_var in row, "Identifier column not found in result row"
                assert row[id_var].typ == "uri"
                assert row[text_var].typ == "literal"
                assert row[type_var].typ == "literal"

                identifier = row[id_var].identifier()
                if identifier not in info:
                    info[identifier] = {}

                typ = row[type_var].value
                assert typ in {"label", "alias", "info", "other"}
                if typ == "label":
                    # only keep one label
                    info[identifier]["label"] = row[text_var].value
                    continue
                elif typ == "info":
                    # for backwards compatibility
                    typ = "other"

                # keep list for other types
                if typ not in info[identifier]:
                    info[identifier][typ] = []

                text = row[text_var].value
                info[identifier][typ].append(text)

        except Exception as e:
            self.logger.warning(
                "Failed to retrieve info for identifiers using "
                f"info sparql: {e}\n\nSPARQL:\n{info_sparql}"
            )

        return info

    def get_info_for_identifiers_from_index(
        self,
        identifiers: Iterable[str],
        index_name: str,
    ) -> dict[str, dict]:
        info_sparql = self.get_info_sparql(index_name)
        data = self.get_data(index_name)
        return self.get_info_for_identifiers(identifiers, info_sparql, data)

    def get_info_for_identifiers(
        self,
        identifiers: Iterable[str],
        info_sparql: None | str = None,
        data: None | Data = None,
    ) -> dict[str, dict]:
        # try live SPARQL first
        if info_sparql is not None and not self.disable_info_retrieval:
            info = self.retrieve_info_for_identifiers(identifiers, info_sparql)
        else:
            info = {}

        if data is None:
            return info

        # try and fill up remaining from local data
        for identifier in identifiers:
            id = data.id_from_identifier(identifier)
            if id is None:
                continue

            current_info = info.get(identifier, {})

            if not current_info.get("label"):
                current_info["label"] = data.main_field(id)

            if not current_info.get("alias"):
                current_info["alias"] = data.fields(id)

            info[identifier] = current_info

        return info

    def autocomplete_prefix(
        self,
        prefix: str,
        limit: int | None = None,
    ) -> tuple[str, str, Position]:
        return autocomplete_prefix(prefix, self.sparql_parser, limit)

    def format_selections(self, selections: list[Selection]) -> str:
        rename_obj_type = [
            (ObjType.ENTITY, "entities"),
            (ObjType.PROPERTY, "properties"),
            (ObjType.UNINDEXED, "other (non-indexed) items"),
        ]

        grouped = group_selections(selections)

        return "\n\n".join(
            f"Using {name}:\n"
            + format_list(
                alt.get_selection_string(include_variants=variants)
                for alt, variants in grouped[obj_type]
            )
            for obj_type, name in rename_obj_type
            if obj_type in grouped
        )


DEFAULT_DESCRIPTIONS = {
    "entities": "Entities indexed by their labels and synonyms",
    "properties": "Properties indexed by their labels, synonyms, and IRIs",
}


def try_load_index(
    kg: str,
    index_name: str,
    index_type: str,
    logger: logging.Logger | None = None,
) -> Index | None:
    index_dir = os.path.join(get_index_dir(kg), index_name)

    index = try_load_search_index(index_dir, index_type, logger)
    if index is None:
        return None

    description = load_index_description(index_dir, logger)
    info_sparql = load_info_sparql(index_dir, logger)

    if index_name == "properties" and kg.startswith("wikidata"):
        normalizer = WikidataPropertyNormalizer()
    else:
        normalizer = None

    return Index(
        description=description
        or DEFAULT_DESCRIPTIONS.get(index_name, "No description available"),
        index=index,
        data=index.data(),
        info_sparql=info_sparql,
        normalizer=normalizer,
    )


def load_kg_manager(
    cfg: KgConfig,
    skip_indices: bool = False,
) -> KgManager:
    logger = get_logger(f"{cfg.kg.upper()} KG MANAGER LOADER")

    prefixes, description = load_kg_info(cfg.kg, logger)
    indices: dict[str, Index] = {}

    if skip_indices:
        logger.info("Skipping loading of indices")
        return KgManager(
            cfg.kg,
            prefixes,
            indices,
            cfg.endpoint,
            description,
        )

    ent_index = try_load_index(
        cfg.kg,
        "entities",
        cfg.entities_type or "fuzzy",
        logger,
    )
    if ent_index is not None:
        indices["entities"] = ent_index

    prop_index = try_load_index(
        cfg.kg,
        "properties",
        cfg.properties_type or "embedding",
        logger,
    )
    if prop_index is not None:
        indices["properties"] = prop_index

    others = load_other_indices(cfg.kg, cfg.indices)
    for name, index in others.items():
        if name in indices:
            logger.warning(
                f"Index '{name}' already loaded as a default index, skipping "
                f"custom index with the same name"
            )
            continue

        indices[name] = index

    return KgManager(
        cfg.kg,
        prefixes,
        indices,
        cfg.endpoint,
        description,
    )


def format_kgs(managers: list[KgManager], notes: dict[str, list[str]]) -> str:
    return format_list(
        format_kg(manager, notes.get(manager.kg)) for manager in managers
    )


def format_kg(manager: KgManager, notes: list[str] | None = None) -> str:
    msg = f'"{manager.kg}" at {manager.endpoint}'
    if manager.description:
        msg += f": {manager.description}"

    indices = []
    for name in manager.index_names:
        index = manager.get(name)
        indices.append(
            f'"{name}" index ({format_index_meta(index.index)}): {index.description}'
        )

    if indices:
        msg += "\n  Search indices:\n" + format_list(indices, indent=4)

    if manager.kg_prefixes:
        msg += f"\n  SPARQL prefixes:\n{format_prefixes(manager.kg_prefixes, indent=4)}"

    if notes:
        msg += "\n  Notes:\n" + format_list(notes, indent=4)

    return msg
