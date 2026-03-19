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
from grasp.manager.cache import Cache
from grasp.manager.normalizer import Normalizer
from grasp.manager.utils import (
    EmbeddingModel,
    Index,
    SearchIndex,
    load_image_from_url,
    format_index_meta,
    get_embedding_model_key,
    load_embedding_model,
    load_kg_indices,
    load_kg_info_caches,
    load_kg_info_sparqls,
    load_kg_normalizers,
    load_kg_prefixes,
    load_other_indices,
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
    has_iri,
    load_entity_info_sparql,
    load_iri_and_literal_parser,
    load_property_info_sparql,
    load_sparql_parser,
    prettify,
    query_type,
    wrap_iri,
)
from grasp.utils import clip, format_list, ordered_unique


class KgManager:
    def __init__(
        self,
        kg: str,
        entity_normalizer: Normalizer,
        property_normalizer: Normalizer,
        entity_index: SearchIndex | None = None,
        property_index: SearchIndex | None = None,
        entity_info_sparql: str | None = None,
        property_info_sparql: str | None = None,
        entity_cache: Cache | None = None,
        property_cache: Cache | None = None,
        prefixes: dict[str, str] | None = None,
        indices: dict[str, Index] | None = None,
        endpoint: str | None = None,
    ):
        self.kg = kg

        self.entity_index = entity_index
        self.entity_data = entity_index.data() if entity_index else None
        self.entity_cache = entity_cache

        self.property_index = property_index
        self.property_data = property_index.data() if property_index else None
        self.property_cache = property_cache

        self.entity_normalizer = entity_normalizer
        self.property_normalizer = property_normalizer

        self.sparql_parser = load_sparql_parser()
        self.iri_literal_parser = load_iri_and_literal_parser()

        self.prefixes = prefixes or {}

        self.entity_info_sparql = entity_info_sparql or load_entity_info_sparql()
        self.property_info_sparql = property_info_sparql or load_property_info_sparql()
        self.disable_info_retrieval = False

        self.endpoint = endpoint or get_endpoint(self.kg)

        self.indices = indices or {}

        self.embedding_models: dict[str, EmbeddingModel] = {}

        self.logger = get_logger(f"{self.kg.upper()} KG MANAGER")

    def load_models(
        self,
        models: dict[str, EmbeddingModel] | None = None,
    ) -> dict[str, EmbeddingModel]:
        if models is None:
            models = {}

        if self.entity_index is not None:
            models = load_embedding_model(self.entity_index, models)

        if self.property_index is not None:
            models = load_embedding_model(self.property_index, models)

        for sub in self.indices.values():
            models = load_embedding_model(sub.index, models)

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
        max_retries: int = 0,
        force_select_result: bool = False,
        read_timeout: float | None = READ_TIMEOUT,
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
            max_retries,
            read_timeout,
        )

    def format_sparql_result(
        self,
        result: SelectResult | AskResult,
        show_top_rows: int = 5,
        show_bottom_rows: int = 5,
        show_left_columns: int = 5,
        show_right_columns: int = 5,
        column_names: list[str] | None = None,
    ) -> str:
        # run sparql against endpoint, format result as string
        if isinstance(result, AskResult):
            return str(result.boolean)

        if result.num_rows == 0:
            return f"Got no rows and {result.num_columns:,} columns"

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
                    formatted = clip(val.value)
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
                    obj_type = ObjType.ENTITY
                    norm = self.normalize(identifier, obj_type)

                    if norm is None or self.label(norm[0], obj_type) is None:
                        obj_type = ObjType.PROPERTY
                        norm = self.normalize(identifier, obj_type)

                    # still not found, just output the formatted iri
                    if norm is None or self.label(norm[0], obj_type) is None:
                        formatted_row.append(formatted)
                        continue

                    label = self.label(norm[0], obj_type)
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
            f"{result.num_columns:,} column{'s' * (result.num_columns != 1)}"
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

    def format_iri(self, iri: str, base_uri: str | None = None) -> str:
        return format_iri(iri, self.prefixes, base_uri)

    def fix_prefixes(
        self,
        sparql: str,
        is_prefix: bool = False,
        remove_known: bool = False,
        sort: bool = False,
    ) -> str:
        return fix_prefixes(
            sparql,
            self.sparql_parser,
            self.prefixes,
            is_prefix,
            remove_known,
            sort,
        )

    def normalizer(self, name: str) -> Normalizer:
        if name == "entity":
            return self.entity_normalizer
        elif name == "property":
            return self.property_normalizer
        elif name in self.indices:
            return Normalizer()
        else:
            raise ValueError(f"Unknown index name '{name}'")

    def index(self, name: str) -> SearchIndex:
        if name == "entity":
            assert self.entity_index is not None, "Entity index is not loaded"
            return self.entity_index
        elif name == "property":
            assert self.property_index is not None, "Property index is not loaded"
            return self.property_index
        elif name in self.indices:
            return self.indices[name].index
        else:
            raise ValueError(f"Unknown index name '{name}'")

    def data(self, name: str) -> Data:
        if name == "entity":
            assert self.entity_data is not None, "Entity data is not loaded"
            return self.entity_data
        elif name == "property":
            assert self.property_data is not None, "Property data is not loaded"
            return self.property_data
        elif name in self.indices:
            return self.indices[name].index.data()
        else:
            raise ValueError(f"Unknown index name '{name}'")

    def get_info_sparql(self, name: str) -> str | None:
        if name == "entity":
            return self.entity_info_sparql
        elif name == "property":
            return self.property_info_sparql
        elif name in self.indices:
            return self.indices[name].info_sparql
        else:
            return None

    def get_info_cache(self, name: str) -> Cache | None:
        if name == "entity":
            return self.entity_cache
        elif name == "property":
            return self.property_cache
        elif name in self.indices:
            return self.indices[name].cache
        else:
            return None

    @property
    def index_names(self) -> list[str]:
        names = []
        if self.entity_index is not None:
            names.append("entity")
        if self.property_index is not None:
            names.append("property")
        names.extend(self.indices.keys())
        return names

    def normalize(
        self,
        identifier: str,
        index_name: str,
    ) -> tuple[str, str | None] | None:
        return self.normalizer(index_name).normalize(identifier)

    def denormalize(
        self,
        identifier: str,
        index_name: str,
        variant: str | None = None,
    ) -> str | None:
        return self.normalizer(index_name).denormalize(identifier, variant)

    def check_identifier(
        self,
        identifier: str,
        index_name: str,
    ) -> bool:
        return self.data(index_name).id_from_identifier(identifier) is not None

    def label(
        self,
        identifier: str,
        obj_type: ObjType,
    ) -> str | None:
        if obj_type == ObjType.ENTITY:
            data = self.entity_data
        elif obj_type == ObjType.PROPERTY:
            data = self.property_data
        else:
            data = None

        if data is None:
            return None

        id = data.id_from_identifier(identifier)
        if id is None:
            return None

        return data.main_field(id) or data.field(id, 0)

    def build_alternative_with_infos(
        self,
        identifier: str,
        infos: dict | None = None,
        variants: list[str] | None = None,
        matched_via: str | None = None,
    ) -> Alternative:
        if infos is None:
            infos = {}

        # extract needed data from infos dict
        label = infos.get("label")
        aliases = infos.get("alias", [])
        added_infos = infos.get("info", [])

        return self.build_alternative(
            identifier,
            label,
            aliases,
            added_infos,
            variants,
            matched_via,
        )

    def build_alternative(
        self,
        identifier: str,
        label: str | None = None,
        aliases: list[str] | None = None,
        infos: list[str] | None = None,
        variants: list[str] | None = None,
        matched_via: str | None = None,
    ) -> Alternative:
        # preprocess some fields
        if variants is not None:
            variants = ordered_unique(variants)

        if aliases is not None:
            aliases = ordered_unique(aliases, filter=lambda alias: alias != label)

        if infos is not None:
            infos = ordered_unique(infos)

        return Alternative(
            identifier=identifier,
            short_identifier=self.format_iri(identifier),
            label=label,
            variants=variants,
            aliases=aliases,
            infos=infos,
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
        index = self.index(index_name)
        data = self.data(index_name)
        normalizer = self.normalizer(index_name)

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
        info_cache = self.get_info_cache(index_name)
        infos = self.get_infos_for_identifiers(
            identifiers, info_sparql, info_cache, data
        )

        alternatives = []
        for identifier in identifiers:
            if identifier_map is not None:
                variants = identifier_map.get(identifier)
            else:
                variants = normalizer.default_variants()

            matched_via = None
            if field_map:
                matched_via = field_map.get(identifier)

            alternative = self.build_alternative_with_infos(
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
        timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 0,
    ) -> dict[str, list[str]]:
        typ = query_type(sparql, self.sparql_parser)
        if typ != "select":
            raise SPARQLException("SPARQL query is not a SELECT query")

        if not has_iri(sparql, self.sparql_parser):
            raise SPARQLException("SPARQL query contains no IRIs to constrain with")

        self.logger.debug(
            f"Getting candidate IDs for index '{index_name}' with {sparql}"
        )
        result = self.execute_sparql(sparql, timeout, max_retries)

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

        normalizer = self.normalizer(index_name)
        data = self.data(index_name)

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

    def retrieve_infos_for_identifiers(
        self,
        identifiers: Iterable[str],
        info_sparql: str,
    ) -> dict[str, dict]:
        infos = {}

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
                if identifier not in infos:
                    infos[identifier] = {}

                typ = row[type_var].value
                assert typ in {"label", "alias", "info"}
                if typ == "label":
                    # only keep one label
                    infos[identifier]["label"] = row[text_var].value
                    continue

                # keep list for other types
                if typ not in infos[identifier]:
                    infos[identifier][typ] = []

                text = row[text_var].value
                infos[identifier][typ].append(text)

        except Exception as e:
            self.logger.warning(f"Failed to retrieve infos for identifiers: {e}")

        return infos

    def get_infos_for_identifiers_of_type(
        self,
        identifiers: Iterable[str],
        obj_type: ObjType,
    ) -> dict[str, dict]:
        if obj_type == ObjType.ENTITY:
            info_sparql = self.entity_info_sparql
            info_cache = self.entity_cache
            data = self.entity_data
        elif obj_type == ObjType.PROPERTY:
            info_sparql = self.property_info_sparql
            info_cache = self.property_cache
            data = self.property_data
        else:
            self.logger.warning(f"No info retrieval for object type '{obj_type}'")
            return {}

        return self.get_infos_for_identifiers(
            identifiers,
            info_sparql,
            info_cache,
            data,
        )

    def get_infos_for_identifiers(
        self,
        identifiers: Iterable[str],
        info_sparql: None | str = None,
        info_cache: None | Cache = None,
        data: None | Data = None,
    ) -> dict[str, dict]:
        infos = {}

        # try cache first
        if info_cache is not None:
            left = []
            for identifier in identifiers:
                info = info_cache.get(identifier)
                if info is None:
                    left.append(identifier)
                    continue

                infos[identifier] = info

            identifiers = left

        if not identifiers:
            return infos

        # try live SPARQL next
        if info_sparql is not None and not self.disable_info_retrieval:
            live_infos = self.retrieve_infos_for_identifiers(
                identifiers,
                info_sparql,
            )
            infos.update(live_infos)

        # try and fill up remaining from local data
        if data is not None:
            for identifier in identifiers:
                if identifier in infos:
                    continue

                id = data.id_from_identifier(identifier)
                if id is None:
                    continue

                info = {}
                label = data.main_field(id)
                if label is not None:
                    info["label"] = label

                aliases = data.fields(id)
                if aliases:
                    info["alias"] = aliases

                if info:
                    infos[identifier] = info

        return infos

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


def load_kg_manager(
    cfg: KgConfig,
    skip_indices: bool = False,
    skip_caches: bool = False,
) -> KgManager:
    ent_index = prop_index = None
    indices = {}
    if not skip_indices:
        ent_index, prop_index = load_kg_indices(
            cfg.kg,
            cfg.entities_type,
            cfg.properties_type,
        )
        indices = load_other_indices(cfg.kg, cfg.indices)

    prefixes = load_kg_prefixes(cfg.kg, cfg.endpoint)
    ent_norm, prop_norm = load_kg_normalizers(cfg.kg)
    ent_info_sparql, prop_info_sparql = load_kg_info_sparqls(cfg.kg)

    ent_cache = prop_cache = None
    if not skip_caches:
        ent_cache, prop_cache = load_kg_info_caches(cfg.kg)

    return KgManager(
        cfg.kg,
        ent_norm,
        prop_norm,
        ent_index,
        prop_index,
        ent_info_sparql,
        prop_info_sparql,
        ent_cache,
        prop_cache,
        prefixes,
        indices,
        cfg.endpoint,
    )


def format_kgs(managers: list[KgManager], kg_notes: dict[str, list[str]]) -> str:
    if not managers:
        return "No knowledge graphs available"

    return format_list(
        format_kg(
            manager,
            kg_notes.get(manager.kg, []),
        )
        for manager in managers
    )


def format_kg(manager: KgManager, notes: list[str]) -> str:
    msg = f'"{manager.kg}" at {manager.endpoint}'

    parts = []
    if manager.entity_index is not None:
        parts.append(
            f'"entity" index ({format_index_meta(manager.entity_index)}): '
            "Entities indexed by their labels and synonyms"
        )
    if manager.property_index is not None:
        parts.append(
            f'"property" index ({format_index_meta(manager.property_index)}): '
            "Properties indexed by their labels, synonyms, and identifiers"
        )

    if manager.indices:
        other_items = []
        for name, idx in manager.indices.items():
            other_items.append(
                f'"{name}" ({format_index_meta(idx.index)}): {idx.description}'
            )
        parts.append("Other indices:\n" + format_list(other_items, indent=4))

    if notes:
        parts.append("Notes:\n" + format_list(notes, indent=4))

    if parts:
        msg += "\n" + format_list(parts, indent=2)

    return msg
