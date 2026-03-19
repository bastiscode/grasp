import math
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Iterable

import validators
from search_rdf import EmbeddingIndex
from universal_ml_utils.ops import partition_by

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.manager.normalizer import Normalizer
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.sparql.item import parse_into_binding
from grasp.sparql.types import (
    Alternative,
    AskResult,
    Binding,
    ObjType,
    Selection,
    SelectResult,
    SelectRow,
)
from grasp.sparql.utils import find_all, parse_string, wrap_iri
from grasp.utils import FunctionCallException

if TYPE_CHECKING:
    from grasp.tasks.base import GraspTask

# maximum number of results for constraining with sub indices
MAX_RESULTS = 131072


def _has_modality(managers: list[KgManager], modality: str) -> bool:
    for manager in managers:
        for name in manager.index_names:
            index = manager.index(name)
            if isinstance(index, EmbeddingIndex):
                modalities = index.modality or ["text"]
                if modality in modalities:
                    return True
    return False


def kg_functions(managers: list[KgManager], fn_set: str) -> list[dict]:
    assert fn_set in [
        "base",
        "search",
        "search_extended",
        "search_filter",
        "search_constraints",
        "all",
    ], f"Unknown function set {fn_set}"
    kgs = [manager.kg for manager in managers]

    # collect all available index names across managers
    all_index_names: list[str] = []
    seen = set()
    for manager in managers:
        for name in manager.index_names:
            if name not in seen:
                all_index_names.append(name)
                seen.add(name)

    # check if any index supports image queries
    has_image = _has_modality(managers, "image")

    fns = [
        {
            "name": "execute",
            "description": """\
Execute a SPARQL query and retrieve its results as a table if successful, \
and an error message otherwise.

For example, to execute a SPARQL query over Wikidata to find the jobs of \
Albert Einstein, do the following:
execute(kg="wikidata", sparql="SELECT ?job WHERE { wd:Q937 wdt:P106 ?job }")""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to query",
                    },
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to execute",
                    },
                },
                "required": ["kg", "sparql"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    if fn_set == "base":
        return fns

    fns.append(
        {
            "name": "list",
            "description": """\
List triples from the knowledge graph satisfying the given subject, property, \
and object constraints.

For example, to find triples with Albert Einstein as the subject in Wikidata, \
do the following:
list(kg="wikidata", subject="wd:Q937")

Or to find examples of how the property "place of birth" is used in Wikidata, \
do the following:
list(kg="wikidata", property="wdt:P19")""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to use",
                    },
                    "subject": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the subject (null if not constrained)",
                    },
                    "property": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the property (null if not constrained)",
                    },
                    "object": {
                        "type": ["string", "null"],
                        "description": "IRI or literal for constraining the object (null if not constrained)",
                    },
                },
                "required": ["kg", "subject", "property", "object"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    )

    if fn_set in ["search", "search_extended", "all"]:
        search_entity_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
        }
        search_entity_required = ["kg", "query"]

        search_property_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
        }
        search_property_required = ["kg", "query"]

        if has_image:
            query_type_prop = {
                "type": "string",
                "enum": ["text", "image"],
                "description": 'How to interpret the query string. "text" for text search, "image" for an image URL.',
            }
            search_entity_props["query_type"] = query_type_prop
            search_entity_required.append("query_type")
            search_property_props["query_type"] = dict(query_type_prop)
            search_property_required.append("query_type")

        fns.extend(
            [
                {
                    "name": "search_entity",
                    "description": """\
Search for entities in the knowledge graph with a search query. \
This function uses the index type for entities of the \
given knowledge graph internally.

For example, to search for the entity Albert Einstein in Wikidata, \
do the following:
search_entity(kg="wikidata", query="albert einstein")""",
                    "parameters": {
                        "type": "object",
                        "properties": search_entity_props,
                        "required": search_entity_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "name": "search_property",
                    "description": """\
Search for properties in the knowledge graph with a search query. \
This function uses the index type for properties of the \
given knowledge graph internally.

For example, to search for properties related to birth in Wikidata, do the following:
search_property(kg="wikidata", query="birth")""",
                    "parameters": {
                        "type": "object",
                        "properties": search_property_props,
                        "required": search_property_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ]
        )

    if fn_set in ["search_extended", "all"]:
        search_prop_of_ent_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "entity": {
                "type": "string",
                "description": "The entity to search properties for",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
        }
        search_prop_of_ent_required = ["kg", "entity", "query"]

        search_obj_of_prop_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "property": {
                "type": "string",
                "description": "The property to search objects for",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
        }
        search_obj_of_prop_required = ["kg", "property", "query"]

        if has_image:
            query_type_prop = {
                "type": "string",
                "enum": ["text", "image"],
                "description": 'How to interpret the query string. "text" for text search, "image" for an image URL.',
            }
            search_prop_of_ent_props["query_type"] = query_type_prop
            search_prop_of_ent_required.append("query_type")
            search_obj_of_prop_props["query_type"] = dict(query_type_prop)
            search_obj_of_prop_required.append("query_type")

        fns.extend(
            [
                {
                    "name": "search_property_of_entity",
                    "description": """\
Search for properties of a given entity in the knowledge graph. \
This function uses the index type for properties of the \
given knowledge graph internally.

For example, to search for properties related to birth for Albert Einstein \
in Wikidata, do the following:
search_property_of_entity(kg="wikidata", entity="wd:Q937", query="birth")""",
                    "parameters": {
                        "type": "object",
                        "properties": search_prop_of_ent_props,
                        "required": search_prop_of_ent_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "name": "search_object_of_property",
                    "description": """\
Search for objects (entities or literals) for a given property in the knowledge graph. \
This function uses the index type for entities of the \
given knowledge graph and a temporary prefix index for literals internally.

For example, to search for football jobs in Wikidata, do the following:
search_object_of_property(kg="wikidata", property="wdt:P106", query="football")""",
                    "parameters": {
                        "type": "object",
                        "properties": search_obj_of_prop_props,
                        "required": search_obj_of_prop_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ]
        )

    if fn_set in ["search_filter", "all"]:
        search_filter_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "index": {
                "type": "string",
                "enum": all_index_names,
                "description": "The index to search in",
            },
            "sparql": {
                "type": "string",
                "description": "The SPARQL query with ?search variable",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
        }
        search_filter_required = ["kg", "index", "sparql", "query"]

        if has_image:
            search_filter_props["query_type"] = {
                "type": "string",
                "enum": ["text", "image"],
                "description": 'How to interpret the query string. "text" for text search, "image" for an image URL.',
            }
            search_filter_required.append("query_type")

        fns.append(
            {
                "name": "search_with_filter",
                "description": """\
Search for knowledge graph items in a context-sensitive way by specifying a constraining \
SPARQL query together with a search query. The SPARQL query must be a SELECT query \
returning a single column of IRIs. The search is then restricted to knowledge graph items \
matching those IRIs in the specified index. The index parameter specifies which index to \
search in ("entity", "property", or any other index name).

For example, to search for Albert Einstein at the subject position in \
Wikidata, do the following:
search_with_filter(kg="wikidata", index="entity", sparql="SELECT DISTINCT ?s WHERE { ?s ?p ?o }", query="albert einstein")

Or to search for properties of Albert Einstein related to his birth in \
Wikidata, do the following:
search_with_filter(kg="wikidata", index="property", sparql="SELECT DISTINCT ?p WHERE { wd:Q937 ?p ?o }", query="birth")""",
                "parameters": {
                    "type": "object",
                    "properties": search_filter_props,
                    "required": search_filter_required,
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )

    if fn_set in ["search_constraints", "all"]:
        search_constraints_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "index": {
                "type": "string",
                "enum": all_index_names,
                "description": "The index to search in",
            },
            "position": {
                "type": "string",
                "enum": ["subject", "property", "object"],
                "description": "The position/type of item to look for",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "constraints": {
                "type": ["object", "null"],
                "description": "Constraints for the search, \
can be null if there are none",
                "properties": {
                    "subject": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the subject (null if not constrained)",
                    },
                    "property": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the property (null if not constrained)",
                    },
                    "object": {
                        "type": ["string", "null"],
                        "description": "IRI or literal for constraining the object (null if not constrained)",
                    },
                },
                "required": ["subject", "property", "object"],
                "additionalProperties": False,
            },
        }
        search_constraints_required = [
            "kg",
            "index",
            "position",
            "query",
            "constraints",
        ]

        if has_image:
            search_constraints_props["query_type"] = {
                "type": "string",
                "enum": ["text", "image"],
                "description": 'How to interpret the query string. "text" for text search, "image" for an image URL.',
            }
            search_constraints_required.append("query_type")

        fns.append(
            {
                "name": "search_with_constraints",
                "description": """\
Search for knowledge graph items at a particular position (subject, property, or object) \
with optional constraints. If constraints are provided, they are used to limit the search \
space accordingly. The index parameter specifies which index to search in ("entity", \
"property", or any other index name). For position "property", the index must be "property". \
For positions "subject" or "object", the index can be "entity" or any sub-index name.

For example, to search for the subject Albert Einstein in Wikidata, do the following:
search_with_constraints(kg="wikidata", index="entity", position="subject", query="albert einstein")

Or to search for properties of Albert Einstein related to his birth in Wikidata, \
do the following:
search_with_constraints(kg="wikidata", index="property", position="property", query="birth", \
constraints={"subject": "wd:Q937"})""",
                "parameters": {
                    "type": "object",
                    "properties": search_constraints_props,
                    "required": search_constraints_required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        )

    return fns


def find_manager(
    managers: list[KgManager],
    kg: str,
) -> tuple[KgManager, list[KgManager]]:
    managers, others = partition_by(managers, lambda m: m.kg == kg)
    if not managers:
        raise FunctionCallException(f"Unknown knowledge graph {kg}")
    elif len(managers) > 1:
        raise FunctionCallException(f"Multiple managers found for knowledge graph {kg}")
    return managers[0], others


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    task: "GraspTask | None" = None,
    example_indices: dict | None = None,
) -> str:
    if fn_name == "execute":
        return execute_sparql(
            managers,
            fn_args["kg"],
            fn_args["sparql"],
            config.result_max_rows,
            config.result_max_columns,
            known,
            config.know_before_use,
        ).formatted  # type: ignore

    elif fn_name == "list":
        return list_triples(
            managers,
            fn_args["kg"],
            fn_args.get("subject"),
            fn_args.get("property"),
            fn_args.get("object"),
            config.list_k,
            known,
        )

    elif fn_name == "search_entity":
        return search_entity(
            managers,
            fn_args["kg"],
            fn_args["query"],
            config.search_top_k,
            known,
            fn_args.get("query_type", "text"),
        )

    elif fn_name == "search_property":
        return search_property(
            managers,
            fn_args["kg"],
            fn_args["query"],
            config.search_top_k,
            known,
            fn_args.get("query_type", "text"),
        )

    elif fn_name == "search_property_of_entity":
        return search_with_constraints(
            managers,
            fn_args["kg"],
            "property",
            fn_args["query"],
            {"subject": fn_args["entity"]},
            config.search_top_k,
            known,
            "property",
            fn_args.get("query_type", "text"),
        )

    elif fn_name == "search_object_of_property":
        return search_with_constraints(
            managers,
            fn_args["kg"],
            "object",
            fn_args["query"],
            {"property": fn_args["property"]},
            config.search_top_k,
            known,
            "entity",
            fn_args.get("query_type", "text"),
        )

    elif fn_name == "search_with_constraints":
        return search_with_constraints(
            managers,
            fn_args["kg"],
            fn_args["position"],
            fn_args["query"],
            fn_args.get("constraints"),
            config.search_top_k,
            known,
            fn_args.get("index", "entity"),
            fn_args.get("query_type", "text"),
        )

    elif fn_name == "search_with_filter":
        return search_with_filter(
            managers,
            fn_args["kg"],
            fn_args["sparql"],
            fn_args["query"],
            config.search_top_k,
            known,
            fn_args.get("index", "entity"),
            fn_args.get("query_type", "text"),
        )

    elif task is not None:
        return task.call_function(fn_name, fn_args, known, example_indices)

    else:
        raise ValueError(f"Unknown function {fn_name}")


def search_entity(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    alts = manager.search_index(
        "entity",
        query=query,
        k=k,
        query_type=query_type,
        **search_kwargs,
    )

    # update known items
    normalizer = manager.normalizer("entity")
    update_known_from_alts(known, alts, normalizer)

    return format_index_alternatives(alts, "entity", k)


def search_property(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    alts = manager.search_index(
        "property",
        query=query,
        k=k,
        query_type=query_type,
        **search_kwargs,
    )

    # update known items
    normalizer = manager.normalizer("property")
    update_known_from_alts(known, alts, normalizer)

    return format_index_alternatives(alts, "property", k)


COMMON_PREFIXES = get_common_sparql_prefixes()


def check_known(manager: KgManager, sparql: str, known: set[str]):
    parse, _ = parse_string(sparql, manager.sparql_parser)
    in_query = set()

    for iri in find_all(parse, {"IRIREF", "PNAME_NS", "PNAME_LN"}, skip={"Prologue"}):
        binding = parse_into_binding(
            iri["value"],
            manager.iri_literal_parser,
            manager.prefixes,
        )
        assert binding is not None, f"Failed to parse binding from {iri['value']}"
        assert binding.typ == "uri", f"Expected IRI, got {binding.typ}"

        identifier = binding.identifier()

        longest = manager.find_longest_prefix(identifier)
        if longest is None or longest[0] not in COMMON_PREFIXES:
            # unknown or uncommon prefix, should be known before use
            in_query.add(identifier)

    unknown_in_query = in_query - known
    if unknown_in_query:
        not_seen = "\n".join(manager.format_iri(iri) for iri in unknown_in_query)
        raise FunctionCallException(f"""\
The following knowledge graph items are used in the SPARQL query \
without being known from previous function call results. \
This does not mean they are invalid, but you should verify \
that they indeed exist in the knowledge graphs before executing the SPARQL \
query again:
{not_seen}""")


def update_known_from_iris(
    known: set[str],
    iris: Iterable[str],
    normalizer: Normalizer | None = None,
):
    for iri in iris:
        known.add(iri)
        if normalizer is None:
            continue

        norm = normalizer.normalize(iri)
        if norm is None:
            continue

        # also add normalized identifier
        known.add(norm[0])


def update_known_from_alts(
    known: set[str],
    alts: Iterable[Alternative],
    normalizer: Normalizer | None = None,
):
    for alt in alts:
        known.add(alt.identifier)
        if normalizer is None or not alt.variants:
            continue

        for var in alt.variants:
            denorm = normalizer.denormalize(alt.identifier, var)
            if denorm is None:
                continue
            known.add(denorm)


def update_known_from_rows(
    known: set[str],
    rows: Iterable[SelectRow],
    normalizer: Normalizer | None = None,
):
    update_known_from_iris(
        known,
        (
            binding.identifier()
            for row in rows
            for binding in row.values()
            if binding.typ == "uri"
        ),
        normalizer,
    )


def update_known_from_alternatives(
    known: set[str],
    alternatives: dict[ObjType, list[Alternative]],
    manager: KgManager,
):
    # entities
    update_known_from_alts(
        known,
        alternatives.get(ObjType.ENTITY, []),
        manager.entity_normalizer,
    )

    # properties
    update_known_from_alts(
        known,
        alternatives.get(ObjType.PROPERTY, []),
        manager.property_normalizer,
    )

    # other
    update_known_from_alts(
        known,
        alternatives.get(ObjType.UNINDEXED, []),
    )


def update_known_from_selections(
    known: set[str],
    selections: list[Selection],
    manager: KgManager,
):
    # entities
    update_known_from_alts(
        known,
        (sel.alternative for sel in selections if sel.obj_type == ObjType.ENTITY),
        manager.entity_normalizer,
    )

    # properties
    update_known_from_alts(
        known,
        (sel.alternative for sel in selections if sel.obj_type == ObjType.PROPERTY),
        manager.property_normalizer,
    )


@dataclass
class ExecutionResult:
    sparql: str
    formatted: str
    result: SelectResult | AskResult | None = None


def execute_sparql(
    managers: list[KgManager],
    kg: str,
    sparql: str,
    max_rows: int,
    max_columns: int,
    known: set[str] | None = None,
    know_before_use: bool = False,
) -> ExecutionResult:
    manager, others = find_manager(managers, kg)

    # fix prefixes with managers
    sparql = manager.fix_prefixes(sparql)
    for other in others:
        sparql = other.fix_prefixes(sparql)

    sparql = manager.prettify(sparql)

    if know_before_use and known is not None:
        check_known(manager, sparql, known)

    try:
        result = manager.execute_sparql(sparql)
    except Exception as e:
        error = f"SPARQL execution failed:\n{e}"
        return ExecutionResult(sparql, error)

    half_rows = math.ceil(max_rows / 2)
    half_columns = math.ceil(max_columns / 2)

    if isinstance(result, SelectResult) and known is not None:
        # only update with the bindings shown to the model
        shown_vars = result.variables[:half_columns] + result.variables[-half_columns:]
        rows = (
            {var: row[var] for var in shown_vars if var in row}
            for row in chain(
                result.rows(end=half_rows),
                result.rows(start=max(0, len(result) - half_rows)),
            )
        )

        # entity mapping
        update_known_from_rows(known, rows, manager.entity_normalizer)

        # property mapping
        update_known_from_rows(known, rows, manager.property_normalizer)

    formatted = manager.format_sparql_result(
        result,
        half_rows,
        half_rows,
        half_columns,
        half_columns,
    )
    return ExecutionResult(sparql, formatted, result)


def is_iri_or_literal(iri: str, manager: KgManager) -> bool:
    try:
        _ = parse_string(iri, manager.iri_literal_parser)
        return True
    except Exception:
        return False


def verify_iri_or_literal(input: str, position: str, manager: KgManager) -> str | None:
    if is_iri_or_literal(input, manager):
        return input

    url = validators.url(input)

    if position == "object" and not url:
        # check first if it is a string literal
        input = f'"{input}"'
        if is_iri_or_literal(input, manager):
            return input

    elif not url:
        return None

    # url like, so add < and > and check again
    input = wrap_iri(input)
    if is_iri_or_literal(input, manager):
        return input
    else:
        return None


def list_triples(
    managers: list[KgManager],
    kg: str,
    subject: str | None,
    property: str | None,
    obj: str | None,
    k: int,
    known: set[str],
) -> str:
    manager, _ = find_manager(managers, kg)

    triple = []
    bindings = []
    for pos, const in [("subject", subject), ("property", property), ("object", obj)]:
        if const is None:
            triple.append(f"?{pos[0]}")
            continue

        ver_const = verify_iri_or_literal(const, pos, manager)
        if ver_const is None:
            expected = "IRI" if pos != "object" else "IRI or literal"
            raise FunctionCallException(
                f'Constraint "{const}" for {pos} position \
is not a valid {expected}. IRIs can be given in prefixed form, like "wd:Q937", \
as URIs, like "http://www.wikidata.org/entity/Q937", \
or in full form, like "<http://www.wikidata.org/entity/Q937>".'
            )

        bindings.append(f"BIND({ver_const} AS ?{pos[0]})")
        triple.append(ver_const)

    triple = " ".join(triple)
    bindings = "\n".join(bindings)
    sparql = f"""\
SELECT ?s ?p ?o WHERE {{
    {triple}
    {bindings}
}} LIMIT {MAX_RESULTS + 1}"""

    try:
        result = manager.execute_sparql(sparql)
    except Exception as e:
        raise FunctionCallException(f"Failed to list triples with error:\n{e}") from e

    assert isinstance(result, SelectResult)
    result.truncate(MAX_RESULTS)

    # functions to get scores for properties and entities
    def prop_rank(prop: Binding) -> int:
        if not manager.property_data:
            return 0

        norm = manager.property_normalizer.normalize(prop.identifier())
        if norm is None:
            return len(manager.property_data)

        id = manager.property_data.id_from_identifier(norm[0])
        if id is None:
            return len(manager.property_data)

        # lower id means more popular property
        return id

    def ent_rank(ent: Binding) -> int:
        if not manager.entity_data:
            return 0

        norm = manager.entity_normalizer.normalize(ent.identifier())
        if norm is None:
            return len(manager.entity_data)

        id = manager.entity_data.id_from_identifier(norm[0])
        if id is None:
            return len(manager.entity_data)

        # lower id means more popular entity
        return id

    # make sure that rows presented are diverse and that
    # we show the ones with popular properties or subjects / objects
    # first
    def sort_key(row: SelectRow) -> tuple[int, int]:
        # property score
        ps = prop_rank(row["p"])

        # entity score
        es = min(ent_rank(row["s"]), ent_rank(row["o"]))

        # sort first by properties, then by subjects or objects
        return ps, es

    # rows are now sorted by popularity, lowest rank first
    sorted_rows = sorted(
        enumerate(result.rows()),
        key=lambda item: sort_key(item[1]),
    )

    def normalize_prop(prob: Binding) -> str:
        identifier = prob.identifier()
        norm = manager.property_normalizer.normalize(identifier)
        return norm[0] if norm is not None else identifier

    def normalize_ent(ent: Binding) -> str:
        identifier = ent.identifier()
        norm = manager.entity_normalizer.normalize(identifier)
        return norm[0] if norm is not None else identifier

    # now make sure that we show a diverse set of rows
    # triples with unseen properties or subjects / objects
    # should come first
    probs_seen = set()
    ents_seen = set()
    permutation = []

    for i, row in sorted_rows:
        # normalize
        s = normalize_ent(row["s"])
        p = normalize_prop(row["p"])
        o = normalize_ent(row["o"])

        key = (p in probs_seen, s in ents_seen or o in ents_seen)
        permutation.append((key, i))

        probs_seen.add(p)
        ents_seen.add(s)
        ents_seen.add(o)

    # sort by number of seen columns
    # since sort is stable, we keep relative popularity order from before
    permutation = sorted(permutation, key=lambda item: item[0])
    result.data = [result.data[i] for _, i in permutation]

    # update known
    update_known_from_rows(known, result.rows(end=k), manager.entity_normalizer)
    update_known_from_rows(known, result.rows(end=k), manager.property_normalizer)

    # override column names
    column_names = ["subject", "property", "object"]

    return manager.format_sparql_result(
        result,
        show_top_rows=k,
        show_bottom_rows=0,
        show_left_columns=3,
        show_right_columns=0,
        column_names=column_names,
    )


def search_with_constraints(
    managers: list[KgManager],
    kg: str,
    position: str,
    query: str,
    constraints: dict[str, str | None] | None,
    k: int,
    known: set[str],
    index: str = "entity",
    query_type: str = "text",
    max_results: int = MAX_RESULTS,
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    # validate index vs position
    if index == "property":
        assert position == "property", "index='property' requires position='property'"
    elif position == "property":
        # position=property implies property index
        index = "property"

    if constraints is None:
        constraints = {}

    target_constr = constraints.get(position)
    if target_constr is not None:
        raise FunctionCallException(
            f'Cannot look for {position} and constrain it to \
"{target_constr}" at the same time.'
        )

    num_constraints = sum(c is not None for c in constraints.values())
    if num_constraints > 2:
        raise FunctionCallException(
            "At most two of subject, property, and \
object should be constrained at once."
        )

    identifier_map = None
    info = ""
    if num_constraints > 0:
        pos_values = {}
        for pos in ["subject", "property", "object"]:
            const = constraints.get(pos)
            if const is None:
                pos_values[pos] = f"?{pos[0]}"
                continue

            elif pos == position:
                pos_values[pos] = "?search"
                continue

            ver_const = verify_iri_or_literal(const, pos, manager)
            if ver_const is None:
                expected = "IRI" if pos != "object" else "IRI or literal"
                raise FunctionCallException(
                    f'Constraint "{const}" for {pos} position \
is not a valid {expected}. IRIs can be given in prefixed form, like "wd:Q937", \
as URIs, like "http://www.wikidata.org/entity/Q937", \
or in full form, like "<http://www.wikidata.org/entity/Q937>".'
                )

            pos_values[pos] = ver_const

        select_var = f"?{position[0]}"

        sparql = f"""\
SELECT DISTINCT {select_var} WHERE {{
    {pos_values["subject"]}
    {pos_values["property"]}
    {pos_values["object"]}
}} LIMIT {MAX_RESULTS + 1}"""

        try:
            identifier_map = manager.get_candidate_ids(
                index,
                sparql,
                max_results,
            )
        except Exception as e:
            info = f"""\
Falling back to an unconstrained search on the full \
search index due to:
{e}

"""

    alternatives = manager.search_index(
        index,
        query,
        k,
        identifier_map,
        query_type=query_type,
        **search_kwargs,
    )

    # update known items
    normalizer = manager.normalizer(index)
    update_known_from_alts(known, alternatives, normalizer)

    return info + format_index_alternatives(alternatives, index, k)


def format_index_alternatives(
    alternatives: list[Alternative],
    index_name: str,
    k: int,
) -> str:
    if not alternatives:
        return f"No {index_name} alternatives found"

    top_k_string = "\n".join(
        f"{i + 1}. {alt.get_selection_string()}" for i, alt in enumerate(alternatives)
    )
    return f"Top {k} {index_name} alternatives:\n{top_k_string}"


def search_with_filter(
    managers: list[KgManager],
    kg: str,
    sparql: str,
    query: str,
    k: int,
    known: set[str],
    index: str = "entity",
    query_type: str = "text",
    max_results: int = MAX_RESULTS,
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    identifier_map = None
    info = ""
    try:
        identifier_map = manager.get_candidate_ids(
            index,
            sparql,
            max_results,
        )
    except Exception as e:
        info = f"""\
Falling back to an unconstrained search on the full \
search index due to:
{e}

"""

    alternatives = manager.search_index(
        index,
        query,
        k=k,
        identifier_map=identifier_map,
        query_type=query_type,
        **search_kwargs,
    )

    # update known items
    normalizer = manager.normalizer(index)
    update_known_from_alts(known, alternatives, normalizer)

    return info + format_index_alternatives(alternatives, index, k)
