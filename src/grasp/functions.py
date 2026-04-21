import math
import time
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Iterable

from grammar_utils.parse import LR1Parser  # type: ignore
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
    Position,
    Selection,
    SelectResult,
    SelectRow,
)
from grasp.sparql.utils import (
    READ_TIMEOUT,
    REQUEST_TIMEOUT,
    find_all,
    has_scheme,
    parse_string,
    wrap_iri,
)
from grasp.utils import FunctionCallException, format_list

if TYPE_CHECKING:
    from grasp.tasks.base import GraspTask

# maximum number of results for constraining with sub indices
MAX_RESULTS = 131072

MODALITY_QUERY_TYPES = {
    "text": [("text", "textual search query")],
    "image": [("image", "URL pointing to an image")],
}


def kg_functions(managers: list[KgManager], fn_set: str, list_k: int) -> list[dict]:
    assert fn_set in [
        "base",
        "search",
        "search_extended",
        "search_filter",
        "search_constraints",
        "all",
    ], f"Unknown function set {fn_set}"
    kgs = [manager.kg for manager in managers]

    known_indices = set()
    known_modalities = set()
    for manager in managers:
        known_indices.update(manager.index_names)

        for idx in manager.indices.values():
            if not isinstance(idx.index, EmbeddingIndex):
                continue
            known_modalities.update(idx.index.modality)

    assert all(mod in MODALITY_QUERY_TYPES for mod in known_modalities), (
        f"Unknown modality in {known_modalities}"
    )
    index_names = sorted(known_indices)

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
        },
        {
            "name": "list",
            "description": f"""\
List triples from the knowledge graph satisfying the given subject, property, \
and object constraints. At most {list_k} results are returned per page (use \
pagintion to see more results).

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
                        "description": "IRI for constraining the subject (null for unconstrained)",
                    },
                    "property": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the property (null for unconstrained)",
                    },
                    "object": {
                        "type": ["string", "null"],
                        "description": "IRI or literal for constraining the object (null for unconstrained)",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-indexed) for paginating results (default should be 1)",
                    },
                    "unclipped": {
                        "type": "boolean",
                        "description": "Whether to show full unclipped literal values (default should be false, typically only needed to inspect very long string literals)",
                    },
                },
                "required": [
                    "kg",
                    "page",
                    "subject",
                    "property",
                    "object",
                    "unclipped",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    if fn_set == "base":
        return fns

    has_entity_index = "entities" in known_indices
    has_property_index = "properties" in known_indices

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

        if has_entity_index:
            fns.append(
                {
                    "name": "search_entity",
                    "description": """\
Search for entities in the knowledge graph with a search query.

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
            )

        if has_property_index:
            fns.append(
                {
                    "name": "search_property",
                    "description": """\
Search for properties in the knowledge graph with a search query.

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

        if has_property_index:
            fns.append(
                {
                    "name": "search_property_of_entity",
                    "description": """\
Search for properties of a given entity in the knowledge graph.

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
            )

        if has_entity_index:
            fns.append(
                {
                    "name": "search_object_of_property",
                    "description": """\
Search for entities at the object position for a given property in the knowledge graph.

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
            )

    # prepare query type arg
    query_types = [typ for mod in known_modalities for typ in MODALITY_QUERY_TYPES[mod]]
    query_type_prop = {
        "type": "string",
        "enum": sorted(qt for qt, _ in query_types),
        "description": "How to interpret the query string: "
        + ", ".join(f'"{qt}" for {desc}' for qt, desc in query_types),
    }

    if fn_set in ["search_filter", "all"] and index_names:
        search_filter_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "index": {
                "type": "string",
                "enum": index_names,
                "description": "The index to search in",
            },
            "sparql": {
                "type": ["string", "null"],
                "description": "The SPARQL query for filtering or null for an unconstrained search",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
        }
        search_filter_required = ["kg", "index", "sparql", "query"]

        if len(query_types) > 1:
            search_filter_props["query_type"] = query_type_prop
            search_filter_required.append("query_type")

        search_filter_name = (
            "search" if fn_set == "search_filter" else "search_with_filter"
        )
        fns.append(
            {
                "name": search_filter_name,
                "description": f"""\
Search for knowledge graph items in a context-sensitive way by specifying a filter \
SPARQL query together with a search query. The SPARQL query must be a SELECT query \
returning a single column of IRIs. The search is then restricted to knowledge graph items \
matching those IRIs in the specified index. The SPARQL query can be null, in which case \
a search over the full index is performed.

For example, to search for Albert Einstein in Wikidata, do the following:
{search_filter_name}(kg="wikidata", index="entities", query="albert einstein")

Or to search for properties of Albert Einstein related to his birth in \
Wikidata, do the following:
{search_filter_name}(kg="wikidata", index="properties", sparql="SELECT DISTINCT ?p WHERE {{ wd:Q937 ?p ?o }}", query="birth")""",
                "parameters": {
                    "type": "object",
                    "properties": search_filter_props,
                    "required": search_filter_required,
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )

    if fn_set == "search_constraints" and index_names:
        search_constraints_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "index": {
                "type": "string",
                "enum": index_names,
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
                "description": "Constraints for the search (null for unconstrained)",
                "properties": {
                    "subject": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the subject (null for unconstrained)",
                    },
                    "property": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the property (null for unconstrained)",
                    },
                    "object": {
                        "type": ["string", "null"],
                        "description": "IRI or literal for constraining the object (null for unconstrained)",
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

        if len(query_types) > 1:
            search_constraints_props["query_type"] = query_type_prop
            search_constraints_required.append("query_type")

        search_constraints_name = (
            "search" if fn_set == "search_constraints" else "search_with_constraints"
        )
        fns.append(
            {
                "name": search_constraints_name,
                "description": f"""\
Search for knowledge graph items at a particular position (subject, property, or object) \
with optional constraints. If constraints are provided, they are used to limit the search \
space accordingly.

For example, to search for the subject Albert Einstein in Wikidata, do the following:
{search_constraints_name}(kg="wikidata", index="entities", position="subject", query="albert einstein")

Or to search for properties of Albert Einstein related to his birth in Wikidata, \
do the following:
{search_constraints_name}(kg="wikidata", index="properties", position="property", query="birth", \
constraints={{"subject": "wd:Q937"}})""",
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
            config.sparql_request_timeout,
            config.sparql_read_timeout,
        ).formatted  # type: ignore

    elif fn_name == "list":
        return list_triples(
            managers,
            fn_args["kg"],
            fn_args.get("subject"),
            fn_args.get("property"),
            fn_args.get("object"),
            fn_args.get("page") or 1,
            fn_args.get("unclipped") or False,
            config.list_k,
            known,
            config.sparql_request_timeout,
            config.sparql_read_timeout,
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
            "properties",
            "property",
            fn_args["query"],
            {"subject": fn_args["entity"]},
            config.search_top_k,
            known,
            fn_args.get("query_type", "text"),
            config.sparql_request_timeout,
            config.sparql_read_timeout,
        )

    elif fn_name == "search_object_of_property":
        return search_with_constraints(
            managers,
            fn_args["kg"],
            "entities",
            "object",
            fn_args["query"],
            {"property": fn_args["property"]},
            config.search_top_k,
            known,
            fn_args.get("query_type", "text"),
            config.sparql_request_timeout,
            config.sparql_read_timeout,
        )

    elif fn_name == "search_with_constraints" or (
        fn_name == "search" and config.fn_set == "search_constraints"
    ):
        return search_with_constraints(
            managers,
            fn_args["kg"],
            fn_args["index"],
            fn_args["position"],
            fn_args["query"],
            fn_args.get("constraints"),
            config.search_top_k,
            known,
            fn_args.get("query_type", "text"),
            config.sparql_request_timeout,
            config.sparql_read_timeout,
        )

    elif fn_name == "search_with_filter" or (
        fn_name == "search" and config.fn_set == "search_filter"
    ):
        return search_with_filter(
            managers,
            fn_args["kg"],
            fn_args["index"],
            fn_args["sparql"],
            fn_args["query"],
            config.search_top_k,
            known,
            fn_args.get("query_type", "text"),
            config.know_before_use,
            config.sparql_request_timeout,
            config.sparql_read_timeout,
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
        "entities",
        query=query,
        k=k,
        query_type=query_type,
        **search_kwargs,
    )

    # update known items
    normalizer = manager.get_normalizer("entities")
    update_known_from_alts(known, alts, normalizer)

    return format_index_alternatives(alts, "entities", k)


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
        "properties",
        query=query,
        k=k,
        query_type=query_type,
        **search_kwargs,
    )

    # update known items
    normalizer = manager.get_normalizer("properties")
    update_known_from_alts(known, alts, normalizer)

    return format_index_alternatives(alts, "properties", k)


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
    if not unknown_in_query:
        return

    not_seen = []
    for iri in unknown_in_query:
        short_iri = manager.format_iri(iri)
        if short_iri == iri:
            not_seen.append(iri)
        else:
            not_seen.append(f"{iri} ({short_iri})")

    raise FunctionCallException(f"""\
The following knowledge graph items are used in the SPARQL query \
without being known from previous function call results. \
This does not mean they are invalid, but you should verify \
that they indeed exist (e.g., via listing example triples) \
in the knowledge graphs before trying again:
{format_list(not_seen)}""")


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
        manager.get_normalizer(ObjType.ENTITY.index_name),
    )

    # properties
    update_known_from_alts(
        known,
        alternatives.get(ObjType.PROPERTY, []),
        manager.get_normalizer(ObjType.PROPERTY.index_name),
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
        manager.get_normalizer("entities"),
    )

    # properties
    update_known_from_alts(
        known,
        (sel.alternative for sel in selections if sel.obj_type == ObjType.PROPERTY),
        manager.get_normalizer("properties"),
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
    request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
    read_timeout: float | None = READ_TIMEOUT,
) -> ExecutionResult:
    manager, others = find_manager(managers, kg)

    # fix prefixes with managers
    sparql = manager.fix_prefixes(sparql)
    for other in others:
        sparql = other.fix_prefixes(sparql)

    if know_before_use and known is not None:
        check_known(manager, sparql, known)

    try:
        start = time.monotonic()
        result = manager.execute_sparql(sparql, request_timeout, read_timeout)
        end = time.monotonic()
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
        update_known_from_rows(known, rows, manager.get_normalizer("entities"))

        # property mapping
        update_known_from_rows(known, rows, manager.get_normalizer("properties"))

    formatted = manager.format_sparql_result(
        result,
        half_rows,
        half_rows,
        half_columns,
        half_columns,
        time=end - start,
    )
    return ExecutionResult(sparql, formatted, result)


def parse_iri_or_literal(
    input: str,
    parser: LR1Parser,
    prefixes: dict[str, str] | None = None,
) -> Binding | None:
    # parse and resolve percent encoding in IRIs
    binding = parse_into_binding(input, parser, prefixes)

    if binding is None and has_scheme(input):
        # fallback for full IRIs given without angle brackets
        binding = parse_into_binding(wrap_iri(input), parser, prefixes)

    if binding is None:
        # fallback for string literals because they are typically given without quotes
        # but the parser expects them to be quoted
        binding = parse_into_binding(f'"{input}"', parser, prefixes)

    return binding


def format_verification_error(value: str, position: Position) -> str:
    expected = "IRI" if position != Position.OBJECT else "IRI or literal"
    return f'Value "{value}" for {position} is not a valid {expected}. \
IRIs can be given in prefixed form, like wd:Q937, or in full form, \
like <http://www.wikidata.org/entity/Q937> or \
http://www.wikidata.org/entity/Q937. Be aware that not all IRIs can be \
used in prefixed form due to unsupported characters. If the value \
comes from a previous function call result, make sure to specify it exactly \
as given, with proper escaping and quoting.'


def list_triples(
    managers: list[KgManager],
    kg: str,
    subject: str | None,
    property: str | None,
    obj: str | None,
    page: int,
    unclipped: bool,
    k: int,
    known: set[str],
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    manager, _ = find_manager(managers, kg)

    triple = []
    bindings = []
    for pos, const in zip(Position, [subject, property, obj]):
        if const is None:
            triple.append(f"?{pos.value[0]}")
            continue

        ver_const = parse_iri_or_literal(
            const,
            manager.iri_literal_parser,
            manager.prefixes,
        )
        if ver_const is None or (pos != Position.OBJECT and ver_const.typ != "uri"):
            raise FunctionCallException(format_verification_error(const, pos))

        bindings.append(f"BIND({ver_const.sparql()} AS ?{pos.value[0]})")
        triple.append(ver_const.sparql())

    triple = " ".join(triple)
    bindings = "\n".join(bindings)
    sparql = f"""\
SELECT ?s ?p ?o WHERE {{
    {triple}
    {bindings}
}} LIMIT {MAX_RESULTS + 1}"""

    try:
        result = manager.execute_sparql(sparql, request_timeout, read_timeout)
    except Exception as e:
        raise FunctionCallException(f"Failed to list triples with error:\n{e}") from e

    assert isinstance(result, SelectResult)
    result.truncate(MAX_RESULTS)

    # functions to get scores for properties and entities
    prop_index = manager.try_get("properties")
    ent_index = manager.try_get("entities")

    def prop_rank(prop: Binding) -> int:
        if not prop_index:
            return 0

        norm = manager.get_normalizer("properties").normalize(prop.identifier())
        if norm is None:
            return len(prop_index.data)

        id = prop_index.data.id_from_identifier(norm[0])
        if id is None:
            return len(prop_index.data)

        # lower id means more popular property
        return id

    def ent_rank(ent: Binding) -> int:
        if not ent_index or not ent_index.data:
            return 0

        norm = manager.get_normalizer("entities").normalize(ent.identifier())
        if norm is None:
            return len(ent_index.data)

        id = ent_index.data.id_from_identifier(norm[0])
        if id is None:
            return len(ent_index.data)

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

    prop_norm = manager.get_normalizer("properties")
    ent_norm = manager.get_normalizer("entities")

    def normalize(bnd: Binding, normalizer: Normalizer) -> str:
        identifier = bnd.identifier()
        norm = normalizer.normalize(identifier)
        return norm[0] if norm is not None else identifier

    # now make sure that we show a diverse set of rows
    # triples with unseen properties or subjects / objects
    # should come first
    probs_seen = set()
    ents_seen = set()
    permutation = []

    for i, row in sorted_rows:
        # normalize
        s = normalize(row["s"], ent_norm)
        p = normalize(row["p"], prop_norm)
        o = normalize(row["o"], ent_norm)

        key = (p in probs_seen, s in ents_seen or o in ents_seen)
        permutation.append((key, i))

        probs_seen.add(p)
        ents_seen.add(s)
        ents_seen.add(o)

    # sort by number of seen columns
    # since sort is stable, we keep relative popularity order from before
    permutation = sorted(permutation, key=lambda item: item[0])
    result.data = [result.data[i] for _, i in permutation]

    # apply pagination
    start = (page - 1) * k
    end = page * k
    result.data = result.data[start:end]

    # update known
    update_known_from_rows(known, result.rows(), ent_norm)
    update_known_from_rows(known, result.rows(), prop_norm)

    return manager.format_sparql_result(
        result,
        show_top_rows=k,
        show_bottom_rows=0,
        show_left_columns=3,
        show_right_columns=0,
        # override column names
        column_names=["subject", "property", "object"],
        clip_literals=not unclipped,
    )


def search_with_constraints(
    managers: list[KgManager],
    kg: str,
    index: str,
    position: str,
    query: str,
    constraints: dict[str, str | None] | None,
    k: int,
    known: set[str],
    query_type: str = "text",
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

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
        for pos in Position:
            const = constraints.get(pos.value)
            if const is None:
                pos_values[pos] = f"?{pos.value[0]}"
                continue

            elif pos.value == position:
                pos_values[pos] = "?search"
                continue

            ver_const = parse_iri_or_literal(
                const,
                manager.iri_literal_parser,
                manager.prefixes,
            )
            if ver_const is None or (pos != Position.OBJECT and ver_const.typ != "uri"):
                raise FunctionCallException(format_verification_error(const, pos))

            pos_values[pos] = ver_const.sparql()

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
                MAX_RESULTS,
                request_timeout,
                read_timeout,
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
    normalizer = manager.get_normalizer(index)
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
    index: str,
    sparql: str | None,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    know_before_use: bool = False,
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
    **search_kwargs: Any,
) -> str:
    manager, others = find_manager(managers, kg)

    identifier_map = None
    info = ""
    if sparql is not None:
        # fix prefixes with managers
        sparql = manager.fix_prefixes(sparql)
        for other in others:
            sparql = other.fix_prefixes(sparql)

        if know_before_use:
            check_known(manager, sparql, known)

        try:
            identifier_map = manager.get_candidate_ids(
                index,
                sparql,
                MAX_RESULTS,
                request_timeout,
                read_timeout,
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
    normalizer = manager.get_normalizer(index)
    update_known_from_alts(known, alternatives, normalizer)

    return info + format_index_alternatives(alternatives, index, k)
