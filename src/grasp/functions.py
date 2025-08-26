import math
from itertools import chain
from typing import Any, Callable, Iterable

import validators
from universal_ml_utils.ops import partition_by

from grasp.manager import KgManager
from grasp.manager.mapping import Mapping
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.sparql.item import parse_into_binding
from grasp.sparql.types import (
    Alternative,
    Binding,
    ObjType,
    Position,
    Selection,
    SelectResult,
    SelectRow,
)
from grasp.sparql.utils import find_all, parse_string
from grasp.utils import FunctionCallException

# set up some global variables
MAX_RESULTS = 65536
# avoid negative cos sims for fp32 indices, does
# not restrict ubinary indices
MIN_SCORE = 0.5


# a function gets the kg manager, function name, function arguments,
# known entities and properties, and an optional state object;
# it return a string observation and the updated additional state object
TaskHandler = Callable[[list[KgManager], str, dict, set[str], Any | None], str]


# tuple of function definitions as JSON schema, and a handler for executing the
# functions
TaskFunctions = tuple[list[dict], TaskHandler]


def kg_functions(managers: list[KgManager], fn_set: str) -> list[dict]:
    assert fn_set in [
        "base",
        "search",
        "search_extended",
        "search_autocomplete",
        "search_constrained",
    ], f"Unknown function set {fn_set}"
    kgs = [manager.kg for manager in managers]

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
and object constraints. At most two of subject, property, and object should be \
constrained at once. 

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
                        "type": "string",
                        "description": "Optional IRI for constraining the subject",
                    },
                    "property": {
                        "type": "string",
                        "description": "Optional IRI for constraining the property",
                    },
                    "object": {
                        "type": "string",
                        "description": "Optional IRI or literal for constraining the object",
                    },
                },
                "required": ["kg"],
                "additionalProperties": False,
            },
        },
    )

    if fn_set in ["search", "search_extended"]:
        fns.extend(
            [
                {
                    "name": "search_entity",
                    "description": """\
Search for entities in the knowledge graph with a search query. \
This function uses a prefix keyword index internally.

For example, to search for the entity Albert Einstein in Wikidata, \
do the following:
search_entity(kg="wikidata", query="albert einstein")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph to search",
                            },
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                        },
                        "required": ["kg", "query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "name": "search_property",
                    "description": """\
Search for properties in the knowledge graph with a search query. \
This function uses an embedding-based similarity index internally.

For example, to search for properties related to birth in Wikidata, do the following:
search_property(kg="wikidata", query="birth")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph to search",
                            },
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                        },
                        "required": ["kg", "query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ]
        )

    if fn_set == "search_extended":
        fns.extend(
            [
                {
                    "name": "search_property_of_entity",
                    "description": """\
Search for properties of a given entity in the knowledge graph. \
This function uses an embedding-based similarity index internally.

For example, to search for properties related to birth for Albert Einstein \
in Wikidata, do the following:
search_property_of_entity(kg="wikidata", entity="wd:Q937", query="birth")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
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
                        },
                        "required": ["kg", "entity", "query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "name": "search_object_of_property",
                    "description": """\
Search for objects (entities or literals) for a given property in the knowledge graph. \
This function uses a prefix keyword index internally.

For example, to search for football jobs in Wikidata, do the following:
search_object_of_property(kg="wikidata", property="wdt:P106", query="football")""",
                    "parameters": {
                        "type": "object",
                        "properties": {
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
                        },
                        "required": ["kg", "property", "query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ]
        )

    if fn_set == "search_autocomplete":
        fns.append(
            {
                "name": "search",
                "description": """\
Search for knowledge graph items in a context-sensitive way by specifying a constraining \
SPARQL query together with a search query. The SPARQL query must be a SELECT query \
with a variable ?search occurring at least once in the WHERE clause. The search is \
then restricted to knowledge graph items that fit at the ?search positions in the SPARQL \
query.

For the search itself, we use a prefix keyword index for subjects, objects, and \
literals, and an embedding-based similarity index for properties.

For example, to search for Albert Einstein at the subject position in \
Wikidata, do the following:
search(kg="wikidata", sparql="SELECT * WHERE { ?search ?p ?o }", query="albert einstein")

Or to search for properties of Albert Einstein related to his birth in \
Wikidata, do the following:
search(kg="wikidata", sparql="SELECT * WHERE { wd:Q937 ?search ?o }", query="birth")""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to search",
                        },
                        "sparql": {
                            "type": "string",
                            "description": "The SPARQL query with ?search variable",
                        },
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                    },
                    "required": ["kg", "sparql", "query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )

    if fn_set == "search_constrained":
        fns.append(
            {
                "name": "search",
                "description": """\
Search for knowledge graph items at a particular position (subject, property, or object) \
with optional constraints.

If constraints are provided, they are used to limit the search space accordingly. \
For the search itself, we use a prefix keyword index for subjects, objects, \
and literals, and an embedding-based similarity index for properties.

For example, to search for the subject Albert Einstein in Wikidata, do the following:
search(kg="wikidata", position="subject", query="albert einstein")

Or to search for properties of Albert Einstein related to his birth in Wikidata, \
do the following:
search(kg="wikidata", position="property", query="birth", \
constraints={"subject": "wd:Q937"})""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to search",
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
                            "type": "object",
                            "description": "Constraints for the search, \
can be omitted if there are none",
                            "properties": {
                                "subject": {
                                    "type": "string",
                                    "description": "Optional IRI for constraining the subject",
                                },
                                "property": {
                                    "type": "string",
                                    "description": "Optional IRI for constraining the property",
                                },
                                "object": {
                                    "type": "string",
                                    "description": "Optional IRI or literal for constraining the object",
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                    "required": ["kg", "position", "query"],
                    "additionalProperties": False,
                },
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
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    fn_set: str,
    known: set[str],
    task_handler: TaskHandler | None = None,
    **kwargs: Any,
) -> str:
    if fn_name == "execute":
        return execute_sparql(
            managers,
            fn_args["kg"],
            fn_args["sparql"],
            kwargs["result_max_rows"],
            kwargs["result_max_columns"],
            known,
            know_before_use=kwargs["know_before_use"],
        )  # type: ignore

    elif fn_name == "list":
        return list_triples(
            managers,
            fn_args["kg"],
            fn_args.get("subject"),
            fn_args.get("property"),
            fn_args.get("object"),
            kwargs["list_k"],
            known,
        )

    elif fn_name == "search_entity":
        return search_entity(
            managers,
            fn_args["kg"],
            fn_args["query"],
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search_property":
        return search_property(
            managers,
            fn_args["kg"],
            fn_args["query"],
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search_property_of_entity":
        return search_constrained(
            managers,
            fn_args["kg"],
            "property",
            fn_args["query"],
            {"subject": fn_args["entity"]},
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search_object_of_property":
        return search_constrained(
            managers,
            fn_args["kg"],
            "object",
            fn_args["query"],
            {"property": fn_args["property"]},
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search" and fn_set == "search_constrained":
        return search_constrained(
            managers,
            fn_args["kg"],
            fn_args["position"],
            fn_args["query"],
            fn_args.get("constraints"),
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search" and fn_set == "search_autocomplete":
        return search_autocomplete(
            managers,
            fn_args["kg"],
            fn_args["sparql"],
            fn_args["query"],
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif task_handler is not None:
        return task_handler(
            managers,
            fn_name,
            fn_args,
            known,
            kwargs.get("task_state"),
        )

    else:
        raise ValueError(f"Unknown function {fn_name}")


def search_entity(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    alts = manager.get_entity_alternatives(
        query=query,
        k=k,
        **search_kwargs,
    )

    # update known items
    update_known_from_alternatives(
        known,
        {ObjType.ENTITY: alts},
        manager,
    )

    return format_alternatives({ObjType.ENTITY: alts}, k)


def search_property(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    alts = manager.get_property_alternatives(
        query=query,
        k=k,
        **search_kwargs,
    )

    # update known items
    update_known_from_alternatives(known, {ObjType.PROPERTY: alts}, manager)

    return format_alternatives({ObjType.PROPERTY: alts}, k)


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
without being known from previous searches or query executions. \
This does not mean they are invalid, but you should verify \
that they indeed exist in the knowledge graphs before executing the SPARQL \
query again:
{not_seen}""")


def update_known_from_iris(
    known: set[str],
    iris: Iterable[str],
    mapping: Mapping | None = None,
):
    for iri in iris:
        known.add(iri)
        if mapping is None:
            continue

        norm = mapping.normalize(iri)
        if norm is None:
            continue

        # also add normalized identifier
        known.add(norm[0])


def update_known_from_alts(
    known: set[str],
    alts: Iterable[Alternative],
    mapping: Mapping | None = None,
):
    for alt in alts:
        known.add(alt.identifier)
        if mapping is None or not alt.variants:
            continue

        for var in alt.variants:
            denorm = mapping.denormalize(alt.identifier, var)
            if denorm is None:
                continue
            known.add(denorm)


def update_known_from_rows(
    known: set[str],
    rows: Iterable[SelectRow],
    mapping: Mapping | None = None,
):
    update_known_from_iris(
        known,
        (
            binding.identifier()
            for row in rows
            for binding in row.values()
            if binding.typ == "uri"
        ),
        mapping,
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
        manager.entity_mapping,
    )

    # properties
    update_known_from_alts(
        known,
        alternatives.get(ObjType.PROPERTY, []),
        manager.property_mapping,
    )

    # other
    update_known_from_alts(
        known,
        alternatives.get(ObjType.OTHER, []),
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
        manager.entity_mapping,
    )

    # properties
    update_known_from_alts(
        known,
        (sel.alternative for sel in selections if sel.obj_type == ObjType.PROPERTY),
        manager.property_mapping,
    )


def execute_sparql(
    managers: list[KgManager],
    kg: str,
    sparql: str,
    max_rows: int,
    max_columns: int,
    known: set[str] | None = None,
    know_before_use: bool = False,
    return_sparql: bool = False,
) -> str | tuple[str, str]:
    manager, others = find_manager(managers, kg)

    # fix prefixes with managers
    sparql = manager.fix_prefixes(sparql)
    for other in others:
        sparql = other.fix_prefixes(sparql)

    if know_before_use and known is not None:
        check_known(manager, sparql, known)

    try:
        result = manager.execute_sparql(sparql)
    except Exception as e:
        error = f"SPARQL execution failed:\n{e}"
        if return_sparql:
            return error, sparql
        return error

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
        update_known_from_rows(known, rows, manager.entity_mapping)

        # property mapping
        update_known_from_rows(known, rows, manager.property_mapping)

    result = manager.format_sparql_result(
        result,
        half_rows,
        half_rows,
        half_columns,
        half_columns,
    )
    if return_sparql:
        return result, sparql

    return result


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
    input = f"<{input}>"
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

    if subject is not None and property is not None and obj is not None:
        raise FunctionCallException(
            "Only two of subject, property, or object should be provided."
        )

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
}} LIMIT {MAX_RESULTS}"""

    try:
        result = manager.execute_sparql(sparql)
    except Exception as e:
        raise FunctionCallException(f"Failed to list triples with error:\n{e}") from e

    assert isinstance(result, SelectResult)

    # functions to get scores for properties and entities
    def prop_rank(prop: Binding) -> int:
        norm = manager.property_mapping.normalize(prop.identifier())
        if norm is None or norm[0] not in manager.property_mapping:
            return len(manager.property_mapping)

        id = manager.property_mapping[norm[0]]
        # lower id means more popular property
        return id

    def ent_rank(ent: Binding) -> int:
        norm = manager.entity_mapping.normalize(ent.identifier())
        if norm is None or norm[0] not in manager.entity_mapping:
            return len(manager.entity_mapping)

        id = manager.entity_mapping[norm[0]]
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
        norm = manager.property_mapping.normalize(identifier)
        return norm[0] if norm is not None else identifier

    def normalize_ent(ent: Binding) -> str:
        identifier = ent.identifier()
        norm = manager.entity_mapping.normalize(identifier)
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
    update_known_from_rows(known, result.rows(end=k), manager.entity_mapping)
    update_known_from_rows(known, result.rows(end=k), manager.property_mapping)

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


def search_constrained(
    managers: list[KgManager],
    kg: str,
    position: str,
    query: str,
    constraints: dict[str, str | None] | None,
    k: int,
    known: set[str],
    max_results: int = MAX_RESULTS,
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

    if len(constraints) > 2:
        raise FunctionCallException(
            "At most two of subject, property, and \
object should be constrained at once."
        )

    unconstrained = all(c is None for c in constraints.values())

    search_items = manager.get_default_search_items(Position(position))
    info = ""
    if not unconstrained:
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
    {pos_values["subject"]} {pos_values["property"]} {pos_values["object"]} 
}}
LIMIT {MAX_RESULTS + 1}"""

        try:
            search_items = manager.get_search_items(
                sparql,
                Position(position),
                max_results,
            )
        except Exception as e:
            info = f"""\
Falling back to an unconstrained search on the full \
search indices due to an error:
{e}

"""

    alternatives = manager.get_selection_alternatives(
        query,
        search_items,
        k,
        **search_kwargs,
    )

    # update known items
    update_known_from_alternatives(known, alternatives, manager)

    return info + format_alternatives(alternatives, k)


def format_alternatives(alternatives: dict[ObjType, list[Alternative]], k: int) -> str:
    fm = []

    for obj_type, alts in alternatives.items():
        if len(alts) == 0:
            fm.append(f"No {obj_type.value} items found")
            continue

        top_k_string = "\n".join(
            f"{i + 1}. {alt.get_selection_string()}" for i, alt in enumerate(alts)
        )
        fm.append(f"Top {k} {obj_type.value} alternatives:\n{top_k_string}")

    return "\n\n".join(fm)


def search_autocomplete(
    managers: list[KgManager],
    kg: str,
    sparql: str,
    query: str,
    k: int,
    known: set[str],
    max_results: int = MAX_RESULTS,
    **search_kwargs: Any,
) -> str:
    manager, _ = find_manager(managers, kg)

    try:
        sparql, position = manager.autocomplete_sparql(sparql, limit=max_results + 1)
    except Exception as e:
        raise FunctionCallException(f"Invalid SPARQL query: {e}") from e

    info = ""
    try:
        search_items = manager.get_search_items(sparql, position, max_results)
    except Exception as e:
        info = f"""\
Falling back to an unconstrained search on the full \
search indices due to an error:
{e}

"""
        search_items = manager.get_default_search_items(position)

    alternatives = manager.get_selection_alternatives(
        query,
        search_items,
        k=k,
        **search_kwargs,
    )

    # update known items
    update_known_from_alternatives(known, alternatives, manager)

    return info + format_alternatives(alternatives, k)
