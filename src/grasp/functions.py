import json
import math
import random
import re
from copy import deepcopy
from itertools import chain
from typing import Any, Iterable

import validators
from search_index.similarity import SimilarityIndex
from universal_ml_utils.ops import partition_by

from grasp.sparql.constants import (
    Binding,
    ObjType,
    Position,
    SelectResult,
    SelectRow,
)
from grasp.sparql.data import (
    get_sparql_items,
    parse_binding,
    selections_from_items,
)
from grasp.sparql.manager import KgManager
from grasp.sparql.manager.utils import get_common_sparql_prefixes
from grasp.sparql.mapping import Mapping
from grasp.sparql.selection import Alternative
from grasp.sparql.sparql import (
    SPARQLException,
    find_all,
    parse_string,
)

# set up some global variables
MAX_RESULTS = 65536
# avoid negative cos sims for fp32 indices, does
# not restrict ubinary indices
MIN_SCORE = 0.5
# similar examples should be at least have this cos sim
MIN_EXAMPLE_SCORE = 0.5


def get_feedback_functions() -> list[dict]:
    return [
        {
            "name": "give_feedback",
            "description": """\
Provide feedback to the output of the question answering system in the \
context of the user's question.

The feedback status can be one of:
1. done: The output is correct and complete in its current form
2. refine: The output is sensible, but needs some refinement
3. retry: The output is incorrect and needs to be reworked

The feedback message should describe the reasoning behind the chosen status \
and provide suggestions for improving the output if applicable.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["done", "refine", "retry"],
                        "description": "The feedback type",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "The feedback message",
                    },
                },
                "required": ["status", "feedback"],
                "additionalProperties": False,
                "strict": True,
            },
        }
    ]


def get_functions(
    task: str,
    fn_set: str,
    managers: list[KgManager],
    example_indices: dict[str, SimilarityIndex],
    num_examples: int = 3,
    random_examples: bool = False,
) -> list[dict]:
    assert fn_set in [
        "base",
        "search",
        "search_extended",
        "search_autocomplete",
        "search_constrained",
    ], f"Unknown function set {fn_set}"
    kgs = [manager.kg for manager in managers]

    fns = []
    if task == "sparql-qa":
        fns.extend(
            [
                {
                    "name": "answer",
                    "description": """\
Provide your final SPARQL query and answer to the user question based on the \
query results. This function will stop the generation process.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph on which the final SPARQL query \
needs to be executed",
                            },
                            "sparql": {
                                "type": "string",
                                "description": "The final SPARQL query",
                            },
                            "answer": {
                                "type": "string",
                                "description": "The answer to the question based \
on the SPARQL query results",
                            },
                        },
                        "required": ["kg", "sparql", "answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "name": "cancel",
                    "description": """\
If you are unable to find a SPARQL query that answers the question well, \
you can call this function instead of the answer function. This function will \
stop the generation process.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": "A detailed explanation of why you \
could not find a satisfactory SPARQL query",
                            },
                            "best_attempt": {
                                "type": "object",
                                "description": "Your best attempt at a SPARQL query so far, \
can be omitted if there is none",
                                "properties": {
                                    "sparql": {
                                        "type": "string",
                                        "description": "The best SPARQL query so far",
                                    },
                                    "kg": {
                                        "type": "string",
                                        "enum": kgs,
                                        "description": "The knowledge graph on which \
the SPARQL query needs to be executed",
                                    },
                                },
                                "required": ["sparql", "kg"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["explanation"],
                        "additionalProperties": False,
                    },
                },
            ]
        )

    elif task == "general-qa":
        fns.extend(
            [
                {
                    "name": "answer",
                    "description": """\
Provide your final answer to the user question. This function will stop \
the generation process.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The answer to the question",
                            },
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {
                    "name": "cancel",
                    "description": """\
If you are unable to find an answer to the question, \
you can call this function instead of the answer function. \
This function will stop the generation process.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": "A detailed explanation of why you \
could not find a satisfactory answer",
                            },
                            "best_attempt": {
                                "type": "string",
                                "description": "Your best attempt at an answer so far, \
can be omitted if there is none",
                            },
                        },
                        "required": ["explanation"],
                        "additionalProperties": False,
                    },
                },
            ]
        )

    else:
        raise ValueError(f"Unknown task {task}")

    fns.append(
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
    )

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

    if not example_indices:
        return fns

    # at least one example index is provided

    example_kgs = list(example_indices)
    example_info = "\n".join(example_kgs)

    if random_examples:
        fn = {
            "name": "find_examples",
            "description": f"""\
Find examples of SPARQL-question-pairs over the specified knowledge graph. \
At most {num_examples} examples are returned. The examples may help you \
with generating your own SPARQL query.

For example, to find examples of SPARQL-question-pairs over Wikidata, do the following:
find_examples(kg="wikidata")

Currently, examples are available for the following knowledge graphs:
{example_info}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": example_kgs,
                        "description": "The knowledge graph to find examples for",
                    },
                },
                "required": ["kg"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    else:
        fn = {
            "name": "find_similar_examples",
            "description": f"""\
Find SPARQL-question-pairs over the specified knowledge graph that \
try to answer a similar question to the one provided. At most {num_examples} \
examples are returned. The examples may help you with generating \
your own SPARQL query.

For example, to find similar SPARQL-question-pairs to the question \
"What is the capital of France?" over Wikidata, do the following:
find_similar_examples(kg="wikidata", question="What is the capital of France?")

Currently, examples are available for the following knowledge graphs:
{example_info}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": example_kgs,
                        "description": "The knowledge graph to find examples for",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to find examples for",
                    },
                },
                "required": ["kg", "question"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    fns.append(fn)

    return fns


def call_function(
    managers: list[KgManager],
    example_indices: dict[str, SimilarityIndex],
    fn_name: str,
    fn_args: dict,
    fn_set: str,
    known: set[str],
    **kwargs: Any,
) -> str:
    # answer and cancel functions are special, they are not
    # real functions but a signal to stop the generation process
    if fn_name == "answer":
        return "Stopped generation process"
    elif fn_name == "cancel":
        return "Stopped generation process"

    # kg should be there for every function call
    fn_args = deepcopy(fn_args)
    kg = fn_args.pop("kg", None)
    assert kg is not None, "No knowledge graph specified"

    managers, others = partition_by(managers, lambda m: m.kg == kg)

    if len(managers) != 1:
        kgs = "\n".join(manager.kg for manager in managers + others)
        return f"Unknown knowledge graph {kg}, expected one of:\n{kgs}"

    manager = managers[0]

    if fn_name == "find_examples":
        return find_examples(
            manager,
            example_indices,
            kwargs["num_examples"],
            known,
        )

    elif fn_name == "find_similar_examples":
        return find_similar_examples(
            manager,
            example_indices,
            fn_args["question"],
            kwargs["num_examples"],
            known,
            min_score=MIN_EXAMPLE_SCORE,
        )

    elif fn_name == "execute":
        return execute_sparql(
            manager,
            others,
            fn_args["sparql"],
            kwargs["result_max_rows"],
            kwargs["result_max_columns"],
            known,
            know_before_use=kwargs["know_before_use"],
        )  # type: ignore

    elif fn_name == "list":
        return list_triples(
            manager,
            fn_args.get("subject"),
            fn_args.get("property"),
            fn_args.get("object"),
            kwargs["list_k"],
            known,
        )

    elif fn_name == "search_entity":
        return search_entity(
            manager,
            fn_args["query"],
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search_property":
        return search_property(
            manager,
            fn_args["query"],
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search_property_of_entity":
        return search_constrained(
            manager,
            "property",
            fn_args["query"],
            {"subject": fn_args["entity"]},
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search_object_of_property":
        return search_constrained(
            manager,
            "object",
            fn_args["query"],
            {"property": fn_args["property"]},
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search" and fn_set == "search_constrained":
        return search_constrained(
            manager,
            fn_args["position"],
            fn_args["query"],
            fn_args.get("constraints"),
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    elif fn_name == "search" and fn_set == "search_autocomplete":
        return search_autocomplete(
            manager,
            fn_args["sparql"],
            fn_args["query"],
            kwargs["search_top_k"],
            known,
            min_score=MIN_SCORE,
        )

    else:
        raise ValueError(f"Unknown function {fn_name}")


def build_examples(
    manager: KgManager,
    example_index: SimilarityIndex,
    example_ids: list[int],
    known: set[str],
) -> str:
    examples = []
    for id in example_ids:
        q = example_index.get_name(id)
        data = json.loads(example_index.get_val(id, 3))

        try:
            s = data["sparql"]
            s = manager.fix_prefixes(s, remove_known=True)
            s = manager.prettify(s)
            _, items = get_sparql_items(s, manager)
            selections = selections_from_items(items)
            if selections:
                s += "\n\n" + manager.format_selections(selections)
        except Exception:
            continue

        # build alternatives
        alternatives = {
            ObjType.ENTITY: [],
            ObjType.PROPERTY: [],
        }

        for selection in selections:
            if selection.obj_type in alternatives:
                alternatives[selection.obj_type].append(selection.alternative)

        update_known_from_alternatives(known, alternatives, manager)

        examples.append(f"Question:\n{q}\n\nSPARQL:\n{s}")

    if not examples:
        return "No examples found"

    return "\n\n".join(f"Example {i + 1}:\n{ex}" for i, ex in enumerate(examples))


def find_examples(
    manager: KgManager,
    example_indices: dict[str, SimilarityIndex],
    num_examples: int,
    known: set[str],
) -> str:
    if manager.kg not in example_indices:
        # should not happen, but handle anyway
        return f"No example index for knowledge graph {manager.kg}"

    example_index = example_indices[manager.kg]

    # ids are indices from [0, len(example_index))
    indices = list(range(len(example_index)))
    random.shuffle(indices)

    return build_examples(manager, example_index, indices[:num_examples], known)


def find_similar_examples(
    manager: KgManager,
    example_indices: dict[str, SimilarityIndex],
    question: str,
    num_examples: int,
    known: set[str],
    **search_kwargs: Any,
) -> str:
    if manager.kg not in example_indices:
        # should not happen, but handle anyway
        return f"No example index for knowledge graph {manager.kg}"

    example_index = example_indices[manager.kg]

    example_ids = [
        id
        for id, _ in example_index.find_matches(
            question,
            k=num_examples,
            **search_kwargs,
        )
    ]

    return build_examples(manager, example_index, example_ids, known)


def search_entity(
    manager: KgManager,
    query: str,
    k: int,
    known: set[str],
    **search_kwargs: Any,
) -> str:
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
    manager: KgManager,
    query: str,
    k: int,
    known: set[str],
    **search_kwargs: Any,
) -> str:
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
        binding = parse_binding(iri["value"], manager)
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
        raise SPARQLException(f"""\
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


def execute_sparql(
    manager: KgManager,
    others: list[KgManager],
    sparql: str,
    max_rows: int,
    max_columns: int,
    known: set[str],
    know_before_use: bool = False,
    return_sparql: bool = False,
) -> str | tuple[str, str]:
    # fix prefixes with managers
    sparql = manager.fix_prefixes(sparql)
    for other in others:
        sparql = other.fix_prefixes(sparql)

    if know_before_use:
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

    if isinstance(result, SelectResult):
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
    manager: KgManager,
    subject: str | None,
    property: str | None,
    obj: str | None,
    k: int,
    known: set[str],
) -> str:
    if subject is not None and property is not None and obj is not None:
        return "Only two of subject, property, or object should be provided."

    triple = []
    bindings = []
    for pos, const in [("subject", subject), ("property", property), ("object", obj)]:
        if const is None:
            triple.append(f"?{pos[0]}")
            continue

        ver_const = verify_iri_or_literal(const, pos, manager)
        if ver_const is None:
            expected = "IRI" if pos != "object" else "IRI or literal"
            return f'Constraint "{const}" for {pos} position \
is not a valid {expected}. IRIs can be given in prefixed form, like "wd:Q937", \
as URIs, like "http://www.wikidata.org/entity/Q937", \
or in full form, like "<http://www.wikidata.org/entity/Q937>".'

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
        return f"Failed to list triples with error:\n{e}"

    assert isinstance(result, SelectResult)

    # functions to get scores for properties and entities
    def prop_score(prop: Binding) -> int:
        norm = manager.property_mapping.normalize(prop.identifier())
        if norm is None or norm[0] not in manager.property_mapping:
            return 0

        id = manager.property_mapping[norm[0]]
        # score is at column index 1 in property index
        return int(manager.property_index.get_val(id, 1))

    def ent_score(ent: Binding) -> int:
        norm = manager.entity_mapping.normalize(ent.identifier())
        if norm is None or norm[0] not in manager.entity_mapping:
            return 0

        id = manager.entity_mapping[norm[0]]
        # score is at column index 1 in entity index
        return int(manager.entity_index.get_val(id, 1))

    # make sure that rows presented are diverse and that
    # we show the ones with popular properties or subjects / objects
    # first
    def sort_key(row: SelectRow) -> tuple[int, int]:
        # property score
        ps = prop_score(row["p"])

        # entity score
        es = max(ent_score(row["s"]), ent_score(row["o"]))

        # sort first by properties, then by subjects or objects
        return ps, es

    # rows are now sorted by popularity
    sorted_rows = sorted(
        enumerate(result.rows()),
        key=lambda item: sort_key(item[1]),
        reverse=True,
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
    manager: KgManager,
    position: str,
    query: str,
    constraints: dict[str, str | None] | None,
    k: int,
    known: set[str],
    max_results: int = MAX_RESULTS,
    **search_kwargs: Any,
) -> str:
    if constraints is None:
        constraints = {}

    target_constr = constraints.get(position)
    if target_constr is not None:
        return f'Cannot look for {position} and constrain it to \
"{target_constr}" at the same time.'

    if len(constraints) > 2:
        return "At most two of subject, property, and \
object should be constrained at once."

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
                return f'Constraint "{const}" for {pos} position \
is not a valid {expected}. IRIs can be given in prefixed form, like "wd:Q937", \
as URIs, like "http://www.wikidata.org/entity/Q937", \
or in full form, like "<http://www.wikidata.org/entity/Q937>".'

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
Falling back to an unconstrained search on the full precomputed search indices:
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


AUTOCOMPLETE_QUERY_REGEX = re.compile(r"\?search$")


def search_autocomplete(
    manager: KgManager,
    sparql: str,
    query: str,
    k: int,
    known: set[str],
    max_results: int = MAX_RESULTS,
    **search_kwargs: Any,
) -> str:
    try:
        sparql, position = manager.autocomplete_sparql(sparql, limit=max_results + 1)
    except Exception as e:
        return f"Invalid SPARQL query: {e}"

    info = ""
    try:
        search_items = manager.get_search_items(sparql, position, max_results)
    except Exception as e:
        info = f"""\
Falling back to an unconstrained search on the full precomputed search indices:
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
