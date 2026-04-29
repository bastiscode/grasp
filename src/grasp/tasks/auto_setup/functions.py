from grammar_utils.parse import LR1Parser  # type: ignore

from grasp.sparql.utils import (
    find,
    parse_string,
    parse_to_string_with_whitespace,
)

STOP_FUNCTION = {
    "name": "stop",
    "description": "Stop the setup process.",
}


def set_description_function(description: str) -> dict:
    return {
        "name": "set_description",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def index_functions(index_name: str = "entities") -> list[dict]:
    fns: list[dict] = [
        {
            "name": "show_setup",
            "description": "Show the current index and info SPARQL queries for the knowledge graph.",
        },
        {
            "name": "set_query",
            "description": "Set or clear the index or info SPARQL query for the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["index", "info"],
                        "description": "Whether this is an index query or an info query",
                    },
                    "sparql": {
                        "type": ["string", "null"],
                        "description": "The SPARQL query, or null to clear / unset the current query.",
                    },
                },
                "required": ["type", "sparql"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        set_description_function(
            "Set a concise description of what this index contains and how it is built. "
            "Typically a single sentence is sufficient."
        ),
    ]

    # show_entity_setup is only useful while authoring the literals index,
    # where the agent may want to inspect the entity index to avoid
    # duplicating the literals it already covers as entity-associated labels.
    if index_name == "literals":
        fns.append(
            {
                "name": "show_entity_setup",
                "description": "Show the current entity index setup (index SPARQL, info "
                "SPARQL, and description) for this knowledge graph. Use this while "
                "authoring the literals index to see which literals the entity index "
                "already covers, and avoid duplicating that coverage in the literals "
                "index.",
            }
        )

    fns.append(STOP_FUNCTION)
    return fns


def info_functions() -> list[dict]:
    return [
        {
            "name": "show_setup",
            "description": "Show the current prefixes and description of the knowledge graph.",
        },
        {
            "name": "add_prefix",
            "description": "Add a new prefix for the knowledge graph, mapping a short "
            "prefix name to its full IRI namespace (e.g. 'wd' for "
            "'http://www.wikidata.org/entity/'). Only knowledge graph specific "
            "prefixes need to be added - common prefixes like rdf, rdfs, and "
            "xsd are available by default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short": {
                        "type": "string",
                        "description": "The short prefix name (e.g. 'wd')",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The full IRI namespace (e.g. 'http://www.wikidata.org/entity/')",
                    },
                },
                "required": ["short", "namespace"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_prefix",
            "description": "Delete an existing knowledge graph prefix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short": {
                        "type": "string",
                        "description": "The short prefix name to delete",
                    },
                },
                "required": ["short"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "update_prefix",
            "description": "Update the namespace of an existing knowledge graph prefix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short": {
                        "type": "string",
                        "description": "The short prefix name to update",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The new full IRI namespace",
                    },
                },
                "required": ["short", "namespace"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        set_description_function(
            "Set a concise description of what the knowledge graph contains. "
            "Typically a single sentence about the domain and scope of the "
            "knowledge graph is sufficient."
        ),
        STOP_FUNCTION,
    ]


def clean(s: str) -> str:
    return "".join(s.split())


def find_select_clause(parse: dict, enc: bytes) -> str:
    clause = find(parse, "SelectClause", skip={"SubSelect"})
    if clause is None:
        raise ValueError("No SELECT clause found in query")
    return parse_to_string_with_whitespace(clause, enc)


def find_solution_modifier(parse: dict, enc: bytes) -> str:
    sol_mod = find(parse, "SolutionModifier", skip={"SubSelect"})
    if sol_mod is None:
        raise ValueError("No solution modifier found in query")
    return parse_to_string_with_whitespace(sol_mod, enc)


def validate_select_clause(parse: dict, enc: bytes, target: str):
    select = find_select_clause(parse, enc)
    if clean(select) != clean(target):
        raise ValueError(f"SELECT clause must be '{target}', but got '{select}'")


def validate_solution_modifier(parse: dict, enc: bytes, target: str):
    sol_mod = find_solution_modifier(parse, enc)
    if clean(sol_mod) != clean(target):
        raise ValueError(f"Solution modifier must be '{target}', but got '{sol_mod}'")


INDEX_SPARQL_SELECT = "SELECT DISTINCT ?id ?value ?tags"
INDEX_SPARQL_SOL_MOD = "ORDER BY DESC(?score) ?id DESC(?tags)"


def validate_index_sparql(parser: LR1Parser, sparql: str):
    parse, _ = parse_string(sparql, parser)
    enc = sparql.encode()

    validate_select_clause(parse, enc, INDEX_SPARQL_SELECT)
    validate_solution_modifier(parse, enc, INDEX_SPARQL_SOL_MOD)


INFO_SPARQL_SELECT = "SELECT DISTINCT ?id ?value ?type"
INFO_SPARQL_SOL_MOD = "ORDER BY ?id ?type ?value"


def validate_info_sparql(parser: LR1Parser, sparql: str):
    parse, _ = parse_string(sparql, parser)
    enc = sparql.encode()

    validate_select_clause(parse, enc, INFO_SPARQL_SELECT)
    validate_solution_modifier(parse, enc, INFO_SPARQL_SOL_MOD)
