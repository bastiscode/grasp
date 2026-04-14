from grasp.manager import KgManager
from grasp.sparql.utils import find, find_all, parse_string, parse_to_string

STOP_FUNCTION = {
    "name": "stop",
    "description": "Stop the setup process.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
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


def index_functions() -> list[dict]:
    return [
        {
            "name": "show_setup",
            "description": "Show the current index and info SPARQL queries for the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "set_query",
            "description": "Set the index or info SPARQL query for the knowledge graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["index", "info"],
                        "description": "Whether this is an index query or an info query",
                    },
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query",
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
        STOP_FUNCTION,
    ]


def info_functions() -> list[dict]:
    return [
        {
            "name": "show_setup",
            "description": "Show the current prefixes and description of the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
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


def find_select_vars(parse: dict) -> set[str]:
    clause = find(parse, "SelectClause")
    if clause is None:
        raise ValueError("No SELECT clause found in query")

    used = set()
    for var in find_all(clause, "Var"):
        used.add(var["children"][0]["value"].lstrip("?").lstrip("$"))

    return used


def find_order_by(parse: dict) -> str:
    order_by = find(parse, "OrderClause")
    if order_by is None:
        raise ValueError("No ORDER BY clause found in query")
    return parse_to_string(order_by)


def validate_sparql_vars(parse: dict, required: set[str]):
    used = find_select_vars(parse)

    missing = required - used
    if missing:
        missing_str = ", ".join(f"?{v}" for v in sorted(missing))
        raise ValueError(f"Missing required variables {missing_str} in SELECT clause.")


def validate_order_by(parse: dict, target: str):
    order_by = find_order_by(parse)
    if order_by != target:
        raise ValueError(f"ORDER BY clause must be '{target}' instead of '{order_by}'")


INDEX_SPARQL_VARS = {"id", "value", "tags"}
INDEX_SPARQL_ORDER_BY = "ORDER BY DESC ( ?score ) ?id DESC ( ?tags )"


def validate_index_sparql(manager: KgManager, sparql: str):
    parse, _ = parse_string(sparql, manager.sparql_parser)

    validate_sparql_vars(parse, INDEX_SPARQL_VARS)
    validate_order_by(parse, INDEX_SPARQL_ORDER_BY)


INFO_SPARQL_VARS = {"id", "value", "type"}
INFO_SPARQL_ORDER_BY = "ORDER BY ?id ?type ?value"


def validate_info_sparql(manager: KgManager, sparql: str):
    parse, _ = parse_string(sparql, manager.sparql_parser)

    validate_sparql_vars(parse, INFO_SPARQL_VARS)
    validate_order_by(parse, INFO_SPARQL_ORDER_BY)
