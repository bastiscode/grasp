from grasp.functions import find_manager, parse_iri_or_literal
from grasp.manager import KgManager
from grasp.tasks.exploration.functions import (
    note_function_definitions as base_note_function_definitions,
)
from grasp.utils import FunctionCallException, format_enumerate


def note_function_definitions(managers: list[KgManager]) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    functions = base_note_function_definitions(managers)
    # insert the structural-only definitions before the trailing "stop" entry
    stop = functions.pop()
    functions.extend(
        [
            {
                "name": "mark_explored",
                "description": "Mark an IRI as the seed node for this exploration round. "
                "Exactly one seed node must be marked per round.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph of the seed node to mark as explored",
                        },
                        "iri": {
                            "type": "string",
                            "description": "The IRI of the seed node to mark as explored",
                        },
                    },
                    "required": ["kg", "iri"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "show_explored",
                "description": "Show previously explored seed nodes, most recent first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph for which to show explored seed nodes",
                        },
                        "page": {
                            "type": "integer",
                            "description": "Page number (1-indexed) for paginating results (default should be 1)",
                        },
                    },
                    "required": ["kg", "page"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]
    )
    functions.append(stop)
    return functions


def mark_explored(
    managers: list[KgManager],
    kg: str,
    explored: dict[str, list[str]],
    iri: str,
    explored_this_round: bool,
) -> str:
    if explored_this_round:
        raise FunctionCallException(
            "A seed node was already marked as explored this round"
        )

    manager, _ = find_manager(managers, kg)
    ver_iri = parse_iri_or_literal(iri, manager.iri_literal_parser, manager.prefixes)
    if ver_iri is None or ver_iri.typ != "uri":
        raise FunctionCallException(f'"{iri}" is not a valid IRI')

    kg_explored = explored.setdefault(kg, [])
    if ver_iri in kg_explored:
        raise FunctionCallException(
            "This node was already explored in a previous round"
        )

    kg_explored.append(ver_iri.identifier())
    return f'Marked "{iri}" as explored'


def show_explored(
    managers: list[KgManager],
    kg: str,
    explored: dict[str, list[str]],
    page: int,
    k: int,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    kg_explored = explored.get(kg, [])
    if not kg_explored:
        return "None"

    total_pages = (len(kg_explored) + k - 1) // k
    if page > total_pages:
        raise FunctionCallException(f"Page number exceeds maximum page {total_pages}")

    # most recent first
    kg_explored = list(reversed(kg_explored))

    start = (page - 1) * k
    end = page * k
    page_items = kg_explored[start:end]

    header = f"Most recently explored nodes (page {page} of {total_pages}):\n"

    manager, _ = find_manager(managers, kg)
    infos = manager.get_info_for_identifiers_from_index(page_items, "entities")
    items = []
    for iri in page_items:
        alt = manager.build_alternative_with_info(iri, infos.get(iri))
        items.append(alt.get_selection_string())

    return header + format_enumerate(items, start=start)
