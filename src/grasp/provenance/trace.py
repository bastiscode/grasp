"""
Trace provenance: for every knowledge graph item in a final SPARQL query, the
first tool call that showed it, i.e. added it to the known items, and the
first tool call that used it in its arguments. Relies on the known items
recorded per tool call (ToolCall.new_known), so older traces have none.
"""

from typing import Any, Iterator

from grasp.manager import KgManager
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.model import Message
from grasp.provenance.witness import char_offset, parse_prefixes
from grasp.sparql.utils import find_all, parse_into_binding, parse_string

COMMON_PREFIXES = get_common_sparql_prefixes()

# tool calls that end the trace and do not provide information
STOPPING = {"answer", "cancel"}


def message_dict(message: Message | dict) -> dict:
    if isinstance(message, Message):
        return message.model_dump()
    return message


def iter_tool_calls(messages: list[Message | dict]) -> Iterator[dict]:
    for message in map(message_dict, messages):
        content = message.get("content")
        if not isinstance(content, dict):
            continue

        for tool_call in content.get("tool_calls") or []:
            if tool_call["name"] in STOPPING:
                continue
            yield tool_call


def identifiers(manager: KgManager, iri: str) -> set[str]:
    # the IRI and its normalized forms; search results show items in
    # normalized form, e.g. on Wikidata the property wdt:P19 as wd:P19
    ids = {iri}
    for index_name in ["entities", "properties"]:
        try:
            norm = manager.normalize(iri, index_name)
        except Exception:
            norm = None
        if norm is not None:
            ids.add(norm[0])
    return ids


def iri_spans(
    manager: KgManager,
    sparql: str,
    skip_common: bool = True,
) -> list[tuple[str, str, int, int]]:
    # (IRI, as written, start, end) for all IRIs of the query outside of its
    # prologue, by default except common vocabulary
    parse, _ = parse_string(sparql, manager.sparql_parser, skip_empty=True)
    prefixes = {**manager.prefixes, **parse_prefixes(parse)}
    data = sparql.encode()

    spans = []
    for node in find_all(parse, {"IRIREF", "PNAME_NS", "PNAME_LN"}, skip={"Prologue"}):
        binding = parse_into_binding(
            node["value"], manager.iri_literal_parser, prefixes
        )
        if binding is None or binding.typ != "uri":
            continue

        longest = manager.find_longest_prefix(binding.value)
        if skip_common and longest is not None and longest[0] in COMMON_PREFIXES:
            continue

        start, end = node["byte_span"]
        spans.append(
            (
                binding.value,
                node["value"],
                char_offset(data, start),
                char_offset(data, end),
            )
        )
    return spans


def flatten_args(args: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    # nested arguments with dotted keys, as the website shows them
    for key, value in args.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten_args(value, name)
        elif value is not None:
            yield name, value


def arg_usage(
    manager: KgManager,
    args: dict,
) -> tuple[dict[str, set[str]], list[tuple[str, str, int, int]]]:
    # IRIs per argument, and the IRI spans of a SPARQL argument; arguments
    # that are no IRIs or SPARQL queries, e.g. search queries, use none
    used: dict[str, set[str]] = {}
    sparql_spans = []
    for key, value in flatten_args(args):
        if key == "sparql" and isinstance(value, str):
            try:
                sparql_spans = iri_spans(manager, value)
            except Exception:
                pass
            used[key] = {iri for iri, *_ in sparql_spans}
            continue

        used[key] = set()
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, str):
                continue
            binding = parse_into_binding(
                v.strip(), manager.iri_literal_parser, manager.prefixes
            )
            if binding is not None and binding.typ == "uri":
                used[key].add(binding.value)
    return used, sparql_spans


def result_paths(value: Any, ids: set[str], path: str = "") -> Iterator[str]:
    # dotted paths to the parts of a structured result showing one of the
    # identifiers, e.g. items.0 for a search result or rows.2.city for a
    # table value; parts show an item with an identifier or iri field, or
    # are an IRI value of a table row
    if isinstance(value, dict):
        if (
            value.get("identifier") in ids
            or value.get("iri") in ids
            or (value.get("type") == "uri" and value.get("value") in ids)
        ):
            yield path
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return

    for key, child in items:
        yield from result_paths(child, ids, f"{path}.{key}" if path else str(key))


def shown_part(result: dict, path: str) -> dict[str, Any] | None:
    # the part of a structured result at a path, for showing it: a search
    # item, a table row with the variable of the value, or a shown value
    keys = path.split(".")
    parents = []
    node: Any = result
    try:
        for key in keys:
            parents.append(node)
            node = node[int(key)] if isinstance(node, list) else node[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return None

    if not isinstance(node, dict):
        return None
    elif "identifier" in node:
        return {"type": "item", "item": node}
    elif len(keys) >= 3 and keys[-3] == "rows":
        # the value of a table row, parents are the table, its rows and the row
        table = parents[-3]
        return {
            "type": "row",
            "columns": table.get("columns", []),
            "row": parents[-1],
            "variable": keys[-1],
        }
    elif "value" in node:
        return {"type": "text", "text": node["value"]}
    return None


def call_ref(
    manager: KgManager,
    iri: str,
    call: dict,
    step: int,
    usage: tuple[dict[str, set[str]], list[tuple[str, str, int, int]]],
) -> dict[str, Any]:
    ids = identifiers(manager, iri)
    used, sparql_spans = usage
    ref: dict[str, Any] = {
        "id": call.get("id"),
        "step": step,
        "tool": call["name"],
        "args": call.get("args"),
        # where the item is in the arguments: argument names, and spans in the
        # SPARQL argument
        "used_args": sorted(key for key, iris in used.items() if ids & iris),
        "sparql_spans": [[start, end] for i, _, start, end in sparql_spans if i in ids],
    }

    # where the item is in the structured result, for showing and
    # highlighting it
    paths = list(result_paths(call.get("result") or {}, ids))
    if paths:
        ref["result_paths"] = paths
        shown = shown_part(call["result"], paths[0])
        if shown is not None:
            ref["shown"] = shown
    return ref


def trace_provenance(
    manager: KgManager,
    sparql: str,
    messages: list[Message | dict],
) -> dict[str, Any] | None:
    """
    Per knowledge graph item of the query, its spans in the query, the first
    tool call that added it to the known items and the first tool call that
    used it in its arguments. None if the trace has no known items recorded.
    Never raises.
    """
    calls = list(iter_tool_calls(messages))
    if not any(call.get("new_known") is not None for call in calls):
        return None

    try:
        spans = iri_spans(manager, sparql)
    except Exception as e:
        return {"error": f"Failed to find the items of the query: {e}"}

    usages = [arg_usage(manager, call.get("args") or {}) for call in calls]

    items: dict[str, dict[str, Any]] = {}
    for iri, written, start, end in spans:
        if iri in items:
            items[iri]["spans"].append([start, end])
            continue

        item: dict[str, Any] = {
            "identifier": iri,
            "written": written,
            "label": manager.get_label(iri, "entities")
            or manager.get_label(iri, "properties"),
            "spans": [[start, end]],
        }

        ids = identifiers(manager, iri)
        shown_at = next(
            (i for i, call in enumerate(calls) if iri in (call.get("new_known") or [])),
            None,
        )
        used_at = next(
            (
                i
                for i, (used, _) in enumerate(usages)
                if any(ids & u for u in used.values())
            ),
            None,
        )

        if shown_at is not None:
            item["shown_by"] = call_ref(
                manager, iri, calls[shown_at], shown_at + 1, usages[shown_at]
            )
        if used_at is not None:
            item["first_used_by"] = call_ref(
                manager, iri, calls[used_at], used_at + 1, usages[used_at]
            )

        if shown_at is None:
            item["status"] = "unverified"
        elif used_at is None or shown_at < used_at:
            item["status"] = "retrieved"
        else:
            item["status"] = "verified"

        items[iri] = item

    return {"items": list(items.values())}
