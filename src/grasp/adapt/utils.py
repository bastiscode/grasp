import json
import os

from grasp.functions import execute_sparql, find_manager
from grasp.manager import KgManager
from grasp.sparql.item import get_sparql_items, selections_from_items


def format_arguments(args, depth: int = 0) -> str:
    if isinstance(args, list):
        return "[" + ", ".join(format_arguments(i, depth + 1) for i in args) + "]"
    elif isinstance(args, dict):
        return (
            "{" * (depth > 0)
            + ", ".join(
                f"{k}={format_arguments(v, depth + 1)}" for k, v in args.items()
            )
            + "}" * (depth > 0)
        )
    elif isinstance(args, str):
        return f'"{args}"'
    else:
        return str(args)


def format_output(output: dict) -> str:
    tool_call_results = {
        message["tool_call_id"]: message["content"]
        for message in output["messages"]
        if message["role"] == "tool"
    }
    fmt = []
    step = 1
    for message in output["messages"][2:]:
        if message["role"] == "tool":
            continue
        elif message["role"] == "user":
            fmt.append(f"Feedback:\n{message['content']}")
            continue

        assert message["role"] == "assistant"

        content = f"System step {step}:"
        if message.get("reasoning_content"):
            content += f"\n{message['reasoning_content'].strip()}"
        if message.get("content"):
            content += f"\n{message['content'].strip()}"

        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            if tool_call["type"] != "function":
                continue

            tool_call_fn = tool_call["function"]
            tool_calls.append(
                f'Call of "{tool_call_fn["name"]}" function '
                f"with {format_arguments(json.loads(tool_call_fn['arguments']))}:\n"
                f"{tool_call_results[tool_call['id']]}"
            )

        content += "\n" + "\n".join(tool_calls)

        fmt.append(content.strip())
        step += 1

    return "\n\n".join(fmt)


def link(src: str, dst: str) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.lexists(dst):
        os.remove(dst)

    rel = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel, dst)
