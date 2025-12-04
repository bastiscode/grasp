import json
import os
from typing import Iterable

from pydantic import BaseModel
from termcolor import colored

from grasp.model import Message, Response, ToolCall


def get_index_dir(kg: str | None = None) -> str:
    index_dir = os.getenv("GRASP_INDEX_DIR", None)
    if index_dir is None:
        home_dir = os.path.expanduser("~")
        index_dir = os.path.join(home_dir, ".grasp", "index")

    if kg is not None:
        index_dir = os.path.join(index_dir, kg)

    return index_dir


def get_available_knowledge_graphs() -> list[str]:
    index_dir = get_index_dir()
    if not os.path.exists(index_dir):
        return []

    return [
        name
        for name in os.listdir(index_dir)
        if os.path.isdir(os.path.join(index_dir, name))
    ]


class FunctionCallException(Exception):
    pass


def format_prefixes(prefixes: dict[str, str]) -> str:
    if not prefixes:
        return "No prefixes available"

    return format_list(f"{short}: {long}" for short, long in sorted(prefixes.items()))


def format_notes(notes: list[str], indent: int = 0, enumerated: bool = False) -> str:
    if not notes:
        return " " * indent + "No notes available"
    elif enumerated:
        return format_enumerate(notes, indent)
    else:
        return format_list(notes, indent)


def format_list(items: Iterable[str], indent: int = 0) -> str:
    indent_str = " " * indent
    return "\n".join(f"{indent_str}- {item}" for item in items)


def format_enumerate(items: Iterable[str], indent: int = 0) -> str:
    indent_str = " " * indent
    return "\n".join(f"{indent_str}{i + 1}. {item}" for i, item in enumerate(items))


def format_model(model: BaseModel | None) -> str:
    if model is None:
        return "None"
    return model.model_dump_json(indent=2)


def format_error(reason: str, content: str) -> str:
    header = colored(f"ERROR (reason={reason})", "red", attrs=["bold", "underline"])
    return f"{header}\n{content}"


def format_message(message: Message) -> str:
    if isinstance(message.content, str):
        header = colored(f"{message.role.upper()}", "magenta")
        content = (
            json.dumps(message.content, indent=2)
            if isinstance(message.content, dict)
            else message.content
        )
        return f"{header}\n{content}"
    else:
        return format_response(message.content)


def format_response(response: Response) -> str:
    header = colored(f"MODEL (usage={response.usage})", "blue")

    content = ""
    reasoning = response.reasoning
    if reasoning is not None:
        if reasoning.content is not None:
            content += f"Reasoning:\n{reasoning.content}\n\n"
        if reasoning.summary is not None:
            content += f"Reasoning summary:\n{reasoning.summary}\n\n"
        if reasoning.encrypted_content is not None:
            enc = reasoning.encrypted_content
            if len(enc) > 64:
                enc = enc[:64] + "..."
            content += f"Encrypted reasoning content:\n{enc}\n\n"

    if response.message is not None:
        if response.has_reasoning_content:
            content += "Content:\n"

        content += f"{response.message}\n\n"

    for tool_call in response.tool_calls:
        content += format_tool_call(tool_call) + "\n\n"

    return f"{header}\n{content.strip()}"


def format_tool_call(tool_call: ToolCall) -> str:
    name = colored(tool_call.name, "green")
    fn_args_str = colored(json.dumps(tool_call.args, indent=2), "yellow")
    content = f"{name}({fn_args_str})"
    if tool_call.result is not None:
        content += f":\n{tool_call.result}"
    return content


def is_server_error(message: str | None) -> bool:
    if message is None:
        return False

    phrases = [
        "503 Server Error",  # qlever not available
        "502 Server Error",  # proxy error
        "(read timeout=6)",  # qlever not reachable
        "(connect timeout=6)",  # qlever not reachable
        "403 Client Error",  # wrong URL / API key
    ]
    return any(phrase.lower() in message.lower() for phrase in phrases)


def is_invalid_evaluation(evaluation: dict, empty_target_valid: bool = False) -> bool:
    if evaluation["target"]["err"] is not None:
        return True

    elif not empty_target_valid and evaluation["target"]["size"] == 0:
        return True

    elif "prediction" not in evaluation:
        return False

    # no target error, but we have a prediction
    # check whether prediction failed due to server error
    return is_server_error(evaluation["prediction"]["err"])


def is_tool_fail(message: dict) -> bool:
    if message["role"] != "tool":
        return False

    content = message["content"]
    return is_server_error(content)


def is_error(message: dict) -> bool:
    # old error format
    return message["role"] == "error"


def is_invalid_output(
    output: dict | None,
    none_output_invalid: bool = False,
) -> bool:
    if output is None:
        return True

    has_error = output.get("error") is not None
    if has_error:
        return True

    if none_output_invalid and output.get("output", None) is None:
        return True

    for message in output.get("messages", []):
        try:
            # new format
            msg = Message(**message)
            if not isinstance(msg.content, Response):
                continue

            if any(
                is_server_error(tool_call.result)
                for tool_call in msg.content.tool_calls
            ):
                return True

            continue
        except Exception:
            pass

        # old format
        if is_tool_fail(message) or is_error(message):
            return True

    return False


def parse_parameters(headers: list[str]) -> dict[str, str]:
    # each parameter is formatted as key:value
    header_dict = {}
    for header in headers:
        key, value = header.split(":", 1)
        header_dict[key.strip()] = value.strip()
    return header_dict


def clip(s: str, max_len: int = 128, respect_word_boundaries: bool = True) -> str:
    if len(s) <= max_len:
        return s

    elif not respect_word_boundaries:
        if max_len <= 3:
            return s[:max_len]

        half = (max_len - 3) // 2
        return s[:half] + "..." + s[-half:]

    if max_len <= 5:
        return s[:max_len]

    half = (max_len - 5) // 2  # account for spaces around "..."
    first = half
    while first > 0 and not s[first].isspace():
        first -= 1

    last = len(s) - half
    while last < len(s) and last > 0 and not s[last - 1].isspace():
        last += 1

    if first <= 0 or last >= len(s):
        # at least 1 word on either side, fall back
        # to character clipping otherwise
        return clip(s, max_len, respect_word_boundaries=False)

    return s[:first] + " ... " + s[last:]


def ordered_unique(lst: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in lst:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique
