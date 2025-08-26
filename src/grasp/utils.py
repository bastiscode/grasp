import json
import os
from typing import Any

from pydantic import BaseModel
from termcolor import colored


def get_index_dir(kg: str | None = None) -> str:
    index_dir = os.getenv("GRASP_INDEX_DIR", None)
    if index_dir is None:
        home_dir = os.path.expanduser("~")
        index_dir = os.path.join(home_dir, ".grasp", "index")

    if kg is not None:
        index_dir = os.path.join(index_dir, kg)

    return index_dir


class FunctionCallException(Exception):
    pass


def format_prefixes(prefixes: dict[str, str]) -> str:
    if not prefixes:
        return "No prefixes defined"

    return "\n".join(f"{short}: {long}" for short, long in prefixes.items())


def format_notes(notes: list[str]) -> str:
    if not notes:
        return "No notes available"
    else:
        return format_list(notes)


def format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def format_model(model: BaseModel | None) -> str:
    if model is None:
        return "None"
    return model.model_dump_json(indent=2)


def format_message(message: dict) -> str:
    role = message["role"].upper()

    content = ""

    if message.get("reasoning_content"):
        content += f"Reasoning:\n{message['reasoning_content'].strip()}\n\n"

    content += message.get("content", "").strip()
    content += "\n\n"

    for tool_call in message.get("tool_calls", []):
        call = tool_call["function"]

        fn_name = call["name"]
        fn_args = json.loads(call["arguments"])
        content += format_function_call(fn_name, fn_args) + "\n"

    header = colored(role, "blue")
    return f"{header}\n{content.strip()}"


def format_function_call(fn_name: str, fn_args: dict) -> str:
    fn_name = colored(fn_name, "green")
    fn_args_str = colored(json.dumps(fn_args, indent=2), "yellow")
    return f"{fn_name}({fn_args_str})"


class Sample(BaseModel):
    id: str | None = None
    question: str
    sparql: str
    paraphrases: list[str] = []
    info: dict[str, Any] = {}


def is_server_error(message: str | None) -> bool:
    if message is None:
        return False

    phrases = [
        "503 Server Error",  # qlever not available
        "502 Server Error",  # proxy error
        "Read timed out. (read timeout=6)",  # qlever not reachable
        "403 Client Error: Forbidden for url",  # wrong URL / API key
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


def is_invalid_model_output(model_output: dict | None) -> bool:
    if model_output is None:
        return True

    has_error = model_output.get("error") is not None

    return has_error or any(
        is_tool_fail(message) or is_error(message)
        for message in model_output.get("messages", [])
    )


def parse_parameters(headers: list[str]) -> dict[str, str]:
    # each header is formatted as key:value
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
