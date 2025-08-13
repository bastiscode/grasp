import json
import re
from itertools import dropwhile
from typing import Any

from pydantic import BaseModel, ValidationError
from search_index.similarity import EmbeddingModel
from termcolor import colored

from grasp.sparql.manager import KgManager


class FunctionCallException(Exception):
    pass


def find_embedding_model(managers: list[KgManager]) -> EmbeddingModel | None:
    return next(
        dropwhile(
            lambda m: m is None,
            (manager.get_embedding_model() for manager in managers),
        ),
        None,
    )


def format_enumerate(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def format_model(model: BaseModel | None) -> str:
    if model is None:
        return "None"
    return model.model_dump_json(indent=2)


def format_message(message: dict) -> str:
    role = message["role"].upper()

    content = ""

    if message.get("reasoning_content"):
        content += f"Reasoning:\n{message['reasoning_content'].strip()}\n\n"

    content += message.get("content", "No content").strip()

    header = colored(role, "blue")
    return f"{header}\n{content}"


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


class SparqlQaAnswerModel(BaseModel):
    kg: str
    sparql: str
    answer: str


class GeneralQaAnswerModel(BaseModel):
    answer: str


class AnswerCallModel(BaseModel):
    name: str
    arguments: SparqlQaAnswerModel | GeneralQaAnswerModel


class SparqlQaBestAttemptModel(BaseModel):
    sparql: str
    kg: str


class CancelModel(BaseModel):
    explanation: str
    best_attempt: SparqlQaBestAttemptModel | str | None = None


class CancelCallModel(BaseModel):
    name: str
    arguments: CancelModel


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


def get_tool_call_from_message(message: str) -> str | None:
    # sometimes the model fails to call the answer function, but
    # provides the output in one of the following formats:
    # 1) within <tool_call>...</tool_call> tags:
    #    in this case check whether the content is a valid answer JSON like
    #    {"name": "answer", "arguments": "{...}"}
    # 2) as JSON in ```json...``` code block:
    #    do as in 1)

    # check for tool_call tags
    tool_call_match = re.search(
        r"<tool_call>(.*?)</tool_call>",
        message,
        re.IGNORECASE | re.DOTALL,
    )
    if tool_call_match is None:
        # fall back to JSON code block
        tool_call_match = re.search(
            r"```json\s*(.*?)\s*```",
            message,
            re.IGNORECASE | re.DOTALL,
        )

    if tool_call_match is None:
        return None
    else:
        return tool_call_match.group(1).strip()


def get_answer_from_message(task: str, message: str | None) -> dict | None:
    if message is None:
        return None

    tool_call = get_tool_call_from_message(message)
    if tool_call is None:
        return None

    try:
        return AnswerCallModel.model_validate_json(tool_call).arguments.model_dump()
    except ValidationError:
        pass

    try:
        if task == "sparql-qa":
            return SparqlQaAnswerModel.model_validate_json(tool_call).model_dump()
        elif task == "general-qa":
            return GeneralQaAnswerModel.model_validate_json(tool_call).model_dump()
        else:
            raise ValueError(f"Unknown task: {task}")
    finally:
        return None


def get_cancel_from_message(message: str | None) -> dict | None:
    if message is None:
        return None

    tool_call = get_tool_call_from_message(message)
    if tool_call is None:
        return None

    try:
        return CancelCallModel.model_validate_json(tool_call).arguments.model_dump()
    except ValidationError:
        pass

    try:
        return CancelModel.model_validate_json(tool_call).model_dump()
    finally:
        return None


def get_sparql_from_message(message: str | None) -> dict | None:
    if message is None:
        return None

    # Check for SPARQL code blocks
    sparql_match = re.search(
        r"```sparql\s*(.*?)\s*```",
        message,
        re.IGNORECASE | re.DOTALL,
    )
    if sparql_match:
        sparql_query = sparql_match.group(1).strip()
        return {"kg": None, "sparql": sparql_query, "answer": message}

    return None


def get_answer_or_cancel(
    task: str,
    messages: list[dict],
) -> tuple[dict | None, dict | None]:
    last_message: str | None = None
    last_answer: dict | None = None
    last_cancel: str | None = None
    last_execute: dict | None = None
    assert messages[0]["role"] == "system", "First message should be system"
    assert messages[1]["role"] == "user", "Second message should be user"
    for message in messages[2:]:
        if message["role"] == "user" and message != messages[-1]:
            # reset stuff after intermediate user feedback
            last_answer = None
            last_cancel = None
            last_message = None
            last_execute = None

        if message["role"] != "assistant":
            continue

        if "content" in message:
            last_message = message["content"]

        if "tool_calls" not in message:
            continue

        for tool_call in message["tool_calls"]:
            if tool_call["type"] != "function":
                continue

            tool_call = tool_call["function"]
            name = tool_call["name"]
            try:
                args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                continue

            if name == "answer":
                last_answer = args
                # reset last cancel
                last_cancel = None

            elif tool_call["name"] == "cancel":
                last_cancel = args
                # reset last answer
                last_answer = None

            elif tool_call["name"] == "execute":
                last_execute = args

    # try to parse answer from last message if neither are set
    if last_answer is None and last_cancel is None:
        last_answer = get_answer_from_message(task, last_message)

    # try to parse cancel from last message if both are still None
    if last_answer is None and last_cancel is None:
        last_cancel = get_cancel_from_message(last_message)

    # try to parse SPARQL from last message if both are still None
    if last_answer is None and last_cancel is None:
        last_answer = get_sparql_from_message(last_message)

    # try last execute function call for SPARQL QA
    if (
        task == "sparql-qa"
        and last_answer is None
        and last_cancel is None
        and last_execute is not None
    ):
        last_answer = {
            **last_execute,
            "answer": last_message or "No answer provided",
        }

    # and last message for general QA
    elif (
        task == "general-qa"
        and last_answer is None
        and last_cancel is None
        and last_message is not None
    ):
        last_answer = {
            "answer": last_message,
        }

    return last_answer, last_cancel
