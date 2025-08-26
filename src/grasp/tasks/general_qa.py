import json
from typing import Any

from grasp.functions import TaskFunctions
from grasp.manager import KgManager
from grasp.tasks.utils import (
    get_answer_from_message,
    get_cancel_from_message,
    get_sparql_from_message,
)
from grasp.utils import FunctionCallException


def system_information() -> str:
    return """\
You are a question answering assistant. \
Your job is to answer a given user question using the knowledge graphs \
and functions available to you.

You should follow a step-by-step approach to answer the question:
1. Determine the information needed from the knowledge graphs to \
answer the user question and think about how it might be represented with \
entities and properties.
2. Search for the entities and properties in the knowledge graphs. Where \
applicable, constrain the searches with already identified entities and properties.
3. Gradually build up the answer by querying the knowledge graphs using the \
identified entities and properties. You may need to refine or rethink your \
current plan based on the query results and go back to step 2 if needed, \
possibly multiple times.
4. Use the answer or cancel function to finalize your answer and stop the \
generation process."""


def rules() -> list[str]:
    return [
        "Your answers preferably should be based on the information available in the \
knowledge graphs. If you do not need them to answer the question, e.g. if \
you know the answer by heart, still try to verify it with the knowledge graphs.",
        "Do not use 'SERVICE wikibase:label { bd:serviceParam wikibase:language ...' \
in SPARQL queries. It is not SPARQL standard and unsupported by the used QLever \
SPARQL endpoints. Use rdfs:label or similar properties to get labels instead.",
    ]


def functions() -> TaskFunctions:
    fns = [
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
                },
                "required": ["explanation"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    return fns, call_function


def call_function(
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: Any = None,
    **kwargs: Any,
) -> str:
    if fn_name == "answer":
        return "Stopping"

    elif fn_name == "cancel":
        return "Stopping"

    raise FunctionCallException(f"Unknown function: {fn_name}")


def get_answer_or_cancel(messages: list[dict]) -> tuple[dict | None, dict | None]:
    last_message: str | None = None
    last_answer: dict | None = None
    last_cancel: str | None = None
    assert messages[0]["role"] == "system", "First message should be system"
    assert messages[1]["role"] == "user", "Second message should be user"
    for message in messages[2:]:
        if message["role"] == "user" and message != messages[-1]:
            # reset stuff after intermediate user feedback
            last_answer = None
            last_cancel = None
            last_message = None

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

    # try to parse answer from last message if neither are set
    if last_answer is None and last_cancel is None:
        last_answer = get_answer_from_message("general-qa", last_message)

    # try to parse cancel from last message if both are still None
    if last_answer is None and last_cancel is None:
        last_cancel = get_cancel_from_message(last_message)

    # try to parse SPARQL from last message if both are still None
    if last_answer is None and last_cancel is None:
        last_answer = get_sparql_from_message(last_message)

    # try last message for general QA
    if last_answer is None and last_cancel is None and last_message is not None:
        last_answer = {"answer": last_message}

    return last_answer, last_cancel  # type: ignore


def output(messages: list[dict]) -> dict | None:
    answer, cancel = get_answer_or_cancel(messages)
    if answer is None and cancel is None:
        return None

    elif answer is not None:
        return {
            "type": "answer",
            "answer": answer["answer"],
        }

    else:
        assert cancel is not None
        return {
            "type": "cancel",
            "explanation": cancel["explanation"],
        }
