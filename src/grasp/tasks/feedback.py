import json
from logging import Logger
from typing import Any

import litellm
from universal_ml_utils.logging import get_logger

from grasp.configs import Config
from grasp.manager import KgManager
from grasp.model import call_model
from grasp.tasks.sparql_qa import (
    feedback_instructions as sparql_qa_feedback_instructions,
)
from grasp.tasks.sparql_qa import (
    feedback_system_message as sparql_qa_feedback_system_instructions,
)
from grasp.utils import format_message


def format_feedback(feedback: dict) -> str:
    status = feedback["status"]
    return f"Feedback (status={status}):\n{feedback['feedback']}"


def functions() -> list[dict]:
    return [
        {
            "name": "give_feedback",
            "description": """\
Provide feedback on the output of the system for the \
specified task.

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


def system_instructions(
    task: str,
    managers: list[KgManager],
    notes: list[str],
) -> str:
    if task == "sparql-qa":
        return sparql_qa_feedback_system_instructions(managers, notes)

    raise ValueError(f"System message not implemented for task: {task}")


def feedback_instructions(task: str, questions: list[str], output: Any) -> str:
    if task == "sparql-qa":
        return sparql_qa_feedback_instructions(questions, output)

    raise ValueError(f"Feedback not implemented for task: {task}")


def generate_feedback(
    task: str,
    managers: list[KgManager],
    config: Config,
    notes: list[str],
    questions: list[str],
    output: dict,
    logger: Logger = get_logger("GRASP FEEDBACK"),
) -> dict | None:
    api_messages: list[dict] = [
        {
            "role": "system",
            "content": system_instructions(task, managers, notes),
        },
        {
            "role": "user",
            "content": feedback_instructions(task, questions, output),
        },
    ]
    for msg in api_messages:
        logger.debug(format_message(msg))

    try:
        response = call_model(api_messages, functions(), config)
    except litellm.exceptions.Timeout:
        logger.error("LLM API timed out during feedback generation")
        return None

    choice = response.choices[0]  # type: ignore
    msg = choice.message.model_dump(exclude_none=True)  # type: ignore
    logger.debug(format_message(msg))

    try:
        assert len(choice.message.tool_calls) == 1, "No tool call found"  # type: ignore
        tool_call = choice.message.tool_calls[0]  # type: ignore
        assert tool_call.type == "function", "Tool call is not a function call"
        fn_name = tool_call.function.name
        assert fn_name == "give_feedback", "Feedback function not called"
        return json.loads(tool_call.function.arguments)
    except Exception as e:
        logger.debug(f"Failed to parse feedback:\n{e}")
        return None
