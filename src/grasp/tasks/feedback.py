from logging import Logger
from typing import Any

import litellm
from universal_ml_utils.logging import get_logger

from grasp.model import Message, call_model
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.utils import format_message, format_response


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
            },
            "strict": True,
        }
    ]


def generate_feedback(
    task: GraspTask,
    kg_notes: dict[str, list[str]],
    notes: list[str],
    inputs: list[str],
    output: Any,
    logger: Logger = get_logger("GRASP FEEDBACK"),
) -> dict | None:
    if not isinstance(task, FeedbackTask):
        return None

    messages: list[Message] = [
        Message(
            role="system",
            content=task.feedback_system_message(kg_notes, notes),
        ),
        Message(
            role="user",
            content=task.feedback_instructions(inputs, output),
        ),
    ]

    for msg in messages:
        logger.debug(format_message(msg))

    try:
        response = call_model(messages, functions(), task.config)
    except litellm.exceptions.Timeout:
        logger.error("LLM API timed out during feedback generation")
        return None

    logger.debug(format_response(response))

    try:
        assert len(response.tool_calls) == 1, "No tool call found"  # type: ignore
        tool_call = response.tool_calls[0]  # type: ignore
        assert tool_call.name == "give_feedback", "Feedback function not called"
        return tool_call.args
    except Exception as e:
        logger.debug(f"Failed to parse feedback:\n{e}")
        return None
