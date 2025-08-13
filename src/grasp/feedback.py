import json
from logging import Logger

import litellm
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import partition_by

from grasp.configs import Config
from grasp.model import call_model
from grasp.functions import execute_sparql
from grasp.sparql.data import get_sparql_items, selections_from_items
from grasp.sparql.manager import KgManager
from grasp.utils import format_function_call, format_message


def format_feedback(feedback: dict) -> str:
    status = feedback["status"]
    return f"Feedback (status={status}):\n{feedback['feedback']}"


def get_feedback_functions() -> list[dict]:
    return [
        {
            "name": "give_feedback",
            "description": """\
Provide feedback to the output of the question answering system in the \
context of the user's question.

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


def get_sparql_qa_feedback_system_message(
    managers: list[KgManager],
    task_rules: list[str],
) -> dict:
    kgs = "\n".join(f"{manager.kg} at {manager.endpoint}" for manager in managers)

    rules = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(task_rules))
    return {
        "role": "system",
        "content": f"""\
You are a question answering assistant providing feedback on the \
output of a SPARQL-based question answering system for a given user question.

The system has access to the following knowledge graphs:
{kgs}

There are two possible cases:

1) The system was able to find an answer
You are given the final SPARQL query, the knowledge graph it has to be executed \
against, and a human-readable answer to the question. You are also given some \
additional information about the SPARQL query, like the entities and properties \
it uses, and its execution result.

2) The system failed to find an answer
You are given the system's explanation for why it failed to find an answer. \
Optionally, you are provided with the system's best attempt at a SPARQL query \
so far including the same additional information as in case 1.

In any case, make sure that the following rules for SPARQL queries are followed:
{rules}

Provide your feedback with the give_feedback function.""",
    }


def get_sparql_qa_feedback_prompt(
    managers: list[KgManager],
    questions: list[str],
    answer: dict | None,
    cancel: dict | None,
    max_rows: int,
    max_columns: int,
) -> str:
    sparql = None
    kg = None

    if not questions:
        raise ValueError("At least one question is required for feedback")
    elif len(questions) > 1:
        prompt = (
            "Previous questions:\n"
            + "\n\n".join(q.strip() for q in questions[:-1])
            + "\n\n"
        )
    else:
        prompt = ""

    prompt += f"Question:\n{questions[-1].strip()}"

    if answer is not None:
        prompt += f"""

1) The system was able to find an answer

Answer:
{answer["answer"].strip()}"""
        sparql = answer["sparql"]
        kg = answer["kg"]

    else:
        assert cancel is not None, "No answer or cancel info provided"
        prompt += f"""

2) The system failed to find an answer

Explanation:
{cancel["explanation"].strip()}"""
        best_attempt = cancel.get("best_attempt")
        if best_attempt:
            sparql = best_attempt.get("sparql")
            kg = best_attempt.get("kg")

    if sparql is not None:
        if kg is None:
            # can happen if answer is parsed from message
            # use the first manager's kg
            kg = managers[0].kg

        managers, others = partition_by(managers, lambda m: m.kg == kg)
        if len(managers) != 1:
            raise ValueError(f"Unknown knowledge graph {kg}")

        manager = managers[0]
        try:
            result = execute_sparql(
                manager,
                others,
                sparql,
                max_rows,
                max_columns,
                known=set(),
            )
        except Exception as e:
            result = f"Failed to execute SPARQL query:\n{e}"

        try:
            _, items = get_sparql_items(sparql, manager)
            selections = selections_from_items(items)
            selections = manager.format_selections(selections)
        except Exception as e:
            selections = f"Failed to determine used entities and properties:\n{e}"

        prompt += f"""

Knowledge graph:
{kg}

SPARQL query:
{sparql.strip()}"""

        if selections:
            prompt += f"\n\n{selections}"

        assert isinstance(result, str)
        prompt += f"\n\nExecution result:\n{result.strip()}"

    else:
        prompt += "\n\nNo SPARQL query provided"

    return prompt


def generate_sparql_qa_feedback(
    config: Config,
    managers: list[KgManager],
    questions: list[str],
    answer: dict | None,
    cancel: dict | None,
    task_rules: list[str],
    logger: Logger = get_logger("GRASP FEEDBACK GENERATION"),
) -> dict | None:
    # give feedback to answer or last message from generate_sparql

    api_messages: list[dict] = [
        get_sparql_qa_feedback_system_message(managers, task_rules),
        {
            "role": "user",
            "content": get_sparql_qa_feedback_prompt(
                managers,
                questions,
                answer,
                cancel,
                config.result_max_rows,
                config.result_max_columns,
            ),
        },
    ]
    for msg in api_messages:
        logger.debug(format_message(msg))

    try:
        response = call_model(api_messages, get_feedback_functions(), config)
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
        fn_args = json.loads(tool_call.function.arguments)
        msg = {
            "role": "tool call",
            "content": format_function_call(fn_name, fn_args),
        }
        logger.debug(format_message(msg))
        return fn_args
    except Exception as e:
        logger.debug(f"Failed to parse feedback:\n{e}")
        return None
