import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from grasp.configs import GraspConfig
from grasp.functions import TaskFunctions, find_manager
from grasp.manager import KgManager, format_kgs
from grasp.model import Message, ToolCall
from grasp.tasks.sparql_qa.examples import (
    SparqlQaExampleIndex,
    find_random_examples,
    find_similar_examples,
)
from grasp.tasks.sparql_qa.examples import (
    functions as example_functions,
)
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import FunctionCallException, format_list, format_notes


def system_information() -> str:
    return """\
You are a question answering assistant. \
Your job is to generate a SPARQL query to answer a given user question.

You should follow a step-by-step approach to generate the SPARQL query:
1. Determine possible entities and properties implied by the user question.
2. Search for the entities and properties in the knowledge graphs. Where \
applicable, constrain the searches with already identified entities and properties.
3. Gradually build up the SPARQL query using the identified entities \
and properties. Start with simple queries and add more complexity as needed. \
Execute intermediate queries to get feedback and to verify your assumptions. \
You may need to refine or rethink your current plan based on the query \
results and go back to step 2 if needed, possibly multiple times.
4. Use the answer or cancel function to finalize your answer and stop the \
generation process."""


def rules() -> list[str]:
    return [
        "Always execute your final SPARQL query before giving an answer to \
make sure it returns the expected results.",
        "The SPARQL query should always return the actual \
identifiers / IRIs of the items in its result. It additionally may return \
labels or other human-readable information, but they are optional and should be \
put within optional clauses unless explicitly requested by the user.",
        "Do not stop early if there are still obvious improvements to be made \
to the SPARQL query. For example, keep refining your SPARQL query if its result \
contains irrelevant items or is missing items you expected.",
        "Do not perform additional computation (e.g. filtering, sorting, calculations) \
on the result of the SPARQL query to determine the answer. All computation should \
be done solely within SPARQL.",
        'For questions with a "True" or "False" answer the SPARQL query \
should be an ASK query.',
    ]


def functions(managers: list[KgManager], config: GraspConfig) -> TaskFunctions:
    kgs = [manager.kg for manager in managers]
    fns = [
        {
            "name": "answer",
            "description": """\
Provide your final SPARQL query and answer to the user question based on the \
query results. This function will stop the generation process.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph on which the final SPARQL query \
needs to be executed",
                    },
                    "sparql": {
                        "type": "string",
                        "description": "The final SPARQL query",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer to the question based \
on the SPARQL query results",
                    },
                },
                "required": ["kg", "sparql", "answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "cancel",
            "description": """\
If you are unable to find a SPARQL query that answers the question well, \
you can call this function instead of the answer function. This function will \
stop the generation process.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "A detailed explanation of why you \
could not find a satisfactory SPARQL query",
                    },
                    "best_attempt": {
                        "type": ["object", "null"],
                        "description": "Your best attempt at a SPARQL query so far, \
can be omitted if there is none",
                        "properties": {
                            "sparql": {
                                "type": "string",
                                "description": "The best SPARQL query so far",
                            },
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph on which \
the SPARQL query needs to be executed",
                            },
                        },
                        "required": ["sparql", "kg"],
                        "additionalProperties": False,
                    },
                },
                "required": ["explanation", "best_attempt"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    fns.extend(example_functions(config))

    return fns, call_function


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: Any = None,
    example_indices: dict[str, SparqlQaExampleIndex] | None = None,
) -> str:
    if fn_name == "answer":
        return "Stopping"

    elif fn_name == "cancel":
        return "Stopping"

    elif fn_name == "find_examples" and example_indices is not None:
        return find_random_examples(
            managers,
            example_indices,
            fn_args["kg"],
            config.num_examples,
            known,
            config.result_max_rows,
            config.result_max_columns,
        )

    elif fn_name == "find_similar_examples" and example_indices is not None:
        return find_similar_examples(
            managers,
            example_indices,
            fn_args["kg"],
            fn_args["question"],
            config.num_examples,
            known,
            config.result_max_rows,
            config.result_max_columns,
        )

    raise FunctionCallException(f"Unknown function: {fn_name}")


class AnswerModel(BaseModel):
    kg: str
    sparql: str
    answer: str


class AnswerCallModel(BaseModel):
    name: str
    arguments: AnswerModel


class BestAttemptModel(BaseModel):
    sparql: str
    kg: str


class CancelModel(BaseModel):
    explanation: str
    best_attempt: BestAttemptModel | str | None = None


class CancelCallModel(BaseModel):
    name: str
    arguments: CancelModel


def get_raw_tool_call_from_message(message: str) -> str | None:
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


def get_answer_from_message(message: str | None) -> ToolCall | None:
    if message is None:
        return None

    tool_call = get_raw_tool_call_from_message(message)
    if tool_call is None:
        return None

    try:
        answer_call = AnswerCallModel.model_validate_json(tool_call)
        return ToolCall(
            id=uuid4().hex,
            name=answer_call.name,
            args=answer_call.arguments.model_dump(),
        )
    except ValidationError:
        pass

    try:
        args = AnswerModel.model_validate_json(tool_call).model_dump()
        return ToolCall(id=uuid4().hex, name="answer", args=args)
    finally:
        return None


def get_cancel_from_message(message: str | None) -> ToolCall | None:
    if message is None:
        return None

    tool_call = get_raw_tool_call_from_message(message)
    if tool_call is None:
        return None

    try:
        cancel_call = CancelCallModel.model_validate_json(tool_call)
        return ToolCall(
            id=uuid4().hex,
            name=cancel_call.name,
            args=cancel_call.arguments.model_dump(),
        )
    except ValidationError:
        pass

    try:
        args = CancelModel.model_validate_json(tool_call).model_dump()
        return ToolCall(id=uuid4().hex, name="cancel", args=args)
    finally:
        return None


def get_sparql_from_message(message: str | None) -> ToolCall | None:
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
        return ToolCall(
            id=uuid4().hex,
            name="answer",
            args={"kg": None, "sparql": sparql_query, "answer": message},
        )

    return None


def get_answer_or_cancel(
    messages: list[Message],
) -> tuple[ToolCall | None, ToolCall | None]:
    last_message: str | None = None
    last_answer: ToolCall | None = None
    last_cancel: ToolCall | None = None
    last_execute: ToolCall | None = None
    assert messages[0].role == "system", "First message should be system"
    assert messages[1].role == "user", "Second message should be user"
    for message in messages[2:]:
        is_intermediate_feedback = (
            message.role == "feedback" and message != messages[-1]
        )
        is_user_message = message.role == "user"
        if is_intermediate_feedback or is_user_message:
            # reset stuff after intermediate feedback or user message
            last_answer = None
            last_cancel = None
            last_message = None
            last_execute = None

        if isinstance(message.content, str):
            # not assistant message
            continue

        last_message = message.content.message

        for tool_call in message.content.tool_calls:
            if tool_call.name == "answer":
                last_answer = tool_call
                # reset last cancel
                last_cancel = None

            elif tool_call.name == "cancel":
                last_cancel = tool_call
                # reset last answer
                last_answer = None

            elif tool_call.name == "execute":
                last_execute = tool_call

    # try to parse answer from last message if neither are set
    if last_answer is None and last_cancel is None:
        last_answer = get_answer_from_message(last_message)

    # try to parse cancel from last message if both are still None
    if last_answer is None and last_cancel is None:
        last_cancel = get_cancel_from_message(last_message)  # type: ignore

    # try to parse SPARQL from last message if both are still None
    if last_answer is None and last_cancel is None:
        last_answer = get_sparql_from_message(last_message)

    # try last execute function call for SPARQL QA
    if last_answer is None and last_cancel is None and last_execute is not None:
        last_answer = ToolCall(
            id="dummy",
            name="answer",
            args={**last_execute.args, "answer": last_message or "No answer provided"},
        )

    return last_answer, last_cancel  # type: ignore


def output(
    messages: list[Message],
    managers: list[KgManager],
    max_rows: int,
    max_cols: int,
) -> dict | None:
    answer, cancel = get_answer_or_cancel(messages)
    if answer is None and cancel is None:
        return None

    output: dict[str, Any] = {
        "sparql": None,
        "kg": None,
        "selections": None,
        "result": None,
        "endpoint": None,
    }

    if answer is not None:
        output["type"] = "answer"
        output["answer"] = answer.args["answer"].strip()
        output["sparql"] = answer.args["sparql"]
        output["kg"] = answer.args["kg"]
        output["formatted"] = output["answer"]

    else:
        assert cancel is not None
        output["type"] = "cancel"
        output["explanation"] = cancel.args["explanation"].strip()
        output["formatted"] = output["explanation"]

        best_attempt = cancel.args.get("best_attempt")
        if best_attempt:
            output["sparql"] = best_attempt.get("sparql")
            output["kg"] = best_attempt.get("kg")

    if output["sparql"] is not None:
        if output["kg"] is None:
            output["kg"] = managers[0].kg

        result, selections = prepare_sparql_result(
            output["sparql"],
            output["kg"],
            managers,
            max_rows,
            max_cols,
        )
        manager, _ = find_manager(managers, output["kg"])

        output["sparql"] = result.sparql
        output["selections"] = manager.format_selections(selections)
        output["result"] = result.formatted
        output["endpoint"] = manager.endpoint

        output["formatted"] += "\n\n"
        output["formatted"] += format_sparql_result(manager, result, selections)

    return output


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    return f"""\
You are a question answering assistant providing feedback on the \
output of a SPARQL-based question answering system for a given user question.

The system has access to the following knowledge graphs:
{format_kgs(managers, kg_notes)}

The system was provided the following notes across all knowledge graphs:
{format_notes(notes)}

The system was provided the following rules to follow:
{format_list(rules())}

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

Provide your feedback with the give_feedback function."""


def feedback_instructions(questions: list[str], output: dict) -> str:
    assert questions, "At least one question is required for feedback"

    if len(questions) > 1:
        prompt = (
            "Previous questions:\n"
            + "\n\n".join(q.strip() for q in questions[:-1])
            + "\n\n"
        )

    else:
        prompt = ""

    prompt += f"Question:\n{questions[-1].strip()}"

    if output["type"] == "answer":
        # terminated with answer call
        prompt += f"""

1) The system was able to find an answer

Answer:
{output["formatted"]}"""

    else:
        prompt += f"""

2) The system failed to find an answer

Explanation:
{output["formatted"]}"""

    return prompt
