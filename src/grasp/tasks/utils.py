import re

from pydantic import BaseModel, ValidationError

from grasp.functions import execute_sparql, find_manager, update_known_from_selections
from grasp.manager import KgManager
from grasp.sparql.item import get_sparql_items, selections_from_items


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


def prepare_sparql_result(
    sparql: str,
    kg: str,
    managers: list[KgManager],
    max_rows: int,
    max_columns: int,
    known: set[str] | None = None,
) -> tuple[str, str, str]:
    manager, _ = find_manager(managers, kg)

    try:
        result, sparql = execute_sparql(
            managers,
            kg,
            sparql,
            max_rows,
            max_columns,
            known,
            return_sparql=True,
        )
        sparql = manager.prettify(sparql)
    except Exception as e:
        result = f"Failed to execute SPARQL query:\n{e}"

    try:
        _, items = get_sparql_items(sparql, manager)
        selections = selections_from_items(items)
        if known is not None:
            update_known_from_selections(known, selections, manager)
        selections = manager.format_selections(selections)
    except Exception as e:
        selections = f"Failed to determine used entities and properties:\n{e}"

    return sparql, selections, result


def format_sparql_result(
    sparql: str,
    kg: str,
    selections: str,
    result: str,
) -> str:
    fmt = f"SPARQL query over {kg}:\n{sparql.strip()}"

    if selections:
        fmt += f"\n\n{selections}"

    fmt += f"\n\nExecution result:\n{result.strip()}"
    return fmt
