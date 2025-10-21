from enum import StrEnum
from typing import Any, Type

from grasp.configs import GraspConfig
from grasp.functions import TaskFunctions
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.cea import functions as cea_functions
from grasp.tasks.cea import input_and_state as cea_input_and_state
from grasp.tasks.cea import output as cea_output
from grasp.tasks.cea import rules as cea_rules
from grasp.tasks.cea import system_information as cea_system_information
from grasp.tasks.examples import Sample
from grasp.tasks.exploration import ExplorationState
from grasp.tasks.exploration import functions as exploration_functions
from grasp.tasks.exploration import input as exploration_input
from grasp.tasks.exploration import output as exploration_output
from grasp.tasks.exploration import rules as exploration_rules
from grasp.tasks.exploration import system_information as exploration_system_information
from grasp.tasks.general_qa import output as general_qa_output
from grasp.tasks.general_qa import rules as general_qa_rules
from grasp.tasks.general_qa import system_information as general_qa_system_information
from grasp.tasks.sparql_qa import functions as sparql_qa_functions
from grasp.tasks.sparql_qa import output as sparql_qa_output
from grasp.tasks.sparql_qa import rules as sparql_qa_rules
from grasp.tasks.sparql_qa import system_information as sparql_qa_system_information
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.tasks.wikidata_query_logs import functions as wdql_functions
from grasp.tasks.wikidata_query_logs import input_and_state as wdql_input_and_state
from grasp.tasks.wikidata_query_logs import output as wdql_output
from grasp.tasks.wikidata_query_logs import rules as wdql_rules
from grasp.tasks.wikidata_query_logs import (
    system_information as wdql_system_information,
)


# official tasks supported by GRASP, excluding exploration
# which is a special task for note taking
class Task(StrEnum):
    SPARQL_QA = "sparql-qa"
    GENERAL_QA = "general-qa"
    CEA = "cea"
    WDQL = "wikidata-query-logs"


def rules() -> list[str]:
    return [
        "Explain your thought process before and after each step \
and function call.",
        "Do not ask the user for clarification until you have made \
an initial attempt at completing the task. When the task input is incomplete or \
ambiguous, proceed based on reasonable assumptions.",
        'Do not use "SERVICE wikibase:label { bd:serviceParam wikibase:language ..." \
in SPARQL queries. It is unsupported by the used QLever SPARQL endpoints. \
Use rdfs:label or similar properties to get labels instead.',
    ]


def task_rules(task: str) -> list[str]:
    if task == "sparql-qa":
        return sparql_qa_rules()
    elif task == "general-qa":
        return general_qa_rules()
    elif task == "cea":
        return cea_rules()
    elif task == "wikidata-query-logs":
        return wdql_rules()
    elif task == "exploration":
        return exploration_rules()

    raise ValueError(f"Unknown task {task}")


def task_system_information(task: str, config: GraspConfig) -> str:
    if task == "sparql-qa":
        return sparql_qa_system_information()
    elif task == "general-qa":
        return general_qa_system_information()
    elif task == "cea":
        return cea_system_information()
    elif task == "wikidata-query-logs":
        return wdql_system_information(config)
    elif task == "exploration":
        return exploration_system_information(config)

    raise ValueError(f"Unknown task {task}")


def task_functions(
    managers: list[KgManager],
    task: str,
    config: GraspConfig,
) -> TaskFunctions | None:
    if task == "sparql-qa":
        return sparql_qa_functions(managers, config)
    elif task == "general-qa":
        # general qa has no additional functions
        # and does not support examples
        return None
    elif task == "cea":
        # cea supports no examples
        return cea_functions(managers)
    elif task == "wikidata-query-logs":
        return wdql_functions()
    elif task == "exploration":
        return exploration_functions(managers)

    raise ValueError(f"Unknown task {task}")


def task_done(task: str, fn_name: str) -> bool:
    if task == "sparql-qa":
        return fn_name == "answer" or fn_name == "cancel"
    elif task == "general-qa":
        # general qa can never be stopped by a function call
        return False
    elif task == "cea":
        return fn_name == "stop"
    elif task == "wikidata-query-logs":
        return fn_name == "answer" or fn_name == "cancel"
    elif task == "exploration":
        return fn_name == "stop"

    raise ValueError(f"Unknown task {task}")


def task_setup(
    task: str,
    input: Any,
    managers: list[KgManager],
    config: GraspConfig,
) -> tuple[str, Any]:
    if task == "sparql-qa" or task == "general-qa":
        assert isinstance(input, str), (
            f"Input for task {task} must be a string (question)"
        )
        return input, None
    elif task == "cea":
        return cea_input_and_state(input, **config.task_kwargs)
    elif task == "wikidata-query-logs":
        assert isinstance(input, str), (
            "Input for wikidata-query-logs must be a string (SPARQL query)"
        )
        return wdql_input_and_state(
            input,
            managers,
            config.result_max_rows,
            config.result_max_columns,
        )
    elif task == "exploration":
        # exploration has no setup
        assert isinstance(input, ExplorationState), (
            "Input for exploration must already be an ExplorationState"
        )
        return exploration_input(input), input

    raise ValueError(f"Unknown task {task}")


def default_input_field(task: str) -> str | None:
    if task == "sparql-qa" or task == "general-qa":
        # inputs are typically question-sparql or question-answer pairs
        # with question and sparql/answer fields
        return "question"
    elif task == "cea":
        # input is typically a json dict with a table field and optional
        # metadata fields
        return "table"
    elif task == "wikidata-query-logs":
        return "sparql"
    elif task == "exploration":
        return None

    raise ValueError(f"Unknown task {task}")


def task_output(
    task: str,
    messages: list[Message],
    managers: list[KgManager],
    config: GraspConfig,
    task_state: Any = None,
) -> dict | None:
    if task == "sparql-qa":
        return sparql_qa_output(
            messages,
            managers,
            config.result_max_rows,
            config.result_max_columns,
        )
    elif task == "general-qa":
        return general_qa_output(messages)
    elif task == "cea":
        return cea_output(task_state)
    elif task == "wikidata-query-logs":
        return wdql_output(
            messages,
            managers,
            config.result_max_rows,
            config.result_max_columns,
        )
    elif task == "exploration":
        return exploration_output(task_state)

    raise ValueError(f"Unknown task {task}")


def task_to_sample(task: str) -> Type[Sample]:
    if task == "sparql-qa":
        return SparqlQaSample

    raise ValueError(f"Unsupported or unknown task {task}")
