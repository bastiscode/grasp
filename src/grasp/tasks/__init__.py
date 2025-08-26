from typing import Any

from grasp.configs import Config
from grasp.functions import TaskFunctions
from grasp.manager import KgManager
from grasp.tasks.general_qa import functions as general_qa_functions
from grasp.tasks.general_qa import output as general_qa_output
from grasp.tasks.general_qa import rules as general_qa_rules
from grasp.tasks.general_qa import system_information as general_qa_system_information
from grasp.tasks.sparql_qa import functions as sparql_qa_functions
from grasp.tasks.sparql_qa import output as sparql_qa_output
from grasp.tasks.sparql_qa import rules as sparql_qa_rules
from grasp.tasks.sparql_qa import system_information as sparql_qa_system_information


def rules() -> list[str]:
    return [
        "Explain your thought process before and after each step \
and function call.",
        "Do not just use or make up entity or property identifiers \
without verifying their existence in the knowledge graphs first.",
        'Do not use "SERVICE wikibase:label { bd:serviceParam wikibase:language ..." \
in SPARQL queries. It is not SPARQL standard and unsupported by the used QLever \
SPARQL endpoints. Use rdfs:label or similar properties to get labels instead.',
    ]


def task_rules(task: str) -> list[str]:
    if task == "sparql-qa":
        return sparql_qa_rules()
    elif task == "general-qa":
        return general_qa_rules()

    raise ValueError(f"Unknown task {task}")


def task_system_information(task: str) -> str:
    if task == "sparql-qa":
        return sparql_qa_system_information()
    elif task == "general-qa":
        return general_qa_system_information()

    raise ValueError(f"Unknown task {task}")


def task_state(task: str) -> Any:
    return None


def task_functions(managers: list[KgManager], task: str) -> TaskFunctions:
    if task == "sparql-qa":
        return sparql_qa_functions(managers)
    elif task == "general-qa":
        return general_qa_functions()

    raise ValueError(f"Unknown task {task}")


def task_done(task: str, fn_name: str) -> bool:
    if task == "sparql-qa" or task == "general-qa":
        return fn_name == "answer" or fn_name == "cancel"

    raise ValueError(f"Unknown task {task}")


def task_output(
    task: str,
    messages: list[dict],
    managers: list[KgManager],
    config: Config,
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

    raise ValueError(f"Unknown task {task}")
