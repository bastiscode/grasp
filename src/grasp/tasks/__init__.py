from enum import StrEnum

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.tasks.base import GraspTask
from grasp.tasks.cea import CeaTask
from grasp.tasks.exploration import ExplorationTask
from grasp.tasks.general_qa import GeneralQaTask
from grasp.tasks.sparql_qa import SparqlQaTask
from grasp.tasks.wikidata_query_logs import WdqlTask


# official tasks supported by GRASP, excluding exploration
# which is a special task for note taking
class Task(StrEnum):
    SPARQL_QA = "sparql-qa"
    GENERAL_QA = "general-qa"
    CEA = "cea"
    WDQL = "wikidata-query-logs"


_REGISTRY: dict[str, type[GraspTask]] = {
    cls.name: cls
    for cls in [SparqlQaTask, GeneralQaTask, CeaTask, WdqlTask, ExplorationTask]
}


def get_task(task: str, managers: list[KgManager], config: GraspConfig) -> GraspTask:
    if task not in _REGISTRY:
        raise ValueError(f"Unknown task {task}")
    return _REGISTRY[task](managers, config)


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
