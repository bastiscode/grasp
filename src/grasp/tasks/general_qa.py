from typing import Any

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message, Response
from grasp.tasks.sparql_qa import SparqlQaTask
from grasp.tasks.sparql_qa.examples import (
    SparqlQaExampleIndex,
)
from grasp.tasks.sparql_qa.examples import (
    call_function as sparql_qa_example_call_function,
)
from grasp.tasks.sparql_qa.examples import (
    functions as sparql_qa_example_functions,
)


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
4. Output your final answer to the question and stop."""


def functions(config: GraspConfig) -> list[dict]:
    return sparql_qa_example_functions(config)


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: Any | None = None,
    example_indices: dict[str, SparqlQaExampleIndex] | None = None,
) -> str:
    return sparql_qa_example_call_function(
        config,
        managers,
        fn_name,
        fn_args,
        known,
        example_indices,
    )


def rules() -> list[str]:
    return [
        "Your answers preferably should be based on the information available in the \
knowledge graphs. If you do not need them to answer the question, e.g. if \
you know the answer by heart, still try to verify it with the knowledge graphs.",
    ]


_module_call_function = call_function


def output(messages: list[Message]) -> dict | None:
    last_response: Response | None = None
    for message in reversed(messages):
        if isinstance(message.content, Response):
            last_response = message.content
            break

    if last_response is None or last_response.message is None:
        return None

    return {
        "type": "output",
        "output": last_response.message,
        "formatted": last_response.message,
    }


class GeneralQaTask(SparqlQaTask):
    name = "general-qa"

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return functions(self.config)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        state: Any,
        example_indices: dict | None,
    ) -> str:
        return _module_call_function(
            self.config, self.managers, fn_name, fn_args, known, state, example_indices
        )

    def done(self, fn_name: str) -> bool:
        return False

    def output(self, messages: list[Message], state: Any) -> dict | None:
        return output(messages)
