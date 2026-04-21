from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig, NotesConfig, NotesFromExplorationConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.exploration import shared_rules
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.functions import note_function_definitions
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.utils import format_kg_notes, format_notes


class FunctionalExplorationState(BaseModel):
    notes: list[str] = []
    kg_notes: dict[str, list[str]] = {}


def rules() -> list[str]:
    return shared_rules() + [
        "The note-specific functions as well as find_frequent are only available \
during exploration, but not for downstream tasks, so do not take notes about \
them and their usage.",
    ]


def system_information(config: GraspConfig) -> str:
    assert isinstance(config, NotesFromExplorationConfig)
    return f"""\
You are a note-taking assistant. Your task is to \
explore knowledge graphs and take notes about them using the \
provided functions.

You are limited to a maximum of {config.max_notes} notes \
per knowledge graph, plus {config.max_notes} general notes for insights that apply \
across knowledge graphs. Each note is limited to a maximum of \
{config.max_note_length} characters to ensure it is concise and to the point.

Your notes should help you to better understand and navigate the \
knowledge graphs in the future. The notes should generalize to new unseen \
questions, rather than being specific to the ones you come up with \
during the exploration.

You should follow a step-by-step approach to take notes:
1. Determine the scope and domain of the knowledge graphs and what types \
of questions a user might want to answer with them. Look at the current notes \
and figure out well-covered and underexplored areas.
2. Come up with a potential user question over one or more knowledge graphs, \
preferably targeting an underexplored area. Then build a SPARQL query to answer \
the question and take notes about your findings along the way. Try to use \
all of the provided functions at least once during your exploration.
3. Repeat steps 1 and 2 until you have explored at least {config.questions_per_round} \
different potential user questions or you run out of ideas.
4. Before stopping, make sure to check all notes (not only the ones touched in this exploration) \
for the above-mentioned criteria and clean them if needed.

Examples of potentially useful types of notes:
- tips for when and how to use certain functions
- strategies when encountering certain types of questions or errors
- quirks and peculiarities of both knowledge graphs and functions"""


def output(state: FunctionalExplorationState) -> dict:
    formatted = f"""\
Exploration completed.

Knowledge graph specific notes:
{format_kg_notes(state.kg_notes)}

General notes across knowledge graphs:
{format_notes(state.notes)}"""

    return {
        "type": "output",
        "notes": state.notes,
        "kg_notes": state.kg_notes,
        "formatted": formatted,
    }


class FunctionalExplorationTask(GraspTask):
    name = "exploration_functional"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = note_function_definitions(self.managers)
        functions.append(find_frequent_function_definition(kgs, self.config.list_k))
        return functions

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.config, NotesConfig)
        assert self.state is not None, "State must be provided for exploration task"
        if fn_name == "find_frequent":
            return find_frequent(
                self.managers,
                fn_args["kg"],
                fn_args["position"],
                fn_args.get("subject"),
                fn_args.get("property"),
                fn_args.get("object"),
                fn_args.get("page", 1),
                self.config.list_k,
                known,
                self.config.sparql_request_timeout,
                self.config.sparql_read_timeout,
            )

        return call_note_function(
            self.state.kg_notes,
            self.state.notes,
            fn_name,
            fn_args,
            self.config.max_notes,
            self.config.max_note_length,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, FunctionalExplorationState), (
            "Input for exploration must already be a FunctionalExplorationState"
        )
        self.state = input
        return "Explore the available knowledge graphs. Add to, delete from, or \
update the current notes along the way."

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
