from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig, NotesConfig, NotesFromExplorationConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.functions import note_functions
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.utils import format_list, format_notes


class ExplorationState(BaseModel):
    notes: list[str] = []
    kg_notes: dict[str, list[str]] = {}


def rules() -> list[str]:
    return [
        "The questions you come up with should be diverse and cover different \
parts of the knowledge graphs.",
        "As you hit the limits on the number of notes and their length, \
gradually generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
        "The find_frequent function is only available during exploration, "
        "but not for downstream tasks, so do not take notes about it and its usage.",
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
and figure out well covered and underexplored areas.
2. Come up with a potential user question over one or more knowledge graphs, \
preferably in an underexplored area. Try to build a SPARQL query to answer \
the question and take notes about your findings along the way. Try to use all \
of the provided functions during your exploration.
3. Repeat steps 1 and 2 until you explored at least {config.questions_per_round} \
different potential user questions or you run out of ideas.
4.Before stopping, make sure to check all notes (not only the ones touched in this exploration) \
for the above mentioned criteria and clean them if needed.

Examples of potentially useful types of notes include:
- overall structure, domain coverage, and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions"""


def output(state: ExplorationState) -> dict:
    kg_specific_notes = format_list(
        f"{kg}:\n{format_notes(kg_specific_notes, indent=2)}"
        for kg, kg_specific_notes in sorted(state.kg_notes.items())
    )
    formatted = f"""\
Exploration completed.

Knowledge graph specific notes:
{kg_specific_notes}

General notes across knowledge graphs:
{format_notes(state.notes)}"""

    return {
        "type": "output",
        "notes": state.notes,
        "kg_notes": state.kg_notes,
        "formatted": formatted,
    }


class ExplorationTask(GraspTask):
    name = "exploration"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = note_functions(self.managers)
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
        assert isinstance(input, ExplorationState), (
            "Input for exploration must already be an ExplorationState"
        )
        self.state = input
        return "Explore the available knowledge graphs. Add to, delete from, or \
update the current notes along the way."

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
