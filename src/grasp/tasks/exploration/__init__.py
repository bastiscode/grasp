from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig, NotesConfig, NotesFromExplorationConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.functions import note_functions
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
    ]


def system_information(config: GraspConfig) -> str:
    assert isinstance(config, NotesFromExplorationConfig)
    return f"""\
You are a note-taking assistant. Your task is to \
explore knowledge graphs and take notes about them using the \
provided functions.

You should follow a step-by-step approach to take notes:
1. Think about what domains the knowledge graphs might cover and what types of \
questions a user might want to answer with them. Take into account already \
existing notes to focus on unexplored areas.
2. Come up with a potential user question over one or more knowledge graphs. \
Try to build a SPARQL query to answer the question and take notes \
about your findings along the way. Try to use all of the \
provided functions during your exploration.
3. Repeat steps 1 and 2 until you explored at least {config.questions_per_round} \
different potential user questions or you run out of ideas.

You can take notes specific to a certain knowledge graph, as well as general notes \
that might be useful across knowledge graphs.

You are only allowed {config.max_notes} notes at max per knowledge graph and for the \
general notes, such that you are forced to prioritize and to keep them as widely \
applicable as possible. Notes are limited to {config.max_note_length} characters to \
ensure they are concise and to the point.

Examples of potentially useful types of notes include:
- overall structure, domain coverage, and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions"""


def input(state: ExplorationState) -> str:
    kg_specific_notes = format_list(
        f"{kg}:\n{format_notes(kg_specific_notes, indent=2, enumerated=True)}"
        for kg, kg_specific_notes in sorted(state.kg_notes.items())
    )
    return f"""\
Explore the available knowledge graphs. Add to, delete from, or update the following \
notes along the way.

Knowledge graph specific notes:
{kg_specific_notes}

General notes across knowledge graphs:
{format_notes(state.notes, enumerated=True)}"""


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


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: ExplorationState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(config, NotesConfig)
    assert state is not None, "State must be provided for exploration task"
    return call_note_function(
        state.kg_notes,
        state.notes,
        fn_name,
        fn_args,
        config.max_notes,
        config.max_note_length,
    )


# ── Task class ──────────────────────────────────────────────────────────────


from grasp.tasks.base import GraspTask  # noqa: E402

# save reference before it is shadowed by the method parameter name
_exploration_input = input


class ExplorationTask(GraspTask):
    name = "exploration"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return note_functions(self.managers)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        state: Any,
        example_indices: dict | None,
    ) -> str:
        return call_function(
            self.config, self.managers, fn_name, fn_args, known, state, example_indices
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> tuple[str, Any]:
        assert isinstance(input, ExplorationState), (
            "Input for exploration must already be an ExplorationState"
        )
        return _exploration_input(input), input

    def output(self, messages: list[Message], state: Any) -> dict:
        return output(state)
