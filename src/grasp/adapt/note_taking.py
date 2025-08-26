import json
from copy import deepcopy
from logging import Logger

from litellm.exceptions import Timeout

from grasp.adapt.notes import (
    MAX_NOTE_LENGTH,
    MAX_NOTES,
    call_function,
    note_functions,
)
from grasp.adapt.utils import format_output
from grasp.configs import Adapt, Config
from grasp.core import call_model
from grasp.manager import KgManager, format_kgs
from grasp.tasks.utils import prepare_sparql_result
from grasp.utils import Sample, format_list, format_message, format_notes

MAX_MESSAGES = 50


def rules() -> list[str]:
    return [
        "Do not take notes on things that are already handled well by the system.",
        "Avoid notes about entity or property identifiers just for the sake of not \
having to look them up again.",
        "As you hit the limits on the number of notes and their length, \
gradually generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
    ]


def system_instructions() -> str:
    return f"""\
You are a note-taking assistant. Your task is to \
inspect the traces of a knowledge graph question answering system and \
take notes about the system's outputs as well as the used knowledge \
graphs and functions. Before calling a note-taking function, \
provide reasoning for what you are doing and why.

Your notes should help the system to better understand and \
navigate the task and knowledge graphs in the future. For a specific knowledge \
graph, they should generalize across questions, rather than being specific to \
a single question or output. You can also take general notes that might be \
useful across knowledge graphs. \
You are only allowed {MAX_NOTES} notes at max per knowledge graph and for the \
general notes, such that you are forced to prioritize and to keep them as widely \
applicable as possible. Notes are limited to {MAX_NOTE_LENGTH} characters to \
ensure they are concise and to the point.

Examples of potentially useful types of notes include:
- overall structure and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions

Additional rules:
{format_list(rules())}"""


def note_taking_instructions(
    managers: list[KgManager],
    notes: list[str],
    config: Config,
    inputs: list[tuple[str, Sample]],
    outputs: list[dict],
) -> str:
    formatted = []
    for i, ((kg, sample), output) in enumerate(zip(inputs, outputs)):
        messages = output["messages"]
        assert messages[1]["role"] == "user"
        question = messages[1]["content"]

        gt = prepare_sparql_result(
            sample.sparql,
            kg,
            managers,
            config.result_max_rows,
            config.result_max_columns,
        )

        content = f"""\
Question {i + 1} over {kg} knowledge graph:
{question}

System output:
{format_output(output)}

Ground truth:
{gt}"""

        formatted.append(content)

    outputs_formatted = "\n\n".join(formatted)

    return f"""\
Add to, delete from, or update the following notes \
based on the provided questions and outputs below.

Knowledge graph specific notes:
{format_kgs(managers)}

Notes across all knowledge graphs:
{format_notes(notes)}

{outputs_formatted}"""


def take_notes(
    inputs: list[tuple[str, Sample]],
    outputs: list[dict],
    managers: list[KgManager],
    notes: list[str],
    config: Adapt,
    logger: Logger,
) -> None:
    api_messages = [
        {"role": "system", "content": system_instructions()},
        {
            "role": "user",
            "content": note_taking_instructions(
                managers, notes, config, inputs, outputs
            ),
        },
    ]

    for msg in api_messages:
        logger.debug(format_message(msg))

    functions = note_functions(managers)

    num_messages = len(api_messages)

    # copy config to avoid modifying the original
    config = deepcopy(config)
    config.model = config.adapt_model or config.model
    config.model_endpoint = config.adapt_model_endpoint or config.model_endpoint
    config.temperature = config.adapt_temperature or config.temperature
    config.top_p = config.adapt_top_p or config.top_p
    config.reasoning_effort = config.adapt_reasoning_effort or config.reasoning_effort

    while len(api_messages) - num_messages < MAX_MESSAGES:
        try:
            response = call_model(api_messages, functions, config)
        except Timeout:
            return

        choice = response.choices[0]  # type: ignore
        msg = choice.message.model_dump(exclude_none=True)  # type: ignore
        api_messages.append(msg)
        logger.debug(format_message(msg))

        if not choice.message.tool_calls:  # type: ignore
            return

        for tool_call in choice.message.tool_calls or []:  # type: ignore
            fn_name: str = tool_call.function.name  # type: ignore
            fn_args = json.loads(tool_call.function.arguments)

            try:
                result = call_function(managers, notes, fn_name, fn_args)
            except Exception as e:
                result = f"Call to function {fn_name} returned an error:\n{e}"

            tool_msg = {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            api_messages.append(tool_msg)
            logger.debug(format_message(tool_msg))

            if fn_name == "stop":
                return
