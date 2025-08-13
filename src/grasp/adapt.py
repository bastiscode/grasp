from copy import deepcopy
import json
import os
import random
from logging import Logger

import litellm
import yaml
from tqdm import tqdm, trange
from universal_ml_utils.io import dump_json, load_jsonl
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import partition_by

from grasp.configs import Adapt, Config
from grasp.core import call_model, format_kg, generate, setup
from grasp.functions import execute_sparql, get_functions
from grasp.notes import (
    MAX_NOTE_LENGTH,
    MAX_NOTES,
    call_function,
    format_general_notes,
    get_note_functions,
)
from grasp.sparql.data import get_sparql_items, selections_from_items
from grasp.sparql.manager import KgManager, load_general_notes
from grasp.utils import Sample, format_enumerate, format_function_call, format_message

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


def get_note_taking_system_message() -> dict:
    content = f"""\
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
{format_enumerate(rules())}"""

    return {"role": "system", "content": content}


def format_arguments(args, depth: int = 0) -> str:
    if isinstance(args, list):
        return "[" + ", ".join(format_arguments(i, depth + 1) for i in args) + "]"
    elif isinstance(args, dict):
        return (
            "{" * (depth > 0)
            + ", ".join(
                f"{k}={format_arguments(v, depth + 1)}" for k, v in args.items()
            )
            + "}" * (depth > 0)
        )
    elif isinstance(args, str):
        return f'"{args}"'
    else:
        return str(args)


def format_output(output: dict) -> str:
    tool_call_results = {
        message["tool_call_id"]: message["content"]
        for message in output["messages"]
        if message["role"] == "tool"
    }
    fmt = []
    step = 1
    for message in output["messages"][2:]:
        if message["role"] == "tool":
            continue
        elif message["role"] == "user":
            fmt.append(f"Feedback:\n{message['content']}")
            continue

        assert message["role"] == "assistant"

        content = f"System step {step}:"
        if message.get("reasoning_content"):
            content += f"\n{message['reasoning_content'].strip()}"
        if message.get("content"):
            content += f"\n{message['content'].strip()}"

        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            if tool_call["type"] != "function":
                continue

            tool_call_fn = tool_call["function"]
            tool_calls.append(
                f'Call of "{tool_call_fn["name"]}" function '
                f"with {format_arguments(json.loads(tool_call_fn['arguments']))}:\n"
                f"{tool_call_results[tool_call['id']]}"
            )

        content += "\n" + "\n".join(tool_calls)

        fmt.append(content.strip())
        step += 1

    return "\n\n".join(fmt)


def get_note_taking_instructions(
    managers: list[KgManager],
    notes: list[str],
    config: Config,
    inputs: list[tuple[str, Sample]],
    outputs: list[dict],
) -> dict:
    kgs = "\n".join(format_kg(manager) for manager in managers)

    formatted = []
    for i, ((kg, sample), output) in enumerate(zip(inputs, outputs)):
        messages = output["messages"]
        assert messages[1]["role"] == "user"
        question = messages[1]["content"]

        gt = prepare_sparql(kg, sample.sparql, managers, config)

        content = f"""\
Question {i + 1} over {kg} knowledge graph:
{question}

System output:
{format_output(output)}

Ground truth:
{gt}"""

        formatted.append(content)

    outputs_formatted = "\n\n".join(formatted)

    content = f"""\
Add to, delete from, or update the following notes \
based on the provided questions and outputs below.

Knowledge graph specific notes:
{kgs}

{format_general_notes(notes)}

{outputs_formatted}"""

    return {"role": "user", "content": content}


def prepare_sparql(
    kg: str,
    sparql: str,
    managers: list[KgManager],
    config: Config,
) -> str:
    manager, others = partition_by(managers, lambda m: m.kg == kg)
    assert len(manager) == 1, (
        f"Expected exactly one manager for kg {kg}, got {len(manager)}"
    )
    manager = manager[0]

    try:
        result, sparql = execute_sparql(
            manager,
            others,
            sparql,
            config.result_max_rows,
            config.result_max_columns,
            set(),
            return_sparql=True,
        )
        sparql = manager.prettify(sparql)
    except Exception as e:
        result = f"Failed to execute SPARQL query:\n{e}"

    try:
        _, items = get_sparql_items(sparql, manager)
        selections = selections_from_items(items)
        selections = manager.format_selections(selections)
    except Exception as e:
        selections = f"Failed to determine used entities and properties:\n{e}"

    fmt = f"SPARQL query:\n{sparql.strip()}"

    if selections:
        fmt += f"\n\n{selections}"

    fmt += f"\n\nExecution result:\n{result.strip()}"
    return fmt


def take_notes(
    inputs: list[tuple[str, Sample]],
    outputs: list[dict],
    managers: list[KgManager],
    notes: list[str],
    config: Adapt,
    logger: Logger,
) -> None:
    api_messages = [
        get_note_taking_system_message(),
        get_note_taking_instructions(managers, notes, config, inputs, outputs),
    ]

    for msg in api_messages:
        logger.debug(format_message(msg))

    functions = get_note_functions(managers)

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
        except litellm.exceptions.Timeout:
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

            msg = {
                "role": "tool call",
                "content": format_function_call(fn_name, fn_args),
            }
            logger.debug(format_message(msg))

            try:
                result = call_function(managers, notes, fn_name, fn_args)
            except Exception as e:
                result = f"Call to function {fn_name} returned an error:\n{e}"

            tool_msg = {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            logger.debug(format_message(tool_msg))
            api_messages.append(tool_msg)

            if fn_name == "stop":
                return


def link(src: str, dst: str) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.lexists(dst):
        os.remove(dst)

    rel = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel, dst)


def adapt(
    task: str,
    config: Adapt,
    out_dir: str,
    log_level: str | int | None = None,
) -> None:
    assert config.method == "iterative_note_taking", (
        "Only iterative_note_taking method is supported for adaptation"
    )

    logger = get_logger("GRASP ADAPTATION", log_level)

    notes = load_general_notes(config.notes_file)

    managers, example_indices, notes = setup(config)

    assert config.seed is not None, "Seed must be set for adaptation"

    inputs: list[tuple[str, Sample]] = []
    for ipt in config.input:
        samples = [(ipt.kg, Sample(**sample)) for sample in load_jsonl(ipt.file)]
        if config.samples_per_file is not None:
            random.seed(config.seed)
            random.shuffle(samples)
            samples = samples[: config.samples_per_file]
        inputs.extend(samples)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    out_file = os.path.join(out_dir, "notes.general.json")

    for r in trange(config.num_rounds, desc="Adapting GRASP to KGs"):
        random.seed(config.seed + r)
        if inputs:
            samples = random.sample(inputs, min(config.samples_per_round, len(inputs)))
        else:
            raise NotImplementedError

        outputs = []
        for kg, sample in tqdm(samples, desc=f"Round {r + 1} samples", leave=False):
            sel_managers, _ = partition_by(managers, lambda m: m.kg == kg)
            assert len(sel_managers) == 1, (
                f"Expected exactly one manager for kg {kg}, got {len(sel_managers)}"
            )

            functions = get_functions(
                sel_managers,
                task,
                config.fn_set,
                example_indices,
                config.num_examples,
                config.random_examples,
            )

            *_, output = generate(
                task,
                sample.question,
                config,
                sel_managers,
                notes,
                functions,
                example_indices,
            )
            outputs.append(output)

        take_notes(samples, outputs, managers, notes, config, logger)

        for manager in managers:
            out_file = os.path.join(out_dir, f"notes.{manager.kg}.round_{r}.json")
            dump_json(manager.notes, out_file, indent=2)
            link(out_file, os.path.join(out_dir, f"notes.{manager.kg}.json"))

        out_file = os.path.join(out_dir, f"notes.general.round_{r}.json")
        dump_json(notes, out_file, indent=2)
        link(out_file, os.path.join(out_dir, "notes.general.json"))
