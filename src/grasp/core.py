import json
import time
from copy import deepcopy
from logging import Logger
from typing import Any, Iterator

from litellm.exceptions import Timeout
from search_index.similarity import EmbeddingModel
from universal_ml_utils.io import load_json
from universal_ml_utils.logging import get_logger

from grasp.configs import GraspConfig
from grasp.functions import (
    call_function,
    kg_functions,
)
from grasp.manager import KgManager, find_embedding_model, format_kgs, load_kg_manager
from grasp.manager.utils import describe_index
from grasp.model import Message, Response, call_model
from grasp.tasks import rules as general_rules
from grasp.tasks import (
    task_done,
    task_functions,
    task_output,
    task_rules,
    task_setup,
    task_system_information,
)
from grasp.tasks.examples import ExampleIndex
from grasp.tasks.feedback import format_feedback, generate_feedback
from grasp.tasks.sparql_qa.examples import find_examples
from grasp.utils import (
    format_error,
    format_list,
    format_message,
    format_notes,
    format_prefixes,
    format_response,
)


def system_instructions(
    task: str,
    config: GraspConfig,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    index_types = set()
    prefixes = {}
    for manager in managers:
        prefixes.update(manager.prefixes)
        index_types.add(manager.entity_index.get_type())
        index_types.add(manager.property_index.get_type())

    index_infos = []
    for index_type in sorted(index_types):
        title, desc = describe_index(index_type)
        index_infos.append(f"{title}: {desc}")

    system_info = task_system_information(task, config)

    instructions = f"""\
{system_info}

Available knowledge graphs:
{format_kgs(managers, kg_notes)}

Index types used for entities and properties:
{format_list(index_infos)}

"""

    if notes:
        instructions += f"""\
General notes across knowledge graphs:
{format_notes(notes)}

"""

    instructions += f"""\
SPARQL prefixes for use in function calls:
{format_prefixes(prefixes)}

Additional rules to follow:
{format_list(general_rules() + task_rules(task))}"""

    return instructions


def setup(config: GraspConfig) -> list[KgManager]:
    emb_model: EmbeddingModel | None = None
    managers: list[KgManager] = []
    for kg in config.knowledge_graphs:
        if emb_model is None:
            # find and set embedding model
            emb_model = find_embedding_model(managers)

        manager = load_kg_manager(
            kg,
            entities_kwargs={"model": emb_model},
            properties_kwargs={"model": emb_model},
        )
        managers.append(manager)

    return managers


def load_notes(config: GraspConfig) -> tuple[list[str], dict[str, list[str]]]:
    # load notes
    if config.notes_file is None:
        general_notes = []
    else:
        general_notes = load_json(config.notes_file)

    kg_notes = {}
    for kg in config.knowledge_graphs:
        if kg.notes_file is None:
            kg_notes[kg.kg] = []
            continue

        kg_notes[kg.kg] = load_json(kg.notes_file)

    return general_notes, kg_notes  # type: ignore


def generate(
    task: str,
    input: Any,
    config: GraspConfig,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    example_indices: dict[str, ExampleIndex] | None = None,
    past_messages: list[Message] | None = None,
    past_known: set[str] | None = None,
    logger: Logger = get_logger("GRASP"),
) -> Iterator[dict]:
    if task != "sparql-qa":
        # disable examples for tasks other than sparql-qa
        # to avoid errors due to missing implementations
        config = deepcopy(config)
        config.force_examples = None
        logger.debug(f"Disabling examples for {task} task")
    if task == "cea":
        config = deepcopy(config)
        config.know_before_use = True
        logger.debug("Enabling know-before-use for CEA task")

    # setup functions
    fns = kg_functions(managers, config.fn_set)
    additional_fns = task_functions(managers, task, config)
    if additional_fns is not None:
        additional_fns, task_handler = additional_fns
        fns.extend(additional_fns)
    else:
        task_handler = None

    input, task_state = task_setup(task, input, managers, config)
    yield {"type": "input", "input": input}

    if notes is None:
        notes = []
    if kg_notes is None:
        kg_notes = {}

    # setup messages
    system_instruction = system_instructions(task, config, managers, kg_notes, notes)
    yield {
        "type": "system",
        "functions": fns,
        "system_message": system_instruction,
    }

    # log stuff
    config_msg = Message(
        role="config",
        content=config.model_dump_json(indent=2, exclude_none=True),
    )
    logger.debug(format_message(config_msg))

    fn_msg = Message(
        role="functions",
        content=json.dumps([fn["name"] for fn in fns]),
    )
    logger.debug(format_message(fn_msg))

    # handle past
    messages = [Message(role="system", content=system_instruction)]
    if past_messages:
        first, *past = past_messages
        assert first.role == "system", "First past message should be system"
        messages[0].content = first.content
        messages.extend(past)

    known = past_known or set()

    start = time.perf_counter()

    # add user input
    messages.append(Message(role="user", content=input))

    if config.force_examples and example_indices:
        try:
            example_message = find_examples(
                managers,
                example_indices,  # type: ignore
                config.force_examples,
                input,
                config.num_examples,
                config.random_examples,
                known,
                config.result_max_rows,
                config.result_max_columns,
            )

            # add to messages
            messages.append(example_message)

            # yield to user
            assert isinstance(example_message.content, Response)
            content = example_message.content
            yield {"type": "model", **content.get_content()}

            tool_call = content.tool_calls[0]
            yield {
                "type": "tool",
                "name": tool_call.name,
                "args": tool_call.args,
                "result": tool_call.result,
            }

        except Exception:
            logger.warning(
                f"{config.force_examples:=} specified but corresponding manager not found "
                "or without example index, ignoring"
            )

    # log all messages so far
    for msg in messages:
        logger.debug(format_message(msg))

    num_messages = len(messages)
    error: dict | None = None

    # keep track of last serialized message to detect loops
    # if model emits the same message twice, we are stuck
    last_resp_hash: str | None = None
    retries = 0
    while len(messages) - num_messages < config.max_steps:
        try:
            response = call_model(messages, fns, config)
        except Timeout:
            error = {
                "content": "LLM API timed out",
                "reason": "timeout",
            }
            logger.error("LLM API timed out")
            break
        except Exception as e:
            error = {
                "content": f"Failed to generate response:\n{e}",
                "reason": "api",
            }
            logger.error(format_error(**error))
            break

        if response.is_empty:
            error = {
                "content": "Empty response from LLM API",
                "reason": "empty",
            }
            logger.error(format_error(**error))
            break

        messages.append(Message(role="assistant", content=response))

        resp_hash = response.hash()
        if last_resp_hash == resp_hash:
            error = {
                "content": "LLM appears to be stuck in a loop",
                "reason": "loop",
            }
            logger.error(format_error(**error))
            break

        last_resp_hash = resp_hash

        # yield message if there is content
        if response.has_content:
            yield {"type": "model", **response.get_content()}

        # no tool calls mean we should stop
        should_stop = not response.tool_calls

        # execute tool calls
        for tool_call in response.tool_calls:
            try:
                result = call_function(
                    config,
                    managers,
                    tool_call.name,
                    tool_call.args,
                    known,
                    task_handler,
                    task_state,
                    example_indices,
                )
            except Exception as e:
                result = f"Call to function {tool_call.name} returned an error:\n{e}"

            tool_call.result = result

            yield {
                "type": "tool",
                "name": tool_call.name,
                "args": tool_call.args,
                "result": tool_call.result,
            }

            if task_done(task, tool_call.name):
                # we are done
                should_stop = True

        # only log now to show also tool calls results
        logger.debug(format_response(response))

        can_give_feedback = config.feedback and retries < config.max_feedbacks

        if should_stop and not can_give_feedback:
            # done
            break

        elif not should_stop:  # and (choice.message.tool_calls or alternating):
            # not done yet
            continue

        elif not can_give_feedback:
            # no feedback possible, despite answer or cancel
            break

        # get latest output
        output = task_output(task, messages, managers, config, task_state)
        if output is None:
            break

        # provide feedback
        try:
            inputs = [message.content for message in messages if message.role == "user"]
            feedback = generate_feedback(
                task,
                managers,
                config,
                kg_notes,
                notes,
                inputs,  # type: ignore
                output,
                logger,
            )
        except Exception as e:
            error = {
                "content": f"Failed to generate feedback:\n{e}",
                "reason": "feedback",
            }
            logger.error(format_error(**error))
            break

        if feedback is None:
            # no feedback
            break

        messages.append(Message(role="feedback", content=format_feedback(feedback)))
        yield {
            "type": "feedback",
            "status": feedback["status"],
            "feedback": feedback["feedback"],
        }

        if feedback["status"] == "done":
            break

        # if not done, continue
        retries += 1
        # reset loop detection
        last_resp_hash = None

    output = task_output(task, messages, managers, config, task_state)

    out_msg = Message(
        role="output",
        content="No output"
        if output is None
        else output.get("formatted", json.dumps(output)),
    )
    logger.info(format_message(out_msg))

    end = time.perf_counter()
    yield {
        "type": "output",
        "task": task,
        "output": output,
        "elapsed": end - start,
        "error": error,
        "messages": [message.model_dump(exclude_defaults=True) for message in messages],
        "known": list(known),
    }
