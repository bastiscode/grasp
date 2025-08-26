import json
import time
from copy import deepcopy
from logging import Logger
from typing import Iterator

from litellm.exceptions import Timeout
from search_index.similarity import EmbeddingModel
from universal_ml_utils.logging import get_logger

from grasp.configs import Config
from grasp.functions import (
    call_function,
    kg_functions,
)
from grasp.manager import KgManager, find_embedding_model, format_kgs, load_kg_manager
from grasp.manager.utils import load_general_notes
from grasp.model import call_model
from grasp.tasks import (
    rules as general_rules,
)
from grasp.tasks import (
    task_done,
    task_functions,
    task_output,
    task_rules,
    task_state,
    task_system_information,
)
from grasp.tasks.feedback import format_feedback, generate_feedback
from grasp.tasks.sparql_qa_examples import find_examples
from grasp.utils import (
    format_list,
    format_message,
    format_notes,
    format_prefixes,
)

MAX_FEEDBACKS = 2
MAX_MESSAGES = 200


def system_instructions(task: str, managers: list[KgManager], notes: list[str]) -> str:
    prefixes = {}
    for manager in managers:
        prefixes.update(manager.prefixes)

    system_info = task_system_information(task)

    return f"""\
{system_info}

You have access to the following knowledge graphs:
{format_kgs(managers)}

You are provided with the following notes across all knowledge graphs:
{format_notes(notes)}

You can use the following SPARQL prefixes implicitly in all functions:
{format_prefixes(prefixes)}

You should follow these rules:
{format_list(general_rules() + task_rules(task))}"""


def setup(config: Config) -> tuple[list[KgManager], list[str]]:
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
            example_index_kwargs={"model": emb_model},
        )
        managers.append(manager)

    notes = load_general_notes(config.notes_file)

    return managers, notes


def generate(
    task: str,
    question: str,
    config: Config,
    managers: list[KgManager],
    notes: list[str],
    past_questions: list[str] | None = None,
    past_messages: list[dict] | None = None,
    past_known: set[str] | None = None,
    logger: Logger = get_logger("GRASP"),
) -> Iterator[dict]:
    if task != "sparql-qa":
        # disable feedback and examples for tasks other than sparql-qa
        # to avoid errors due to missing implementations
        config = deepcopy(config)
        config.feedback = False
        config.force_examples = None
        logger.debug(f"Disabling feedback and examples for {task} task")

    # setup functions
    fns = kg_functions(managers, config.fn_set)
    task_fns, task_handler = task_functions(managers, task)
    fns.extend(task_fns)

    state = task_state(task)

    # setup messages
    system_instruction = system_instructions(task, managers, notes)
    yield {
        "type": "system",
        "functions": fns,
        "system_message": system_instruction,
    }

    # log stuff
    logger.debug(
        format_message(
            {
                "role": "config",
                "content": config.model_dump_json(indent=2, exclude_none=True),
            }
        )
    )
    logger.debug(
        format_message(
            {
                "role": "functions",
                "content": json.dumps([fn["name"] for fn in fns]),
            }
        )
    )

    # handle past
    api_messages = [{"role": "system", "content": system_instruction}]
    if past_messages:
        # overwrite system message because new set of
        # knowledge graphs or functions might be present
        assert past_messages[0]["role"] == "system", (
            "First past message should be system"
        )

        api_messages.extend(past_messages[1:])

    questions = past_questions or []
    questions.append(question)
    known = past_known or set()

    start = time.perf_counter()

    # add user question
    api_messages.append({"role": "user", "content": question})

    if config.force_examples:
        try:
            example_messages = find_examples(
                managers,
                config.force_examples,
                question,
                config.num_examples,
                config.random_examples,
                known,
                config.result_max_rows,
                config.result_max_columns,
            )

            # add to messages
            api_messages.extend(example_messages)

            # yield to user
            content = example_messages[0]["content"]
            yield {"type": "model", "content": content}

            tool_call = example_messages[0]["tool_calls"][0]["function"]
            content = example_messages[1]["content"]
            yield {
                "type": "tool",
                "name": tool_call["name"],
                "args": json.loads(tool_call["arguments"]),
                "result": content,
            }

        except Exception:
            logger.warning(
                f"{config.force_examples:=} specified but corresponding manager not found "
                "or without example index, ignoring"
            )

    # log all messages so far
    for msg in api_messages:
        logger.debug(format_message(msg))

    error: dict | None = None

    retries = 0
    while len(api_messages) < MAX_MESSAGES:
        try:
            response = call_model(api_messages, fns, config)
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
            logger.error(format_message({"role": "error", **error}))
            break

        if not response.choices:  # type: ignore
            error = {
                "content": "No choices from LLM API",
                "reason": "no_choices",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        choice = response.choices[0]  # type: ignore
        usage = response.usage.model_dump(exclude_none=True)  # type: ignore

        msg = choice.message.model_dump(exclude_none=True)  # type: ignore
        api_messages.append(msg)

        # display usage info for assistant messages
        fmt_msg = deepcopy(msg)
        fmt_msg["role"] += f" (usage={usage})"
        logger.debug(format_message(fmt_msg))

        # yield message
        content = ""
        if msg.get("reasoning_content"):
            content += f"Reasoning:\n{msg['reasoning_content'].strip()}\n\n"
        content += msg.get("content", "").strip()
        if content:
            yield {"type": "model", "content": content}

        if choice.finish_reason not in ["tool_calls", "stop", "length"]:
            error = {
                "content": f"Unexpected finish reason {choice.finish_reason}",
                "reason": "invalid_finish_reason",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        elif choice.finish_reason == "length":
            break

        # no tool calls mean we should stop
        should_stop = not choice.message.tool_calls  # type: ignore

        # execute tool calls
        for tool_call in choice.message.tool_calls or []:  # type: ignore
            fn_name: str = tool_call.function.name  # type: ignore
            fn_args = json.loads(tool_call.function.arguments)

            try:
                result = call_function(
                    managers,
                    fn_name,
                    fn_args,
                    config.fn_set,
                    known,
                    task_handler,
                    result_max_rows=config.result_max_rows,
                    result_max_columns=config.result_max_columns,
                    list_k=config.list_k,
                    search_top_k=config.search_top_k,
                    num_examples=config.num_examples,
                    know_before_use=config.know_before_use,
                )
            except Exception as e:
                result = f"Call to function {fn_name} returned an error:\n{e}"

            tool_msg = {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            api_messages.append(tool_msg)
            logger.debug(format_message(tool_msg))

            yield {
                "type": "tool",
                "name": fn_name,
                "args": fn_args,
                "result": result,
            }

            if task_done(task, fn_name):
                # we are done
                should_stop = True
                break

        can_give_feedback = config.feedback and retries < MAX_FEEDBACKS

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
        output = task_output(task, api_messages, managers, config, state)
        if output is None:
            break

        # provide feedback
        try:
            feedback = generate_feedback(
                task,
                managers,
                config,
                notes,
                questions,
                output,
                logger,
            )
        except Exception as e:
            error = {
                "content": f"Failed to generate feedback:\n{e}",
                "reason": "feedback",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        if feedback is None:
            # no feedback
            break

        msg = {
            "role": "user",
            "content": format_feedback(feedback),
        }
        logger.debug(format_message(msg))
        api_messages.append(msg)
        yield {
            "type": "feedback",
            "status": feedback["status"],
            "feedback": feedback["feedback"],
        }

        if feedback["status"] == "done":
            break

        # if not done, continue
        retries += 1

    output = task_output(task, api_messages, managers, config, state)

    logger.info(
        format_message({"role": "output", "content": json.dumps(output, indent=2)})
    )

    end = time.perf_counter()
    yield {
        "type": "output",
        "task": task,
        "output": output,
        "elapsed": end - start,
        "error": error,
        "questions": questions,
        "messages": api_messages,
        "known": list(known),
    }
