import json
import time
from copy import deepcopy
from logging import Logger
from typing import Iterator
from uuid import uuid4

import litellm
from search_index.similarity import EmbeddingModel, SimilarityIndex
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import partition_by

from grasp.configs import Config
from grasp.feedback import format_feedback, generate_sparql_qa_feedback
from grasp.functions import (
    MIN_EXAMPLE_SCORE,
    call_function,
    execute_sparql,
    find_examples,
    find_similar_examples,
)
from grasp.model import call_model
from grasp.notes import format_general_notes
from grasp.rules import general_rules, task_rules
from grasp.sparql.manager import (
    KgManager,
    load_example_index,
    load_general_notes,
    load_kg_manager,
)
from grasp.utils import (
    find_embedding_model,
    format_enumerate,
    format_function_call,
    format_message,
    get_answer_or_cancel,
)

MAX_FEEDBACKS = 2
MAX_MESSAGES = 200


def format_kg(manager: KgManager) -> str:
    msg = f"{manager.kg} at {manager.endpoint}"
    if not manager.notes:
        return msg + " without notes"

    msg += " with notes:\n" + format_enumerate(manager.notes)
    return msg


def get_system_message(task: str, managers: list[KgManager], notes: list[str]) -> dict:
    prefixes = {}
    for manager in managers:
        prefixes.update(manager.prefixes)

    prefixes = "\n".join(f"{short}: {long}" for short, long in prefixes.items())

    kgs = "\n".join(format_kg(manager) for manager in managers)

    rules = format_enumerate(general_rules() + task_rules(task))

    if task == "sparql-qa":
        content = f"""\
You are a question answering assistant. Your job is to generate a SPARQL query \
to answer a given user question.

You have access to the following knowledge graphs:
{kgs}

{format_general_notes(notes)}

You can use the following SPARQL prefixes implicitly in all functions:
{prefixes}

You should follow a step-by-step approach to generate the SPARQL query:
1. Determine possible entities and properties implied by the user question.
2. Search for the entities and properties in the knowledge graphs. Where \
applicable, constrain the searches with already identified entities and properties.
3. Gradually build up the SPARQL query using the identified entities \
and properties. Start with simple queries and add more complexity as needed. \
Execute intermediate queries to get feedback and to verify your assumptions. \
You may need to refine or rethink your current plan based on the query \
results and go back to step 2 if needed, possibly multiple times.
4. Use the answer or cancel function to finalize your answer and stop the \
generation process.

Additional rules:
{rules}"""

    elif task == "general-qa":
        content = f"""\
You are a question answering assistant. Your job is to answer a given user \
question using the knowledge graphs and functions available to you.

You have access to the following knowledge graphs:
{kgs}

{format_general_notes(notes)}

You can use the following SPARQL prefixes implicitly in all functions:
{prefixes}

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
4. Use the answer or cancel function to finalize your answer and stop the \
generation process.

Additional rules:
{rules}"""

    else:
        raise ValueError(f"Unknown task {task}")

    return {"role": "system", "content": content}


def setup(
    config: Config,
) -> tuple[list[KgManager], dict[str, SimilarityIndex], list[str]]:
    if config.force_examples:
        assert len(config.knowledge_graphs) == 1, (
            "Forcing examples only works with a single knowledge graph"
        )

    example_indices = {}
    for kg_config in config.knowledge_graphs:
        if kg_config.example_index is None:
            continue

        example_index = load_example_index(kg_config.example_index)
        example_indices[kg_config.kg] = example_index

    emb_model: EmbeddingModel | None = None
    managers: list[KgManager] = []
    for kg in config.knowledge_graphs:
        if emb_model is None:
            # find and set embedding model
            emb_model = find_embedding_model(managers)

        manager = load_kg_manager(
            **kg.model_dump(exclude={"example_index"}),
            # pass model as kwargs for the correct index types
            entities_kwargs={"model": emb_model},
            properties_kwargs={"model": emb_model},
        )
        managers.append(manager)

    notes = load_general_notes(config.notes_file)

    return managers, example_indices, notes


def has_content(message: dict) -> bool:
    return bool(message.get("content") or message.get("reasoning_content"))


def generate(
    task: str,
    question: str,
    config: Config,
    managers: list[KgManager],
    notes: list[str],
    functions: list[dict],
    example_indices: dict[str, SimilarityIndex] | None = None,
    past_questions: list[str] | None = None,
    past_messages: list[dict] | None = None,
    past_known: set[str] | None = None,
    logger: Logger = get_logger("GRASP SPARQL GENERATION"),
) -> Iterator[dict]:
    if task == "general-qa":
        # disable feedback
        config = deepcopy(config)
        config.feedback = False
        config.force_examples = False
        example_indices = {}
        logger.debug(f"Disabling feedback and examples for {task} task")

    elif example_indices is None:
        example_indices = {}

    start = time.perf_counter()

    if past_questions is None:
        questions = [question]
    else:
        questions = past_questions + [question]

    known = set() if past_known is None else deepcopy(past_known)

    if past_messages:
        api_messages = deepcopy(past_messages)
    else:
        api_messages = [get_system_message(task, managers, notes)]

    api_messages.append({"role": "user", "content": question})

    # log messages
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
                "content": json.dumps([fn["name"] for fn in functions]),
            }
        )
    )
    for msg in api_messages:
        logger.debug(format_message(msg))

    if config.force_examples:
        # add forced tool call
        fn_args = {"kg": managers[0].kg}
        if config.random_examples:
            tool_result = find_examples(
                managers[0],
                example_indices,
                config.num_examples,
                known,
            )
            fn_name = "find_examples"
            content = "Let's start by looking at some examples."

        else:
            tool_result = find_similar_examples(
                managers[0],
                example_indices,
                question,
                config.num_examples,
                known,
                min_score=MIN_EXAMPLE_SCORE,
            )
            fn_name = "find_similar_examples"
            fn_args["question"] = question
            content = "Let's start by looking at some similar examples."

        tool_call_id = uuid4().hex
        api_messages.extend(
            [
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": json.dumps(fn_args, indent=2),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call_id,
                },
            ]
        )

        logger.debug(format_message(api_messages[-2]))
        logger.debug(
            format_message(
                {"role": "tool call", "content": format_function_call(fn_name, fn_args)}
            )
        )
        logger.debug(format_message(api_messages[-1]))

        yield {
            "typ": "tool",
            "name": fn_name,
            "args": fn_args,
            "result": tool_result,
        }

    error: dict | None = None

    retries = 0
    while len(api_messages) < MAX_MESSAGES:
        try:
            response = call_model(
                api_messages,
                functions,
                config,
            )
        except litellm.exceptions.Timeout:
            error = {
                "content": "LLM API timed out during SPARQL generation",
                "reason": "timeout",
            }
            logger.error("LLM API timed out during SPARQL generation")
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

        if has_content(msg):
            # yield message
            content = ""

            if msg.get("reasoning_content"):
                content += f"Reasoning:\n{msg['reasoning_content'].strip()}\n\n"

            content += msg.get("content", "").strip()

            yield {"typ": "model", "content": content}

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

            msg = {
                "role": "tool call",
                "content": format_function_call(fn_name, fn_args),
            }
            logger.debug(format_message(msg))

            try:
                result = call_function(
                    managers,
                    fn_name,
                    fn_args,
                    config.fn_set,
                    known,
                    example_indices,
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
            logger.debug(format_message(tool_msg))
            api_messages.append(tool_msg)

            yield {
                "typ": "tool",
                "name": fn_name,
                "args": fn_args,
                "result": result,
            }

            if fn_name in ["answer", "cancel"]:
                # answer or cancel function means we are done
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

        # get latest answer and message
        answer, cancel = get_answer_or_cancel(task, api_messages)
        if answer is None and cancel is None:
            # need at least one of them for feedback
            break

        # provide feedback
        try:
            feedback = generate_sparql_qa_feedback(
                config,
                managers,
                questions,
                answer,
                cancel,
                task_rules(task),
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
            "typ": "feedback",
            "status": feedback["status"],
            "feedback": feedback["feedback"],
        }

        if feedback["status"] == "done":
            break

        # if not done, continue
        retries += 1

    answer, cancel = get_answer_or_cancel(task, api_messages)

    if task == "general-qa":
        ans: str | None = None
        content = "Failed to answer the question"
        if answer is not None:
            content = ans = answer["answer"]
        elif cancel is not None:
            content = cancel["explanation"]
            if cancel.get("best_attempt"):
                ans = cancel["best_attempt"]
                content += f"\n\nBest answer attempt:\n{ans}"

        logger.info(
            format_message(
                {
                    "role": "output",
                    "content": f"Question:\n{question}\n\nAnswer:\n{content}",
                }
            )
        )

        end = time.perf_counter()
        yield {
            "typ": "output",
            "task": task,
            "answer": answer,
            "content": content,
            "elapsed": end - start,
            "error": error,
            "questions": questions,
            "messages": api_messages,
            "known": list(known),
        }

        return

    assert task == "sparql-qa", "Unknown task, expected 'sparql-qa'"

    # setup some variables for the SPARQL qa output
    sparql = None
    kg = None
    endpoint = None
    result = None
    content = "Failed to answer the question"

    if answer is not None:
        sparql = answer["sparql"]
        kg = answer["kg"]
        content = answer["answer"]

    elif cancel is not None:
        content = cancel["explanation"]
        best_attempt = cancel.get("best_attempt")
        if best_attempt:
            sparql = best_attempt.get("sparql")
            kg = best_attempt.get("kg")

    if sparql is not None:
        if kg is None:
            # use the first manager's kg
            kg = managers[0].kg

        try:
            managers, others = partition_by(managers, lambda m: m.kg == kg)
            assert len(managers) == 1, (
                f"Unknown knowledge graph {kg} for final SPARQL query"
            )
            manager = managers[0]
            result, sparql = execute_sparql(
                manager,
                others,
                sparql,
                config.result_max_rows,
                config.result_max_columns,
                known,
                return_sparql=True,
            )
            sparql = manager.prettify(sparql)
            endpoint = manager.endpoint
        except Exception as e:
            logger.error(f"Failed to execute and prettify final SPARQL query:\n{e}")

    logger.info(
        format_message(
            {
                "role": "output",
                "content": f"Question:\n{question}\n\nSPARQL:\n{sparql}"
                f"\n\nResult:\n{result}\n\nContent:\n{content}",
            }
        )
    )

    end = time.perf_counter()
    yield {
        "typ": "output",
        "task": task,
        "sparql": sparql,
        "result": result,
        "endpoint": endpoint,
        "content": content,
        "elapsed": end - start,
        "error": error,
        "questions": questions,
        "messages": api_messages,
        "known": list(known),
    }
