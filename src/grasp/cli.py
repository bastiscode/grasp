import argparse
import json
import os
import random
import time
from copy import deepcopy
from logging import Logger
from typing import Any, Iterator
from uuid import uuid4

import litellm
from fastapi import WebSocketDisconnect
from litellm import completion
from pydantic import BaseModel, conlist
from search_index.similarity import SimilarityIndex
from termcolor import colored
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import extract_field, partition_by

from grasp.configs import Config
from grasp.functions import (
    MIN_SCORE,
    call_function,
    execute_sparql,
    find_examples,
    find_similar_examples,
    get_feedback_functions,
    get_functions,
)
from grasp.sparql.data import get_sparql_items, selections_from_items
from grasp.sparql.manager import KgManager, load_kg_manager
from grasp.utils import get_answer_or_cancel, is_invalid_model_output

MAX_FEEDBACKS = 2
MAX_MESSAGES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="grasp",
        description="GRASP: Generic Reasoning and SPARQL generation across Knowledge Graphs",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["sparql-qa", "general-qa"],
        default="sparql-qa",
        help="Task to perform, either 'sparql-qa' for SPARQL question answering or "
        "'general-qa' for general knowledge graph assisted question answering (ignored"
        " if --serve is used)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--serve",
        type=int,
        default=None,
        help="Start a WebSocket server on this port",
    )
    input_group.add_argument(
        "-q",
        "--question",
        type=str,
        help="Question to answer",
    )
    input_group.add_argument(
        "-f",
        "--file",
        type=str,
        help="File containing questions to translate to SPARQL",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="File to write the output to",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the input questions",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Number of questions to take",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="question",
        help="Field to extract as question from the input JSON object",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed questions (only used with --file and --output-file)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level for the logger",
    )
    parser.add_argument(
        "--all-loggers",
        action="store_true",
        help="Enable logging for all loggers, not only the GRASP-specific ones",
    )
    return parser.parse_args()


def general_rules() -> list[str]:
    return [
        "Explain your thought process before and after each step \
and function call.",
        "Do not assume or make up any entity or property identifiers \
without verifying them with the knowledge graphs and functions.",
    ]


def task_rules(task: str) -> list[str]:
    if task == "sparql-qa":
        return [
            "The SPARQL query should always return the actual \
identifiers / IRIs of the items in its result. It additionally may return \
labels or other human-readable information, but they are optional and should be \
put within optional clauses unless explicitly requested by the user.",
            "Do not stop early if there are still obvious improvements to be made \
to the SPARQL query. For example, keep refining your SPARQL query if its result \
contains irrelevant items or is missing items you expected.",
            "Do not perform additional computation (e.g. filtering, sorting, calculations) \
on the result of the SPARQL query to determine the answer. All computation should \
be done solely within SPARQL.",
            'For questions with a "True" or "False" answer the SPARQL query \
should be an ASK query.',
            "Do not use 'SERVICE wikibase:label { bd:serviceParam wikibase:language ...' \
in SPARQL queries. It is not SPARQL standard and unsupported by the used QLever \
SPARQL endpoints. Use rdfs:label or similar properties to get labels instead.",
        ]

    elif task == "general-qa":
        return [
            "Your answers should be based on the information available in the \
knowledge graphs. If you do not need them to answer the question, e.g. if \
you know the answer by heart, still try to verify it with the knowledge graphs.",
            "Do not use 'SERVICE wikibase:label { bd:serviceParam wikibase:language ...' \
in SPARQL queries. It is not SPARQL standard and unsupported by the used QLever \
SPARQL endpoints. Use rdfs:label or similar properties to get labels instead.",
        ]

    else:
        raise ValueError(f"Unknown task {task}")


def get_system_message(task: str, managers: list[KgManager]) -> dict:
    prefixes = {}
    for manager in managers:
        prefixes.update(manager.prefixes)

    prefixes = "\n".join(f"{short}: {long}" for short, long in prefixes.items())

    kgs = "\n".join(f"{manager.kg} at {manager.endpoint}" for manager in managers)

    rules = "\n".join(
        f"{i + 1}. {rule}" for i, rule in enumerate(general_rules() + task_rules(task))
    )

    if task == "sparql-qa":
        content = f"""\
You are a question answering assistant. Your job is to generate a SPARQL query \
to answer a given user question.

You have access to the following knowledge graphs:
{kgs}

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

    return {
        "role": "system",
        "content": content,
    }


def format_message(message: dict) -> str:
    role = message["role"].upper()

    content = ""

    if message.get("reasoning_content"):
        content += f"Reasoning:\n{message['reasoning_content'].strip()}\n\n"

    content += message.get("content", "No content").strip()

    header = colored(role, "blue")
    return f"{header}\n{content}"


def format_function_call(fn_name: str, fn_args: dict) -> str:
    fn_name = colored(fn_name, "green")
    fn_args_str = colored(json.dumps(fn_args, indent=2), "yellow")
    return f"{fn_name}({fn_args_str})"


def write_jsonl_file(items: list, file: str) -> None:
    with open(file, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def run(args: argparse.Namespace) -> None:
    logger = get_logger("GRASP", args.log_level)
    config = Config(**load_config(args.config))

    if config.force_examples:
        assert len(config.knowledge_graphs) == 1, (
            "Forcing examples only works with a single knowledge graph"
        )

    example_indices = {}
    for kg_config in config.knowledge_graphs:
        if kg_config.example_index is None:
            continue

        example_data = os.path.join(kg_config.example_index, "data.tsv")
        example_index = SimilarityIndex.load(example_data, kg_config.example_index)
        example_indices[kg_config.name] = example_index

    managers = [
        load_kg_manager(**kg.model_dump(exclude={"example_index"}))
        for kg in config.knowledge_graphs
    ]

    functions = get_functions(
        args.task,
        config.fn_set,
        managers,
        example_indices,
        config.num_examples,
        config.random_examples,
    )
    system_message = get_system_message(args.task, managers)

    outputs = []
    if args.file is not None:
        random.seed(args.seed)
        assert args.output_file is not None, "Output file is required with --file"

        inputs = load_jsonl(args.file)
        if args.shuffle:
            random.shuffle(inputs)

        take = args.take or len(inputs)
        inputs = inputs[:take]

        exists = os.path.exists(args.output_file)
        if exists and not args.overwrite:
            outputs = load_jsonl(args.output_file)

        # save this in a config file
        output_stem, _ = os.path.splitext(args.output_file)
        config_file = output_stem + ".config.json"
        with open(config_file, "w") as f:
            json.dump(
                {
                    "config": config.model_dump(),
                    "functions": functions,
                    "system_message": system_message,
                },
                f,
                indent=2,
            )

    else:
        inputs = [{"id": 0, args.input_field: args.question}]

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id")
        question = extract_field(ipt, args.input_field)
        assert id is not None and question is not None, "id and question are required"

        if i < len(outputs):
            output = outputs[i]
            if not args.retry_failed or not is_invalid_model_output(output):
                continue

        *_, output = generate(
            args.task,
            question,
            config,
            managers,
            example_indices,
            functions,
            logger=logger,
        )

        output["id"] = id
        if args.question:
            print(json.dumps(output))
            return

        if i < len(outputs):
            outputs[i] = output
        else:
            outputs.append(output)

        write_jsonl_file(outputs, args.output_file)


def get_sparql_qa_feedback_system_message(managers: list[KgManager]) -> dict:
    kgs = "\n".join(f"{manager.kg} at {manager.endpoint}" for manager in managers)

    rules = "\n".join(
        f"{i + 1}. {rule}" for i, rule in enumerate(task_rules("sparql-qa"))
    )
    return {
        "role": "system",
        "content": f"""\
You are a question answering assistant providing feedback on the \
output of a SPARQL-based question answering system for a given user question.

The system has access to the following knowledge graphs:
{kgs}

There are two possible cases:

1) The system was able to find an answer
You are given the final SPARQL query, the knowledge graph it has to be executed \
against, and a human-readable answer to the question. You are also given some \
additional information about the SPARQL query, like the entities and properties \
it uses, and its execution result.

2) The system failed to find an answer
You are given the system's explanation for why it failed to find an answer. \
Optionally, you are provided with the system's best attempt at a SPARQL query \
so far including the same additional information as in case 1.

In any case, make sure that the following rules for SPARQL queries are followed:
{rules}

Provide your feedback with the give_feedback function.""",
    }


def get_sparql_qa_feedback_prompt(
    managers: list[KgManager],
    question: str,
    answer: dict | None,
    cancel: dict | None,
    max_rows: int,
    max_columns: int,
) -> str:
    sparql = None
    kg = None

    prompt = f"Question:\n{question.strip()}"

    if answer is not None:
        prompt += f"""

1) The system was able to find an answer

Answer:
{answer["answer"].strip()}"""
        sparql = answer["sparql"]
        kg = answer["kg"]

    else:
        assert cancel is not None, "No answer or cancel info provided"
        prompt += f"""

2) The system failed to find an answer

Explanation:
{cancel["explanation"].strip()}"""
        best_attempt = cancel.get("best_attempt")
        if best_attempt:
            sparql = best_attempt.get("sparql")
            kg = best_attempt.get("kg")

    if sparql is not None:
        if kg is None:
            # can happen if answer is parsed from message
            # use the first manager's kg
            kg = managers[0].kg

        managers, others = partition_by(managers, lambda m: m.kg == kg)
        if len(managers) != 1:
            raise ValueError(f"Unknown knowledge graph {kg}")

        manager = managers[0]
        try:
            result = execute_sparql(
                manager,
                others,
                sparql,
                max_rows,
                max_columns,
                known=set(),
            )
        except Exception as e:
            result = f"Failed to execute SPARQL query:\n{e}"

        try:
            _, items = get_sparql_items(sparql, manager)
            selections = selections_from_items(items)
            selections = manager.format_selections(selections)
        except Exception as e:
            selections = f"Failed to determine used entities and properties:\n{e}"

        prompt += f"""

Knowledge graph:
{kg}

SPARQL query:
{sparql}"""

        if selections:
            prompt += f"\n\n{selections}"

        assert isinstance(result, str)
        prompt += f"\n\nExecution result:\n{result.strip()}"

    else:
        prompt += "\n\nNo SPARQL query provided"

    return prompt


def generate_sparql_qa_feedback(
    config: Config,
    managers: list[KgManager],
    question: str,
    answer: dict | None,
    cancel: dict | None,
    logger: Logger = get_logger("GRASP FEEDBACK GENERATION"),
) -> dict | None:
    # give feedback to answer or last message from generate_sparql

    api_messages: list[dict] = [
        get_sparql_qa_feedback_system_message(managers),
        {
            "role": "user",
            "content": get_sparql_qa_feedback_prompt(
                managers,
                question,
                answer,
                cancel,
                config.result_max_rows,
                config.result_max_columns,
            ),
        },
    ]
    for msg in api_messages:
        logger.debug(format_message(msg))

    fns = get_feedback_functions()

    try:
        response = completion(
            model=config.model,
            messages=api_messages,
            tools=[{"type": "function", "function": fn} for fn in fns],
            parallel_tool_calls=False,
            tool_choice="auto",
            # decoding parameters
            temperature=config.temperature,
            top_p=config.top_p,
            reasoning_effort=config.reasoning_effort,  # type: ignore
            # should be set to more than enough until the next function call
            max_completion_tokens=config.max_completion_tokens,
            base_url=config.model_endpoint,
            api_key=config.api_key,
            # drop unsupported parameters
            drop_params=True,
        )
    except litellm.exceptions.Timeout:
        logger.error("LLM API timed out during feedback generation")
        return None

    choice = response.choices[0]  # type: ignore
    msg = choice.message.model_dump(exclude_none=True)  # type: ignore
    logger.debug(format_message(msg))

    try:
        assert len(choice.message.tool_calls) == 1, "No tool call found"  # type: ignore
        tool_call = choice.message.tool_calls[0]  # type: ignore
        assert tool_call.type == "function", "Tool call is not a function call"
        fn_name = tool_call.function.name
        assert fn_name == "give_feedback", "Feedback function not called"
        fn_args = json.loads(tool_call.function.arguments)
        msg = {
            "role": "tool call",
            "content": format_function_call(fn_name, fn_args),
        }
        logger.debug(format_message(msg))
        return fn_args
    except Exception as e:
        logger.debug(f"Failed to parse feedback:\n{e}")
        return None


def format_feedback(feedback: dict) -> str:
    status = feedback["status"]
    return f"Feedback (status={status}):\n{feedback['feedback']}"


def has_content(message: dict[str, Any]) -> bool:
    return bool(message.get("content") or message.get("reasoning_content"))


def generate(
    task: str,
    question: str,
    config: Config,
    managers: list[KgManager],
    example_indices: dict[str, SimilarityIndex],
    functions: list[dict],
    past_messages: list[dict] | None = None,
    logger: Logger = get_logger("GRASP SPARQL GENERATION"),
) -> Iterator[dict]:
    if task == "general-qa":
        # check that certain config options are not set
        assert not config.feedback, "Feedback is not supported for general QA"
        assert not config.force_examples, (
            "Forced examples are not supported for general QA"
        )
        assert not example_indices, "Examples are not supported for general QA"

    start = time.perf_counter()

    known = set()
    retries = 0

    if past_messages:
        api_messages = deepcopy(past_messages)
    else:
        api_messages = [get_system_message(task, managers)]

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
                min_score=MIN_SCORE,
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

    while len(api_messages) < MAX_MESSAGES:
        try:
            response = completion(
                model=config.model,
                messages=api_messages,
                tools=[{"type": "function", "function": fn} for fn in functions],  # type: ignore
                parallel_tool_calls=False,
                tool_choice="auto",
                # decoding parameters
                temperature=config.temperature,
                top_p=config.top_p,
                reasoning_effort=config.reasoning_effort,  # type: ignore
                # should be more than enough until the next function call
                max_completion_tokens=config.max_completion_tokens,
                base_url=config.model_endpoint,
                api_key=config.api_key,
                timeout=config.completion_timeout,
                # drop unsupported parameters
                drop_params=True,
            )  # type: ignore
        except litellm.exceptions.Timeout:
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
                    example_indices,
                    fn_name,
                    fn_args,
                    config.fn_set,
                    known,
                    result_max_rows=config.result_max_rows,
                    result_max_columns=config.result_max_columns,
                    list_k=config.list_k,
                    search_top_k=config.search_top_k,
                    num_examples=config.num_examples,
                    know_before_use=config.know_before_use,
                )
            except Exception as e:
                result = f"Failed to call function {fn_name}:\n{e}"
                # useful for debugging
                # import traceback
                #
                # logger.debug(traceback.format_exc())

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
                question,
                answer,
                cancel,
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
        yield {"typ": "user", "content": msg["content"]}

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
            "messages": api_messages,
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
        "messages": api_messages,
    }


# keep track of connections and limit to 10 concurrent connections
active_connections = 0
MAX_CONNECTIONS = 10
# maximum duration for a query in seconds
MAX_QUERY_DURATION = 120.0


class Request(BaseModel):
    task: str
    question: str
    knowledge_graphs: conlist(str, min_length=1)  # type: ignore
    past_messages: conlist(dict, min_length=1) | None = None  # type: ignore


def serve(args: argparse.Namespace) -> None:
    # create a fast api websocket server to serve the generate_sparql function
    import uvicorn
    from fastapi import FastAPI, WebSocket

    app = FastAPI()
    logger = get_logger("GRASP SERVER", args.log_level)

    # add cors
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = Config(**load_config(args.config))

    example_indices = {}
    for kg_config in config.knowledge_graphs:
        if kg_config.example_index is None:
            continue

        example_data = os.path.join(kg_config.example_index, "data.tsv")
        example_index = SimilarityIndex.load(example_data, kg_config.example_index)
        example_indices[kg_config.name] = example_index

    managers = [
        load_kg_manager(**kg.model_dump(exclude={"example_index"}))
        for kg in config.knowledge_graphs
    ]
    kgs = [manager.kg for manager in managers]

    @app.get("/knowledge_graphs")
    async def _knowledge_graphs():
        return kgs

    @app.get("/config")
    async def _config():
        return config.model_dump()

    @app.websocket("/live")
    async def _live(websocket: WebSocket):
        global active_connections

        # Check if we've reached the maximum number of connections
        if active_connections >= MAX_CONNECTIONS:
            await websocket.close(code=1008)  # HTTP Status 503: Service Unavailable
            return

        await websocket.accept()
        active_connections += 1

        try:
            while True:
                data = await websocket.receive_json()
                try:
                    request = Request(**data)
                except Exception:
                    await websocket.send_json({"error": "Invalid request format"})
                    continue

                sel = request.knowledge_graphs
                if not sel or not all(kg in kgs for kg in sel):
                    await websocket.send_json(
                        {"error": "Unsupported knowledge graph selection"}
                    )
                    continue

                sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)
                sel_example_indices = {
                    kg: example_indices[kg] for kg in sel if kg in example_indices
                }

                functions = get_functions(
                    request.task,
                    config.fn_set,
                    sel_managers,
                    sel_example_indices,
                    config.num_examples,
                    config.random_examples,
                )

                system_message = get_system_message(request.task, sel_managers)
                past_messages = []
                if request.past_messages is None:
                    past_messages.append(system_message)
                else:
                    # overwrite system message because new set of
                    # knowledge graphs might be present
                    past_messages = request.past_messages
                    past_messages[0] = system_message

                await websocket.send_json(
                    {
                        "typ": "system",
                        "functions": functions,
                        "system_message": system_message["content"],
                    }
                )
                await websocket.receive_json()

                # Setup generator
                generator = generate(
                    request.task,
                    request.question,
                    config,
                    sel_managers,
                    sel_example_indices,
                    functions,
                    past_messages,
                    logger=logger,
                )

                # Track start time for timeout
                start_time = time.perf_counter()

                # Process generator outputs with timeout check
                for output in generator:
                    # Check if we've exceeded the time limit
                    current_time = time.perf_counter()
                    if current_time - start_time > MAX_QUERY_DURATION:
                        # Send timeout message to client
                        await websocket.send_json(
                            {
                                "error": f"Operation timed out after {MAX_QUERY_DURATION} seconds",
                            }
                        )
                        break

                    # Process the output normally
                    await websocket.send_json(output)
                    data = await websocket.receive_json()

                    # Check if client requested cancellation
                    if data.get("cancel", False):
                        # Send cancellation confirmation to client
                        await websocket.send_json({"cancelled": True})
                        break

        except WebSocketDisconnect:
            pass

        except Exception as e:
            await websocket.send_json({"error": f"Failed to handle request:\n{e}"})

        finally:
            active_connections -= 1

    uvicorn.run(app, host="0.0.0.0", port=args.serve)


def main():
    args = parse_args()
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.serve is None:
        run(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
