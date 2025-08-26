import json
import random
from uuid import uuid4

from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import Sample

# similar examples should be at least have this cos sim
MIN_EXAMPLE_SCORE = 0.5


def functions(
    managers: list[KgManager],
    num_examples: int = 3,
    random_examples: bool = False,
) -> list[dict]:
    example_kgs = [
        manager.kg for manager in managers if manager.example_index is not None
    ]
    if not example_kgs:
        return []

    example_info = "\n".join(example_kgs)

    if random_examples:
        fn = {
            "name": "find_examples",
            "description": f"""\
Find examples of SPARQL-question-pairs over the specified knowledge graph. \
At most {num_examples} examples are returned. The examples may help you \
with generating your own SPARQL query.

For example, to find examples of SPARQL-question-pairs over Wikidata, do the following:
find_examples(kg="wikidata")

Currently, examples are available for the following knowledge graphs:
{example_info}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": example_kgs,
                        "description": "The knowledge graph to find examples for",
                    },
                },
                "required": ["kg"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    else:
        fn = {
            "name": "find_similar_examples",
            "description": f"""\
Find SPARQL-question-pairs over the specified knowledge graph that \
try to answer a similar question to the one provided. At most {num_examples} \
examples are returned. The examples may help you with generating \
your own SPARQL query.

For example, to find similar SPARQL-question-pairs to the question \
"What is the capital of France?" over Wikidata, do the following:
find_similar_examples(kg="wikidata", question="What is the capital of France?")

Currently, examples are available for the following knowledge graphs:
{example_info}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": example_kgs,
                        "description": "The knowledge graph to find examples for",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to find examples for",
                    },
                },
                "required": ["kg", "question"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    return [fn]


def format_examples(
    kg: str,
    managers: list[KgManager],
    examples: list[Sample],
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> str:
    exs = []
    for example in examples:
        try:
            sparql, selections, result = prepare_sparql_result(
                example.sparql,
                kg,
                managers,
                max_rows,
                max_cols,
                known,
            )
        except Exception:
            continue

        exs.append(
            f"Question:\n{example.question}\n\n{format_sparql_result(sparql, kg, selections, result)}"
        )

    if not exs:
        return "No examples found"

    return "\n\n".join(f"Example {i + 1}:\n{ex}" for i, ex in enumerate(exs))


def find_random_examples(
    managers: list[KgManager],
    kg: str,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> str:
    manager, _ = find_manager(managers, kg)

    if manager.example_index is None:
        return f"No example index for knowledge graph {manager.kg}"

    examples = random.sample(
        manager.example_index.samples,
        min(num_examples, len(manager.example_index)),
    )

    return format_examples(
        kg,
        managers,
        examples,
        known,
        max_rows,
        max_cols,
    )


def find_similar_examples(
    managers: list[KgManager],
    kg: str,
    question: str,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> str:
    manager, _ = find_manager(managers, kg)

    if manager.example_index is None:
        # should not happen, but handle anyway
        return f"No example index for knowledge graph {manager.kg}"

    examples = manager.example_index.find_matches(
        question,
        num_examples,
        min_score=MIN_EXAMPLE_SCORE,
    )

    return format_examples(
        kg,
        managers,
        examples,
        known,
        max_rows,
        max_cols,
    )


def find_examples(
    managers: list[KgManager],
    kg: str,
    question: str,
    num_examples: int,
    random_examples: bool,
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> list[dict]:
    if random_examples:
        tool_result = find_random_examples(
            managers,
            kg,
            num_examples,
            known,
            max_rows,
            max_cols,
        )
        fn_name = "find_examples"
        fn_args = {"kg": kg}
        content = "Let's start by looking at some examples."

    else:
        tool_result = find_similar_examples(
            managers,
            kg,
            question,
            num_examples,
            known,
            max_rows,
            max_cols,
        )
        fn_name = "find_similar_examples"
        fn_args = {"kg": kg, "question": question}
        content = "Let's start by looking at some similar examples."

    tool_call_id = uuid4().hex
    return [
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
