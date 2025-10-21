import random
from typing import Any
from uuid import uuid4

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message, Response, ToolCall
from grasp.tasks.examples import ExampleIndex
from grasp.tasks.utils import Sample, format_sparql_result, prepare_sparql_result


class SparqlQaSample(Sample):
    question: str
    sparql: str
    paraphrases: list[str] = []
    info: dict[str, Any] = {}

    def input(self) -> str:
        return self.question

    def queries(self) -> list[str]:
        return [self.question] + self.paraphrases


class SparqlQaExampleIndex(ExampleIndex):
    sample_cls = SparqlQaSample


# similar examples should be at least have this cos sim
MIN_EXAMPLE_SCORE = 0.5


def functions(config: GraspConfig) -> list[dict]:
    example_indices = [
        kg.kg for kg in config.knowledge_graphs if kg.example_index is not None
    ]

    if not example_indices:
        return []

    example_kgs = list(example_indices)
    example_info = "\n".join(example_kgs)

    if config.random_examples:
        fn = {
            "name": "find_examples",
            "description": f"""\
Find examples of SPARQL-question-pairs over the specified knowledge graph. \
At most {config.num_examples} examples are returned. The examples may help you \
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
try to answer a similar question to the one provided. At most {config.num_examples} \
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
    examples: list[SparqlQaSample],
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
    example_indices: dict[str, SparqlQaExampleIndex],
    kg: str,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> str:
    if kg not in example_indices:
        return f"No example index for knowledge graph {kg}"

    example_index = example_indices[kg]
    examples = random.sample(
        example_index.samples,
        min(num_examples, len(example_index)),
    )

    return format_examples(
        kg,
        managers,
        examples,  # type: ignore
        known,
        max_rows,
        max_cols,
    )


def find_similar_examples(
    managers: list[KgManager],
    example_indices: dict[str, SparqlQaExampleIndex],
    kg: str,
    question: str,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> str:
    if kg not in example_indices:
        return f"No example index for knowledge graph {kg}"

    example_index = example_indices[kg]

    examples = example_index.find_matches(
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
    example_indices: dict[str, SparqlQaExampleIndex],
    kg: str,
    question: str,
    num_examples: int,
    random_examples: bool,
    known: set[str],
    max_rows: int,
    max_cols: int,
) -> Message:
    if random_examples:
        tool_result = find_random_examples(
            managers,
            example_indices,
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
            example_indices,
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

    response_id = uuid4().hex
    tool_call_id = uuid4().hex
    return Message(
        role="assistant",
        content=Response(
            id=response_id,
            message=content,
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    name=fn_name,
                    args=fn_args,
                    result=tool_result,
                )
            ],
        ),
    )
