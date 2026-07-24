from pydantic import BaseModel

from grasp.functions import execute_sparql
from grasp.manager import KgManager
from grasp.sparql.types import AskResult, SelectResult
from grasp.utils import FunctionCallException, clip, format_enumerate

DIFFICULTIES = ["easier", "similar", "harder"]


class Proposal(BaseModel):
    kg: str
    question: str
    sparql: str
    intent: str
    difficulty: str
    # backlink to the root competency question this proposal derives from
    cq_id: str | None = None
    # immediate predecessor pair, None for first-generation proposals
    parent_id: str | None = None
    # size of the verified execution result
    result_size: int | None = None


def function_definitions(kgs: list[str]) -> list[dict]:
    return [
        {
            "name": "submit_proposal",
            "description": "Submit a question-SPARQL pair derived from the "
            "competency question. The SPARQL query is verified by execution "
            "and the proposal is rejected if the query fails, returns an "
            "empty result, or the question duplicates a previous proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph the proposal is "
                        "intended for",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question as a real user would "
                        "phrase it; self-contained and answerable solely "
                        "from the knowledge graph",
                    },
                    "sparql": {
                        "type": "string",
                        "description": "The reference SPARQL query answering "
                        "the question",
                    },
                    "intent": {
                        "type": "string",
                        "description": "A short description of what this "
                        "proposal covers and how it differs from previous "
                        "proposals",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": DIFFICULTIES,
                        "description": "Estimated difficulty for the student "
                        "agent relative to the source pair (or the "
                        "competency question if no pair is given)",
                    },
                },
                "required": ["kg", "question", "sparql", "intent", "difficulty"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_proposals",
            "description": "Show previous proposals with their intents, most "
            "recent first. Use this to avoid duplicates and to diversify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-indexed) for paginating "
                        "results (default should be 1)",
                    },
                },
                "required": ["page"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "stop",
            "description": "Stop the distillation process.",
        },
    ]


def format_proposal(proposal: Proposal) -> str:
    return (
        f'[{proposal.kg}] "{proposal.question}" '
        f"(intent: {clip(proposal.intent, 128)}, "
        f"difficulty: {proposal.difficulty})"
    )


def submit_proposal(
    proposals: list[Proposal],
    num_given: int,
    managers: list[KgManager],
    kg: str,
    question: str,
    sparql: str,
    intent: str,
    difficulty: str,
    cq_id: str | None,
    parent_id: str | None,
    max_rows: int,
    max_columns: int,
    request_timeout: float | tuple[float, float],
    read_timeout: float,
    sparql_result_max_rows: int | None,
) -> str:
    question = question.strip()
    if not question:
        raise FunctionCallException("Question must not be empty")

    intent = intent.strip()
    if not intent:
        raise FunctionCallException("Intent must not be empty")

    if difficulty not in DIFFICULTIES:
        raise FunctionCallException(
            f"Difficulty must be one of {', '.join(DIFFICULTIES)}"
        )

    if any(question.lower() == p.question.lower() for p in proposals):
        raise FunctionCallException(
            "Rejected: an identical question was already proposed. "
            "Check previous proposals with show_proposals and diversify."
        )

    try:
        result = execute_sparql(
            managers,
            kg,
            sparql,
            max_rows,
            max_columns,
            request_timeout=request_timeout,
            read_timeout=read_timeout,
            sparql_result_max_rows=sparql_result_max_rows,
        )
    except Exception as e:
        raise FunctionCallException(
            f"Rejected: verifying the SPARQL query failed:\n{e}"
        )

    if result.result is None:
        raise FunctionCallException(
            f"Rejected: verifying the SPARQL query failed:\n{result.formatted}"
        )

    if isinstance(result.result, AskResult):
        raise FunctionCallException(
            "Rejected: boolean (ASK) queries are not used as training tasks; "
            "their yes/no result is trivially gameable and a poor reward signal. "
            "Propose a query whose result is a set of answers."
        )

    if isinstance(result.result, SelectResult) and len(result.result) == 0:
        raise FunctionCallException(
            "Rejected: the SPARQL query returns an empty result. Reference "
            "queries must return a non-empty result."
        )

    if isinstance(result.result, SelectResult):
        result_size = len(result.result)
    else:
        result_size = 1

    proposals.append(
        Proposal(
            kg=kg,
            question=question,
            # store the prefix-fixed query as returned by execution
            sparql=result.sparql,
            intent=intent,
            difficulty=difficulty,
            cq_id=cq_id,
            parent_id=parent_id,
            result_size=result_size,
        )
    )
    return (
        f"Accepted proposal {len(proposals) - num_given}: {clip(question, 64)}\n\n"
        f"{result.formatted}"
    )


def show_proposals(
    proposals: list[Proposal],
    page: int,
    k: int,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    if not proposals:
        return "None"

    total_pages = (len(proposals) + k - 1) // k
    if page > total_pages:
        raise FunctionCallException(f"Page number exceeds maximum page {total_pages}")

    # most recent first
    ordered = [format_proposal(p) for p in reversed(proposals)]

    start = (page - 1) * k
    end = page * k
    page_items = ordered[start:end]

    header = f"Most recent proposals (page {page} of {total_pages}):\n"
    return header + format_enumerate(page_items, start=start)
