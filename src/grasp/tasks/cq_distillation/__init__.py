from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.cq_distillation.functions import (
    Proposal,
    format_proposal,
    function_definitions,
    show_proposals,
    submit_proposal,
)
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import format_list, format_section


class CompetencyQuestion(BaseModel):
    sparql: str
    question: str | None = None
    id: str | None = None
    kg: str | None = None


class DistilledPair(BaseModel):
    question: str
    sparql: str
    id: str | None = None
    intent: str | None = None


class CqDistillationState(BaseModel):
    cq: CompetencyQuestion
    # previously distilled pair derived from the cq, if any
    pair: DistilledPair | None = None
    # pre-rendered traces of the student agent attempting the pair, if any
    traces: list[str] = []
    # recent proposals given for context (deduplication and diversification);
    # proposals submitted during the run are appended after them
    proposals: list[Proposal] = []
    num_given: int = 0

    def submitted(self) -> list[Proposal]:
        return self.proposals[self.num_given :]


def rules() -> list[str]:
    return [
        "Never submit a proposal whose SPARQL query you have not "
        "successfully executed yourself first.",
        "The reference SPARQL query is the ground truth the student is "
        "trained against; its correctness matters more than its elegance.",
        "Judge difficulty from the student's perspective: how many entities "
        "and properties need to be found, how complex the query structure "
        "is, and how much knowledge-graph-specific knowledge is required, "
        "not how long the query text is.",
        "Aim for variants the student solves ROUGHLY HALF THE TIME. Its "
        "traces state the execution F1 it reached, so treat a variant it "
        "scores 0 on in every attempt as still too hard, however much simpler "
        "it looks than its source, and one it solves every time as too easy. "
        "Only a variant it sometimes succeeds and sometimes fails at teaches "
        "it anything; 'easier' is a direction, this is the destination. Your "
        "proposals are given to the student immediately, and ones it cannot "
        "solve at all are discarded, so aim at this level deliberately rather "
        "than easing by a token amount each time.",
        "Diversify proposals across entities, domains, and SPARQL features; "
        "avoid near-duplicates of previous proposals.",
        "Questions must read as a real user would phrase them and must not "
        "mention SPARQL syntax or gratuitously expose implementation details.",
        "Adjust difficulty by simplifying the reference query, NOT by naming "
        "the entities and properties the question is about. The student has "
        "search functions for those and learning to wield them is most of the "
        "skill being trained, so an IRI handed over teaches nothing that "
        "transfers to the next question and reads like nothing a real user "
        "would ask.",
        "You MAY explain a knowledge-graph construction pattern the student "
        "cannot reasonably discover on its own -- how normalized quantity "
        "values are reached, that qualifiers hang off statement nodes, an "
        "endpoint-specific service or function. State such a pattern "
        "GENERICALLY, with placeholders rather than the question's own IRIs "
        "(e.g. \"normalized quantities are reached via "
        "p:<prop>/psn:<prop>/wikibase:quantityAmount\"), so the student still "
        "has to find the properties itself. Showing the shape of a solution "
        "is teaching; filling it in for the student is not.",
        "Phrase questions so that the expected result is uniquely "
        "determined: name the information the answer should contain "
        "(e.g. names, heights, counts) instead of leaving it implicit, and "
        "avoid formulations that allow several reasonable interpretations.",
    ]


def system_information(config: GraspConfig) -> str:
    task_kwargs = config.task_kwargs.get("cq-distillation", {})
    max_proposals = task_kwargs.get("max_proposals", 3)
    return f"""\
You are a knowledge graph and SPARQL expert creating training tasks for a \
smaller knowledge graph question answering agent, called the student.

You are given a competency question (CQ): a reference SPARQL query, \
optionally with a natural language question, exemplifying a capability that \
is important for working with a particular knowledge graph. Optionally, you \
are also given a question-SPARQL pair previously distilled from this CQ, \
traces of the student attempting to solve it, and a list of recent proposals.

Your task is to propose up to {max_proposals} new question-SPARQL pairs \
derived from the CQ. Each pair consists of a natural language question, as a \
real user would ask it, and a reference SPARQL query answering it. The \
student is trained by solving your questions and being rewarded based on \
comparing its query results against your reference query results, so the \
question must be self-contained, unambiguous, and answerable solely from the \
knowledge graph, and the reference query must be correct and return a \
non-empty, sensible result.

You should follow a step-by-step approach:
1. Analyze the CQ and, if given, the previous pair and the student traces. \
If traces are given, judge how the student did: if it failed or struggled, \
propose easier variants that isolate what it got wrong; if it \
succeeded with ease, propose harder variants. Without traces, propose fresh variations of \
the CQ; if no pair was distilled from the CQ yet, the first proposal should \
typically be a faithful natural language phrasing of the CQ query itself.
2. Check previous proposals to avoid duplicates and to find \
underrepresented angles, e.g. other entities, domains, or query features.
3. Explore the knowledge graph with the provided functions to ground new \
entities and properties, and to verify your assumptions.
4. Draft the SPARQL query and execute it. Iterate until you are satisfied \
with the query and its result, then formulate the matching question.
5. Submit the pair with submit_proposal. It verifies the query by execution \
and rejects broken, empty, or duplicate proposals; fix and resubmit if \
rejected.
6. Repeat until you have submitted roughly {max_proposals} proposals for \
this round, then stop."""


def output(state: CqDistillationState) -> dict:
    submitted = state.submitted()
    formatted = format_list([format_proposal(p) for p in submitted])

    return {
        "type": "output",
        "proposals": [p.model_dump() for p in submitted],
        "formatted": formatted or "No proposals",
    }


class CqDistillationTask(GraspTask):
    name = "cq-distillation"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        return function_definitions(kgs)

    def resolve_kg(self, kg: str | None) -> str:
        if kg is not None:
            return kg
        assert len(self.managers) == 1, (
            "Knowledge graph must be specified if multiple are available"
        )
        return self.managers[0].kg

    def format_sparql(self, sparql: str, kg: str | None) -> str:
        kg = self.resolve_kg(kg)
        result, selections = prepare_sparql_result(
            sparql,
            kg,
            self.managers,
            self.config.result_max_rows,
            self.config.result_max_columns,
            request_timeout=self.config.sparql_request_timeout,
            read_timeout=self.config.sparql_read_timeout,
            sparql_result_max_rows=self.config.sparql_result_max_rows,
        )
        manager, *_ = (m for m in self.managers if m.kg == kg)
        return format_sparql_result(manager, result, selections)

    def setup(self, input: Any) -> str:
        assert isinstance(input, CqDistillationState), (
            "Input for cq-distillation must be a CqDistillationState"
        )
        self.state = input
        self.state.num_given = len(self.state.proposals)

        cq = self.state.cq
        cq_fmt = ""
        if cq.question is not None:
            cq_fmt += f"Question: {cq.question}\n\n"
        cq_fmt += self.format_sparql(cq.sparql, cq.kg)
        sections = [format_section("Competency question", cq_fmt, 3)]

        pair = self.state.pair
        if pair is not None:
            pair_fmt = f"Question: {pair.question}\n\n"
            if pair.intent is not None:
                pair_fmt += f"Intent: {pair.intent}\n\n"
            pair_fmt += self.format_sparql(pair.sparql, cq.kg)
            sections.append(
                format_section("Previously distilled pair", pair_fmt, 3)
            )

        for i, trace in enumerate(self.state.traces):
            sections.append(
                format_section(f"Student attempt {i + 1} at the pair", trace, 3)
            )

        if self.state.num_given > 0:
            recent = [
                format_proposal(p)
                for p in reversed(self.state.proposals[-self.config.list_k :])
            ]
            sections.append(
                format_section(
                    "Recent proposals, most recent first",
                    format_list(recent),
                    3,
                )
            )

        sections.append("Start proposing question-SPARQL pairs.")
        return "\n\n".join(sections)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert self.state is not None, (
            "State must be provided for cq-distillation task"
        )

        if fn_name == "stop":
            return "Stopping"

        elif fn_name == "submit_proposal":
            pair = self.state.pair
            return submit_proposal(
                self.state.proposals,
                self.state.num_given,
                self.managers,
                fn_args["kg"],
                fn_args["question"],
                fn_args["sparql"],
                fn_args["intent"],
                fn_args["difficulty"],
                self.state.cq.id,
                pair.id if pair is not None else None,
                self.config.result_max_rows,
                self.config.result_max_columns,
                self.config.sparql_request_timeout,
                self.config.sparql_read_timeout,
                self.config.sparql_result_max_rows,
            )

        elif fn_name == "show_proposals":
            return show_proposals(
                self.state.proposals,
                fn_args["page"],
                self.config.list_k,
            )

        raise ValueError(f"Unknown function {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
