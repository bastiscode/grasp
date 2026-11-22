import re
from collections import Counter
from dataclasses import dataclass
from logging import Logger
from typing import Any

from pydantic import BaseModel, Field

from grasp.configs import GraspConfig
from grasp.functions import execute_sparql, query_iris
from grasp.manager import KgManager, find_manager
from grasp.model import FunctionDefinition, Message, ToolCallResult
from grasp.sparql.types import AskResult, SelectResult
from grasp.sparql.utils import (
    find,
    find_all,
    parse_into_binding,
    parse_prefixes,
    parse_string,
    parse_to_string,
    query_type,
)
from grasp.tasks.base import GraspTask
from grasp.tasks.construct.checks import (
    Report,
    Source,
    Vocabulary,
    check_output,
    format_report,
    schema_terms,
    source_profile,
)
from grasp.tasks.construct.sample import SampleConfig
from grasp.tasks.construct.scratch import (
    GRAPHS,
    INVALID_IRI,
    SAMPLE_GRAPH,
    Scratch,
    Triple,
    construct,
    draft_graph,
    invalid_iris,
    parse_ntriples,
)
from grasp.utils import FunctionCallException, clip, format_list

NAME = re.compile(r"[A-Za-z0-9_-]+")

MAX_EXAMPLES = 3

# namespaces of rdf, rdfs, owl and xsd, their terms are not looked up in
# the other knowledge graphs
CORE = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
}

# stands in for a property to test its effect on the output of a query
ABLATED = f"<{GRAPHS}ablated>"


class ConstructConfig(BaseModel):
    # task_kwargs of construct: the knowledge graph to remodel and the
    # QLever server for the sample and the output of the queries on it
    kg: str
    scratch_endpoint: str
    sample: SampleConfig = SampleConfig()
    # questions the output should be able to answer, checked with the feedback
    competency_questions: list[str] = []
    # for a query on the sample and on the full knowledge graph, in seconds
    timeout: float = 60.0
    full_timeout: float = 600.0


def construct_config(config: GraspConfig) -> ConstructConfig:
    return ConstructConfig(**config.task_kwargs.get("construct", {}))


class TestConstructArgs(BaseModel):
    name: str = Field(
        description="Short name of the query, with letters, digits, - and _ only"
    )
    sparql: str = Field(description="The SPARQL CONSTRUCT query")


class NameArgs(BaseModel):
    name: str = Field(description="Name of the saved query")


class ExecuteOutputArgs(BaseModel):
    sparql: str = Field(description="The SPARQL SELECT or ASK query to execute")


class Explanation(BaseModel):
    property: str = Field(description="Full IRI of the property")
    explanation: str = Field(
        description="Why its values are not in the output, or how they are changed"
    )


class SubmitConstructArgs(BaseModel):
    explanations: list[Explanation] = Field(
        description="Explanations for the properties of the knowledge graph that "
        "check_output reports as errors, empty if there are none"
    )


def execute_output_definition() -> FunctionDefinition:
    return FunctionDefinition(
        "execute_output",
        """\
Execute a SPARQL SELECT or ASK query on the output of the saved CONSTRUCT \
queries, as produced by their last tests on the sample of the knowledge graph. \
The outputs of all queries form one graph.

Use it to verify the output as a whole, for example whether links between \
entities from different queries resolve, how many entities of a class there \
are, or which values a property has.""",
        ExecuteOutputArgs,
    )


def functions() -> list[FunctionDefinition]:
    return [
        FunctionDefinition(
            "test_construct",
            """\
Run a SPARQL CONSTRUCT query on the sample of the knowledge graph and save it \
under the given name together with its output. Testing a query under an \
existing name replaces the query and its output. A query that fails or \
produces no triples is not saved.

Returns a summary of the produced triples, the UNION branches that match \
nothing on the sample, and the link properties of the knowledge graph that the \
query mentions without effect on its output. The query has to run on the full \
knowledge graph later, so it must not have a LIMIT, and it cannot have FROM \
or GRAPH clauses.""",
            TestConstructArgs,
        ),
        execute_output_definition(),
        FunctionDefinition(
            "check_output",
            """\
Check the output of all saved queries against the sample of the knowledge \
graph. Returns errors that prevent submitting, warnings, the properties of the \
knowledge graph whose values are not all in the output, and the classes and \
properties of the output.""",
        ),
        FunctionDefinition(
            "list_constructs",
            "List the saved queries with the size of their outputs.",
        ),
        FunctionDefinition(
            "get_construct",
            "Get the text of a saved query.",
            NameArgs,
        ),
        FunctionDefinition(
            "remove_construct",
            "Remove a saved query and its output.",
            NameArgs,
        ),
        FunctionDefinition(
            "submit_construct",
            """\
Submit all saved queries as the final result. This is only possible if \
check_output reports no errors. A property of the knowledge graph without \
any value in the output, or with links that no query uses, is an error, unless \
you explain it here, for example because its values are converted to another \
format. A property is only redundant if all of its information is in the \
output in another way, verify that before you explain it as such.""",
            SubmitConstructArgs,
        ),
    ]


def system_information(kg: str, others: list[str]) -> str:
    info = f"""\
You are an RDF modeling assistant. Your task is to remodel the knowledge graph \
"{kg}" as a well-formed and semantically meaningful knowledge graph, by writing \
one or more SPARQL CONSTRUCT queries that produce the new graph from it. The \
knowledge graph is typically a direct mapping of structured data to RDF, e.g. \
with one class per table, one entity per row and one property per column. \
Where it already models something well, keep or refine that. Together, the \
queries should carry over all of its information. The form it has there is \
not to be kept: datatypes, names, keys and references given as strings often \
stem from the format of the original data and not from what the data means.

You should follow a step-by-step approach:
1. Explore the knowledge graph with the available functions. Find out which \
classes and properties exist, what the values look like, how missing values \
are marked, and how entities refer to each other.
2. Design the new model: which entities exist and what their classes are, \
how their IRIs are built from stable identifiers in a namespace that fits \
the domain, which properties they have, which values are links to other \
entities, and which values are entities of their own.
3. Write the queries and test each one with test_construct. Tests run on a \
sample of the knowledge graph, which has some instances of every class \
together with what they link to. Inspect the output of all saved queries with \
execute_output and check_output, and fix what you find.
4. Submit the saved queries with submit_construct once they are complete. \
They are then run on the full knowledge graph."""
    if others:
        info += f"""

The other knowledge graphs ({", ".join(others)}) provide existing vocabulary \
for the new model. Before you define a class or property, search them for one \
with the same meaning, and use it under its IRI wherever its definition, \
domain and range fit. Define your own classes and properties only for what \
they do not cover, and relate these to existing ones where possible, e.g. \
with rdfs:subClassOf or rdfs:subPropertyOf. The CONSTRUCT queries themselves \
only read from {kg}."""
    return info


def rules() -> list[str]:
    return [
        "Give every entity an rdf:type and an rdfs:label, but never make the "
        "label a precondition for its other triples.",
        "Where a value can be missing for an entity, produce it independently, "
        "e.g. with OPTIONAL, so that it does not suppress the other triples of "
        "the entity. Never output a marker for a missing value as a value.",
        "Model references to other entities as links to their IRIs, not as "
        "literals. A value that names a thing which several entities share or "
        "which can have properties of its own is such an entity, also if the "
        "knowledge graph only has it as a string.",
        "Name classes and properties after what they mean, so that a triple "
        "reads correctly from subject to object. Verify the name and direction "
        "of each property on some of its triples in the output.",
        "Build the IRIs of an entity type in exactly the same way in every "
        "query, and use the same properties and namespace for the same concept "
        "across queries, so that the outputs combine into one graph. Encode "
        "values within IRIs, e.g. with ENCODE_FOR_URI.",
        "Give literals the datatype that fits what they mean, which can "
        "differ from the one in the knowledge graph, e.g. xsd:integer for a "
        "count, xsd:gYear for a year or xsd:date for a date. Keep information "
        "that only the name of a property carries, like a unit, currency or "
        "scale.",
        "Describe the classes and properties that you define yourself, at "
        "least with an rdfs:label each. A query with an empty WHERE clause "
        "can produce such triples.",
        "Do not rely on specific instances being part of the sample, queries "
        "have to work on the full knowledge graph.",
        "After feedback, change a query by testing its new version under the "
        "same name, remove queries that are no longer needed, and submit again.",
    ]


@dataclass
class Draft:
    sparql: str
    triples: list[Triple]
    # prefixes and IRIs of the query
    prefixes: dict[str, str]
    iris: set[str]
    # link properties of the sample with an effect on the output, with
    # the number of triples that depend on them
    effects: dict[str, int]
    # UNION branches without matches on the sample
    empty: list[str]


def prepare_query(manager: KgManager, sparql: str) -> Draft:
    # checks that the query can run unchanged on the sample and on the
    # full knowledge graph
    try:
        sparql = manager.fix_prefixes(sparql)
        parse, _ = parse_string(sparql, manager.sparql_parser)
    except Exception as e:
        raise FunctionCallException(f"Failed to parse the query:\n{e}") from e

    if query_type(sparql, manager.sparql_parser) != "construct":
        raise FunctionCallException("The query is not a CONSTRUCT query")

    query = find(parse, "ConstructQuery")
    assert query is not None
    limit = None
    for child in query["children"]:
        if child["name"] == "DatasetClauseOptional" and child.get("children"):
            raise FunctionCallException("The query must not have FROM clauses")
        elif child["name"] == "SolutionModifier":
            limit = find(child, "LimitOffsetClausesOptional")
    if limit is not None and limit.get("children"):
        raise FunctionCallException(
            "The query must not have a LIMIT or OFFSET, it has to cover all "
            "of the knowledge graph"
        )
    if find(parse, "GraphGraphPattern") is not None:
        raise FunctionCallException("The query must not have GRAPH clauses")

    return Draft(
        sparql,
        [],
        parse_prefixes(find(parse, "Prologue")),
        query_iris(manager, sparql),
        {},
        [],
    )


def without_property(manager: KgManager, sparql: str, prop: str) -> str:
    # the query with the property replaced by one that matches nothing
    parse, _ = parse_string(sparql, manager.sparql_parser)
    prefixes = {**manager.prefixes, **parse_prefixes(find(parse, "Prologue"))}
    for node in find_all(parse, {"IRIREF", "PNAME_LN"}, skip={"Prologue"}):
        binding = parse_into_binding(
            node["value"],
            manager.iri_literal_parser,
            prefixes,
        )
        if binding is not None and binding.identifier() == prop:
            node["value"] = ABLATED
    return parse_to_string(parse)


def union_branches(manager: KgManager, sparql: str) -> list[str]:
    # each branch as an ASK query
    parse, _ = parse_string(sparql, manager.sparql_parser)
    prologue = find(parse, "Prologue")
    prefix = "" if prologue is None else parse_to_string(prologue)
    branches = []

    def collect(node: dict, in_union: bool) -> None:
        children = node.get("children", [])
        if node["name"] != "GroupOrUnionGraphPattern":
            for child in children:
                collect(child, False)
            return

        # the rest of a union is the last child
        is_union = len(children) > 1
        if is_union or in_union:
            branches.append(f"{prefix} ASK {parse_to_string(children[0])}")
        collect(children[0], False)
        if is_union:
            collect(children[-1], True)

    collect(parse, False)
    return branches


def namespace(iri: str) -> str:
    return iri[: max(iri.rfind("#"), iri.rfind("/")) + 1]


def defines(manager: KgManager, iris: list[str], timeout: float) -> set[str]:
    # the IRIs the knowledge graph says something about
    result = manager.execute_sparql(
        "SELECT DISTINCT ?s WHERE { VALUES ?s { "
        + " ".join(f"<{iri}>" for iri in iris)
        + " } ?s ?p ?o }",
        request_timeout=(6.0, timeout),
        read_timeout=timeout,
    )
    assert isinstance(result, SelectResult)
    return {row["s"].value for row in result.rows()}


def has_namespace(manager: KgManager, prefix: str, timeout: float) -> bool:
    result = manager.execute_sparql(
        f'ASK {{ ?s ?p ?o FILTER(STRSTARTS(STR(?s), "{prefix}")) }}',
        request_timeout=(6.0, timeout),
        read_timeout=timeout,
    )
    assert isinstance(result, AskResult)
    return result.boolean


def output_manager(
    scratch: Scratch,
    manager: KgManager,
    queries: dict[str, str],
) -> KgManager:
    # for queries over the outputs of the given queries on the sample,
    # with the prefixes of the queries
    prefixes = dict(manager.kg_prefixes)
    for sparql in queries.values():
        parse, _ = parse_string(sparql, manager.sparql_parser)
        prefixes.update(parse_prefixes(find(parse, "Prologue")))

    return scratch.manager(
        "output",
        [draft_graph(name) for name in queries],
        prefixes,
    )


def execute_output(
    output: KgManager,
    config: GraspConfig,
    fn_args: dict,
) -> ToolCallResult:
    sparql = ExecuteOutputArgs.model_validate(fn_args).sparql
    if query_type(sparql, output.sparql_parser) not in ("select", "ask"):
        raise FunctionCallException("Only SELECT and ASK queries are supported")

    result = execute_sparql(
        [output],
        output.kg,
        sparql,
        config.result_max_rows,
        config.result_max_columns,
        request_timeout=config.sparql_request_timeout,
        read_timeout=config.sparql_read_timeout,
        sparql_result_max_rows=config.sparql_result_max_rows,
    )
    return result.tool_result()


def format_triples(name: str, triples: list[Triple]) -> str:
    subjects = {subj for subj, _, _ in triples}
    counts = Counter(prop for _, prop, _ in triples)
    examples: dict[str, list[str]] = {}
    for subj, prop, obj in triples:
        prop_examples = examples.setdefault(prop, [])
        if len(prop_examples) < MAX_EXAMPLES:
            prop_examples.append(f"{subj.sparql()} {obj.sparql()}")

    lines = [
        f"<{prop}>: {count:,} triples, e.g. " + " | ".join(examples[prop])
        for prop, count in sorted(counts.items())
    ]
    return f"""\
Saved query {name}. On the sample it produces {len(triples):,} triples for \
{len(subjects):,} subjects, by property with subject and object of some triples:
{format_list(lines)}"""


class ConstructTask(GraspTask):
    name = "construct"
    feedback_task = "construct-feedback"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.construct_config = construct_config(self.config)
        self.manager = find_manager(self.managers, self.construct_config.kg)
        self.scratch = Scratch(
            self.construct_config.scratch_endpoint,
            self.construct_config.timeout,
        )
        self.others = [manager for manager in self.managers if manager != self.manager]
        # classes and properties of the output that the other knowledge
        # graphs define, and which namespaces they have, as looked up so far
        self.defined: dict[str, str] = {}
        self.namespaces: dict[tuple[str, str], bool] = {}
        self.source: Source | None = None
        self.drafts: dict[str, Draft] = {}
        self.output_manager: KgManager | None = None
        # of the last accepted submission
        self.submitted = False
        self.explanations: dict[str, str] = {}
        self.report: Report | None = None

    def system_information(self) -> str:
        others = [manager.kg for manager in self.others]
        return system_information(self.manager.kg, others)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[FunctionDefinition]:
        return functions()

    def setup(self, input: Any) -> str:
        # starts without outputs of earlier runs
        graphs = self.scratch.graphs()
        if SAMPLE_GRAPH not in graphs:
            raise ValueError(
                f"No sample of {self.manager.kg} on the scratch server at "
                f"{self.scratch.endpoint}, build it first"
            )
        for graph in graphs:
            if graph != SAMPLE_GRAPH:
                self.scratch.drop(graph)

        sample = self.scratch.construct(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
            [SAMPLE_GRAPH],
        )
        self.source = source_profile(parse_ntriples(sample))
        classes = format_list(
            f"<{cls}>: {count:,}" for cls, count in sorted(self.source.classes.items())
        )
        return f"""\
{super().setup(input)}

Classes of the knowledge graph with their number of instances in the sample:
{classes}"""

    def check(self, explanations: dict[str, str]) -> Report:
        assert self.source is not None
        # as one graph, queries can produce the same triple
        triples = list(
            dict.fromkeys(
                triple for draft in self.drafts.values() for triple in draft.triples
            )
        )
        effects: dict[str, int] = {}
        for draft in self.drafts.values():
            for prop, dependent in draft.effects.items():
                effects[prop] = max(effects.get(prop, 0), dependent)
        empty = {
            name: draft.empty for name, draft in self.drafts.items() if draft.empty
        }
        return check_output(
            self.source,
            triples,
            effects,
            explanations,
            empty,
            self.vocabulary(triples),
        )

    def vocabulary(self, triples: list[Triple]) -> Vocabulary:
        # which classes and properties of the output, and of those they
        # are related to, the other knowledge graphs define
        timeout = self.construct_config.timeout
        terms = sorted(
            term
            for term in schema_terms(triples)
            if INVALID_IRI.search(term) is None and namespace(term) not in CORE
        )
        vocabulary = Vocabulary()
        new = [term for term in terms if term not in self.defined]
        for manager in self.others if new else []:
            try:
                found = defines(manager, new, timeout)
            except Exception:
                vocabulary.failed.append(manager.kg)
                continue
            for term in found:
                self.defined.setdefault(term, manager.kg)

        for term in terms:
            if term in self.defined:
                vocabulary.defined[term] = self.defined[term]
                continue

            prefix = namespace(term)
            for manager in self.others:
                key = (manager.kg, prefix)
                if key not in self.namespaces:
                    try:
                        self.namespaces[key] = has_namespace(manager, prefix, timeout)
                    except Exception:
                        vocabulary.failed.append(manager.kg)
                        continue
                if self.namespaces[key]:
                    vocabulary.undefined[term] = manager.kg
                    break
        return vocabulary

    def analyze(self, draft: Draft, output: str) -> list[str]:
        # finds parts of the query without effect on the sample, which a
        # query that only mentions a property does not show
        assert self.source is not None
        notes = []
        branches = union_branches(self.manager, draft.sparql)
        for branch in branches:
            if not self.scratch.ask(branch, [SAMPLE_GRAPH]):
                draft.empty.append(clip(branch[branch.index(" ASK ") + 5 :], 200))
        if draft.empty:
            notes.append(
                f"{len(draft.empty):,} of {len(branches):,} UNION branches match "
                f"nothing on the sample:\n{format_list(draft.empty)}"
            )

        lines = set(output.split("\n"))
        unused = []
        for prop in sorted(draft.iris):
            source = self.source.properties.get(prop)
            if source is None or not source.links:
                continue
            ablated = self.scratch.construct(
                without_property(self.manager, draft.sparql, prop),
                [SAMPLE_GRAPH],
            )
            without = set(ablated.split("\n"))
            if without != lines:
                draft.effects[prop] = len(lines - without)
            else:
                unused.append(f"<{prop}>")
        if unused:
            notes.append(
                "The output is the same without these link properties of the "
                "knowledge graph, the patterns they are in match nothing or do "
                f"not contribute:\n{format_list(unused)}"
            )
        return notes

    def get_draft(self, fn_args: dict) -> tuple[str, Draft]:
        name = NameArgs.model_validate(fn_args).name
        if name not in self.drafts:
            raise FunctionCallException(
                f"No saved query {name}, saved are: {', '.join(self.drafts) or 'none'}"
            )
        return name, self.drafts[name]

    def test_construct(self, fn_args: dict) -> str:
        args = TestConstructArgs.model_validate(fn_args)
        if NAME.fullmatch(args.name) is None:
            raise FunctionCallException(
                "The name can only have letters, digits, - and _"
            )

        draft = prepare_query(self.manager, args.sparql)
        try:
            text = self.scratch.construct(draft.sparql, [SAMPLE_GRAPH])
            draft.triples = parse_ntriples(text)
        except Exception as e:
            raise FunctionCallException(f"The query failed on the sample:\n{e}") from e

        if not draft.triples:
            raise FunctionCallException("The query produces no triples on the sample")

        invalid = invalid_iris(draft.triples)
        if invalid:
            raise FunctionCallException(f"""\
The query produces {len(invalid):,} invalid IRIs, encode the values they are \
built from, e.g. with ENCODE_FOR_URI:
{format_list(f"<{iri}>" for iri in invalid[:MAX_EXAMPLES])}""")

        try:
            notes = self.analyze(draft, text)
        except Exception as e:
            raise FunctionCallException(
                f"Failed to analyze the query on the sample:\n{e}"
            ) from e

        self.scratch.load(draft_graph(args.name), text)
        self.drafts[args.name] = draft
        self.output_manager = None
        self.submitted = False
        return "\n\n".join([format_triples(args.name, draft.triples), *notes])

    def submit_construct(self, fn_args: dict) -> str:
        self.submitted = False
        args = SubmitConstructArgs.model_validate(fn_args)
        if not self.drafts:
            raise FunctionCallException("There are no saved queries to submit")

        explanations = {
            explanation.property.strip("<>"): explanation.explanation
            for explanation in args.explanations
        }
        report = self.check(explanations)
        # nothing can be fixed in the last step
        if report.errors and not self.last_step:
            errors = [
                f"{error.message}: {', '.join(error.examples)}"
                for error in report.errors
            ]
            raise FunctionCallException(
                f"Not submitted because of errors:\n{format_list(errors)}"
            )

        self.submitted = True
        self.explanations = explanations
        self.report = report
        return f"Submitted {len(self.drafts):,} queries"

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str | ToolCallResult:
        if fn_name == "test_construct":
            return self.test_construct(fn_args)

        elif fn_name == "submit_construct":
            return self.submit_construct(fn_args)

        elif fn_name == "execute_output":
            if not self.drafts:
                raise FunctionCallException("There are no saved queries yet")
            if self.output_manager is None:
                self.output_manager = output_manager(
                    self.scratch,
                    self.manager,
                    {name: draft.sparql for name, draft in self.drafts.items()},
                )
            return execute_output(self.output_manager, self.config, fn_args)

        elif fn_name == "check_output":
            if not self.drafts:
                raise FunctionCallException("There are no saved queries yet")
            return format_report(self.check({}))

        elif fn_name == "list_constructs":
            if not self.drafts:
                return "No saved queries"
            return "Saved queries:\n" + format_list(
                f"{name}: {len(draft.triples):,} triples on the sample"
                for name, draft in self.drafts.items()
            )

        elif fn_name == "get_construct":
            name, draft = self.get_draft(fn_args)
            return f"Query {name}:\n```sparql\n{draft.sparql}\n```"

        elif fn_name == "remove_construct":
            name, _ = self.get_draft(fn_args)
            self.scratch.drop(draft_graph(name))
            self.drafts.pop(name)
            self.output_manager = None
            self.submitted = False
            return f"Removed query {name}"

        raise ValueError(f"Unknown function {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "submit_construct" and self.submitted

    def limit_message(self) -> str:
        return (
            "The step or time limit is reached. You have one last step, "
            "submit the saved queries now."
        )

    def limit_functions(self) -> list[str] | None:
        return ["submit_construct"]

    def output(self, messages: list[Message]) -> dict | None:
        if not self.drafts:
            return None

        queries = [
            {"name": name, "sparql": draft.sparql, "triples": len(draft.triples)}
            for name, draft in self.drafts.items()
        ]
        return {
            "type": "output",
            "kg": self.manager.kg,
            # the saved queries are kept if a run ends without submitting
            "submitted": self.submitted,
            "queries": queries,
            "explanations": self.explanations,
            "report": None if self.report is None else self.report.model_dump(),
            "formatted": format_queries(queries),
        }


def format_queries(queries: list[dict]) -> str:
    return "\n\n".join(
        f"Query {query['name']} ({query['triples']:,} triples on the sample):\n"
        f"```sparql\n{query['sparql']}\n```"
        for query in queries
    )


def materialize(
    manager: KgManager,
    queries: list[dict],
    path: str,
    timeout: float,
    logger: Logger,
) -> None:
    # runs the queries on the full knowledge graph, their outputs as
    # N-Triples in one file
    with open(path, "wb") as out:
        for query in queries:
            num_triples = 0
            response = construct(manager, query["sparql"], timeout, stream=True)
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
                num_triples += chunk.count(b"\n")
            logger.info(f"Query {query['name']} produced {num_triples:,} triples")
