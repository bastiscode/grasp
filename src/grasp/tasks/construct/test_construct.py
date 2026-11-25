import uuid

import pytest

import grasp.tasks.construct as construct
import grasp.tasks.construct.feedback as construct_feedback
from grasp.configs import GraspConfig, KgConfig, KgInfo
from grasp.core import generate
from grasp.manager import KgManager
from grasp.model import Model, Response, ToolCall
from grasp.sparql.types import AskResult, Binding, SelectResult
from grasp.tasks import get_task
from grasp.tasks.construct import (
    ABLATED,
    ConstructTask,
    prepare_query,
    union_branches,
    without_property,
)
from grasp.tasks.construct.checks import (
    Vocabulary,
    check_output,
    format_report,
    source_profile,
    value_keys,
)
from grasp.tasks.construct.feedback import ConstructFeedbackTask
from grasp.tasks.construct.scratch import (
    SAMPLE_GRAPH,
    draft_graph,
    invalid_iris,
    parse_ntriples,
)
from grasp.utils import FunctionCallException

SAMPLE = """\
<http://s/row/1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://s/City> .
<http://s/row/1> <http://s/name> "Freiburg" .
<http://s/row/1> <http://s/population> "230000.0"^^<http://www.w3.org/2001/XMLSchema#decimal> .
<http://s/row/1> <http://s/founded> "1120-01-01 00:00:00" .
<http://s/row/1> <http://s/country> <http://s/row/2> .
<http://s/row/2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://s/Country> .
<http://s/row/2> <http://s/name> "Germany" .
<http://s/City> <http://www.w3.org/2000/01/rdf-schema#label> "City table" .
"""

OUTPUT = """\
<http://o/city/Freiburg> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://o/City> .
<http://o/city/Freiburg> <http://www.w3.org/2000/01/rdf-schema#label> "Freiburg"@de .
<http://o/city/Freiburg> <http://o/population> "230000"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://o/city/Freiburg> <http://o/country> <http://o/country/Germany> .
"""

CITIES = """\
PREFIX o: <http://o/>
CONSTRUCT { ?city a o:City ; o:country ?country }
WHERE { ?row a <http://s/City> ; <http://s/country> ?other }"""


class FakeScratch:
    # keeps the graphs in memory, queries on the sample give a fixed output
    graphs_loaded: dict[str, str] = {}
    output = OUTPUT
    # whether UNION branches match and links have an effect on the output
    matches = True

    def __init__(self, endpoint: str, timeout: float = 60.0) -> None:
        self.endpoint = endpoint

    def graphs(self) -> list[str]:
        return sorted(self.graphs_loaded)

    def construct(self, sparql: str, graphs: list[str]) -> str:
        assert graphs == [SAMPLE_GRAPH]
        if sparql == "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }":
            return SAMPLE
        elif ABLATED in sparql and self.matches:
            return "".join(
                line
                for line in self.output.splitlines(keepends=True)
                if "<http://o/country>" not in line
            )
        return self.output

    def ask(self, sparql: str, graphs: list[str]) -> bool:
        assert graphs == [SAMPLE_GRAPH]
        return self.matches

    def load(self, graph: str, ntriples: str, replace: bool = True) -> None:
        self.graphs_loaded[graph] = ntriples

    def drop(self, graph: str) -> None:
        self.graphs_loaded.pop(graph, None)

    def manager(self, kg: str, graphs: list[str], prefixes=None) -> KgManager:
        return KgManager(
            kg,
            prefixes=prefixes,
            endpoint=self.endpoint,
            params={"default-graph-uri": graphs},
        )


@pytest.fixture
def scratch(monkeypatch):
    FakeScratch.graphs_loaded = {SAMPLE_GRAPH: SAMPLE, draft_graph("old"): OUTPUT}
    FakeScratch.output = OUTPUT
    FakeScratch.matches = True
    monkeypatch.setattr(construct, "Scratch", FakeScratch)
    monkeypatch.setattr(construct_feedback, "Scratch", FakeScratch)
    return FakeScratch


def config(**kwargs) -> GraspConfig:
    kg = KgConfig(
        kg="source",
        entities=None,
        properties=None,
        literals=None,
        classes=None,
        shapes=None,
        info=KgInfo(endpoint="http://localhost:1"),
    )
    task_kwargs = {
        "construct": {
            "kg": "source",
            "scratch_endpoint": "http://localhost:2",
            "competency_questions": ["Which cities are in Germany?"],
        }
    }
    return GraspConfig(knowledge_graphs=[kg], task_kwargs=task_kwargs, **kwargs)


def manager() -> KgManager:
    return KgManager("source", endpoint="http://localhost:1")


def call(task, fn_name: str, **args) -> str:
    result = task.call_function(fn_name, args, set(), None)
    return result if isinstance(result, str) else result.formatted


def output_of(task) -> dict:
    output = task.output([])
    assert output is not None
    return output


def test_parse_ntriples():
    text = (
        '<http://x/s> <http://x/p> "a \\"b\\"\\n\\u00e4" .\n'
        '_:b1 <http://x/p> "1"^^<http://x/int> .\n\n'
        '<http://x/s> <http://x/p> "hi"@en-US .\n'
        "<http://x/s> <http://x/q> <http://x/o> .\n"
    )
    assert parse_ntriples(text) == [
        (Binding("uri", "http://x/s"), "http://x/p", Binding("literal", 'a "b"\nä')),
        (
            Binding("bnode", "b1"),
            "http://x/p",
            Binding("literal", "1", datatype="http://x/int"),
        ),
        (
            Binding("uri", "http://x/s"),
            "http://x/p",
            Binding("literal", "hi", lang="en-US"),
        ),
        (Binding("uri", "http://x/s"), "http://x/q", Binding("uri", "http://x/o")),
    ]
    with pytest.raises(ValueError):
        parse_ntriples("<http://x/s> <http://x/p> .")


def test_invalid_iris():
    triples = parse_ntriples(
        "<http://x/a b> <http://x/p> <http://x/o> .\n"
        '<http://x/s> <http://x/p> "a b" .\n'
    )
    assert invalid_iris(triples) == ["http://x/a b"]


def test_value_keys():
    assert value_keys("1.0") & value_keys(" 1 ")
    assert not value_keys("1.0") & value_keys("10")
    assert value_keys("nan") == {"nan"}


def test_source_profile():
    source = source_profile(parse_ntriples(SAMPLE))
    assert source.classes == {"http://s/City": 1, "http://s/Country": 1}
    # without the triples about the classes
    assert set(source.properties) == {
        "http://s/name",
        "http://s/population",
        "http://s/founded",
        "http://s/country",
    }
    assert source.properties["http://s/name"].values == {"Freiburg", "Germany"}
    assert source.properties["http://s/country"].links == 1


def test_check_output():
    source = source_profile(parse_ntriples(SAMPLE))
    triples = parse_ntriples(OUTPUT)
    report = check_output(source, triples, {}, {}, {}, Vocabulary())
    coverage = {cov.iri: cov for cov in report.coverage}
    # the number in another format, one name as label and one in an IRI
    assert coverage["http://s/population"].matched == 1
    assert coverage["http://s/name"].matched == 2
    assert coverage["http://s/founded"].missing == ["1120-01-01 00:00:00"]
    # no value of the date and no use of the link
    [error] = report.errors
    assert error.examples == ["http://s/country", "http://s/founded"]
    assert report.output_classes == {"http://o/City": 1}

    warnings = [f.message for f in report.findings if f.level == "warning"]
    assert warnings == [
        "3 classes and properties without rdfs:label",
        "1 IRIs linked by <http://o/country> that no query describes "
        "(targets outside of the sample can be a reason)",
    ]

    explanations = {"http://s/founded": "Converted to xsd:date."}
    empty = {"cities": ["{ ?row a <http://s/Town> }"]}
    report = check_output(
        source,
        triples,
        {"http://s/country": 1},
        explanations,
        empty,
        Vocabulary(),
    )
    assert not report.errors
    formatted = format_report(report)
    assert "explanation: Converted to xsd:date." in formatted
    assert "1 links, used" not in formatted
    assert (
        "- 1 UNION branches of query cities match nothing on the sample: "
        "{ ?row a <http://s/Town> }"
    ) in formatted
    assert "(3 other properties have all values in the output" in formatted

    report = check_output(
        source,
        triples,
        {},
        {"http://s/nope": "Unknown."},
        {},
        Vocabulary(),
    )
    assert [error.examples for error in report.errors][0] == ["http://s/nope"]

    # fewer triples depend on the links than there are links
    source.properties["http://s/country"].triples = 5
    report = check_output(
        source,
        triples,
        {"http://s/country": 3},
        explanations,
        {},
        Vocabulary(),
    )
    formatted = format_report(report)
    assert (
        "- 1 properties of the knowledge graph have more triples in the "
        "sample than triples in the output of a query that depend on them, "
        "check that nothing is dropped (several of their triples leading to "
        "the same triple of the output is a harmless reason): "
        "<http://s/country> (3 for 5 triples)"
    ) in formatted
    assert (
        "- <http://s/country>: 1 links, used; 3 triples of the output depend "
        "on its 5 triples"
    ) in formatted


def test_output_findings():
    triples = parse_ntriples(
        "<http://o/a> <http://o/p> _:b .\n"
        '<http://o/a> <http://o/p> "1" .\n'
        '<http://o/p> <http://www.w3.org/2000/01/rdf-schema#label> "p" .\n'
    )
    report = check_output(source_profile([]), triples, {}, {}, {}, Vocabulary())
    assert [finding.message for finding in report.findings] == [
        "1 entities without rdf:type",
        "1 entities without rdfs:label",
        "1 blank nodes, they have no stable identity across queries",
        "<http://o/p> has objects of different kinds",
    ]


def test_value_findings():
    xsd = "http://www.w3.org/2001/XMLSchema#"
    lines = ['<http://o/c/DE> <http://www.w3.org/2000/01/rdf-schema#label> "Germany" .']
    for i in range(10):
        lines += [
            f'<http://o/{i}> <http://o/year> "{1980 + i}.0"^^<{xsd}decimal> .',
            f'<http://o/{i}> <http://o/area> "{i}.5"^^<{xsd}decimal> .',
            f'<http://o/{i}> <http://o/founded> "{1980 + i}.0"^^<{xsd}gYear> .',
            f'<http://o/{i}> <http://o/code> "{i}" .',
            f'<http://o/{i}> <http://o/country> "Germany" .',
            f'<http://o/{i}> <http://o/range> "Range {i % 3}" .',
            f'<http://o/{i}> <http://o/name> "Name {i}" .',
        ]
    report = check_output(
        source_profile([]),
        parse_ntriples("\n".join(lines)),
        {},
        {},
        {},
        Vocabulary(),
    )
    messages = [
        f"{finding.message}: {', '.join(finding.examples)}"
        for finding in report.findings
        if "<http://o/" in finding.message and "linked by" not in finding.message
    ]
    assert messages == [
        "All 10 values of <http://o/code> are strings that look like numbers: "
        '"0", "1", "2"',
        "10 of 10 strings of <http://o/country> are labels of other entities in "
        "the output, check whether they refer to them and should be links: "
        '"Germany"',
        "10 values of <http://o/founded> are not valid for their datatype "
        'xsd:gYear: "1980.0", "1981.0", "1982.0"',
        "<http://o/range> has only 3 distinct strings in 10 triples, check "
        "whether they name things that should be entities of their own: "
        '"Range 0", "Range 1", "Range 2"',
        "All 10 values of <http://o/year> are whole numbers with the datatype "
        "xsd:decimal, check whether they are fractional by nature or need "
        'another datatype: "1980.0", "1981.0", "1982.0"',
    ]


def ontology() -> KgManager:
    # another knowledge graph that defines one class
    other = KgManager("ontology", endpoint="http://localhost:3")

    def execute_sparql(sparql: str, **kwargs):
        if sparql.startswith("ASK"):
            return AskResult(boolean="http://v/" in sparql)
        assert "<http://o/country>" in sparql and "rdf-schema#label" not in sparql
        iris = ["http://v/City"] if "<http://v/City>" in sparql else []
        bindings: list[dict | None] = [
            {"s": {"type": "uri", "value": iri}} for iri in iris
        ]
        return SelectResult(["s"], bindings)

    other.execute_sparql = execute_sparql  # type: ignore
    return other


def test_vocabulary(scratch, monkeypatch):
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    task = get_task("construct", [manager(), ontology()], config())
    assert isinstance(task, ConstructTask)
    assert "knowledge graphs (ontology) provide" in task.system_information()
    task.setup("Remodel.")
    scratch.output = OUTPUT.replace("<http://o/City>", "<http://v/City>").replace(
        "<http://o/population>", "<http://v/population>"
    )
    scratch.output += (
        "<http://o/country> <http://www.w3.org/2000/01/rdf-schema#range> "
        "<http://v/City> .\n"
    )
    sparql = CITIES.replace("o:City", "<http://v/City>")
    call(task, "test_construct", name="cities", sparql=sparql)
    report = call(task, "check_output")
    assert "- <http://v/City>: 1, defined in ontology" in report
    assert (
        "- 1 classes and properties in a namespace of ontology that it does "
        "not define: http://v/population"
    ) in report
    assert "linked by <http://www.w3.org/2000/01/rdf-schema#range>" not in report
    # what another knowledge graph defines needs no label
    assert (
        "- 2 classes and properties without rdfs:label: http://o/country, "
        "http://v/population"
    ) in report

    cfg = config()
    feedback = get_task("construct-feedback", [manager(), ontology()], cfg)
    assert "- Existing vocabulary: The other knowledge graphs (ontology)" in (
        feedback.system_information()
    )


@pytest.mark.parametrize(
    "sparql, error",
    [
        ("SELECT * WHERE { ?s ?p ?o }", "not a CONSTRUCT query"),
        ("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 5", "LIMIT or OFFSET"),
        ("CONSTRUCT { ?s ?p ?o } FROM <http://g> WHERE { ?s ?p ?o }", "FROM"),
        ("CONSTRUCT { ?s ?p ?o } WHERE { GRAPH ?g { ?s ?p ?o } }", "GRAPH"),
        ("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p", "Failed to parse"),
        ("CONSTRUCT { ?s x:p ?o } WHERE { ?s ?p ?o }", "prefixes"),
    ],
)
def test_prepare_query_rejects(sparql: str, error: str):
    with pytest.raises(FunctionCallException, match=error):
        prepare_query(manager(), sparql)


def test_prepare_query():
    # a limit in a subquery is fine
    draft = prepare_query(
        manager(),
        "PREFIX o: <http://o/> CONSTRUCT { ?s o:p ?o } WHERE "
        "{ { SELECT ?s ?o WHERE { ?s <http://s/name> ?o } LIMIT 3 } }",
    )
    assert draft.prefixes["o"] == "http://o/"
    assert draft.iris == {"http://o/p", "http://s/name"}


def test_without_property():
    sparql = """\
PREFIX s: <http://s/>
CONSTRUCT { ?row s:country ?other }
WHERE { ?row <http://s/country> ?other ; s:name ?name . ?other s:country ?x }"""
    assert without_property(manager(), sparql, "http://s/country") == (
        f"PREFIX s: <http://s/> CONSTRUCT {{ ?row {ABLATED} ?other }} "
        f"WHERE {{ ?row {ABLATED} ?other ; s:name ?name . ?other {ABLATED} ?x }}"
    )


def test_union_branches():
    sparql = """\
PREFIX s: <http://s/>
CONSTRUCT { ?s s:p ?o }
WHERE {
  { ?s s:a ?o } UNION { ?s s:b ?o { ?o s:c ?x } UNION { ?o s:d ?x } } UNION { ?s s:e ?o }
  OPTIONAL { { ?s s:f ?y } }
}"""
    assert union_branches(manager(), sparql) == [
        "PREFIX s: <http://s/> ASK { ?s s:a ?o }",
        "PREFIX s: <http://s/> ASK { ?s s:b ?o { ?o s:c ?x } UNION { ?o s:d ?x } }",
        "PREFIX s: <http://s/> ASK { ?o s:c ?x }",
        "PREFIX s: <http://s/> ASK { ?o s:d ?x }",
        "PREFIX s: <http://s/> ASK { ?s s:e ?o }",
    ]
    assert union_branches(manager(), CITIES) == []


def test_query_without_effect(scratch, monkeypatch):
    # the link has no effect on the output and the branches match nothing
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    task = get_task("construct", [manager()], config())
    task.setup("Remodel.")
    scratch.matches = False
    sparql = CITIES.replace(
        "?row a <http://s/City> ;",
        "{ ?row a <http://s/City> } UNION { ?row a <http://s/Town> } ?row",
    )
    result = call(task, "test_construct", name="cities", sparql=sparql)
    assert "2 of 2 UNION branches match nothing on the sample:" in result
    assert "- { ?row a <http://s/Town> }" in result
    assert "The output is the same without these properties" in result
    assert result.endswith("- <http://s/country>")

    report = call(task, "check_output")
    assert "2 UNION branches of query cities match nothing" in report
    assert "<http://s/country>: 1 links, without effect on the output" in report
    with pytest.raises(FunctionCallException, match="http://s/country"):
        call(task, "submit_construct", explanations=[])


def test_construct_task(scratch, monkeypatch):
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    task = get_task("construct", [manager()], config())
    assert isinstance(task, ConstructTask)
    assert task.setup("Remodel.") == (
        "Remodel.\n\nClasses of the knowledge graph with their number of "
        "instances in the sample:\n- <http://s/City>: 1\n- <http://s/Country>: 1"
    )
    # outputs of earlier runs are gone
    assert scratch.graphs_loaded == {SAMPLE_GRAPH: SAMPLE}
    assert task.output([]) is None

    with pytest.raises(FunctionCallException, match="no saved queries"):
        call(task, "submit_construct", explanations=[])
    with pytest.raises(FunctionCallException, match="letters, digits"):
        call(task, "test_construct", name="my query", sparql=CITIES)

    result = call(task, "test_construct", name="cities", sparql=CITIES)
    assert result.startswith(
        "Saved query cities. On the sample it produces 4 triples for 1 subjects"
    )
    assert scratch.graphs_loaded[draft_graph("cities")] == OUTPUT
    assert call(task, "list_constructs") == (
        "Saved queries:\n- cities: 4 triples on the sample"
    )
    assert CITIES in call(task, "get_construct", name="cities")
    assert "Errors, the queries cannot be submitted" in call(task, "check_output")

    # a failed test keeps the saved query
    scratch.output = ""
    with pytest.raises(FunctionCallException, match="no triples"):
        call(task, "test_construct", name="cities", sparql=CITIES)
    scratch.output = "<http://o/a b> <http://o/p> <http://o/c> .\n"
    with pytest.raises(FunctionCallException, match="invalid IRIs"):
        call(task, "test_construct", name="cities", sparql=CITIES)
    assert len(task.drafts["cities"].triples) == 4

    # the link is used by the query, the date is not in the output
    with pytest.raises(FunctionCallException, match="http://s/founded") as error:
        call(task, "submit_construct", explanations=[])
    assert "http://s/country" not in str(error.value)
    assert not task.done("submit_construct")
    assert not output_of(task)["submitted"]

    explanations = [
        {"property": "<http://s/founded>", "explanation": "Converted to xsd:date."}
    ]
    assert call(task, "submit_construct", explanations=explanations) == (
        "Submitted 1 queries"
    )
    assert task.done("submit_construct")
    output = output_of(task)
    assert output["submitted"]
    assert output["queries"] == [{"name": "cities", "sparql": CITIES, "triples": 4}]
    assert output["explanations"] == {"http://s/founded": "Converted to xsd:date."}

    # a change needs a new submission
    scratch.output = OUTPUT
    call(task, "test_construct", name="countries", sparql=CITIES)
    assert not task.done("submit_construct")
    assert call(task, "remove_construct", name="countries") == "Removed query countries"
    assert draft_graph("countries") not in scratch.graphs_loaded
    with pytest.raises(FunctionCallException, match="No saved query countries"):
        call(task, "get_construct", name="countries")


def test_submit_in_last_step(scratch, monkeypatch):
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    task = get_task("construct", [manager()], config())
    task.setup("Remodel.")
    call(task, "test_construct", name="cities", sparql=CITIES)
    assert task.limit_functions() == ["submit_construct"]
    # errors cannot be fixed anymore
    task.last_step = True
    call(task, "submit_construct", explanations=[])
    assert task.done("submit_construct")
    assert output_of(task)["report"]["findings"][0]["level"] == "error"


def test_setup_without_sample(scratch, monkeypatch):
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    scratch.graphs_loaded = {}
    task = get_task("construct", [manager()], config())
    with pytest.raises(ValueError, match="No sample"):
        task.setup("Remodel.")


@pytest.mark.parametrize("max_steps", [1, 20])
def test_feedback_task(scratch, monkeypatch, max_steps: int):
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    cfg = config()
    task = get_task("construct", [manager()], cfg)
    task.setup("Remodel.")
    call(task, "test_construct", name="cities", sparql=CITIES)
    explanations = [{"property": "http://s/founded", "explanation": "Converted."}]
    call(task, "submit_construct", explanations=explanations)
    assert task.feedback_task == ConstructFeedbackTask.name

    cfg.max_steps = max_steps
    feedback = get_task("construct-feedback", [manager()], cfg)
    assert isinstance(feedback, ConstructFeedbackTask)
    inputs = ["Remodel.", "Feedback (status=refine):\nNo labels."]
    instructions = feedback.setup({"inputs": inputs, "output": task.output([])})
    assert "Query cities (4 triples on the sample)" in instructions
    assert "explanation: Converted." in instructions
    assert "1. Which cities are in Germany?" in instructions
    assert instructions.endswith(
        "## Previous feedback\n\nFeedback (status=refine):\nNo labels."
    )

    # the output can only be queried with more than one step
    names = [fn.name for fn in feedback.function_definitions()]
    agentic = max_steps > 1
    assert names == ["give_feedback"] + ["execute_output"] * agentic
    assert ("execute_output" in feedback.system_information()) == agentic
    assert feedback.output_manager is not None
    assert feedback.output_manager.params == {
        "default-graph-uri": [draft_graph("cities")]
    }
    assert feedback.output_manager.prefixes["o"] == "http://o/"
    with pytest.raises(FunctionCallException, match="Only SELECT and ASK"):
        feedback.call_function("execute_output", {"sparql": CITIES}, set(), None)
    args = {"status": "done", "feedback": "Fine."}
    assert feedback.call_function("give_feedback", args, set(), None) == "Stopping"


class ScriptedModel(Model):
    # answers each call with the next tool call of the script
    def __init__(self, config, script: list[tuple[str, dict]]):
        super().__init__(config)
        self.script = list(script)
        self.offered: list[list[str]] = []

    def call(self, messages, fns, config=None) -> Response:
        self.offered.append([fn["name"] for fn in fns])
        name, args = self.script.pop(0)
        return Response(
            id=uuid.uuid4().hex,
            tool_calls=[ToolCall(id=uuid.uuid4().hex, name=name, args=args)],
        )


def test_generate_with_feedback(scratch, monkeypatch):
    monkeypatch.setenv("GRASP_SCRATCH_ACCESS_TOKEN", "token")
    cfg = config(feedback=True, feedback_max_steps=5)
    explanations = [{"property": "http://s/founded", "explanation": "Converted."}]
    model = ScriptedModel(
        cfg,
        [
            ("test_construct", {"name": "cities", "sparql": CITIES}),
            ("submit_construct", {"explanations": explanations}),
            ("give_feedback", {"status": "refine", "feedback": "1. No labels."}),
            ("test_construct", {"name": "cities", "sparql": CITIES + " "}),
            ("submit_construct", {"explanations": explanations}),
            ("give_feedback", {"status": "done", "feedback": "Fine."}),
        ],
    )
    *_, result = generate(
        "construct",
        "Remodel.",
        cfg,
        [manager()],
        custom_model=model,
        yield_output=True,
    )
    assert not model.script
    assert result["error"] is None
    assert result["output"]["submitted"]
    # the feedback run can query the knowledge graph and the output
    assert {"execute", "execute_output", "give_feedback"} <= set(model.offered[2])
    assert "test_construct" not in model.offered[2]

    first, second = [m for m in result["messages"] if "trace" in m]
    assert first["content"] == "Feedback (status=refine):\n1. No labels."
    assert "## Previous feedback" not in first["trace"][1]["content"]
    # the second round gets the feedback of the first
    assert first["content"] in second["trace"][1]["content"]
