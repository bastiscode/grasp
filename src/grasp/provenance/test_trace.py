from types import SimpleNamespace

import pytest

from grasp.functions import check_known, query_iris
from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.provenance.trace import instruction_iris, trace_provenance
from grasp.sparql.utils import load_iri_and_literal_parser, load_sparql_parser
from grasp.utils import FunctionCallException

WALD = "https://wald.cs.uni-freiburg.de/ontology#"
QLPROXY = "https://qlever.cs.uni-freiburg.de/qlproxy/"


@pytest.fixture(scope="module")
def manager():
    prefixes = {**get_common_sparql_prefixes(), "wald": WALD}
    return SimpleNamespace(
        kg="wald",
        prefixes=prefixes,
        sparql_parser=load_sparql_parser(),
        iri_literal_parser=load_iri_and_literal_parser(),
        find_longest_prefix=lambda iri: find_longest_prefix(iri, prefixes),
        normalize=lambda iri, index: None,
        get_label=lambda iri, index: None,
        format_iri=lambda iri: iri,
    )


SERVICE_QUERY = f"""PREFIX qlproxy: <{QLPROXY}>
SELECT ?value WHERE {{
  wald:Betrieb_1 geo:hasGeometry/geo:asWKT ?wkt .
  SERVICE qlproxy: {{
    _:config qlproxy:endpoint <https://qlever.dev/api/wald-tiff> ;
      qlproxy:input-filter_wkt ?wkt ; qlproxy:output-value ?value .
  }}
}}"""


def test_query_iris_resolves_query_prefixes(manager):
    iris = query_iris(manager, SERVICE_QUERY)
    assert iris == {
        WALD + "Betrieb_1",
        QLPROXY,
        QLPROXY + "endpoint",
        "https://qlever.dev/api/wald-tiff",
        QLPROXY + "input-filter_wkt",
        QLPROXY + "output-value",
    }


def test_query_iris_skips_service(manager):
    assert query_iris(manager, SERVICE_QUERY, skip_service=True) == {WALD + "Betrieb_1"}


def test_check_known_ignores_service(manager):
    check_known(manager, SERVICE_QUERY, {WALD + "Betrieb_1"})
    with pytest.raises(FunctionCallException, match="without being known"):
        check_known(manager, SERVICE_QUERY, set())


def test_check_known_undefined_prefix(manager):
    sparql = "SELECT ?x WHERE { ?x foo:bar wald:Betrieb_1 . ?x foo:baz ?y }"
    with pytest.raises(FunctionCallException) as exc:
        check_known(manager, sparql, {WALD + "Betrieb_1"})
    message = str(exc.value)
    assert "foo: (e.g. foo:bar)" in message
    assert "neither declared in the query nor common or wald prefixes" in message


INSTRUCTION = f"""You are a question answering assistant.
- Notes: the raster service is called with PREFIX qlproxy: <{QLPROXY}> and
SERVICE qlproxy: {{ _:config qlproxy:endpoint <https://qlever.dev/api/wald-tiff> }}.
- Stands link via wald:teilfläche; links like https://en.wikipedia.org/wiki/<Title>.
- Use rdfs:label for labels, e.g. wald:Betrieb_<bez>_<nr>."""


def test_instruction_iris(manager):
    iris = instruction_iris(manager, [{"role": "system", "content": INSTRUCTION}])
    assert QLPROXY in iris
    assert QLPROXY + "endpoint" in iris
    assert "https://qlever.dev/api/wald-tiff" in iris
    assert WALD + "teilfläche" in iris
    assert iris[WALD + "teilfläche"].startswith("- Stands link")
    # no relative IRIs, undefined prefixes or common vocabulary
    assert "Title" not in iris
    assert not any(iri.startswith("Notes") for iri in iris)
    assert "http://www.w3.org/2000/01/rdf-schema#label" not in iris


def test_instruction_iris_only_system(manager):
    assert instruction_iris(manager, [{"role": "user", "content": INSTRUCTION}]) == {}


def test_trace_status_from_instruction(manager):
    sparql = (
        f"PREFIX qlproxy: <{QLPROXY}>\n"
        "SELECT ?v WHERE { wald:Betrieb_1 wald:x ?v . ?v qlproxy:endpoint ?e }"
    )
    call = {
        "id": "c1",
        "name": "execute",
        "args": {"kg": "wald", "sparql": "SELECT * WHERE { ?s ?p ?o }"},
        "result": {
            "type": "table",
            "rows": [{"s": {"type": "uri", "value": WALD + "Betrieb_1"}}],
        },
        "new_known": [WALD + "Betrieb_1"],
    }
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "assistant", "content": {"tool_calls": [call]}},
    ]
    trace = trace_provenance(manager, sparql, messages)
    assert trace is not None
    status = {item["identifier"]: item["status"] for item in trace["items"]}
    assert status[WALD + "Betrieb_1"] == "retrieved"
    assert status[QLPROXY + "endpoint"] == "instructed"
    assert status[WALD + "x"] == "unverified"
    item = next(i for i in trace["items"] if i["identifier"] == QLPROXY + "endpoint")
    assert "qlproxy:endpoint" in item["instruction"]["text"]


def test_trace_shown_by_example(manager):
    sparql = (
        f"PREFIX qlproxy: <{QLPROXY}>\nSELECT ?e WHERE {{ ?x qlproxy:endpoint ?e }}"
    )
    call = {
        "id": "c1",
        "name": "search_example",
        "args": {"kg": "wald", "query": "raster"},
        "result": {
            "type": "sparql_qa_examples",
            "examples": [
                {"question": "Other?", "sparql": "SELECT * WHERE { ?s ?p ?o }"},
                {"question": "Raster?", "sparql": SERVICE_QUERY},
            ],
        },
        "new_known": [QLPROXY + "endpoint"],
    }
    messages = [{"role": "assistant", "content": {"tool_calls": [call]}}]
    trace = trace_provenance(manager, sparql, messages)
    assert trace is not None
    item = next(i for i in trace["items"] if i["identifier"] == QLPROXY + "endpoint")
    assert item["status"] == "retrieved"
    assert item["shown_by"]["result_paths"] == ["examples.1"]
    shown = item["shown_by"]["shown"]
    assert shown["type"] == "example" and shown["question"] == "Raster?"
    assert "qlproxy:endpoint" in shown["line"]
