from types import SimpleNamespace

import pytest

from grasp.functions import (
    check_known,
    execute_sparql,
    order_triples,
    query_iris,
    row_iris,
    update_known_from_index_iris,
    update_known_from_selections,
)
from grasp.manager import KgManager
from grasp.manager.normalizer import Normalizer
from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.sparql.types import Binding, KgItem, Selection, SelectResult, TermKind
from grasp.sparql.utils import load_iri_and_literal_parser, load_sparql_parser
from grasp.utils import FunctionCallException


class VariantNormalizer(Normalizer):
    def denormalize(self, iri: str, variant: str | None) -> str | None:
        return f"{iri}/{variant}"


class PropertyNormalizer(Normalizer):
    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        if not iri.startswith("ex:direct/"):
            return None
        return "ex:" + iri.removeprefix("ex:direct/"), "direct"


class Manager:
    def get_normalizer(self, name: str) -> Normalizer:
        if name == "classes":
            return VariantNormalizer()
        elif name == "properties":
            return PropertyNormalizer()
        return Normalizer()


def test_known_from_selections():
    selections = [
        Selection(KgItem(identifier="ex:e"), TermKind.ENTITY),
        Selection(KgItem(identifier="ex:c", variants=["v"]), TermKind.CLASS),
        Selection(KgItem(identifier="ex:p"), TermKind.PROPERTY),
    ]
    known = set()
    update_known_from_selections(known, selections, Manager())  # type: ignore
    assert known == {"ex:e", "ex:c", "ex:c/v", "ex:p"}


def test_known_from_rows_uses_all_indices():
    rows = (
        {"p": Binding("uri", "ex:direct/p"), "v": Binding("literal", "x")}
        for _ in range(2)
    )
    known = set()
    update_known_from_index_iris(known, row_iris(rows), Manager())  # type: ignore
    assert known == {"ex:direct/p", "ex:p"}


class Data:
    def __init__(self, identifiers: list[str]):
        self.ids = {identifier: i for i, identifier in enumerate(identifiers)}

    def id_from_identifier(self, identifier: str) -> int | None:
        return self.ids.get(identifier)


class TripleManager:
    # ids are popularity ranks, the classes are not in the entity index
    data = {
        "properties": Data(["ex:type", "ex:name", "ex:knows"]),
        "entities": Data(["ex:a", "ex:b", "ex:x"]),
        "classes": Data(["ex:C1", "ex:C2"]),
    }

    def normalize(self, iri: str, index_name: str):
        return iri, None

    def try_get_data(self, index_name: str) -> Data | None:
        return self.data.get(index_name)

    def check_identifier(self, identifier: str, index_name: str) -> bool:
        return identifier in self.data[index_name].ids

    def get_normalizer(self, name: str) -> Normalizer:
        return Normalizer()


def test_all_classes_of_an_entity_come_first():
    triples = [
        ("ex:x", "ex:knows", "ex:a"),
        ("ex:x", "ex:knows", "ex:b"),
        ("ex:x", "ex:name", "x"),
        ("ex:x", "ex:type", "ex:C2"),
        ("ex:x", "ex:type", "ex:C1"),
    ]
    rows = [
        {
            "s": Binding("uri", s),
            "p": Binding("uri", p),
            "o": Binding("literal" if o == "x" else "uri", o),
        }
        for s, p, o in triples
    ]
    order = order_triples(rows, TripleManager())  # type: ignore
    # types by class popularity, then the other properties, then repeats
    assert order == [4, 3, 2, 0, 1]


EX = "http://example.org/kg#"
REMOTE = "http://example.org/remote/"


@pytest.fixture(scope="module")
def manager():
    prefixes = {**get_common_sparql_prefixes(), "ex": EX}
    return SimpleNamespace(
        kg="example",
        prefixes=prefixes,
        sparql_parser=load_sparql_parser(),
        iri_literal_parser=load_iri_and_literal_parser(),
        find_longest_prefix=lambda iri: find_longest_prefix(iri, prefixes),
        format_iri=lambda iri: iri,
    )


SERVICE_QUERY = f"""PREFIX remote: <{REMOTE}>
SELECT ?value WHERE {{
  ex:place_1 geo:hasGeometry/geo:asWKT ?wkt .
  SERVICE remote: {{
    _:config remote:endpoint <http://example.org/raster> ;
      remote:input ?wkt ; remote:output ?value .
  }}
}}"""


def test_query_iris_resolves_query_prefixes(manager):
    iris = query_iris(manager, SERVICE_QUERY)
    assert iris == {
        EX + "place_1",
        REMOTE,
        REMOTE + "endpoint",
        "http://example.org/raster",
        REMOTE + "input",
        REMOTE + "output",
    }


def test_query_iris_skips_service(manager):
    iris = query_iris(manager, SERVICE_QUERY, skip_service=True)
    assert iris == {EX + "place_1"}


def test_check_known_ignores_service(manager):
    check_known(manager, SERVICE_QUERY, {EX + "place_1"})
    with pytest.raises(FunctionCallException, match="without being known"):
        check_known(manager, SERVICE_QUERY, set())


def test_check_known_undefined_prefix(manager):
    sparql = "SELECT ?x WHERE { ?x foo:bar ex:place_1 . ?x foo:baz ?y }"
    with pytest.raises(FunctionCallException) as exc:
        check_known(manager, sparql, {EX + "place_1"})
    message = str(exc.value)
    assert "foo: (e.g. foo:bar)" in message
    assert "neither declared in the query nor common or example prefixes" in message


def test_execute_unclipped(monkeypatch):
    manager = KgManager("example", endpoint="http://example.org/sparql")
    value = " ".join(f"word{i}" for i in range(100))
    result = SelectResult.from_json(
        {
            "head": {"vars": ["o"]},
            "results": {"bindings": [{"o": {"type": "literal", "value": value}}]},
        }
    )
    monkeypatch.setattr(manager, "execute_sparql", lambda *args, **kwargs: result)
    sparql = "SELECT ?o WHERE { ?s ?p ?o }"

    clipped = execute_sparql([manager], "example", sparql, 10, 10)
    assert value not in clipped.formatted and "word0" in clipped.formatted

    unclipped = execute_sparql([manager], "example", sparql, 10, 10, unclipped=True)
    assert value in unclipped.formatted


def test_execute_incomplete_result(monkeypatch):
    manager = KgManager("example", endpoint="http://example.org/sparql")
    result = SelectResult(
        variables=["o"],
        data=[{"o": {"type": "literal", "value": f"value{i}"}} for i in range(20)],
        complete=False,
    )
    monkeypatch.setattr(manager, "execute_sparql", lambda *args, **kwargs: result)

    formatted = execute_sparql(
        [manager],
        "example",
        "SELECT ?o WHERE { ?s ?p ?o }",
        10,
        10,
    ).formatted
    assert "more than 20 rows" in formatted and "the first 10 rows" in formatted
    assert "value9" in formatted and "value19" not in formatted


def test_execute_construct_result(monkeypatch):
    manager = KgManager("example", endpoint="http://example.org/sparql")
    result = SelectResult(
        variables=["s", "p", "o"],
        data=[
            {
                "s": {"type": "uri", "value": "http://example.org/s"},
                "p": {"type": "uri", "value": "http://example.org/p"},
                "o": {"type": "literal", "value": f"value{i}"},
            }
            for i in range(10)
        ],
        complete=False,
    )
    calls = []

    def execute(*args, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(manager, "execute_sparql", execute)

    formatted = execute_sparql(
        [manager],
        "example",
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
        10,
        10,
    ).formatted
    assert calls[0]["sparql_result_max_rows"] == 10
    assert "Got the first 10 triples" in formatted
    assert "LIMIT and OFFSET" in formatted
    assert "subject" in formatted and "value9" in formatted
