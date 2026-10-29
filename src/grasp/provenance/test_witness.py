import pytest

from grasp.provenance.trace import flatten_args
from grasp.provenance.witness import char_offset
from grasp.provenance.witness import (
    RDF_TYPE,
    instantiate,
    matches_row,
    plan_witness,
    witness_query,
)
from grasp.sparql.types import Binding
from grasp.sparql.utils import (
    load_iri_and_literal_parser,
    load_sparql_parser,
    parse_string,
)

SPARQL_PARSER = load_sparql_parser()
IRI_PARSER = load_iri_and_literal_parser()

PREFIXES = "PREFIX : <http://ex.org/> "


def uri(name: str) -> Binding:
    return Binding(typ="uri", value=f"http://ex.org/{name}")


def lit(value: str, **kwargs) -> Binding:
    return Binding(typ="literal", value=value, **kwargs)


def plan(body: str):
    return plan_witness(PREFIXES + body, SPARQL_PARSER)


def query(body: str, row=None, limit: int = 4, **kwargs) -> str:
    q = witness_query(plan(body), row, limit, **kwargs)
    # every witness query must be valid SPARQL
    parse_string(q, SPARQL_PARSER)
    return q


def witnesses(body: str, rows):
    p = plan(body)
    return instantiate(p, rows, IRI_PARSER, p.prefixes)


def triples(witness) -> list[tuple]:
    return [
        (
            t.subject.value,
            t.predicate if isinstance(t.predicate, str) else t.predicate.value,
            t.object.value,
        )
        for t in witness.triples
    ]


def test_plain_bgp():
    body = "SELECT DISTINCT ?actor WHERE { ?film :director :Nolan ; :cast ?actor . }"
    p = plan(body)
    assert p.bindable == ["actor"]
    assert [(t.subject, t.predicate, t.object) for t in p.triples] == [
        ("?film", ":director", ":Nolan"),
        ("?film", ":cast", "?actor"),
    ]

    q = query(body, {"actor": uri("Bale")})
    assert "SELECT DISTINCT *" in q
    assert "VALUES (?actor) { (<http://ex.org/Bale>) }" in q
    assert q.endswith("LIMIT 4")

    [w] = witnesses(body, [{"film": uri("Prestige"), "actor": uri("Bale")}])
    assert triples(w) == [
        ("http://ex.org/Prestige", "http://ex.org/director", "http://ex.org/Nolan"),
        ("http://ex.org/Prestige", "http://ex.org/cast", "http://ex.org/Bale"),
    ]


def test_object_list_and_rdf_type():
    body = "SELECT ?x WHERE { ?x a :City ; :name ?a , ?b }"
    p = plan(body)
    assert [(t.predicate, t.object) for t in p.triples] == [
        (f"<{RDF_TYPE}>", ":City"),
        (":name", "?a"),
        (":name", "?b"),
    ]


def test_inverse_predicate():
    body = "SELECT ?x WHERE { ?x ^:capital :Germany }"
    [t] = plan(body).triples
    assert (t.subject, t.predicate, t.object, t.is_path) == (
        ":Germany",
        ":capital",
        "?x",
        False,
    )


def test_aggregate_drops_grouping_and_binds_keys():
    body = (
        "SELECT ?country (COUNT(?city) AS ?n) WHERE { ?city :country ?country } "
        "GROUP BY ?country HAVING (COUNT(?city) > 2) ORDER BY DESC(?n) LIMIT 1"
    )
    p = plan(body)
    assert p.bindable == ["country"]

    q = query(body, {"country": uri("DE"), "n": lit("3")})
    for keyword in ["GROUP", "HAVING", "ORDER", "COUNT", "LIMIT 1"]:
        assert keyword not in q
    assert "VALUES (?country)" in q


def test_count_without_keys_has_no_values():
    body = "SELECT (COUNT(?x) AS ?n) WHERE { ?x a :City }"
    assert plan(body).bindable == []
    assert "VALUES" not in query(body, {"n": lit("5")})


def test_optional_marker():
    body = "SELECT ?x ?y WHERE { ?x a :City OPTIONAL { ?x :mayor ?y } }"
    p = plan(body)
    assert p.triples[1].markers == ("?_prov_m0",)
    assert "BIND(true AS ?_prov_m0)" in query(body, {"x": uri("A")})

    # optional matched
    [w] = witnesses(body, [{"x": uri("A"), "y": uri("M"), "_prov_m0": lit("true")}])
    assert len(w.triples) == 2
    assert "_prov_m0" not in w.bindings

    # optional did not match
    [w] = witnesses(body, [{"x": uri("A")}])
    assert len(w.triples) == 1


def test_optional_with_outer_variable_not_reported_when_unmatched():
    # ?y is bound outside, but the optional did not match, so its pattern
    # must not be reported even though all its variables are bound
    body = "SELECT ?x WHERE { ?x :a ?y OPTIONAL { ?x :b ?y } }"
    [w] = witnesses(body, [{"x": uri("X"), "y": uri("Y")}])
    assert triples(w) == [("http://ex.org/X", "http://ex.org/a", "http://ex.org/Y")]


def test_union_reports_only_matching_branch():
    body = "SELECT ?x WHERE { { :Q :educatedAt ?x } UNION { :Q :employer ?x } }"
    p = plan(body)
    assert [t.markers for t in p.triples] == [("?_prov_m0",), ("?_prov_m1",)]

    [w] = witnesses(body, [{"x": uri("Uni"), "_prov_m1": lit("true")}])
    assert triples(w) == [
        ("http://ex.org/Q", "http://ex.org/employer", "http://ex.org/Uni")
    ]


def test_nested_union_in_optional():
    body = "SELECT ?x WHERE { ?x a :C OPTIONAL { { ?x :a ?y } UNION { ?x :b ?y } } }"
    p = plan(body)
    assert [t.markers for t in p.triples[1:]] == [
        ("?_prov_m0", "?_prov_m1"),
        ("?_prov_m0", "?_prov_m2"),
    ]
    query(body, {"x": uri("X")})


def test_filters_are_conditions():
    body = (
        "SELECT ?x WHERE { ?x :pop ?p FILTER(?p > 100) "
        "FILTER NOT EXISTS { ?x :dissolved ?d } MINUS { ?x a :Village } }"
    )
    p = plan(body)
    # patterns in NOT EXISTS and MINUS are never instantiated
    assert len(p.triples) == 1
    assert len(p.filters) == 2

    [w] = witnesses(body, [{"x": uri("X"), "p": lit("200")}])
    assert len(w.triples) == 1
    assert w.conditions[0].text == "FILTER ( ?p > 100 )"
    assert w.conditions[0].bindings == {"p": lit("200")}


def test_property_path():
    body = "SELECT ?x WHERE { ?x :type/:subclass* :City }"
    [t] = plan(body).triples
    assert t.is_path

    [w] = witnesses(body, [{"x": uri("Berlin")}])
    [wt] = w.triples
    assert wt.is_path and isinstance(wt.predicate, str)


def test_ask():
    body = "ASK { :Q42 :citizen :UK }"
    p = plan(body)
    assert p.query_type == "ask"
    q = query(body, None)
    assert q.startswith(PREFIXES.strip())
    assert "VALUES" not in q

    [w] = witnesses(body, [{}])
    assert triples(w) == [
        ("http://ex.org/Q42", "http://ex.org/citizen", "http://ex.org/UK")
    ]


def test_select_star_binds_all_row_variables():
    body = "SELECT * WHERE { ?x :p ?y }"
    assert plan(body).bindable is None
    q = query(body, {"x": uri("X"), "y": uri("Y")})
    assert "VALUES (?x ?y)" in q


def test_subquery_with_aggregate():
    body = (
        "SELECT ?x ?m WHERE { { SELECT ?x (MAX(?p) AS ?m) WHERE { ?x :pop ?p } "
        "GROUP BY ?x } ?x :name ?m }"
    )
    q = query(body, {"x": uri("X"), "m": lit("5")})
    assert "GROUP BY ?x" in q

    # filters inside the subquery are not conditions of the outer witness
    assert (
        plan(
            "SELECT ?x WHERE { ?x :p ?y FILTER(?y > 1) "
            "{ SELECT (COUNT(?z) AS ?n) WHERE { ?z :p ?w FILTER(?w > 1) } } }"
        )
        .filters[0]
        .text
        == "FILTER ( ?y > 1 )"
    )

    # ?p is hidden inside the subquery, so its pattern stays uninstantiated
    [w] = witnesses(body, [{"x": uri("X"), "m": lit("5")}])
    assert triples(w) == [("http://ex.org/X", "http://ex.org/name", "5")]


def test_union_branch_with_subquery():
    body = "SELECT ?x WHERE { { SELECT ?x WHERE { ?x :a ?y } } UNION { ?x :b ?z } }"
    q = query(body, {"x": uri("X")})
    assert "BIND(true AS ?_prov_m0)" in q
    assert "BIND(true AS ?_prov_m1)" in q


def test_bind_projected_variable():
    # VALUES must join outside the original group, otherwise BIND would
    # assign an already bound variable
    body = 'SELECT ?label WHERE { ?x :name ?n BIND(CONCAT(?n, "!") AS ?label) }'
    q = query(body, {"label": lit("a!")})
    where = q[q.index("WHERE") :]
    assert where.index("BIND") < where.index("VALUES")


def test_literal_escaping_and_bnodes():
    body = "SELECT ?x ?y WHERE { ?x :name ?y }"
    q = query(
        body,
        {"x": Binding(typ="bnode", value="b0"), "y": lit('say "hi"\n', lang="en")},
    )
    assert 'VALUES (?x ?y) { (UNDEF "say \\"hi\\"\\n"@en) }' in q


def test_typed_literal_and_fallback():
    body = "SELECT ?c WHERE { :X :coords ?c }"
    wkt = "http://www.opengis.net/ont/geosparql#wktLiteral"
    row = {"c": lit("POINT(1 2)", datatype=wkt)}
    assert f'"POINT(1 2)"^^<{wkt}>' in query(body, row)
    # without literal binding there is nothing left to bind
    assert "VALUES" not in query(body, row, bind_literals=False)

    p = plan(body)
    assert matches_row(p, row, {"c": lit("POINT(1 2)", datatype=wkt)})
    assert not matches_row(p, row, {"c": lit("POINT(3 4)", datatype=wkt)})


def test_top_level_values_kept():
    body = "SELECT ?x WHERE { ?x :p ?y } VALUES ?y { :A :B }"
    q = query(body, {"x": uri("X")})
    assert "VALUES ?y" in q


def test_no_triples():
    body = "SELECT ?x WHERE { VALUES ?x { :A :B } }"
    assert plan(body).triples == []
    [w] = witnesses(body, [{"x": uri("A")}])
    assert w.triples == []


def test_unsupported_query_types():
    with pytest.raises(Exception):
        plan("CONSTRUCT { ?x :p ?y } WHERE { ?x :p ?y }")


def test_char_offset():
    # UTF-16 code units, characters outside the BMP count twice
    data = "a\u00e4\U0001f600b".encode()
    assert char_offset(data, 1) == 1
    assert char_offset(data, 3) == 2
    assert char_offset(data, 7) == 4


def test_flatten_args():
    args = {"kg": "wd", "constraints": {"subject": "wd:Q1", "object": None}}
    assert list(flatten_args(args)) == [("kg", "wd"), ("constraints.subject", "wd:Q1")]


@pytest.mark.parametrize(
    "num, k, expected",
    [
        (0, 3, []),
        (2, 3, [1, 2]),
        (3, 3, [1, 2, 3]),
        (10, 3, [1, 2, 10]),
        (10, 4, [1, 2, 9, 10]),
        (10, 1, [1]),
    ],
)
def test_select_witnesses(num, k, expected):
    from grasp.provenance import select_witnesses

    assert select_witnesses(num, k) == expected


def test_negation_only_note():
    p = plan("ASK { FILTER NOT EXISTS { :B :partner ?p } }")
    assert p.triples == []
    assert "vacuously true" in p.notes[0]

    p = plan("SELECT ?x WHERE { VALUES ?x { :A } }")
    assert "VALUES or BIND" in p.notes[0]


def test_hidden_subquery_patterns():
    from grasp.provenance import hidden_subquery_patterns

    plan = plan_witness(
        "SELECT (AVG(?n) AS ?avg) WHERE { { SELECT ?person (COUNT(?spouse) AS ?n) "
        "WHERE { ?country <p:head> ?person . ?person <p:type> <c:human> . "
        "OPTIONAL { ?person <p:spouse> ?spouse } } GROUP BY ?person } }",
        SPARQL_PARSER,
    )
    assert all(pattern.in_subquery for pattern in plan.triples)
    rows = [{"person": Binding(typ="uri", value="e:1")}]
    assert hidden_subquery_patterns(plan, rows) == [
        "?country <p:head> ?person",
        "?person <p:spouse> ?spouse",
    ]


def test_pattern_spans():
    sparql = 'PREFIX : <http://ex.org/> SELECT ?x WHERE { ?x :p ?y ; :q "ä" , :Z }'
    p = plan_witness(sparql, SPARQL_PARSER)
    texts = [[sparql[start:end] for start, end in t.spans] for t in p.triples]
    assert texts == [["?x", ":p", "?y"], ["?x", ":q", '"ä"'], ["?x", ":q", ":Z"]]


def test_unbound_row_variables():
    # a row from one UNION branch must not match the other branch
    body = "SELECT ?a ?s WHERE { { :P :advisor ?a } UNION { :P :student ?s } }"
    row = {"s": Binding(typ="uri", value="http://ex.org/S")}
    q = query(body, row)
    assert "VALUES (?s) { (<http://ex.org/S>) }" in q
    assert "FILTER(!BOUND(?a))" in q

    p = plan(body)
    assert matches_row(p, row, {"s": row["s"]})
    assert not matches_row(p, row, {"s": row["s"], "a": Binding(typ="uri", value="x")})
