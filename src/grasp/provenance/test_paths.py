import pytest

from grasp.provenance.paths import (
    PathSyntaxError,
    Step,
    chain_patterns,
    expansion_query,
    path_chains,
)


def test_sequence() -> None:
    assert path_chains("p:P39 / ps:P39", 4) == [(Step("p:P39"), Step("ps:P39"))]


def test_star_shortest_first() -> None:
    chains = path_chains("wdt:P31 / wdt:P279 *", 3)
    assert [len(c) for c in chains] == [1, 2, 3]
    assert chains[2] == (Step("wdt:P31"), Step("wdt:P279"), Step("wdt:P279"))


def test_plus_and_optional() -> None:
    assert [len(c) for c in path_chains("wdt:P279 +", 3)] == [1, 2, 3]
    assert path_chains("wdt:P279 ?", 3) == [(), (Step("wdt:P279"),)]


def test_inverse_and_alternative() -> None:
    chains = path_chains("( wdt:P31 | ^ wdt:P361 ) +", 2)
    assert (Step("wdt:P361", inverse=True),) in chains
    assert len(chains) == 2 + 4
    # inverse of a sequence reverses it
    assert path_chains("^ ( wdt:P1 / wdt:P2 )", 2) == [
        (Step("wdt:P2", True), Step("wdt:P1", True))
    ]


def test_negated() -> None:
    assert path_chains("! ( wdt:P1 | wdt:P2 )", 2) == [
        (Step(None, False, ("wdt:P1", "wdt:P2")),)
    ]
    assert path_chains("! a", 1) == [
        (Step(None, False, ("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",)),)
    ]


def test_syntax_error() -> None:
    with pytest.raises(PathSyntaxError):
        path_chains("( wdt:P1", 2)


def test_patterns() -> None:
    chain = (Step("wdt:P31"), Step("wdt:P361", True), Step(None, False, ("wdt:P1",)))
    patterns = chain_patterns(chain, "<s>", "<o>", 3)
    assert patterns == (
        "<s> wdt:P31 ?_path3_n1 . ?_path3_n2 wdt:P361 ?_path3_n1 . "
        "?_path3_n2 ?_path3_p2 <o> . FILTER(?_path3_p2 NOT IN (wdt:P1))"
    )
    query = expansion_query("PREFIX x: <y>", [(0, (Step("x:a"),))], "<s>", "<o>")
    assert query == (
        "PREFIX x: <y> SELECT * WHERE { { <s> x:a <o> . BIND(0 AS ?_path_chain) } } "
        "LIMIT 1"
    )


def test_pairs_query() -> None:
    from grasp.provenance.paths import pairs_query

    query = pairs_query(
        "", [(1, (Step("x:a"),))], [("<s1>", "<o1>"), ("<s2>", "<o2>")], 10
    )
    assert query == (
        "SELECT * WHERE { { VALUES (?_path_s ?_path_o) { (<s1> <o1>) (<s2> <o2>) } "
        "?_path_s x:a ?_path_o . BIND(1 AS ?_path_chain) } } LIMIT 10"
    )
