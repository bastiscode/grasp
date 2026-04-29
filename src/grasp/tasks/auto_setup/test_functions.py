from grasp.sparql.utils import load_sparql_parser, parse_string
from grasp.tasks.auto_setup.functions import (
    find_select_clause,
    find_solution_modifier,
    validate_index_sparql,
    validate_info_sparql,
)

SPARQL_PARSER = load_sparql_parser()


class TestFindOuterClauses:
    def test_outer_solution_modifier_with_subselect(self):
        sparql = (
            "SELECT ?id ?value ?tags WHERE {\n"
            "  { SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?x LIMIT 5 }\n"
            "} ORDER BY DESC(?score) ?id DESC(?tags)"
        )
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        out = find_solution_modifier(parse, sparql.encode())
        assert out == "ORDER BY DESC(?score) ?id DESC(?tags)"

    def test_outer_select_clause_with_subselect(self):
        sparql = (
            "SELECT ?id ?value ?tags WHERE {\n"
            "  { SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?x }\n"
            "} ORDER BY ?id"
        )
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        out = find_select_clause(parse, sparql.encode())
        assert out == "SELECT ?id ?value ?tags"


class TestValidateIndexSparql:
    def test_canonical_passes(self):
        sparql = (
            "SELECT ?id ?value ?tags WHERE {\n"
            '  ?s ?p ?o . BIND(?s AS ?id) BIND(?o AS ?value) BIND("tag" AS ?tags)\n'
            "  BIND(1.0 AS ?score)\n"
            "} ORDER BY DESC(?score) ?id DESC(?tags)"
        )
        validate_index_sparql(SPARQL_PARSER, sparql)

    def test_wrong_select_rejected(self):
        sparql = (
            "SELECT ?id ?value WHERE { ?s ?p ?o } ORDER BY DESC(?score) ?id DESC(?tags)"
        )
        try:
            validate_index_sparql(SPARQL_PARSER, sparql)
        except ValueError as e:
            assert "SELECT clause must be" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_wrong_solution_modifier_rejected(self):
        sparql = "SELECT ?id ?value ?tags WHERE { ?s ?p ?o } ORDER BY ?id"
        try:
            validate_index_sparql(SPARQL_PARSER, sparql)
        except ValueError as e:
            assert "Solution modifier must be" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestValidateInfoSparql:
    def test_canonical_passes(self):
        sparql = (
            "SELECT ?id ?value ?type WHERE {\n  ?s ?p ?o\n} ORDER BY ?id ?type ?value"
        )
        validate_info_sparql(SPARQL_PARSER, sparql)
