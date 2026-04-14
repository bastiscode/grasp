from grasp.sparql.utils import (
    fix_prefixes,
    load_iri_and_literal_parser,
    load_sparql_parser,
)

SPARQL_PARSER = load_sparql_parser()
IRI_PARSER = load_iri_and_literal_parser()

PREFIXES = {
    "wd": "http://example.org/entity/",
    "wdt": "http://example.org/prop/",
}


def _fix(sparql: str, **kwargs) -> str:
    return fix_prefixes(sparql, SPARQL_PARSER, IRI_PARSER, PREFIXES, **kwargs)


class TestFixPrefixes:
    def test_replaces_iri_with_prefix(self):
        result = _fix("SELECT ?s WHERE { ?s <http://example.org/prop/p1> ?o }")
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE { ?s wdt:p1 ?o }"
        )

    def test_preserves_spaces(self):
        result = _fix(
            "SELECT  ?s  WHERE  {  ?s  <http://example.org/prop/p1>  ?o  }"
        )
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT  ?s  WHERE  {  ?s  wdt:p1  ?o  }"
        )

    def test_preserves_newlines_and_indentation(self):
        result = _fix("SELECT ?s WHERE {\n  ?s <http://example.org/prop/p1> ?o\n}")
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE {\n  ?s wdt:p1 ?o\n}"
        )

    def test_preserves_tabs(self):
        result = _fix(
            "SELECT\t?s\tWHERE\t{\n\t?s\t<http://example.org/prop/p1>\t?o\n}"
        )
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT\t?s\tWHERE\t{\n\t?s\twdt:p1\t?o\n}"
        )

    def test_existing_prefix(self):
        result = _fix(
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT ?s WHERE { ?s wd:e1 ?o }"
        )
        assert result == (
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT ?s WHERE { ?s wd:e1 ?o }"
        )

    def test_existing_prefix_whitespace_preserved(self):
        result = _fix(
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT  ?s  WHERE  {\n  ?s  wd:e1  ?o\n}"
        )
        assert result == (
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT  ?s  WHERE  {\n  ?s  wd:e1  ?o\n}"
        )

    def test_no_prefixes_needed(self):
        result = _fix("SELECT  ?s  WHERE  {\n  ?s  ?p  ?o\n}")
        assert result == "SELECT  ?s  WHERE  {\n  ?s  ?p  ?o\n}"

    def test_remove_known(self):
        result = _fix(
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT ?s WHERE { ?s wd:e1 ?o }",
            remove_known=True,
        )
        assert result == "SELECT ?s WHERE { ?s wd:e1 ?o }"

    def test_sort_prefixes(self):
        result = _fix(
            "SELECT ?s WHERE { "
            "?s <http://example.org/prop/p1> <http://example.org/entity/e1> "
            "}",
            sort=True,
        )
        assert result == (
            "PREFIX wd: <http://example.org/entity/>\n"
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE { ?s wdt:p1 wd:e1 }"
        )

    def test_unknown_iri_not_replaced(self):
        result = _fix("SELECT ?s WHERE { ?s <http://unknown.org/foo> ?o }")
        assert result == "SELECT ?s WHERE { ?s <http://unknown.org/foo> ?o }"

    def test_preserves_comments(self):
        result = _fix(
            "SELECT ?s WHERE {\n"
            "  # find all properties of entity\n"
            "  ?s <http://example.org/prop/p1> ?o\n"
            "}"
        )
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE {\n"
            "  # find all properties of entity\n"
            "  ?s wdt:p1 ?o\n"
            "}"
        )
