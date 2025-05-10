from search_index import PrefixIndex, QGramIndex, SearchIndex


def get_common_sparql_prefixes() -> dict[str, str]:
    return {
        "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "<http://www.w3.org/2000/01/rdf-schema#",
        "owl": "<http://www.w3.org/2002/07/owl#",
        "xsd": "<http://www.w3.org/2001/XMLSchema#",
        "foaf": "<http://xmlns.com/foaf/0.1/",
        "skos": "<http://www.w3.org/2004/02/skos/core#",
        "dct": "<http://purl.org/dc/terms/",
        "dc": "<http://purl.org/dc/elements/1.1/",
        "prov": "<http://www.w3.org/ns/prov#",
        "schema": "<http://schema.org/",
        "geo": "<http://www.opengis.net/ont/geosparql#",
        "geosparql": "<http://www.opengis.net/ont/geosparql#",
        "gn": "<http://www.geonames.org/ontology#",
        "bd": "<http://www.bigdata.com/rdf#",
        "hint": "<http://www.bigdata.com/queryHints#",
        "wikibase": "<http://wikiba.se/ontology#",
        "qb": "<http://purl.org/linked-data/cube#",
        "void": "<http://rdfs.org/ns/void#",
    }


def get_index_desc(index: SearchIndex | None = None) -> str:
    # prefix index is the default
    if isinstance(index, PrefixIndex) or index is None:
        index_type = "Keyword index"
        dist_info = "number of exact and prefix keyword matches"

    elif isinstance(index, QGramIndex):
        assert isinstance(index, QGramIndex)
        if index.distance == "ied":
            dist = "substring"
        else:
            dist = "prefix"
        index_type = "Fuzzy n-gram index"
        dist_info = f"{dist} edit distance"

    else:
        index_type = "Vector embedding index"
        dist_info = "cosine similarity"

    return f"{index_type} ranking by {dist_info}"
