import os
import re
from typing import Any

import requests

from grasp.manager import KgManager
from grasp.sparql.types import Binding

# graphs on the scratch server, the sample of the knowledge graph and one
# graph with the output of each query
GRAPHS = "http://grasp.cs.uni-freiburg.de/construct/"
SAMPLE_GRAPH = GRAPHS + "sample"

ACCESS_TOKEN_ENV = "GRASP_SCRATCH_ACCESS_TOKEN"

Triple = tuple[Binding, str, Binding]

TERM = r'<[^>]*>|_:\S+|"(?:[^"\\]|\\.)*"(?:\^\^<[^>]*>|@[A-Za-z0-9-]+)?'
TRIPLE = re.compile(rf"\s*({TERM})\s+<([^>]*)>\s+({TERM})\s*\.\s*$")
ESCAPE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)")
ESCAPES = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f"}
# characters not allowed in IRIs
INVALID_IRI = re.compile(r'[\x00-\x20<>"{}|^`\\]')


def draft_graph(name: str) -> str:
    return f"{GRAPHS}draft/{name}"


def unescape(value: str) -> str:
    def replace(match: re.Match) -> str:
        escape: str = match.group(1)
        if len(escape) > 1:
            return chr(int(escape[1:], 16))
        return ESCAPES.get(escape, escape)

    return ESCAPE.sub(replace, value)


def parse_term(term: str) -> Binding:
    if term.startswith("<"):
        return Binding("uri", unescape(term[1:-1]))
    elif term.startswith("_:"):
        return Binding("bnode", term[2:])

    end = term.rindex('"')
    value = unescape(term[1:end])
    suffix = term[end + 1 :]
    if suffix.startswith("^^"):
        return Binding("literal", value, datatype=suffix[3:-1])
    elif suffix:
        return Binding("literal", value, lang=suffix[1:])
    return Binding("literal", value)


def parse_ntriples(text: str) -> list[Triple]:
    triples = []
    # not splitlines, literals can contain other line separators
    for line in text.split("\n"):
        if not line.strip():
            continue

        match = TRIPLE.match(line)
        if match is None:
            raise ValueError(f"Invalid N-Triples line: {line}")

        subj, prop, obj = match.groups()
        triples.append((parse_term(subj), unescape(prop), parse_term(obj)))

    return triples


def invalid_iris(triples: list[Triple]) -> list[str]:
    invalid = set()
    for subj, prop, obj in triples:
        iris = [term.value for term in (subj, obj) if term.typ == "uri"]
        invalid.update(iri for iri in [*iris, prop] if INVALID_IRI.search(iri))
    return sorted(invalid)


def check_response(response: requests.Response) -> None:
    if response.ok:
        return

    try:
        message = response.json()["exception"]
    except Exception:
        message = response.text[:1000]
    raise RuntimeError(f"Status {response.status_code}: {message}")


def construct(
    manager: KgManager,
    sparql: str,
    timeout: float,
    stream: bool = False,
) -> requests.Response:
    # the result of a CONSTRUCT query on a knowledge graph as N-Triples
    params: dict[str, Any] = dict(manager.params)
    if manager.backend == "qlever" and not stream:
        # streamed results of long queries run up to the timeout of the
        # server, QLever rejects timeouts above it
        params["timeout"] = f"{round(timeout)}s"

    response = requests.post(
        manager.endpoint,
        data={**params, "query": manager.fix_prefixes(sparql)},
        headers={
            "Accept": "application/n-triples",
            "User-Agent": "grasp-rdf",
            **manager.headers,
        },
        timeout=(6.0, timeout),
        auth=manager.auth,
        stream=stream,
    )
    check_response(response)
    response.encoding = "utf-8"
    return response


class Scratch:
    # QLever server for the sample and the output of the queries on it,
    # written to with its access token
    def __init__(self, endpoint: str, timeout: float = 60.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        token = os.getenv(ACCESS_TOKEN_ENV)
        if token is None:
            raise ValueError(
                f"Set {ACCESS_TOKEN_ENV} to the access token of the scratch "
                f"server at {endpoint}"
            )
        self.token = token

    def manager(
        self,
        kg: str,
        graphs: list[str],
        prefixes: dict[str, str] | None = None,
    ) -> KgManager:
        # queries see the union of the given graphs, the default graph of
        # QLever is the union of all graphs
        return KgManager(
            kg,
            prefixes=prefixes,
            endpoint=self.endpoint,
            params={"default-graph-uri": graphs},
        )

    def construct(self, sparql: str, graphs: list[str]) -> str:
        return construct(self.manager("scratch", graphs), sparql, self.timeout).text

    def load(self, graph: str, ntriples: str, replace: bool = True) -> None:
        # with the graph store protocol, blank nodes in the template of an
        # INSERT are shared between all solutions in QLever
        response = requests.request(
            "PUT" if replace else "POST",
            self.endpoint,
            params={"graph": graph, "access-token": self.token},
            data=ntriples.encode(),
            headers={"Content-Type": "application/n-triples"},
            timeout=self.timeout,
        )
        check_response(response)

    def update(self, sparql: str) -> None:
        response = requests.post(
            self.endpoint,
            data={"update": sparql, "access-token": self.token},
            timeout=self.timeout,
        )
        check_response(response)

    def drop(self, graph: str) -> None:
        self.update(f"DROP SILENT GRAPH <{graph}>")

    def graphs(self) -> list[str]:
        result = self.manager("scratch", []).execute_sparql(
            "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }",
            request_timeout=(6.0, self.timeout),
            read_timeout=self.timeout,
        )
        return sorted(
            row["g"].value
            for row in result.rows()  # type: ignore
            if row["g"].value.startswith(GRAPHS)
        )
