"""
Why-provenance for SPARQL answers via query rewriting.

For one result row of a SELECT query, a witness query keeps the WHERE clause,
drops grouping, aggregation, ordering and slicing, binds the row's projected
variables with VALUES and returns all variables. Substituting each returned
solution into the triple patterns of the WHERE clause yields one witness: a set
of concrete triples from the knowledge graph that produce the row.

OPTIONAL and UNION branches are instrumented with marker variables, so that
only the patterns of branches that actually matched are instantiated. Patterns
in MINUS, (NOT) EXISTS and SERVICE are never instantiated; FILTERs are reported
as conditions that held. Everything here is pure and works on any knowledge
graph; execution and labeling happen in grasp.provenance.
"""

from dataclasses import dataclass, field
from typing import Any

from grammar_utils.parse import LR1Parser  # type: ignore

from grasp.sparql.types import Binding, SelectRow
from grasp.sparql.utils import (
    SPARQLException,
    find,
    find_all,
    find_terminals,
    parse_into_binding,
    parse_string,
    parse_to_string,
    span,
)

# marker variables are recognizable by this prefix and hidden from witnesses
MARKER_PREFIX = "_prov_m"

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# patterns below these nodes never contribute witness triples
EXCLUDED = {
    "MinusGraphPattern",
    "ServiceGraphPattern",
    "ExistsFunc",
    "NotExistsFunc",
    "Bind",
    "InlineData",
}

IRI_TERMINALS = {"IRIREF", "PNAME_LN", "PNAME_NS"}


@dataclass
class TriplePattern:
    # terms as written in the query; variables keep their ? or $ sigil
    subject: str
    predicate: str
    object: str
    # predicate is a property path other than a single (inverse) IRI
    is_path: bool
    # markers of all enclosing OPTIONAL / UNION branches
    markers: tuple[str, ...]
    # inside a subquery, whose variables the outer query may not see
    in_subquery: bool = False
    # character spans of subject, predicate and object in the query
    spans: tuple[tuple[int, int], ...] = ()


@dataclass
class FilterPattern:
    text: str
    variables: list[str]
    markers: tuple[str, ...]


@dataclass
class WitnessPlan:
    query_type: str
    prologue: str
    dataset: str
    # instrumented WHERE clause group graph pattern, including a trailing
    # top-level VALUES clause of the original query if there was one
    where: str
    # projected variables (without sigil) that can be bound via VALUES,
    # None means all variables of a row (SELECT *)
    bindable: list[str] | None
    triples: list[TriplePattern]
    filters: list[FilterPattern]
    prefixes: dict[str, str]
    # parts of the query that are never instantiated
    notes: list[str] = field(default_factory=list)


@dataclass
class WitnessTriple:
    subject: Binding
    predicate: Binding | str
    object: Binding
    # for property paths the predicate is the path as written in the query
    is_path: bool = False
    # names of the variables subject, predicate and object are bound from,
    # None for constants written in the query
    variables: tuple[str | None, str | None, str | None] = (None, None, None)
    # character spans of the triple pattern in the query
    spans: tuple[tuple[int, int], ...] = ()

    def key(self) -> tuple:
        return (self.subject, str(self.predicate), self.object)


@dataclass
class Condition:
    text: str
    bindings: dict[str, Binding]


@dataclass
class Witness:
    triples: list[WitnessTriple]
    conditions: list[Condition]
    bindings: dict[str, Binding]


def char_offset(data: bytes, offset: int) -> int:
    # parse positions are byte offsets, the website needs offsets in UTF-16
    # code units as used by JavaScript strings
    return len(data[:offset].decode(errors="replace").encode("utf-16-le")) // 2


def var_name(var: str) -> str:
    return var[1:]


def is_var(term: str) -> bool:
    return term.startswith("?") or term.startswith("$")


def variables_of(node: dict) -> list[str]:
    seen = []
    for var in find_all(node, {"VAR1", "VAR2"}):
        name = var_name(var["value"])
        if name not in seen:
            seen.append(name)
    return seen


def raw(value: str) -> dict:
    return {"name": "_raw", "value": value}


class Planner:
    def __init__(self, data: bytes) -> None:
        # the query, for the spans of triple patterns
        self.data = data
        self.triples: list[TriplePattern] = []
        self.filters: list[FilterPattern] = []
        self.notes: list[str] = []
        self.num_markers = 0
        # filters inside subqueries constrain bindings that witnesses of the
        # outer query do not see, so they are not reported as conditions
        self.subquery_depth = 0

    def new_marker(self) -> str:
        marker = f"?{MARKER_PREFIX}{self.num_markers}"
        self.num_markers += 1
        return marker

    def walk(self, node: dict, markers: tuple[str, ...]) -> None:
        name = node["name"]

        if name in EXCLUDED:
            if name == "ServiceGraphPattern":
                self.notes.append(
                    f"SERVICE pattern not instantiated: {parse_to_string(node)}"
                )
            return

        elif name == "Filter":
            if self.subquery_depth == 0:
                self.filters.append(
                    FilterPattern(parse_to_string(node), variables_of(node), markers)
                )
            return

        elif name == "SubSelect":
            self.subquery_depth += 1
            for child in node.get("children", []):
                self.walk(child, markers)
            self.subquery_depth -= 1
            return

        elif name == "TriplesSameSubjectPath":
            self.extract_triples(node, markers)
            return

        elif name == "OptionalGraphPattern":
            ggp = find(node, "GroupGraphPattern")
            assert ggp is not None, "OPTIONAL without group graph pattern"
            self.walk_branch(ggp, markers)
            return

        elif name == "GroupOrUnionGraphPattern" and find_child(node, "UNION"):
            for ggp in union_branches(node):
                self.walk_branch(ggp, markers)
            return

        for child in node.get("children", []):
            self.walk(child, markers)

    def walk_branch(self, ggp: dict, markers: tuple[str, ...]) -> None:
        marker = self.new_marker()
        self.walk(ggp, markers + (marker,))
        add_marker(ggp, marker)

    def extract_triples(self, node: dict, markers: tuple[str, ...]) -> None:
        children = node["children"]
        if children[0]["name"] != "VarOrTerm":
            self.notes.append(
                f"Blank node pattern not instantiated: {parse_to_string(node)}"
            )
            return

        subject = parse_to_string(children[0])
        for verb, objects in verb_object_pairs(children[1]):
            predicate, is_path, inverse = verb_term(verb)
            for obj in object_nodes(objects):
                term = obj["children"][0]
                if term["name"] != "VarOrTerm":
                    self.notes.append(
                        f"Blank node pattern not instantiated: {parse_to_string(obj)}"
                    )
                    continue

                s, o = subject, parse_to_string(term)
                if inverse:
                    s, o = o, s

                spans = []
                for part in (children[0], verb, term):
                    part_span = span(part)
                    if part_span is not None:
                        start, end = part_span
                        spans.append(
                            (char_offset(self.data, start), char_offset(self.data, end))
                        )

                self.triples.append(
                    TriplePattern(
                        s,
                        predicate,
                        o,
                        is_path,
                        markers,
                        self.subquery_depth > 0,
                        tuple(spans),
                    )
                )


def find_child(node: dict, name: str) -> dict | None:
    return next((c for c in node.get("children", []) if c["name"] == name), None)


def union_branches(node: dict) -> list[dict]:
    # GroupOrUnionGraphPattern: GroupGraphPattern ('UNION' GroupOrUnionGraphPattern)?
    branches = []
    while node is not None:
        ggp = find_child(node, "GroupGraphPattern")
        assert ggp is not None, "UNION without group graph pattern"
        branches.append(ggp)
        node = find_child(node, "GroupOrUnionGraphPattern")  # type: ignore
    return branches


def add_marker(ggp: dict, marker: str) -> None:
    # bind the marker at the end of the group; it is bound in a solution
    # exactly if the group matched
    bind = raw(f"BIND(true AS {marker})")
    sub = find_child(ggp, "GroupGraphPatternSub")
    if sub is not None:
        sub["children"].append(bind)
        return

    sub_select = find_child(ggp, "SubSelect")
    if sub_select is not None:
        inner = {"name": "GroupGraphPattern", "children": ggp["children"]}
        ggp["children"] = [raw("{"), inner, bind, raw("}")]
        return

    # empty group {}
    ggp["children"] = [raw("{"), bind, raw("}")]


def verb_object_pairs(node: dict) -> list[tuple[dict, dict]]:
    # PropertyListPathNotEmpty: VerbPathOrSimple ObjectListPath
    #   (';' (VerbPathOrSimple ObjectList)?)*
    pairs = []
    verb = None
    for child in node.get("children", []):
        if child["name"] == "VerbPathOrSimple":
            verb = child
        elif child["name"] in {"ObjectListPath", "ObjectList"}:
            assert verb is not None, "object list without verb"
            pairs.append((verb, child))
        elif child["name"] == "VerbPathObjectListOptional":
            pairs.extend(verb_object_pairs(child))
    return pairs


def object_nodes(node: dict) -> list[dict]:
    # ObjectListPath: ObjectPath (',' ObjectListPath)?, same for ObjectList;
    # returns the GraphNodePath / GraphNode nodes
    objects = []
    for child in node.get("children", []):
        if child["name"] in {"ObjectPath", "Object"}:
            objects.append(child["children"][0])
        elif child["name"] in {"ObjectListPath", "ObjectList"}:
            objects.extend(object_nodes(child))
    return objects


def verb_term(verb: dict) -> tuple[str, bool, bool]:
    # returns (predicate, is_path, inverse)
    text = parse_to_string(verb)
    terminals = list(find_terminals(verb))
    if len(terminals) == 1:
        terminal = terminals[0]
        if terminal["name"] in IRI_TERMINALS or terminal["name"] in {"VAR1", "VAR2"}:
            return terminal["value"], False, False
        elif terminal["value"] == "a":
            return f"<{RDF_TYPE}>", False, False

    elif (
        len(terminals) == 2
        and terminals[0]["value"] == "^"
        and terminals[1]["name"] in IRI_TERMINALS
    ):
        return terminals[1]["value"], False, True

    return text, True, False


def parse_prefixes(prologue: dict | None) -> dict[str, str]:
    prefixes = {}
    if prologue is None:
        return prefixes

    for decl in find_all(prologue, "PrefixDecl"):
        pfx = find(decl, "PNAME_NS")
        iri = find(decl, "IRIREF")
        if pfx is None or iri is None:
            continue
        prefixes[pfx["value"][:-1]] = iri["value"][1:-1]
    return prefixes


def plan_witness(sparql: str, parser: LR1Parser) -> WitnessPlan:
    parse, _ = parse_string(sparql, parser, skip_empty=True, collapse_single=False)

    query = find(parse, "Query")
    if query is None:
        raise SPARQLException("only queries are supported", sparql)

    query_type = find(query, "QueryType")
    assert query_type is not None
    typed = query_type["children"][0]
    if typed["name"] not in {"SelectQuery", "AskQuery"}:
        raise SPARQLException("only SELECT and ASK queries are supported", sparql)

    prologue = find_child(query, "Prologue")
    where_clause = find_child(typed, "WhereClause")
    assert where_clause is not None
    ggp = find_child(where_clause, "GroupGraphPattern")
    assert ggp is not None

    datasets = [
        parse_to_string(dataset)
        for dataset in find_all(typed, "DatasetClause", skip={"WhereClause"})
    ]

    # projected variables that are in scope of the WHERE clause; aggregates
    # and expressions are computed afterwards and cannot be bound
    bindable: list[str] | None = None
    if typed["name"] == "SelectQuery":
        select_clause = find_child(typed, "SelectClause")
        assert select_clause is not None
        if find_child(select_clause, "*") is None:
            where_vars = set(variables_of(ggp))
            bindable = [
                var_name(var["children"][0]["value"])
                for select_var in find_all(select_clause, "SelectVar")
                if (var := find_child(select_var, "Var")) is not None
            ]
            bindable = [var for var in bindable if var in where_vars]

    planner = Planner(sparql.encode())
    planner.walk(ggp, ())

    if not planner.triples and find(ggp, "TriplesSameSubjectPath") is not None:
        planner.notes.append(
            "All triple patterns of the query are inside FILTER (NOT) EXISTS, "
            "MINUS or SERVICE, so no knowledge graph triples witness its result. "
            "NOT EXISTS and MINUS also hold if nothing matches at all, so the "
            "result may be vacuously true"
        )
    elif not planner.triples:
        planner.notes.append(
            "The query has no triple patterns, so its result is not derived "
            "from knowledge graph triples (e.g. constants in VALUES or BIND)"
        )

    where = parse_to_string(ggp)
    values = find_child(query, "ValuesClause")
    if values is not None and values.get("children"):
        where = f"{where} {parse_to_string(values)}"

    return WitnessPlan(
        query_type="select" if typed["name"] == "SelectQuery" else "ask",
        prologue=parse_to_string(prologue) if prologue is not None else "",
        dataset=" ".join(datasets),
        where=where,
        bindable=bindable,
        triples=planner.triples,
        filters=planner.filters,
        prefixes=parse_prefixes(prologue),
        notes=planner.notes,
    )


def escape_string(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def binding_to_sparql(binding: Binding) -> str | None:
    # None for values that cannot be bound (blank nodes)
    if binding.typ == "uri":
        return f"<{binding.value}>"
    elif binding.typ == "literal":
        lit = f'"{escape_string(binding.value)}"'
        if binding.lang is not None:
            return f"{lit}@{binding.lang}"
        elif binding.datatype is not None:
            return f"{lit}^^<{binding.datatype}>"
        return lit
    return None


def row_values(
    plan: WitnessPlan,
    row: SelectRow | None,
    bind_literals: bool = True,
) -> str:
    if row is None:
        return ""

    variables = plan.bindable
    if variables is None:
        variables = list(row)

    terms = []
    bound = []
    unbound = []
    for var in variables:
        binding = row.get(var)
        if binding is None:
            # unbound in the row, e.g. from another UNION branch or an OPTIONAL
            # that did not match; UNDEF would allow any value
            unbound.append(f"!BOUND(?{var})")
            continue
        elif binding.typ == "literal" and not bind_literals:
            term = None
        else:
            term = binding_to_sparql(binding)
        bound.append(f"?{var}")
        terms.append(term or "UNDEF")

    parts = []
    if bound and any(term != "UNDEF" for term in terms):
        parts.append(f"VALUES ({' '.join(bound)}) {{ ({' '.join(terms)}) }}")
    if unbound:
        parts.append(f"FILTER({' && '.join(unbound)})")
    return " ".join(parts)


def witness_query(
    plan: WitnessPlan,
    row: SelectRow | None,
    limit: int,
    bind_literals: bool = True,
) -> str:
    # the original WHERE clause is nested as its own group, so the VALUES
    # clause joins with its result instead of interfering with BINDs inside it
    parts = [plan.prologue, "SELECT DISTINCT *"]
    if plan.dataset:
        parts.append(plan.dataset)

    values = row_values(plan, row, bind_literals)
    parts.append(f"WHERE {{ {plan.where} {values} }}")
    parts.append(f"LIMIT {limit}")
    return " ".join(part for part in parts if part)


def matches_row(plan: WitnessPlan, row: SelectRow, witness_row: SelectRow) -> bool:
    # client-side check of the bindable variables, used when literals could
    # not be bound in the witness query itself
    variables = plan.bindable if plan.bindable is not None else list(row)
    for var in variables:
        expected = row.get(var)
        actual = witness_row.get(var)
        if expected is None:
            if actual is not None:
                return False
        elif expected.typ != "bnode" and (
            actual is None or actual.value != expected.value
        ):
            return False
    return True


def is_marker(var: str) -> bool:
    return var.startswith(MARKER_PREFIX)


def resolve(
    term: str,
    row: SelectRow,
    parser: LR1Parser,
    prefixes: dict[str, str],
) -> Binding | None:
    if is_var(term):
        return row.get(var_name(term))

    binding = parse_into_binding(term, parser, prefixes)
    if binding is None:
        # e.g. a numeric or boolean literal the IRI/literal grammar rejects
        return Binding(typ="literal", value=term)
    return binding


def instantiate(
    plan: WitnessPlan,
    rows: list[SelectRow],
    parser: LR1Parser,
    prefixes: dict[str, str],
) -> list[Witness]:
    # parser is the IRI and literal parser; prefixes should include the ones
    # declared in the query (plan.prefixes)
    witnesses = []
    for row in rows:
        active = {f"?{var}" for var in row if is_marker(var)}
        triples: list[WitnessTriple] = []
        seen = set()
        for pattern in plan.triples:
            if not active.issuperset(pattern.markers):
                continue

            subject = resolve(pattern.subject, row, parser, prefixes)
            obj = resolve(pattern.object, row, parser, prefixes)
            if pattern.is_path:
                predicate: Binding | str | None = pattern.predicate
            else:
                predicate = resolve(pattern.predicate, row, parser, prefixes)

            if subject is None or predicate is None or obj is None:
                # unbound variable, e.g. hidden inside a subquery
                continue

            triple = WitnessTriple(
                subject,
                predicate,
                obj,
                pattern.is_path,
                tuple(
                    var_name(term) if is_var(term) else None
                    for term in (pattern.subject, pattern.predicate, pattern.object)
                ),  # type: ignore
                pattern.spans,
            )
            if triple.key() in seen:
                continue
            seen.add(triple.key())
            triples.append(triple)

        conditions = [
            Condition(
                flt.text,
                {var: row[var] for var in flt.variables if var in row},
            )
            for flt in plan.filters
            if active.issuperset(flt.markers)
        ]

        witnesses.append(
            Witness(
                triples,
                conditions,
                {var: b for var, b in row.items() if not is_marker(var)},
            )
        )

    return witnesses


def binding_to_dict(binding: Binding) -> dict[str, Any]:
    data: dict[str, Any] = {"type": binding.typ, "value": binding.value}
    if binding.datatype is not None:
        data["datatype"] = binding.datatype
    if binding.lang is not None:
        data["lang"] = binding.lang
    return data


def types_query(instance_pattern: str, iris: list[str]) -> str:
    # classes of the given entities, using the knowledge graph's instance
    # pattern from the shapes setup (e.g. "?instance wdt:P31 {CLASS}")
    pattern = instance_pattern.replace("{CLASS}", "?class")
    values = " ".join(f"<{iri}>" for iri in iris)
    return (
        "SELECT DISTINCT ?instance ?class WHERE { "
        f"VALUES ?instance {{ {values} }} {{ {pattern} }} FILTER(isIRI(?class)) }}"
    )
