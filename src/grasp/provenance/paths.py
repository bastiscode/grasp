"""
Expansion of property paths in witnesses.

A witness shows a property path only by its two endpoints. To show why the
endpoints are connected, find one concrete chain of triples between them:
enumerate the fixed-length chains the path allows (sequences, alternatives,
inverses, negated property sets, repetitions up to a maximum length), and query
for the shortest one that exists, with both endpoints bound (iterative
deepening, one query per chain length). Engine-agnostic, see PATHS.md.
"""

import itertools
from dataclasses import dataclass

from grasp.sparql.types import Binding, SelectRow

# upper bound on the number of chains tried per path
MAX_CHAINS = 32


@dataclass(frozen=True)
class Step:
    # predicate as written in the query, None for a negated property set
    predicate: str | None
    inverse: bool = False
    # predicates excluded by a negated property set
    excluded: tuple[str, ...] = ()


Chain = tuple[Step, ...]


class PathSyntaxError(Exception):
    pass


class PathParser:
    # recursive descent over the whitespace-separated tokens of a path as
    # produced by parse_to_string, e.g. "( wdt:P31 | ^ wdt:P361 ) +"
    def __init__(self, text: str) -> None:
        self.tokens = text.split()
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> str:
        token = self.peek()
        if token is None:
            raise PathSyntaxError("unexpected end of path")
        self.pos += 1
        return token

    def expect(self, token: str) -> None:
        if self.next() != token:
            raise PathSyntaxError(f"expected {token}")

    def parse(self) -> tuple:
        node = self.alternative()
        if self.peek() is not None:
            raise PathSyntaxError(f"unexpected token {self.peek()}")
        return node

    def alternative(self) -> tuple:
        options = [self.sequence()]
        while self.peek() == "|":
            self.next()
            options.append(self.sequence())
        return options[0] if len(options) == 1 else ("alt", options)

    def sequence(self) -> tuple:
        elements = [self.element()]
        while self.peek() == "/":
            self.next()
            elements.append(self.element())
        return elements[0] if len(elements) == 1 else ("seq", elements)

    def element(self) -> tuple:
        inverse = False
        if self.peek() == "^":
            self.next()
            inverse = True
        node = self.primary()
        if self.peek() in {"*", "+", "?"}:
            node = ("mod", node, self.next())
        return ("inv", node) if inverse else node

    def primary(self) -> tuple:
        token = self.next()
        if token == "(":
            node = self.alternative()
            self.expect(")")
            return node
        elif token == "!":
            return ("neg", self.negated())
        elif token == "a":
            return ("iri", "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
        elif token in {")", "|", "/", "*", "+", "?", "^"}:
            raise PathSyntaxError(f"unexpected token {token}")
        return ("iri", token)

    def negated(self) -> list[tuple[str, bool]]:
        if self.peek() != "(":
            return [self.negated_one()]
        self.next()
        members = [self.negated_one()]
        while self.peek() == "|":
            self.next()
            members.append(self.negated_one())
        self.expect(")")
        return members

    def negated_one(self) -> tuple[str, bool]:
        inverse = False
        if self.peek() == "^":
            self.next()
            inverse = True
        token = self.next()
        if token == "a":
            token = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
        return token, inverse


def invert(chain: Chain) -> Chain:
    return tuple(
        Step(step.predicate, not step.inverse, step.excluded)
        for step in reversed(chain)
    )


def chains(node: tuple, max_length: int) -> list[Chain]:
    # all chains of at most max_length steps that the path node allows
    kind = node[0]
    if kind == "iri":
        return [(Step(node[1]),)] if max_length >= 1 else []

    elif kind == "neg":
        if max_length < 1:
            return []
        forward = tuple(p for p, inverse in node[1] if not inverse)
        backward = tuple(p for p, inverse in node[1] if inverse)
        result = []
        if forward or not backward:
            result.append((Step(None, False, forward),))
        if backward:
            result.append((Step(None, True, backward),))
        return result

    elif kind == "inv":
        return [invert(chain) for chain in chains(node[1], max_length)]

    elif kind == "alt":
        return [c for option in node[1] for c in chains(option, max_length)]

    elif kind == "seq":
        result: list[Chain] = [()]
        for element in node[1]:
            options = chains(element, max_length)
            result = [
                prefix + suffix
                for prefix, suffix in itertools.product(result, options)
                if len(prefix) + len(suffix) <= max_length
            ]
        return result

    assert kind == "mod", f"unknown path node {kind}"
    inner = [c for c in chains(node[1], max_length) if c]
    if node[2] == "?":
        return [()] + inner

    # repetitions: concatenations of inner chains up to max_length steps
    result = [()] if node[2] == "*" else []
    frontier: list[Chain] = [()]
    while frontier:
        frontier = [
            prefix + suffix
            for prefix in frontier
            for suffix in inner
            if len(prefix) + len(suffix) <= max_length
        ]
        result.extend(frontier)
        if len(result) > MAX_CHAINS * 4:
            break
    return result


def path_chains(path: str, max_length: int) -> list[Chain]:
    """
    Distinct chains of at most max_length steps for a property path, shortest
    first, at most MAX_CHAINS of them. Raises PathSyntaxError for paths that
    cannot be parsed.
    """
    node = PathParser(path).parse()
    unique = list(dict.fromkeys(chains(node, max_length)))
    unique.sort(key=len)
    return unique[:MAX_CHAINS]


def chain_patterns(chain: Chain, subject: str, obj: str, index: int) -> str:
    # triple patterns for one chain between the given endpoint terms; fresh
    # variables are named after the chain index and step
    nodes = [subject] + [f"?_path{index}_n{i}" for i in range(1, len(chain))] + [obj]
    patterns = []
    for i, step in enumerate(chain):
        s, o = nodes[i], nodes[i + 1]
        if step.inverse:
            s, o = o, s
        if step.predicate is not None:
            patterns.append(f"{s} {step.predicate} {o} .")
        else:
            var = f"?_path{index}_p{i}"
            patterns.append(f"{s} {var} {o} .")
            if step.excluded:
                patterns.append(f"FILTER({var} NOT IN ({', '.join(step.excluded)}))")
    return " ".join(patterns)


def expansion_query(
    prologue: str,
    chains_: list[tuple[int, Chain]],
    subject: str,
    obj: str,
) -> str:
    # one query for several chains of the same length; ?_path_chain tells
    # which of them matched
    branches = [
        f"{{ {chain_patterns(chain, subject, obj, index)} "
        f"BIND({index} AS ?_path_chain) }}"
        for index, chain in chains_
    ]
    where = " UNION ".join(branches)
    return f"{prologue} SELECT * WHERE {{ {where} }} LIMIT 1".strip()


def chain_steps(
    chain: Chain,
    index: int,
    subject: Binding,
    obj: Binding,
    row: SelectRow,
) -> list[tuple[Binding, Binding]] | None:
    # (from, to) nodes of each step in chain order, from subject to object,
    # for a result row of the expansion query
    nodes: list[Binding | None] = (
        [subject]
        + [row.get(f"_path{index}_n{i}") for i in range(1, len(chain))]
        + [obj]
    )
    steps = []
    for start, end in zip(nodes, nodes[1:]):
        if start is None or end is None:
            return None
        steps.append((start, end))
    return steps


def pairs_query(
    prologue: str,
    chains_: list[tuple[int, Chain]],
    pairs: list[tuple[str, str]],
    limit: int,
) -> str:
    # like expansion_query, but for many endpoint pairs at once, bound via
    # VALUES to ?_path_s and ?_path_o; returns all matching chains of all
    # pairs. VALUES is repeated inside each branch, so that engines join it
    # with the chain patterns first (QLever otherwise evaluates the chain
    # patterns for all endpoints)
    values = " ".join(f"({s} {o})" for s, o in pairs)
    branches = [
        f"{{ VALUES (?_path_s ?_path_o) {{ {values} }} "
        f"{chain_patterns(chain, '?_path_s', '?_path_o', index)} "
        f"BIND({index} AS ?_path_chain) }}"
        for index, chain in chains_
    ]
    where = " UNION ".join(branches)
    return f"{prologue} SELECT * WHERE {{ {where} }} LIMIT {limit}".strip()
