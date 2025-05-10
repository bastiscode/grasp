import os
import signal
import uuid
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from importlib import resources
from typing import Iterator
from urllib.parse import quote_plus, urljoin

import requests
import validators
from grammar_utils.parse import LR1Parser

from grasp.sparql.constants import (
    READ_TIMEOUT,
    REQUEST_TIMEOUT,
    AskResult,
    Position,
    SelectResult,
)


class SPARQLException(Exception):
    pass


def get_index_dir() -> str | None:
    return os.getenv("KG_INDEX_DIR", None)


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = resources.read_text("grasp.sparql.grammar", "sparql.y")
    sparql_lexer = resources.read_text("grasp.sparql.grammar", "sparql.l")
    return sparql_grammar, sparql_lexer


def load_sparql_parser() -> LR1Parser:
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Parser(sparql_grammar, sparql_lexer)


def load_iri_and_literal_grammar() -> tuple[str, str]:
    il_grammar = resources.read_text("grasp.sparql.grammar", "iri_literal.y")
    il_lexer = resources.read_text("grasp.sparql.grammar", "iri_literal.l")
    return il_grammar, il_lexer


def load_iri_and_literal_parser() -> LR1Parser:
    iri_and_literal_grammar, iri_and_literal_lexer = load_iri_and_literal_grammar()
    return LR1Parser(iri_and_literal_grammar, iri_and_literal_lexer)


def find_longest_prefix(iri: str, prefixes: dict[str, str]) -> tuple[str, str] | None:
    longest = None
    for short, long in prefixes.items():
        if not iri.startswith(long):
            continue
        if longest is None or len(long) > len(longest[1]):
            longest = short, long
    return longest


def parse_to_string(parse: dict) -> str:
    def _flatten(parse: dict) -> str:
        if "value" in parse:
            return parse["value"]
        elif "children" in parse:
            children = []
            for p in parse["children"]:
                child = _flatten(p)
                if child != "":
                    children.append(child)
            return " ".join(children)
        else:
            return ""

    return _flatten(parse)


def parse_string(
    input: str,
    parser: LR1Parser,
    collapse_single: bool = False,
    skip_empty: bool = False,
    is_prefix: bool = False,
) -> tuple[dict, str]:
    if is_prefix:
        parse, rest = parser.prefix_parse(
            input.encode(),
            skip_empty=skip_empty,
            collapse_single=collapse_single,
        )
        rest_str = bytes(rest).decode(errors="replace")
    else:
        parse = parser.parse(
            input,
            skip_empty=skip_empty,
            collapse_single=collapse_single,
        )
        rest_str = ""

    return parse, rest_str


def find(
    parse: dict,
    name: str | set[str],
    skip: set[str] | None = None,
    last: bool = False,
) -> dict | None:
    all = find_all(parse, name, skip)
    if not last:
        return next(all, None)
    else:
        last = None
        for item in all:
            last = item
        return last


def find_all(
    parse: dict,
    name: str | set[str],
    skip: set[str] | None = None,
) -> Iterator[dict]:
    if skip is not None and parse["name"] in skip:
        return
    elif isinstance(name, str) and parse["name"] == name:
        yield parse
    elif isinstance(name, set) and parse["name"] in name:
        yield parse
    else:
        for child in parse.get("children", []):
            yield from find_all(child, name, skip)


def find_terminals(parse: dict) -> Iterator[dict]:
    if "value" in parse:
        yield parse
    else:
        for child in parse.get("children", []):
            yield from find_terminals(child)


def normalize(sparql: str, parser: LR1Parser, is_prefix: bool = False) -> str:
    # normalize SPARQL by changing variable names to ?v1, ?v2, ...
    parse, rest = parse_string(
        sparql + " " * is_prefix,
        parser,
        skip_empty=True,
        collapse_single=True,
        is_prefix=is_prefix,
    )

    var_rename = {}
    for var in find_all(parse, {"VAR1", "VAR2"}):
        var_name = var["value"][1:]  # remove ? or $
        if var_name not in var_rename:
            var_rename[var_name] = len(var_rename) + 1

        var["value"] = f"?v{var_rename[var_name]}"

    if is_prefix and rest and rest[-1] == " ":
        rest = rest[:-1]

    return parse_to_string(parse) + rest


def has_iri(sparql: str, parser: LR1Parser) -> bool:
    parse, _ = parse_string(
        sparql,
        parser,
        skip_empty=True,
        collapse_single=True,
    )

    return (
        find(
            parse,
            {"IRIREF", "PNAME_NS", "PNAME_LN"},
            skip={"BaseDecl", "PrefixDecl"},
        )
        is not None
    )


def autocomplete_sparql(
    sparql: str,
    parser: LR1Parser,
    var: str,
    limit: int | None = None,
) -> tuple[str, Position]:
    """
    Autocomplete the SPARQL by checking that the target variable is
    selected with SELECT DISTINCT and that it occurrs at least once
    in the WHERE clause. Optionally add a LIMIT clause to the query.
    """
    try:
        parse, rest = parse_string(sparql, parser)
    except Exception as e:
        raise SPARQLException("SPARQL query is not valid") from e

    # check if query is a select query
    query = find(parse, "QueryType")
    assert query is not None, "SPARQL query has no type"

    query = query["children"][0]
    if query["name"] != "SelectQuery":
        raise SPARQLException("SPARQL query is not a select query")

    select_clause = query["children"][0]
    select_clause["children"][1:] = [
        {"name": "DISTINCT", "value": "DISTINCT", "byte_span": (0, 0)},
        {"name": "VAR1", "value": f"?{var}", "byte_span": (0, 0)},
    ]

    body = query["children"][2]
    autocomp_vars = list(
        filter(
            lambda x: x["value"] == f"?{var}",
            find_all(body, "VAR1", skip={"SubSelect"}),
        )
    )
    if not autocomp_vars:
        raise SPARQLException(
            f"Variable ?{var} must occurr in the WHERE clause at least once"
        )

    for autocomp_var in autocomp_vars:
        autocomp_start, _ = autocomp_var["byte_span"]

        # set autocomp var and all values after it to empty string
        autocomp_parse = deepcopy(parse)
        for p in find_terminals(autocomp_parse):
            start, _ = p["byte_span"]
            if start >= autocomp_start:
                p["value"] = ""

        prefix = parse_to_string(autocomp_parse)

        try:
            _, position = autocomplete_prefix(prefix, parser)
        except Exception:
            continue

        if limit is not None:
            # set limit
            lim_off_clause = find(
                parse,
                "LimitOffsetClausesOptional",
                skip={"SubSelect"},
            )
            assert lim_off_clause is not None, "Failed to find limit clause"
            lim_off_clause.pop("children", None)
            lim_off_clause["value"] = f"LIMIT {limit}"

        return parse_to_string(parse), position

    raise SPARQLException(
        f"Failed to determine position (subject, property, or object) "
        f"of ?{var} in the query"
    )


def autocomplete_prefix(
    prefix: str,
    parser: LR1Parser,
    limit: int | None = None,
) -> tuple[str, Position]:
    """
    Autocomplete the SPARQL prefix by running
    it against the SPARQL grammar parser.
    Assumes the prefix is somewhere in a triple block.
    Optionally add a LIMIT clause to the query.
    Returns the full SPARQL query and the current position
    in the query triple block (subject, property, object).
    """
    # autocomplete by adding 1 to 3 variables to the query,
    # completing and then parsing it to find the current position
    # in the query triple block
    parse, rest = parse_string(
        prefix + " ",
        parser,
        is_prefix=True,
    )

    # build bracket stack to fix brackets later
    bracket_stack = []
    for item in find_all(
        parse,
        {"{", "}", "(", ")"},
    ):
        if item["name"] in ["{", "("]:
            bracket_stack.append(item["name"])
            continue

        assert bracket_stack, "bracket stack is empty"
        last = bracket_stack[-1]
        expected = "(" if item["name"] == ")" else "{"
        assert (
            last == expected
        ), f"expected {expected} bracket in the stack but got {last}"
        bracket_stack.pop()

    if rest in ["{", "("]:
        bracket_stack.append(rest)

    def close_brackets(s: str) -> str:
        for b in reversed(bracket_stack):
            if b == "{":
                s += " }"
            else:
                s += " )"
        return s

    def fix_last_subselect(parse: dict, var: str):
        subsel = find(parse, "SubSelect", last=True)
        if not subsel:
            return

        selclause = find(subsel, "SelectClause")
        if selclause is None or len(selclause["children"]) != 3:
            return

        selclause["children"][-1] = {"name": "Var", "value": f"?{var}"}
        whereclause = find(subsel, "WhereClause")
        if whereclause is None:
            return

        fix_last_subselect(whereclause, var)

    for i, position in enumerate(Position):
        vars = [uuid.uuid4().hex for _ in range(3 - i)]

        full_query = prefix.strip() + " " + " ".join(f"?{v}" for v in vars)
        full_query = close_brackets(full_query)

        try:
            parse, _ = parse_string(full_query, parser)
        except Exception:
            continue

        # replace all select vars with the last one
        select_var = vars[0]

        # replace first select or ask with select var
        query = find(parse, "QueryType")
        assert query is not None
        query = query["children"][0]
        if query["name"] == "SelectQuery":
            select_clause = query["children"][0]
            select_clause["children"][1:] = [
                {"name": "DISTINCT", "value": "DISTINCT"},
                {"name": "VAR1", "value": f"?{select_var}"},
            ]
        elif query["name"] == "AskQuery":
            # ask to select here
            query["name"] = "SelectQuery"
            query["children"][0] = {
                "name": "SelectClause",
                "children": [
                    {"name": "SELECT", "value": "SELECT"},
                    {"name": "DISTINCT", "value": "DISTINCT"},
                    {"name": "VAR1", "value": f"?{select_var}"},
                ],
            }
        else:
            continue

        # fix subselects
        fix_last_subselect(parse, select_var)

        final_query = parse_to_string(parse)
        if limit is not None:
            final_query += f" LIMIT {limit}"

        return final_query, position

    raise SPARQLException("Failed to autocomplete prefix")


def query_type(sparql: str, parser: LR1Parser, is_prefix: bool = False) -> str:
    try:
        parse, _ = parse_string(sparql + " " * is_prefix, parser, is_prefix=is_prefix)
    except Exception:
        # if query is not parsable, return select as default
        return "select"

    query_type = find(parse, "QueryType")
    assert query_type is not None, "Cannot find query type of SPARQL query"

    query_type = query_type["children"][0]
    name = query_type["name"]
    match name:
        case "SelectQuery":
            return "select"
        case "ConstructQuery":
            return "construct"
        case "DescribeQuery":
            return "describe"
        case "AskQuery":
            return "ask"
        case _:
            raise SPARQLException(f'Unknown SPARQL query type "{name}"')


def ask_to_select(
    sparql: str,
    parser: LR1Parser,
    limit: int | None = None,
) -> str | None:
    parse = parser.parse(sparql)

    sub_parse = find(parse, "QueryType")
    assert sub_parse is not None

    ask_query = sub_parse["children"][0]
    if ask_query["name"] != "AskQuery":
        return None

    # find all triples
    triples = list(find_all(ask_query, "TriplesSameSubjectPath"))
    for triple in triples:
        # find first var in triple
        var = find(triple, "Var")
        if var is not None:
            continue

        iri = find(triple, "iri")
        assert iri is not None

        # triple block does not have a var
        # introduce one in VALUES clause and replace iri with var
        var = uuid.uuid4().hex
        triple["children"].append(
            {
                "name": "ValuesClause",
                "children": [
                    {"name": "VALUES", "value": "VALUES"},
                    {"name": "Var", "value": f"?{var}"},
                    {"name": "{", "value": "{"},
                    deepcopy(iri),
                    {"name": "}", "value": "}"},
                ],
            }
        )
        iri.pop("children")
        iri["name"] = "Var"
        iri["value"] = f"?{var}"

    # ask query has a var, convert to select
    ask_query["name"] = "SelectQuery"
    # replace ASK terminal with SelectClause
    ask_query["children"][0] = {
        "name": "SelectClause",
        "children": [
            {"name": "SELECT", "value": "SELECT"},
            {"name": "*", "value": "*"},
        ],
    }
    # return if no limit is to be added
    if not limit:
        return parse_to_string(parse)

    limit_clause = find(ask_query, "LimitClause", skip={"SubSelect"})
    if limit_clause is None:
        return parse_to_string(parse) + " LIMIT 1"
    else:
        limit_clause["children"] = [
            {
                "name": "LIMIT",
                "value": "LIMIT",
            },
            {
                "name": "INTEGER",
                "value": "1",
            },
        ]
        return parse_to_string(parse)


def fix_prefixes(
    sparql: str,
    parser: LR1Parser,
    prefixes: dict[str, str],
    is_prefix: bool = False,
    remove_known: bool = False,
    sort: bool = False,
) -> str:
    parse, rest = parse_string(
        sparql + " " * is_prefix,
        parser,
        is_prefix=is_prefix,
    )

    reverse_prefixes = {long: short for short, long in prefixes.items()}

    exist = {}
    for prefix_decl in find_all(parse, "PrefixDecl"):
        assert len(prefix_decl["children"]) == 3
        first = prefix_decl["children"][1]["value"]
        second = prefix_decl["children"][2]["value"]

        short = first.split(":", 1)[0]
        long = second[:-1]
        exist[short] = long

    base_decl = find(parse, "BaseDecl", last=True)
    if base_decl:
        base_uri = base_decl["children"][1]["value"]
    else:
        base_uri = None

    skip = {"Prologue", "PrefixDecl", "BaseDecl"}

    seen = set()
    for iri in find_all(parse, "IRIREF", skip=skip):
        formatted = format_iri(
            iri["value"],
            prefixes,
            base_uri=base_uri,
        )
        if is_iri(formatted):
            continue

        pfx, _ = formatted.split(":", 1)
        iri["value"] = formatted
        iri["name"] = "PNAME_LN"
        seen.add(pfx)

    for pfx in find_all(parse, {"PNAME_NS", "PNAME_LN"}, skip=skip):
        short, val = pfx["value"].split(":", 1)
        long = exist.get(short, "")

        if reverse_prefixes.get(long, short) != short:
            # replace existing short forms with our own short form
            short = reverse_prefixes[long]
            pfx["value"] = f"{short}:{val}"

        seen.add(short)

    updated_prologue = []
    for pfx in seen:
        if pfx in prefixes:
            if remove_known:
                continue
            long = prefixes[pfx]
        elif pfx in exist:
            long = exist[pfx]
        else:
            continue

        updated_prologue.append(
            {
                "name": "PrefixDecl",
                "children": [
                    {"name": "PREFIX", "value": "PREFIX"},
                    {"name": "PNAME_NS", "value": f"{pfx}:"},
                    {"name": "IRIREF", "value": f"{long}>"},
                ],
            }
        )

    if sort:
        updated_prologue.sort(key=lambda pfx: pfx["children"][1]["value"])

    prologue = find(parse, "Prologue")
    if prologue:
        prologue["children"] = updated_prologue
    else:
        parse = {"name": "Prologue", "children": updated_prologue}

    return (parse_to_string(parse) + rest).strip()


def prettify(
    sparql: str,
    parser: LR1Parser,
    indent: int = 2,
    is_prefix: bool = False,
) -> str:
    parse, rest = parse_string(
        sparql + " " * is_prefix,
        parser,
        skip_empty=True,
        collapse_single=True,
        is_prefix=is_prefix,
    )

    # some simple rules for prettifing:
    # 1. new lines after prologue (PrologueDecl) and triple blocks
    # (TriplesBlock)
    # 2. new lines after { and before }
    # 3. increase indent after { and decrease before }

    assert indent > 0, "indent step must be positive"
    current_indent = 0
    s = ""

    def _pretty(parse: dict) -> bool:
        nonlocal current_indent
        nonlocal s
        newline = False

        if "value" in parse:
            if parse["name"] in ["UNION", "MINUS"]:
                s = s.rstrip() + " "

            elif parse["name"] == "}":
                current_indent -= indent
                s = s.rstrip()
                s += "\n" + " " * current_indent

            elif parse["name"] == "{":
                current_indent += indent

            s += parse["value"]

        elif len(parse["children"]) == 1:
            newline = _pretty(parse["children"][0])

        else:
            for i, child in enumerate(parse["children"]):
                if i > 0 and not newline and child["name"] != "(":
                    s += " "

                newline = _pretty(child)

        if not newline and parse["name"] in [
            "{",
            "}",
            ".",
            "PrefixDecl",
            "BaseDecl",
            "TriplesBlock",
            "GroupClause",
            "HavingClause",
            "OrderClause",
            "LimitClause",
            "OffsetClause",
            "GraphPatternNotTriples",
        ]:
            s += "\n" + " " * current_indent
            newline = True

        return newline

    newline = _pretty(parse)
    if newline:
        s = s.rstrip()

    return (s.strip() + " " + rest).strip()


def set_limit(sparql: str, parser: LR1Parser, limit: int) -> str:
    parse, _ = parse_string(sparql, parser)
    limit_clause = find(parse, "LimitClause", skip={"SubSelect"})
    if limit_clause is None:
        return sparql

    limit_clause["children"] = [
        {
            "name": "LIMIT",
            "value": "LIMIT",
        },
        {
            "name": "INTEGER",
            "value": str(limit),
        },
    ]
    return parse_to_string(parse)


def _timeout(signum, frame, message: str):
    raise TimeoutError(message)


@contextmanager
def timeout(seconds: float | None = None, message: str = "Took too long"):
    if seconds is None:
        # just yield and return
        yield
        return

    seconds = max(0, round(seconds))
    message = f"Timeout after {seconds} seconds: {message}"
    original_handler = signal.signal(signal.SIGALRM, partial(_timeout, message=message))
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Cancel the alarm and restore original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def execute(
    sparql: str,
    endpoint: str,
    request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
    max_retries: int = 1,
    read_timeout: float | None = READ_TIMEOUT,
) -> SelectResult | AskResult:
    max_retries = max(1, max_retries)
    for i in range(max_retries):
        try:
            with timeout(read_timeout, message="Took too long to read SPARQL result"):
                response = requests.post(
                    endpoint,
                    headers={
                        "Content-type": "application/x-www-form-urlencoded",
                        "Accept": "application/sparql-results+json",
                        "User-Agent": "sparql-kgqa-bot",
                    },
                    data={"query": sparql},
                    timeout=request_timeout,
                )
            response.raise_for_status()

        except TimeoutError as e:
            # retry if not last retry
            if i < max_retries - 1:
                continue

            raise e

        except requests.RequestException as e:
            # try to get qlever exception
            try:
                status = e.response.status_code
                body = e.response.json()
            except Exception:
                status = None
                body = None

            client_error = status and int(status / 100) == 4
            qlever_ex = body["exception"] if body and "exception" in body else None

            # immediately return on client error
            if client_error and qlever_ex:
                raise requests.RequestException(qlever_ex) from e
            elif client_error:
                raise e
            # retry on server error if not last retry
            elif i < max_retries - 1:
                continue
            elif qlever_ex:
                raise requests.RequestException(qlever_ex) from e
            else:
                raise e

        res = response.json()

        if "boolean" in res:
            return AskResult(res["boolean"])
        else:
            return SelectResult.from_json(res)

    raise requests.RequestException(f"Maximum retries ({max_retries}) reached")


def is_iri(iri: str) -> bool:
    return iri.startswith("<") and iri.endswith(">")


def is_fq_iri(iri: str) -> bool:
    return is_iri(iri) and validators.url(iri[1:-1])


def format_iri(
    iri: str,
    prefixes: dict[str, str],
    base_uri: str | None = None,
) -> str:
    if not is_iri(iri):
        return iri

    elif not is_fq_iri(iri):
        assert base_uri is not None, (
            f"Could not find a scheme in the IRI {iri}, it seems "
            f"you provided a relative IRI without a BASE URI"
        )
        iri = "<" + urljoin(base_uri[1:-1], iri[1:-1]) + ">"

    longest = find_longest_prefix(iri, prefixes)
    if longest is None:
        return iri

    short, long = longest
    val = iri[len(long) : -1]

    # check if no bad characters are in the short form
    # by url encoding it and checking if it is still the same
    if quote_plus(val) == val:
        return short + ":" + val
    else:
        return iri


def clip(s: str, max_len: int = 64) -> str:
    if len(s) <= max_len + 3:  # 3 for "..."
        return s

    # clip string to max_len  + 3 by stripping out middle part
    half = max_len // 2
    first = s[:half]
    last = s[-half:]
    return first + "..." + last
