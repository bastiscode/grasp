import json
import time
import uuid
from copy import deepcopy
from importlib import resources
from typing import Any, Iterator
from urllib.parse import quote_plus, urlparse, urlunparse

import requests
from grammar_utils.parse import LR1Parser
from requests.exceptions import JSONDecodeError

from grasp.sparql.types import AskResult, Binding, Position, SelectResult

# default request timeout
# 6 seconds for establishing a connection, 30 seconds for processing query
# and beginning to receive the response
REQUEST_TIMEOUT = (6, 30)

# default read timeout
# if you cannot read the full response in 10 seconds, it is likely too large
READ_TIMEOUT = 10

QLEVER_API = "https://qlever.dev/api"


def get_endpoint(kg: str) -> str:
    return f"{QLEVER_API}/{kg}"


class SPARQLException(Exception):
    def __init__(self, message: str, query: str | None = None) -> None:
        super().__init__(message)
        self.query = query


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


def format_literal(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        s = s.strip('"')
    elif s.startswith("'") and s.endswith("'"):
        s = s.strip("'")

    return s.encode().decode("unicode_escape")


def parse_into_binding(
    input: str,
    parser: LR1Parser,
    prefixes: dict[str, str],
) -> Binding | None:
    try:
        parse, _ = parse_string(
            input,
            parser,
            skip_empty=True,
            collapse_single=True,
        )
    except Exception:
        return None

    match parse["name"]:
        case "IRIREF":
            # already an IRI
            return Binding(
                typ="uri",
                value=input[1:-1],
            )

        case "PNAME_LN" | "PNAME_NS":
            pfx, name = input.split(":", 1)
            if pfx not in prefixes:
                return None

            uri = prefixes[pfx] + name

            # prefixed IRI
            return Binding(
                typ="uri",
                value=uri,
            )

        case lit if lit.startswith("STRING_LITERAL"):
            # string literal -> strip quotes
            return Binding(
                typ="literal",
                value=format_literal(parse["value"]),
            )

        case lit if lit.startswith("INTEGER"):
            # integer literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#int",
            )

        case lit if lit.startswith("DECIMAL"):
            # decimal literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#decimal",
            )

        case lit if lit.startswith("DOUBLE"):
            # double literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#double",
            )

        case lit if lit in ["true", "false"]:
            # boolean literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#boolean",
            )

        case "RDFLiteral":
            if len(parse["children"]) == 2:
                # langtag
                lit, langtag = parse["children"]

                return Binding(
                    typ="literal",
                    value=format_literal(lit["value"]),
                    lang=langtag["value"][1:],
                )

            elif len(parse["children"]) == 3:
                # datatype
                lit, _, datatype = parse["children"]
                if datatype["name"] == "IRIREF":
                    datatype = datatype["value"][1:-1]
                else:
                    pfx, name = datatype["value"].split(":", 1)
                    if pfx not in prefixes:
                        return None

                    datatype = prefixes[pfx] + name

                return Binding(
                    typ="literal",
                    value=format_literal(lit["value"]),
                    datatype=datatype,
                )

        case other:
            raise ValueError(
                f"Unexpected type {other} for IRI or literal: {input}",
            )


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
        last_item = None
        for item in all:
            last_item = item
        return last_item


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


def autocomplete_prefix(
    prefix: str,
    parser: LR1Parser,
    limit: int | None = None,
) -> tuple[str, str, Position]:
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
        assert last == expected, (
            f"expected {expected} bracket in the stack but got {last}"
        )
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

    def find_top_level_triples(parse: dict) -> list[str]:
        blocks = []
        for triples in find_all(
            parse,
            "TriplesSameSubjectPath",
            skip={"GraphPatternNotTriples"},
        ):
            blocks.append(parse_to_string(triples))
        return blocks

    for i, position in enumerate(Position):
        vars = [uuid.uuid4().hex for _ in range(3 - i)]

        full_query = prefix.strip() + " " + " ".join(f"?{v}" for v in vars)
        full_query = close_brackets(full_query)

        # check if query is valid now
        try:
            parse, _ = parse_string(full_query, parser)
            query_type = find(parse, "QueryType")
            assert query_type is not None
            query_type = query_type["children"][0]
            # strip "Query" suffix
            query_type = query_type["name"][:-5].lower()
        except Exception:
            continue

        select_var = vars[0]

        triple_blocks = find_top_level_triples(parse)
        if not any(select_var in block for block in triple_blocks):
            # reset to empty query if selected var is not in triple blocks
            # because then the result wouuld always be empty
            triple_blocks = []

        final_query = (
            "SELECT DISTINCT ?"
            + select_var
            + " WHERE { "
            + " . ".join(triple_blocks)
            + " }"
        )
        if limit is not None:
            final_query += f" LIMIT {limit}"

        return final_query, query_type, position

    raise SPARQLException("Failed to autocomplete prefix", prefix)


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
    return name[:-5].lower()  # remove "Query" suffix


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
        long = second[1:-1]
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
                    {"name": "IRIREF", "value": wrap_iri(long)},
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
    last = None

    def _pretty(parse: dict) -> bool:
        nonlocal current_indent
        nonlocal s
        nonlocal last
        newline = False

        if parse["name"] == "(" or (last and last["name"] == "("):
            s = s.rstrip()
        elif parse["name"] == ")":
            s = s.rstrip()

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
                if i > 0 and not newline:  # and child["name"] != "(":
                    s += " "

                newline = _pretty(child)

        if not newline and parse["name"] in [
            "{",
            "}",
            ".",
            "PrefixDecl",
            "BaseDecl",
            "TriplesBlock",
            "GraphPatternNotTriples",
            "GroupClause",
            "HavingClause",
            "OrderClause",
            "LimitClause",
            "OffsetClause",
        ]:
            s += "\n" + " " * current_indent
            newline = True

        last = parse

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


class SPARQLExecuteException(SPARQLException):
    def __init__(
        self,
        message: str,
        query: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, query)
        self.status_code = status_code

    @property
    def is_other_error(self) -> bool:
        return self.status_code is None

    @property
    def is_client_error(self) -> bool:
        return self.status_code is not None and int(self.status_code / 100) == 4

    @property
    def is_server_error(self) -> bool:
        return self.status_code is not None and int(self.status_code / 100) == 5


def _stream_with_timeout(
    sparql: str,
    response: requests.Response,
    seconds: float | None = None,
) -> Any:
    start = time.perf_counter()
    chunks: list[bytes] = []
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue
        chunks.append(chunk)
        if seconds and time.perf_counter() - start > seconds:
            raise SPARQLExecuteException(
                f"Took longer than {seconds} seconds to read SPARQL result",
                sparql,
            )

    full = b"".join(chunks)
    decoded = full.decode(response.encoding or "utf-8")
    return json.loads(decoded)


def execute(
    sparql: str,
    endpoint: str,
    request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
    max_retries: int = 0,
    read_timeout: float | None = READ_TIMEOUT,
    **kwargs: Any,
) -> SelectResult | AskResult:
    max_retries = max(0, max_retries)
    for i in range(max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": "grasp-rdf",
                },
                data={"query": sparql},
                timeout=request_timeout,
                stream=True,
                **kwargs,
            )

            response.raise_for_status()

            res = _stream_with_timeout(sparql, response, read_timeout)
            if "boolean" in res:
                return AskResult(res["boolean"])
            else:
                return SelectResult.from_json(res)

        except SPARQLExecuteException as e:
            # retry if not last retry
            if i < max_retries:
                continue

            raise e

        except requests.Timeout as e:
            # retry if not last retry
            if i < max_retries:
                continue

            raise SPARQLExecuteException(
                f"SPARQL query timed out after {request_timeout} seconds",
                sparql,
            ) from e

        except requests.RequestException as e:
            # try to get qlever exception
            status = None
            body = None
            qlever_ex = None
            if e.response is not None:
                status = e.response.status_code  # type: ignore
                try:
                    body = e.response.json()  # type: ignore
                    qlever_ex = body["exception"] if "exception" in body else None
                except JSONDecodeError:
                    body = e.response.text

            client_error = status and int(status / 100) == 4

            # immediately return on client error
            if client_error and qlever_ex:
                raise SPARQLExecuteException(
                    qlever_ex,
                    sparql,
                    status_code=status,
                ) from e
            elif client_error:
                raise SPARQLExecuteException(
                    body if body else "Client error",
                    sparql,
                    status_code=status,
                ) from e
            # retry on server error if not last retry
            elif i < max_retries - 1:
                continue
            elif qlever_ex:
                raise SPARQLExecuteException(
                    qlever_ex,
                    sparql,
                    status_code=status,
                ) from e
            else:
                raise SPARQLExecuteException(
                    body if body else "Server error",
                    sparql,
                    status_code=status,
                ) from e

    raise SPARQLExecuteException(
        f"Maximum retries reached ({max_retries})",
        sparql,
    )


def is_iri(iri: str) -> bool:
    return iri.startswith("<") and iri.endswith(">")


def wrap_iri(iri: str) -> str:
    return f"<{iri}>"


def format_iri(
    iri: str,
    prefixes: dict[str, str],
    base_uri: str | None = None,
) -> str:
    # strip angle brackets if present (e.g. from SPARQL parse tree IRIREF nodes)
    wrapped = is_iri(iri)
    if wrapped:
        iri = iri[1:-1]

    if "://" not in iri:
        return wrap_iri(iri) if wrapped else iri

    longest = find_longest_prefix(iri, prefixes)
    if longest is None:
        return wrap_iri(iri) if wrapped else iri

    short, long = longest
    val = iri[len(long):]

    # check if no bad characters are in the short form
    # by url encoding it and checking if it is still the same
    if quote_plus(val) == val:
        return short + ":" + val
    else:
        return wrap_iri(iri) if wrapped else iri


def load_qlever_prefixes(endpoint: str) -> dict[str, str]:
    parse = urlparse(endpoint)
    parse.encode()
    split = parse.path.split("/")
    assert len(split) >= 1, "Endpoint path must contain at least one segment"
    split.insert(len(split) - 1, "prefixes")
    path = "/".join(split)
    parse = parse._replace(path=path)
    prefix_url = urlunparse(parse)

    response = requests.get(prefix_url)
    response.raise_for_status()
    prefixes = {}
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        assert line.startswith("PREFIX "), "Each line must start with 'PREFIX '"
        _, rest = line.split(" ", 1)
        prefix, uri = rest.split(":", 1)
        uri = uri.strip()
        assert is_iri(uri), "Prefix must be in IRI format"
        prefixes[prefix.strip()] = uri[1:-1]

    return prefixes


def load_entity_index_sparql() -> str:
    return resources.read_text("grasp.sparql.queries", "entity.index.sparql").strip()


def load_property_index_sparql() -> str:
    return resources.read_text("grasp.sparql.queries", "property.index.sparql").strip()


def load_entity_info_sparql() -> str:
    return resources.read_text("grasp.sparql.queries", "entity.info.sparql").strip()


def load_property_info_sparql() -> str:
    return resources.read_text("grasp.sparql.queries", "property.info.sparql").strip()
