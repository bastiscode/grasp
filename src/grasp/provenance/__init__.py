import os
from typing import Any

from grasp.manager import KgManager
from grasp.provenance.paths import (
    Chain,
    PathSyntaxError,
    chain_steps,
    expansion_query,
    pairs_query,
    path_chains,
)
from grasp.provenance.witness import (
    Witness,
    WitnessTriple,
    binding_to_dict,
    WitnessPlan,
    binding_to_sparql,
    instantiate,
    is_marker,
    is_var,
    matches_row,
    plan_witness,
    resolve,
    types_query,
    var_name,
    witness_query,
)
from grasp.shapes import load_setup_patterns
from grasp.sparql.types import AskResult, Binding, SelectResult, SelectRow
from grasp.utils import clip, get_index_dir

# witness queries should be cheap, so use shorter timeouts than for the
# query itself by default
WITNESS_REQUEST_TIMEOUT = (6.0, 10.0)
WITNESS_READ_TIMEOUT = 5.0

# number of candidate witnesses fetched per row; the first and last of them
# are shown, like the first and last rows of a query result
WITNESS_POOL = 100


def has_literal(row: SelectRow) -> bool:
    return any(binding.typ == "literal" for binding in row.values())


# instance pattern per knowledge graph, None if it has no shapes setup
INSTANCE_PATTERNS: dict[str, str | None] = {}


def load_instance_pattern(manager: KgManager) -> str | None:
    # read from the shapes directory even if shapes are disabled in the
    # config, since provenance only needs the pattern, not the shape index
    if manager.shapes is not None and manager.shapes.instance_pattern:
        return manager.shapes.instance_pattern

    if manager.kg not in INSTANCE_PATTERNS:
        shapes_dir = os.path.join(get_index_dir(manager.kg), "shapes")
        instance_pattern, _ = load_setup_patterns(shapes_dir)
        INSTANCE_PATTERNS[manager.kg] = instance_pattern
    return INSTANCE_PATTERNS[manager.kg]


def compute_types(
    manager: KgManager,
    iris: list[str],
    request_timeout: float | tuple[float, float] | None,
    read_timeout: float | None,
) -> dict[str, list[str]]:
    # class IRIs per entity; empty if there is no instance pattern or the
    # query fails, types are only a hint
    instance_pattern = load_instance_pattern(manager)
    if not iris or instance_pattern is None:
        return {}

    try:
        result = manager.execute_sparql(
            types_query(instance_pattern, iris),
            request_timeout,
            read_timeout,
        )
    except Exception:
        return {}

    assert isinstance(result, SelectResult)
    types: dict[str, list[str]] = {}
    for row in result.rows():
        if "instance" in row and "class" in row:
            types.setdefault(row["instance"].value, []).append(row["class"].value)
    return types


def label_class(manager: KgManager, iri: str) -> str:
    return manager.get_label(iri, "entities") or manager.format_iri(iri, wrap=True)


def entity_variables(plan: WitnessPlan) -> list[str]:
    # variables in subject or object position, whose values are entities
    # worth typing; predicates and marker variables are excluded
    variables = []
    for pattern in plan.triples:
        for term in (pattern.subject, pattern.object):
            if is_var(term) and var_name(term) not in variables:
                variables.append(var_name(term))
    return variables


def type_summary(
    manager: KgManager,
    values: dict[str, set[str]],
    types: dict[str, list[str]],
    max_classes: int = 5,
) -> dict[str, list[tuple[str, int]]]:
    # per variable with more than one entity value: number of entities per
    # class, most frequent first
    summary = {}
    for var, iris in values.items():
        if len(iris) < 2:
            continue
        counts: dict[str, int] = {}
        for iri in iris:
            for cls in types.get(iri) or ["(no class)"]:
                counts[cls] = counts.get(cls, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        summary[var] = [
            (cls if cls == "(no class)" else label_class(manager, cls), count)
            for cls, count in ranked[:max_classes]
        ]
        if len(ranked) > max_classes:
            summary[var].append(("other classes", len(ranked) - max_classes))
    return summary


def expand_path(
    manager: KgManager,
    plan: WitnessPlan,
    triple: WitnessTriple,
    prefixes: dict[str, str],
    max_length: int,
    request_timeout: float | tuple[float, float] | None,
    read_timeout: float | None,
) -> dict[str, Any]:
    # the shortest concrete chain between the endpoints of a path triple, by
    # iterative deepening over the chain length; returns the steps from
    # subject to object (empty for a zero-length path) or a note why there
    # are none
    assert isinstance(triple.predicate, str)
    subject = binding_to_sparql(triple.subject)
    obj = binding_to_sparql(triple.object)
    if subject is None or obj is None:
        return {"note": "path endpoints cannot be bound"}

    try:
        chains = path_chains(triple.predicate, max_length)
    except PathSyntaxError as e:
        return {"note": f"path not expanded: {e}"}

    by_length: dict[int, list[tuple[int, Chain]]] = {}
    for index, chain in enumerate(chains):
        by_length.setdefault(len(chain), []).append((index, chain))

    for length in sorted(by_length):
        if length == 0:
            if triple.subject == triple.object:
                return {"steps": []}
            continue

        query = expansion_query(plan.prologue, by_length[length], subject, obj)
        try:
            result = manager.execute_sparql(query, request_timeout, read_timeout)
        except Exception:
            return {"note": "path expansion failed or timed out"}

        assert isinstance(result, SelectResult)
        row = next(result.rows(), None)
        if row is None:
            continue

        index = int(row["_path_chain"].value)
        chain = chains[index]
        steps = chain_steps(chain, index, triple.subject, triple.object, row)
        if steps is None:
            return {"note": "path expansion failed"}

        expanded = []
        for step, (start, end) in zip(chain, steps):
            predicate = step.predicate
            binding = (
                row.get(f"_path{index}_p{len(expanded)}")
                if predicate is None
                else resolve(predicate, {}, manager.iri_literal_parser, prefixes)
            )
            expanded.append(
                {
                    "from": label_binding(manager, start),
                    "predicate": label_binding(manager, binding or "?"),
                    "to": label_binding(manager, end),
                    "inverse": step.inverse,
                }
            )
        return {"steps": expanded}

    return {"note": f"no chain of at most {max_length} steps found"}


def expand_witness_paths(
    manager: KgManager,
    plan: WitnessPlan,
    witnesses: list[Witness],
    prefixes: dict[str, str],
    max_length: int,
    request_timeout: float | tuple[float, float] | None,
    read_timeout: float | None,
) -> list[dict[int, dict[str, Any]]]:
    # per witness, expansions by triple index; identical path triples across
    # witnesses are expanded once
    cache: dict[tuple, dict[str, Any]] = {}
    expansions = []
    for witness in witnesses:
        current = {}
        for i, triple in enumerate(witness.triples):
            if not triple.is_path:
                continue
            key = triple.key()
            if key not in cache:
                cache[key] = expand_path(
                    manager,
                    plan,
                    triple,
                    prefixes,
                    max_length,
                    request_timeout,
                    read_timeout,
                )
            current[i] = cache[key]
        expansions.append(current)
    return expansions


def summarize_paths(
    manager: KgManager,
    plan: WitnessPlan,
    witness_rows: list[SelectRow],
    prefixes: dict[str, str],
    max_length: int,
    request_timeout: float | tuple[float, float] | None,
    read_timeout: float | None,
    max_via: int = 3,
) -> list[dict[str, Any]]:
    # for each property path pattern, how its endpoint pairs over all
    # candidate witnesses are connected: number of pairs per shortest chain
    # length, and the most frequent first intermediate nodes of longer chains;
    # one query per chain length for all pairs not matched by shorter chains
    summaries = []
    for pattern in plan.triples:
        if not pattern.is_path:
            continue

        pairs: dict[tuple[str, str], None] = {}
        for row in witness_rows:
            active = {f"?{var}" for var in row if is_marker(var)}
            if not active.issuperset(pattern.markers):
                continue
            subject = resolve(
                pattern.subject, row, manager.iri_literal_parser, prefixes
            )
            obj = resolve(pattern.object, row, manager.iri_literal_parser, prefixes)
            if subject is None or obj is None:
                continue
            s, o = binding_to_sparql(subject), binding_to_sparql(obj)
            if s is not None and o is not None:
                pairs[(s, o)] = None

        if len(pairs) < 2:
            # a single pair is expanded in the shown witnesses already
            continue

        try:
            chains = path_chains(pattern.predicate, max_length)
        except PathSyntaxError:
            continue

        by_length: dict[int, list[tuple[int, Chain]]] = {}
        for index, chain in enumerate(chains):
            by_length.setdefault(len(chain), []).append((index, chain))

        remaining = set(pairs)
        lengths: dict[int, int] = {}
        via: dict[str, int] = {}
        failed = False
        for length in sorted(by_length):
            if not remaining:
                break
            if length == 0:
                matched = {pair for pair in remaining if pair[0] == pair[1]}
                if matched:
                    lengths[0] = len(matched)
                remaining -= matched
                continue

            query = pairs_query(
                plan.prologue,
                by_length[length],
                sorted(remaining),
                limit=10_000,
            )
            try:
                result = manager.execute_sparql(query, request_timeout, read_timeout)
            except Exception:
                failed = True
                break

            assert isinstance(result, SelectResult)
            first: dict[tuple[str, str], SelectRow] = {}
            for row in result.rows():
                s = binding_to_sparql(row["_path_s"]) if "_path_s" in row else None
                o = binding_to_sparql(row["_path_o"]) if "_path_o" in row else None
                if s is not None and o is not None and (s, o) in remaining:
                    first.setdefault((s, o), row)

            for row in first.values():
                if length < 2:
                    continue
                index = int(row["_path_chain"].value)
                node = row.get(f"_path{index}_n1")
                if node is not None:
                    label = label_binding(manager, node)["label"]
                    via[label] = via.get(label, 0) + 1

            if first:
                lengths[length] = len(first)
            remaining -= set(first)

        ranked = sorted(via.items(), key=lambda item: (-item[1], item[0]))
        summaries.append(
            {
                "pattern": f"{pattern.subject} {pattern.predicate} {pattern.object}",
                "pairs": len(pairs),
                "lengths": sorted(lengths.items()),
                "via": ranked[:max_via],
                "more_via": max(0, len(ranked) - max_via),
                # longer than max_length, or the summary failed
                "unmatched": len(remaining),
                "failed": failed,
            }
        )
    return summaries


def hidden_subquery_patterns(
    plan: WitnessPlan,
    witness_rows: list[SelectRow],
) -> list[str]:
    # patterns inside subqueries with a variable that no witness binds,
    # because the subquery does not return it
    bound = {var for row in witness_rows for var in row}
    hidden = []
    for pattern in plan.triples:
        if not pattern.in_subquery:
            continue
        terms = [pattern.subject, pattern.object]
        if not pattern.is_path:
            terms.append(pattern.predicate)
        if any(is_var(term) and var_name(term) not in bound for term in terms):
            hidden.append(f"{pattern.subject} {pattern.predicate} {pattern.object}")
    return hidden


def label_binding(
    manager: KgManager,
    binding: Binding | str,
    types: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(binding, str):
        # property path as written in the query
        return {"type": "path", "value": binding, "label": binding}

    data = binding_to_dict(binding)
    if binding.typ == "uri":
        short = manager.format_iri(binding.value, wrap=True)
        label = manager.get_label(binding.value, "entities") or manager.get_label(
            binding.value, "properties"
        )
        data["short"] = short
        data["label"] = f"{label} ({short})" if label else short
        if types:
            data["types"] = types
            data["label"] += f" [{', '.join(types)}]"
    elif binding.typ == "literal":
        data["label"] = clip(manager.format_literal(binding.identifier()), 128)
    else:
        data["label"] = binding.identifier()
    return data


def triple_to_dict(
    manager: KgManager,
    triple: WitnessTriple,
    types: dict[str, list[str]],
    expansion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # only entities bound from variables get types, constants of the query
    # are known to the model already; types maps entities to class labels
    subject_var, _, object_var = triple.variables
    subject_types = types.get(triple.subject.value) if subject_var else None
    object_types = types.get(triple.object.value) if object_var else None
    return {
        "subject": label_binding(manager, triple.subject, subject_types),
        "predicate": label_binding(manager, triple.predicate),
        "object": label_binding(manager, triple.object, object_types),
        "path": triple.is_path,
        # variables subject, predicate and object are bound from, None for
        # constants of the query
        "variables": list(triple.variables),
        "spans": [list(span) for span in triple.spans],
        **({"expansion": expansion} if expansion is not None else {}),
    }


def typed_iris(witness: Witness) -> list[str]:
    iris = []
    for triple in witness.triples:
        for binding, from_var in [
            (triple.subject, triple.variables[0] is not None),
            (triple.object, triple.variables[2] is not None),
        ]:
            if from_var and binding.typ == "uri":
                iris.append(binding.value)
    return iris


def witness_to_dict(
    manager: KgManager,
    witness: Witness,
    types: dict[str, list[str]] | None = None,
    expansions: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expansions = expansions or {}
    return {
        "triples": [
            triple_to_dict(manager, t, types or {}, expansions.get(i))
            for i, t in enumerate(witness.triples)
        ],
        "conditions": [
            {
                "text": cond.text,
                "bindings": {
                    var: label_binding(manager, b) for var, b in cond.bindings.items()
                },
            }
            for cond in witness.conditions
        ],
    }


def row_to_dict(manager: KgManager, row: SelectRow) -> dict[str, Any]:
    return {var: label_binding(manager, b) for var, b in row.items()}


def compute_witnesses(
    manager: KgManager,
    sparql: str,
    result: SelectResult | AskResult | None,
    max_rows: int = 3,
    max_witnesses: int = 3,
    request_timeout: float | tuple[float, float] | None = WITNESS_REQUEST_TIMEOUT,
    read_timeout: float | None = WITNESS_READ_TIMEOUT,
    rows: list[tuple[int, SelectRow]] | None = None,
    witness_pool: int = WITNESS_POOL,
    max_types: int = 3,
    max_path_length: int = 4,
) -> dict[str, Any]:
    """
    Why-provenance for the first rows of a query result: for each row, up to
    max_witnesses sets of knowledge graph triples that produce it, the first
    and last ones of up to witness_pool candidates. With rows, for the given
    (number, row) pairs instead, e.g. rows of an earlier execution; the result
    is not needed then. Property paths in shown witnesses are expanded into a
    chain of at most max_path_length steps (0 to disable).
    Never raises; failures are reported in the returned dict.
    """
    try:
        plan = plan_witness(sparql, manager.sparql_parser)
    except Exception as e:
        return {"error": f"Failed to plan witness queries: {e}"}

    prefixes = {**manager.prefixes, **plan.prefixes}

    selected: list[tuple[int, SelectRow | None]]
    if rows is not None:
        selected = list(rows)
    elif isinstance(result, AskResult):
        if not result.boolean:
            return {
                "rows": [],
                "notes": plan.notes + ["ASK query is false, there is no witness"],
            }
        selected = [(1, None)]
    elif result is None:
        return {"error": "Neither a result nor rows given"}
    else:
        selected = list(enumerate(result.rows(end=max_rows), 1))

    provenance = []
    instantiated: list[
        tuple[dict[str, Any], list[tuple[int, Witness, dict[int, dict[str, Any]]]]]
    ] = []
    # entity values per variable over all candidate witnesses of a row
    pool_values: dict[int, dict[str, set[str]]] = {}
    all_witness_rows: list[SelectRow] = []
    for number, row in selected:
        query = witness_query(plan, row, witness_pool)
        entry: dict[str, Any] = {
            "number": number,
            "row": row_to_dict(manager, row) if row is not None else None,
            "witness_sparql": query,
        }
        try:
            witness_result = manager.execute_sparql(
                query,
                request_timeout,
                read_timeout,
            )
            assert isinstance(witness_result, SelectResult)
            witness_rows = list(witness_result.rows())
            if not witness_rows and row is not None and has_literal(row):
                # some literals (e.g. WKT or other typed literals) do not
                # join via VALUES, so bind IRIs only and compare literals
                # on the client
                query = witness_query(
                    plan,
                    row,
                    witness_pool,
                    bind_literals=False,
                )
                entry["witness_sparql"] = query
                witness_result = manager.execute_sparql(
                    query,
                    request_timeout,
                    read_timeout,
                )
                assert isinstance(witness_result, SelectResult)
                witness_rows = [
                    witness_row
                    for witness_row in witness_result.rows()
                    if matches_row(plan, row, witness_row)
                ]

            numbers = select_witnesses(len(witness_rows), max_witnesses)
            witnesses = instantiate(
                plan,
                [witness_rows[i - 1] for i in numbers],
                manager.iri_literal_parser,
                prefixes,
            )
            expansions: list[dict[int, dict[str, Any]]] = [{} for _ in witnesses]
            if max_path_length > 0:
                expansions = expand_witness_paths(
                    manager,
                    plan,
                    witnesses,
                    prefixes,
                    max_path_length,
                    request_timeout,
                    read_timeout,
                )
                path_summary = summarize_paths(
                    manager,
                    plan,
                    witness_rows,
                    prefixes,
                    max_path_length,
                    request_timeout,
                    read_timeout,
                )
                if path_summary:
                    entry["path_summary"] = path_summary
            instantiated.append((entry, list(zip(numbers, witnesses, expansions))))
            all_witness_rows.extend(witness_rows)
            pool_values[id(entry)] = {
                var: {
                    witness_row[var].value
                    for witness_row in witness_rows
                    if var in witness_row and witness_row[var].typ == "uri"
                }
                for var in entity_variables(plan)
            }
            entry["num_witnesses"] = len(witness_rows)
            # the pool was exhausted, so there may be more witnesses
            entry["more"] = len(witness_rows) >= witness_pool
        except Exception as e:
            entry["error"] = f"Failed to execute witness query: {e}"

        provenance.append(entry)

    # classes of all shown and candidate witness entities in one query
    types: dict[str, list[str]] = {}
    if max_types > 0:
        iris = {
            iri for _, ws in instantiated for _, w, _ in ws for iri in typed_iris(w)
        }
        for values in pool_values.values():
            iris.update(iri for var_iris in values.values() for iri in var_iris)
        types = compute_types(manager, sorted(iris), request_timeout, read_timeout)

    labels = {
        iri: [label_class(manager, cls) for cls in classes[:max_types]]
        for iri, classes in types.items()
    }
    for entry, ws in instantiated:
        entry["witnesses"] = [
            {"number": number, **witness_to_dict(manager, w, labels, expansion)}
            for number, w, expansion in ws
        ]
        if types:
            summary = type_summary(manager, pool_values[id(entry)], types)
            if summary:
                entry["type_summary"] = summary

    notes = list(plan.notes)
    hidden = (
        hidden_subquery_patterns(plan, all_witness_rows) if all_witness_rows else []
    )
    if hidden:
        notes.append(
            "Triple patterns inside a subquery are not shown, because the "
            f"subquery does not return their variables: {'; '.join(hidden)}"
        )

    output: dict[str, Any] = {"rows": provenance, "notes": notes}
    if isinstance(result, SelectResult):
        output["total_rows"] = len(result)
    elif isinstance(result, AskResult):
        output["total_rows"] = 1
    return output


def select_witnesses(num: int, k: int) -> list[int]:
    # 1-based numbers of the first ceil(k/2) and last floor(k/2) witnesses
    if num <= k:
        return list(range(1, num + 1))
    first = (k + 1) // 2
    last = k // 2
    return list(range(1, first + 1)) + list(range(num - last + 1, num + 1))
