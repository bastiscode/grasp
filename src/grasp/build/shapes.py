import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from universal_ml_utils.io import dump_text
from universal_ml_utils.logging import get_logger

from grasp.configs import ShapeConfig
from grasp.manager import KgManager
from grasp.shapes import ShapeSample
from grasp.sparql.types import AskResult, SelectResult


@dataclass
class PropertyProfile:
    iri: str
    short_iri: str
    triple_count: int = 0
    entity_count: int = 0
    cardinality_tag: str = ""
    literal_datatypes: list[str] = field(default_factory=list)
    target_class_iris: list[str] = field(default_factory=list)
    target_class_short_iris: list[str] = field(default_factory=list)


@dataclass
class ConceptProfile:
    iri: str
    short_iri: str
    total_entities: int = 0
    properties: list[PropertyProfile] = field(default_factory=list)


def class_pattern(pattern: str) -> str:
    return pattern.replace("{CLASS}", "?class")


def object_pattern(pattern: str) -> str:
    p = pattern.replace("?instance", "?o")
    return p.replace("{CLASS}", "?targetClass")


def instance_pattern(pattern: str, class_iri: str) -> str:
    return pattern.replace("{CLASS}", f"<{class_iri}>")


def subquery(pattern: str) -> str:
    cp = class_pattern(pattern)
    return f"  {{\n    SELECT DISTINCT ?class WHERE {{\n      {cp}\n    }}\n  }}"


def membership(pattern: str) -> str:
    cp = class_pattern(pattern)
    lines = [f"  {line}" for line in cp.strip().splitlines()]
    return "\n".join(lines)


def objectmembership(pattern: str) -> str:
    op = object_pattern(pattern)
    lines = [f"  {line}" for line in op.strip().splitlines()]
    return "\n".join(lines)


def build_validation_query(pattern: str) -> str:
    return f"SELECT DISTINCT ?class WHERE {{\n  {class_pattern(pattern)}\n}} LIMIT 1"


def build_property_frequency_query(pattern: str) -> str:
    return (
        f"SELECT ?class ?p (COUNT(*) AS ?tripleCount) "
        f"(COUNT(DISTINCT ?instance) AS ?entityCount)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"  ?instance ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?class ?p\n"
        f"ORDER BY ?class DESC(?tripleCount)"
    )


def build_literal_profile_query(pattern: str) -> str:
    return (
        f"SELECT ?class ?p ?datatype (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"  ?instance ?p ?o .\n"
        f"  FILTER(isLiteral(?o))\n"
        f"  BIND(DATATYPE(?o) AS ?datatype)\n"
        f"}}\n"
        f"GROUP BY ?class ?p ?datatype\n"
        f"ORDER BY ?class ?p DESC(?count)"
    )


def build_range_profile_query(pattern: str) -> str:
    return (
        f"SELECT ?class ?p ?targetClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"  ?instance ?p ?o .\n"
        f"{objectmembership(pattern)}\n"
        f"}}\n"
        f"GROUP BY ?class ?p ?targetClass\n"
        f"ORDER BY ?class ?p DESC(?count)"
    )


def build_total_entities_query(pattern: str) -> str:
    return (
        f"SELECT ?class (COUNT(DISTINCT ?instance) AS ?totalEntities)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"}}\n"
        f"GROUP BY ?class"
    )


def build_per_class_property_frequency_query(pattern: str, class_iri: str) -> str:
    ip = instance_pattern(pattern, class_iri)
    lines = "\n".join(f"  {line}" for line in ip.strip().splitlines())
    return (
        f"SELECT ?p (COUNT(*) AS ?tripleCount) (COUNT(DISTINCT ?instance) AS ?entityCount)\n"
        f"WHERE {{\n"
        f"{lines}\n"
        f"  ?instance ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?p\n"
        f"ORDER BY DESC(?tripleCount)"
    )


def build_per_class_literal_profile_query(pattern: str, class_iri: str) -> str:
    ip = instance_pattern(pattern, class_iri)
    lines = "\n".join(f"  {line}" for line in ip.strip().splitlines())
    return (
        f"SELECT ?p ?datatype (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{lines}\n"
        f"  ?instance ?p ?o .\n"
        f"  FILTER(isLiteral(?o))\n"
        f"  BIND(DATATYPE(?o) AS ?datatype)\n"
        f"}}\n"
        f"GROUP BY ?p ?datatype\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_range_profile_query(pattern: str, class_iri: str) -> str:
    ip = instance_pattern(pattern, class_iri)
    lines = "\n".join(f"  {line}" for line in ip.strip().splitlines())
    op = object_pattern(pattern)
    obj_lines = "\n".join(f"  {line}" for line in op.strip().splitlines())
    return (
        f"SELECT ?p ?targetClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{lines}\n"
        f"  ?instance ?p ?o .\n"
        f"{obj_lines}\n"
        f"}}\n"
        f"GROUP BY ?p ?targetClass\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_total_query(pattern: str, class_iri: str) -> str:
    ip = instance_pattern(pattern, class_iri)
    lines = "\n".join(f"  {line}" for line in ip.strip().splitlines())
    return (
        f"SELECT (COUNT(DISTINCT ?instance) AS ?totalEntities)\nWHERE {{\n{lines}\n}}"
    )


def _property_rank(iri: str, manager: KgManager) -> int:
    data = manager.try_get_data("properties")
    if data is None:
        return 0  # no index: all equal, triple_count tiebreaker takes over
    norm = manager.normalize(iri, "properties")
    if norm is not None:
        iri = norm[0]
    id = data.id_from_identifier(iri)
    return id if id is not None else len(data)  # unknown props sort last


def cardinality_tag(coverage: float, avg_values: float) -> str:
    if coverage >= 0.90 and avg_values <= 1.10:
        return ""
    elif coverage < 0.90 and avg_values <= 1.10:
        return "?"
    elif coverage >= 0.90 and avg_values > 1.10:
        return "+"
    else:
        return "*"


def _label_iri(label: str | None, short_iri: str) -> str:
    return f"{label} ({short_iri})" if label else short_iri


def _target_class_parts(prop: "PropertyProfile", manager: KgManager) -> list[str]:
    if prop.target_class_iris:
        return [
            _label_iri(manager.get_label(t_iri, "entities"), t_short)
            for t_iri, t_short in zip(prop.target_class_iris, prop.target_class_short_iris)
        ]
    return prop.target_class_short_iris[:3]


def emit_pseudo_shex(profile: ConceptProfile, manager: KgManager) -> str:
    class_label = manager.get_label(profile.iri, "entities")
    header = _label_iri(class_label, profile.short_iri)
    lines = [f"{header} {{"]

    for prop in profile.properties:
        prop_label = manager.get_label(prop.iri, "properties")
        prop_str = _label_iri(prop_label, prop.short_iri)
        tag = prop.cardinality_tag
        tag_suffix = f" {tag}" if tag else ""

        if prop.literal_datatypes:
            dtype = prop.literal_datatypes[0]
            lines.append(f"  {prop_str} {dtype}{tag_suffix} ;")
        elif prop.target_class_short_iris:
            target_str = " ".join(_target_class_parts(prop, manager))
            lines.append(f"  {prop_str} [ {target_str} ]{tag_suffix} ;")
        else:
            lines.append(f"  {prop_str} IRI{tag_suffix} ;")

    lines.append("}")
    return "\n".join(lines)


def emit_pseudo_shex_natural(profile: ConceptProfile, manager: KgManager) -> str | None:
    if manager.try_get_data("entities") is None and manager.try_get_data("properties") is None:
        return None

    class_label = manager.get_label(profile.iri, "entities") or profile.short_iri
    lines = [f"{class_label} {{"]

    for prop in profile.properties:
        prop_label = manager.get_label(prop.iri, "properties") or prop.short_iri
        tag = prop.cardinality_tag
        tag_suffix = f" {tag}" if tag else ""

        if prop.literal_datatypes:
            dtype = prop.literal_datatypes[0]
            lines.append(f"  {prop_label} {dtype}{tag_suffix} ;")
        elif prop.target_class_short_iris:
            if prop.target_class_iris:
                parts = [
                    manager.get_label(t_iri, "entities") or t_short
                    for t_iri, t_short in zip(
                        prop.target_class_iris, prop.target_class_short_iris
                    )
                ]
            else:
                parts = prop.target_class_short_iris[:3]
            lines.append(f"  {prop_label} [ {' '.join(parts)} ]{tag_suffix} ;")
        else:
            lines.append(f"  {prop_label} IRI{tag_suffix} ;")

    lines.append("}")
    return "\n".join(lines)


def get_rows(result: SelectResult | AskResult) -> list[dict[str, Any]]:
    if isinstance(result, AskResult):
        return []

    return [
        {var: row[var].value for var in result.variables if var in row}
        for row in result.rows()
    ]


def assemble_profile(
    class_iri: str,
    freq_map: dict[str, dict],
    lit_dtypes: dict[str, list[str]],
    range_map: dict[str, list[str]],
    total_entities: int,
    shape_config: ShapeConfig,
    manager: KgManager,
) -> ConceptProfile:
    properties: list[PropertyProfile] = []

    common_props = sorted(
        freq_map.items(),
        key=lambda item: (
            _property_rank(item[0], manager),
            -item[1].get("triple_count", 0),
        ),
    )

    for p_iri, freq in common_props[: shape_config.max_properties_per_concept]:
        entity_count = freq.get("entity_count", 0)
        triple_count = freq.get("triple_count", 0)

        if total_entities > 0:
            coverage = entity_count / total_entities
            if coverage < shape_config.min_property_coverage:
                continue
            avg_vals = triple_count / entity_count if entity_count > 0 else 1.0
            tag = cardinality_tag(coverage, avg_vals)
        else:
            tag = ""

        p_short = manager.format_iri(p_iri, wrap=True)
        target_iris = range_map.get(p_iri, [])[:5]
        prop = PropertyProfile(
            iri=p_iri,
            short_iri=p_short,
            triple_count=triple_count,
            entity_count=entity_count,
            cardinality_tag=tag,
            literal_datatypes=lit_dtypes.get(p_iri, []),
            target_class_iris=target_iris,
            target_class_short_iris=[manager.format_iri(t, wrap=True) for t in target_iris],
        )
        properties.append(prop)

    return ConceptProfile(
        iri=class_iri,
        short_iri=manager.format_iri(class_iri, wrap=True),
        total_entities=total_entities,
        properties=properties,
    )


def compute_shape(
    class_iri: str,
    pattern: str,
    manager: KgManager,
    shape_config: ShapeConfig | None = None,
) -> str | None:
    if shape_config is None:
        shape_config = ShapeConfig()

    def run(query: str) -> list[dict[str, Any]]:
        result = manager.execute_sparql(
            query,
            shape_config.request_timeout,
            shape_config.read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        return get_rows(result)

    try:
        freq_rows = run(build_per_class_property_frequency_query(pattern, class_iri))
        lit_rows = run(build_per_class_literal_profile_query(pattern, class_iri))
        range_rows = run(build_per_class_range_profile_query(pattern, class_iri))
        total_rows = run(build_per_class_total_query(pattern, class_iri))
    except Exception:
        return None

    total = int(total_rows[0].get("totalEntities", 0)) if total_rows else 0

    freq_map: dict[str, dict] = {}
    for row in freq_rows:
        p = row.get("p", "")
        if p:
            freq_map[p] = {
                "triple_count": int(row.get("tripleCount", 0)),
                "entity_count": int(row.get("entityCount", 0)),
            }

    lit_dtypes: dict[str, list[str]] = defaultdict(list)
    for row in lit_rows:
        p, dt = row.get("p", ""), row.get("datatype", "")
        if p and dt:
            lit_dtypes[p].append(manager.format_iri(dt, wrap=True))

    range_map: dict[str, list[str]] = defaultdict(list)
    for row in range_rows:
        p, tc = row.get("p", ""), row.get("targetClass", "")
        if p and tc and tc not in range_map[p]:
            range_map[p].append(tc)

    profile = assemble_profile(
        class_iri, freq_map, lit_dtypes, range_map, total, shape_config, manager
    )
    return emit_pseudo_shex(profile, manager)


def build_shapes(
    pattern: str,
    shapes_dir: str,
    manager: KgManager,
    shape_config: ShapeConfig = ShapeConfig(),
    max_concepts: int = 500,
    log_level: str | int | None = None,
) -> list[dict]:
    logger = get_logger("GRASP SHAPES BUILD", log_level)

    def run(query: str) -> list[dict[str, Any]]:
        logger.debug(f"Running query:\n{query}")
        result = manager.execute_sparql(
            query, sparql_result_max_rows=shape_config.sparql_result_max_rows
        )
        return get_rows(result)

    logger.info("Validating pattern with test query")
    validation_rows = run(build_validation_query(pattern))
    if not validation_rows:
        raise ValueError(
            "Pattern validation failed: test query returned no results. "
            "Check that the pattern correctly connects instances to class nodes."
        )
    logger.info(f"Validation passed ({len(validation_rows)} row(s) returned)")

    logger.info("Running property frequency query")
    freq_rows = run(build_property_frequency_query(pattern))

    logger.info("Running literal datatype profiling query")
    lit_rows = run(build_literal_profile_query(pattern))

    logger.info("Running object range profiling query")
    range_rows = run(build_range_profile_query(pattern))

    logger.info("Running total entities query")
    total_rows = run(build_total_entities_query(pattern))

    # Build per-class maps
    total_entities: dict[str, int] = {}
    for row in total_rows:
        c, t = row.get("class", ""), row.get("totalEntities", 0)
        if c:
            total_entities[c] = int(t)

    prop_freq: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in freq_rows:
        c, p = row.get("class", ""), row.get("p", "")
        if c and p:
            prop_freq[c][p] = {
                "triple_count": int(row.get("tripleCount", 0)),
                "entity_count": int(row.get("entityCount", 0)),
            }

    lit_dtypes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in lit_rows:
        c, p, dt = row.get("class", ""), row.get("p", ""), row.get("datatype", "")
        if c and p and dt:
            lit_dtypes[c][p].append(manager.format_iri(dt, wrap=True))

    range_map: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in range_rows:
        c, p, tc = row.get("class", ""), row.get("p", ""), row.get("targetClass", "")
        if c and p and tc and tc not in range_map[c][p]:
            range_map[c][p].append(tc)

    all_concept_iris = list(prop_freq.keys())[:max_concepts]
    logger.info(f"Emitting pseudo-ShEx for {len(all_concept_iris)} concepts")

    samples = []
    for c_iri in all_concept_iris:
        total = total_entities.get(c_iri, 0)
        profile = assemble_profile(
            c_iri,
            prop_freq.get(c_iri, {}),
            lit_dtypes.get(c_iri, defaultdict(list)),
            range_map.get(c_iri, defaultdict(list)),
            total,
            shape_config,
            manager,
        )
        shex = emit_pseudo_shex(profile, manager)
        shex_natural = emit_pseudo_shex_natural(profile, manager)
        class_label = manager.get_label(c_iri, "entities")

        samples.append(
            ShapeSample(
                iri=c_iri,
                short_iri=profile.short_iri,
                shex=shex,
                shex_natural=shex_natural,
                label=class_label,
            )
        )

    os.makedirs(shapes_dir, exist_ok=True)
    pattern_file = os.path.join(shapes_dir, "pattern.sparql")
    dump_text(pattern, pattern_file)
    logger.info(f"Wrote {len(samples)} shapes and pattern to {shapes_dir}")
    return samples
