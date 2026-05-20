import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm
from universal_ml_utils.io import dump_text
from universal_ml_utils.logging import get_logger

from grasp.configs import ShapeConfig
from grasp.manager import KgManager
from grasp.shapes import ShapeSample
from grasp.sparql.types import SelectResult
from grasp.utils import derive_label_from_iri, get_local_name_from_iri


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
class ClassProfile:
    iri: str
    short_iri: str
    total_entities: int = 0
    properties: list[PropertyProfile] = field(default_factory=list)
    omitted_properties: int = 0
    filtered_properties: int = 0


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
    inner = "\n".join(f"    {line}" for line in cp.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


def objectmembership(pattern: str) -> str:
    op = object_pattern(pattern)
    inner = "\n".join(f"    {line}" for line in op.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


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


def wrap_pattern(pattern: str) -> str:
    inner = "\n".join(f"    {line}" for line in pattern.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


def build_per_class_property_frequency_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    return (
        f"SELECT ?p (COUNT(*) AS ?tripleCount) (COUNT(DISTINCT ?instance) AS ?entityCount)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?p\n"
        f"ORDER BY DESC(?tripleCount)"
    )


def build_per_class_literal_profile_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    return (
        f"SELECT ?p ?datatype (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"  FILTER(isLiteral(?o))\n"
        f"  BIND(DATATYPE(?o) AS ?datatype)\n"
        f"}}\n"
        f"GROUP BY ?p ?datatype\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_range_profile_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    op = wrap_pattern(object_pattern(pattern))
    return (
        f"SELECT ?p ?targetClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"{op}\n"
        f"}}\n"
        f"GROUP BY ?p ?targetClass\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_total_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    return f"SELECT (COUNT(DISTINCT ?instance) AS ?totalEntities)\nWHERE {{\n{ip}\n}}"


def property_rank(iri: str, manager: KgManager) -> int:
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


def resolve_label(
    iri: str,
    index_name: str,
    short_iri: str,
    manager: KgManager,
) -> str:
    label = manager.get_label(iri, index_name)
    if not label:
        return short_iri
    local_name = get_local_name_from_iri(iri, manager.prefixes)
    derived = derive_label_from_iri(iri, manager.prefixes)
    if local_name.lower() == label.lower() or derived.lower() == label.lower():
        return short_iri
    return f"{short_iri} ({label})"


def get_label_and_aliases(
    iri: str,
    index_name: str,
    manager: KgManager,
) -> tuple[str | None, list[str]]:
    data = manager.try_get_data(index_name)
    if data is None:
        return None, []
    norm = manager.normalize(iri, index_name)
    key = norm[0] if norm is not None else iri
    id = data.id_from_identifier(key)
    if id is None:
        return None, []
    return data.main_field(id) or data.field(id, 0), data.fields(id) or []


def _target_class_parts(prop: "PropertyProfile", manager: KgManager) -> list[str]:
    if prop.target_class_iris:
        return [
            resolve_label(t_iri, "entities", t_short, manager)
            for t_iri, t_short in zip(
                prop.target_class_iris, prop.target_class_short_iris
            )
        ]
    return prop.target_class_short_iris[:3]


def emit_pseudo_shex(profile: ClassProfile, manager: KgManager) -> str:
    header = resolve_label(profile.iri, "entities", profile.short_iri, manager)
    lines = [f"{header} {{"]

    for prop in profile.properties:
        prop_str = resolve_label(prop.iri, "properties", prop.short_iri, manager)
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

    if profile.omitted_properties or profile.filtered_properties:
        parts = []
        if profile.omitted_properties:
            parts.append(f"{profile.omitted_properties:,} omitted (cap)")
        if profile.filtered_properties:
            parts.append(f"{profile.filtered_properties:,} filtered (low coverage)")
        lines.append(f"  # ... {', '.join(parts)}")
    lines.append("}")
    return "\n".join(lines)


def collect_iris(profile: ClassProfile) -> list[str]:
    iris = []
    for prop in profile.properties:
        iris.append(prop.iri)
        iris.extend(prop.target_class_iris)
    return iris


def assemble_profile(
    class_iri: str,
    freq_map: dict[str, dict],
    lit_dtypes: dict[str, list[str]],
    range_map: dict[str, list[str]],
    total_entities: int,
    shape_config: ShapeConfig,
    manager: KgManager,
) -> ClassProfile:
    properties: list[PropertyProfile] = []
    filtered_count = 0

    common_props = sorted(
        freq_map.items(),
        key=lambda item: (
            property_rank(item[0], manager),
            -item[1].get("triple_count", 0),
        ),
    )

    for p_iri, freq in common_props[: shape_config.max_properties_per_class]:
        entity_count = freq.get("entity_count", 0)
        triple_count = freq.get("triple_count", 0)

        if total_entities > 0:
            coverage = entity_count / total_entities
            if coverage < shape_config.min_property_coverage:
                filtered_count += 1
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
            target_class_short_iris=[
                manager.format_iri(t, wrap=True) for t in target_iris
            ],
        )
        properties.append(prop)

    return ClassProfile(
        iri=class_iri,
        short_iri=manager.format_iri(class_iri, wrap=True),
        total_entities=total_entities,
        properties=properties,
        omitted_properties=max(
            0, len(common_props) - shape_config.max_properties_per_class
        ),
        filtered_properties=filtered_count,
    )


def compute_shape(
    class_iri: str,
    pattern: str,
    manager: KgManager,
    shape_config: ShapeConfig | None = None,
) -> ClassProfile:
    if shape_config is None:
        shape_config = ShapeConfig()

    def run(query: str) -> list[dict]:
        result = manager.execute_sparql(
            query,
            shape_config.request_timeout,
            shape_config.read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        assert isinstance(result, SelectResult), "Expected SELECT query result"
        return [
            {var: row[var].value for var in result.variables} for row in result.rows()
        ]

    try:
        freq_rows = run(build_per_class_property_frequency_query(pattern, class_iri))
        lit_rows = run(build_per_class_literal_profile_query(pattern, class_iri))
        range_rows = run(build_per_class_range_profile_query(pattern, class_iri))
        total_rows = run(build_per_class_total_query(pattern, class_iri))
    except Exception as e:
        raise RuntimeError(
            f"One of the queries for computing the shape of '{class_iri}' failed:\n{e}"
        )

    total = int(total_rows[0]["totalEntities"]) if total_rows else 0

    freq_map: dict[str, dict] = {}
    for row in freq_rows:
        freq_map[row["p"]] = {
            "triple_count": int(row["tripleCount"]),
            "entity_count": int(row["entityCount"]),
        }

    lit_dtypes: dict[str, list[str]] = defaultdict(list)
    for row in lit_rows:
        lit_dtypes[row["p"]].append(manager.format_iri(row["datatype"], wrap=True))

    range_map: dict[str, list[str]] = defaultdict(list)
    for row in range_rows:
        p, tc = row["p"], row["targetClass"]
        if tc not in range_map[p]:
            range_map[p].append(tc)

    return assemble_profile(
        class_iri,
        freq_map,
        lit_dtypes,
        range_map,
        total,
        shape_config,
        manager,
    )


def build_shapes(
    pattern: str,
    shapes_dir: str,
    manager: KgManager,
    shape_config: ShapeConfig = ShapeConfig(),
    max_classes: int = 500,
    log_level: str | int | None = None,
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> list[ShapeSample]:
    logger = get_logger("GRASP SHAPES BUILD", log_level)

    def run(query: str) -> list[dict[str, Any]]:
        logger.debug(f"Running query:\n{query}")
        result = manager.execute_sparql(
            query,
            request_timeout,
            read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        assert isinstance(result, SelectResult), "Expected SELECT query result"
        return [
            {var: row[var].value for var in result.variables} for row in result.rows()
        ]

    logger.info("Discovering classes")
    try:
        total_rows = run(build_total_entities_query(pattern))
    except Exception as e:
        raise ValueError(
            f"Pattern validation failed: could not query classes ({e}). "
            "Check that the pattern correctly connects instances to class nodes."
        ) from e
    if not total_rows:
        raise ValueError(
            "Pattern validation failed: no classes found. "
            "Check that the pattern correctly connects instances to class nodes."
        )
    total_entities = {row["class"]: int(row["totalEntities"]) for row in total_rows}

    all_class_iris = list(total_entities.keys())[:max_classes]
    logger.info(f"Building shapes for {len(all_class_iris)} classes")

    samples = []
    skipped = 0
    for c_iri in tqdm(all_class_iris, desc="Profiling classes"):
        try:
            freq_rows = run(build_per_class_property_frequency_query(pattern, c_iri))
            lit_rows = run(build_per_class_literal_profile_query(pattern, c_iri))
            range_rows = run(build_per_class_range_profile_query(pattern, c_iri))
        except Exception as e:
            logger.warning(
                f"Skipping {c_iri}: profiling query failed ({e}). "
                "To retry, increase timeouts in the shape config and re-run with --overwrite."
            )
            skipped += 1
            continue

        freq_map: dict[str, dict] = {}
        for row in freq_rows:
            freq_map[row["p"]] = {
                "triple_count": int(row["tripleCount"]),
                "entity_count": int(row["entityCount"]),
            }

        lit_dtypes: dict[str, list[str]] = defaultdict(list)
        for row in lit_rows:
            lit_dtypes[row["p"]].append(manager.format_iri(row["datatype"], wrap=True))

        range_map: dict[str, list[str]] = defaultdict(list)
        for row in range_rows:
            p, tc = row["p"], row["targetClass"]
            if tc not in range_map[p]:
                range_map[p].append(tc)

        profile = assemble_profile(
            c_iri,
            freq_map,
            lit_dtypes,
            range_map,
            total_entities[c_iri],
            shape_config,
            manager,
        )
        dense_config = shape_config.model_copy(
            update={
                "max_properties_per_class": shape_config.dense_max_properties_per_class
            }
        )
        dense_profile = assemble_profile(
            c_iri,
            freq_map,
            lit_dtypes,
            range_map,
            total_entities[c_iri],
            dense_config,
            manager,
        )
        shex = emit_pseudo_shex(profile, manager)
        dense_shex = emit_pseudo_shex(dense_profile, manager)
        iris = collect_iris(profile)
        dense_iris = collect_iris(dense_profile)
        class_label, class_aliases = get_label_and_aliases(c_iri, "entities", manager)
        if class_label:
            derived = derive_label_from_iri(c_iri, manager.prefixes)
            if derived and derived.lower() == class_label.lower():
                class_label = None

        samples.append(
            ShapeSample(
                iri=c_iri,
                short_iri=profile.short_iri,
                shex=shex,
                dense_shex=dense_shex,
                iris=iris,
                dense_iris=dense_iris,
                label=class_label,
                aliases=class_aliases,
            )
        )

    if skipped:
        logger.warning(f"Skipped {skipped:,} class(es) due to query failures")

    os.makedirs(shapes_dir, exist_ok=True)
    pattern_file = os.path.join(shapes_dir, "pattern.sparql")
    dump_text(pattern, pattern_file)
    logger.info(f"Wrote {len(samples)} shapes and pattern to {shapes_dir}")
    return samples
