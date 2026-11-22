import math
import re
from collections import Counter, defaultdict
from typing import Literal
from urllib.parse import unquote

from pydantic import BaseModel

from grasp.sparql.types import Binding
from grasp.tasks.construct.sample import RDF_TYPE
from grasp.tasks.construct.scratch import Triple
from grasp.utils import clip, format_list

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
# namespace of rdf, rdfs, owl and xsd
W3C = "http://www.w3.org/"

XSD = "http://www.w3.org/2001/XMLSchema#"

MAX_EXAMPLES = 3
MAX_LINES = 50
# values a property needs for findings about all of its values
MIN_VALUES = 5
# strings of a property that repeat are reported up to this mean length
MAX_NAME_LENGTH = 80

INTEGER = re.compile(r"[+-]?\d+")
DECIMAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)")
DOUBLE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?INF|NaN")
ZONE = r"(Z|[+-]\d\d:\d\d)?"
DATE = re.compile(rf"-?\d{{4,}}-\d\d-\d\d{ZONE}")
DATE_TIME = re.compile(rf"-?\d{{4,}}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?{ZONE}")
BOOLEAN = re.compile(r"true|false|yes|no", re.IGNORECASE)
# lexical forms of common datatypes
LEXICAL = {
    "integer": INTEGER,
    "int": INTEGER,
    "long": INTEGER,
    "nonNegativeInteger": re.compile(r"\+?\d+"),
    "decimal": DECIMAL,
    "double": DOUBLE,
    "float": DOUBLE,
    "boolean": re.compile(r"true|false|1|0"),
    "date": DATE,
    "dateTime": DATE_TIME,
    "gYear": re.compile(rf"-?\d{{4,}}{ZONE}"),
}
FRACTIONAL = {XSD + name for name in ["decimal", "double", "float"]}


class SourceProperty(BaseModel):
    iri: str
    # distinct literal values and number of links to other entities
    values: set[str] = set()
    links: int = 0


class Source(BaseModel):
    # what the checks need from the sample of the knowledge graph
    classes: dict[str, int]
    properties: dict[str, SourceProperty]


class Coverage(BaseModel):
    iri: str
    values: int
    matched: int
    missing: list[str] = []
    links: int = 0
    # whether the links have an effect on the output of a query, and the
    # largest number of triples in the output of a query that depend on them
    used: bool = False
    dependent: int = 0
    explanation: str | None = None

    @property
    def unmatched(self) -> bool:
        if self.values:
            return self.matched == 0
        return not self.used

    @property
    def partial(self) -> bool:
        # fewer triples depend on the links than there are links
        return 0 < self.dependent < self.links


class Finding(BaseModel):
    level: Literal["error", "warning"]
    message: str
    examples: list[str] = []


class OutputProperty(BaseModel):
    iri: str
    triples: int
    subjects: list[str]
    objects: list[str]


class Vocabulary(BaseModel):
    # classes and properties of the output by the other knowledge graph
    # that defines them
    defined: dict[str, str] = {}
    # and those in a namespace of another knowledge graph without being
    # defined by it
    undefined: dict[str, str] = {}
    # other knowledge graphs with failed lookups
    failed: list[str] = []


class Report(BaseModel):
    source_classes: dict[str, int]
    output_classes: dict[str, int]
    output_properties: list[OutputProperty]
    coverage: list[Coverage]
    findings: list[Finding]
    vocabulary: Vocabulary = Vocabulary()

    @property
    def errors(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "error"]


def schema_nodes(triples: list[Triple]) -> set[str]:
    # classes and properties
    nodes = {prop for _, prop, _ in triples}
    nodes.update(obj.value for _, prop, obj in triples if prop == RDF_TYPE)
    return nodes


def schema_terms(triples: list[Triple]) -> set[str]:
    # classes and properties together with what is said to relate to them,
    # like the classes they are subclasses of
    schema = schema_nodes(triples)
    related = {
        obj.value
        for subj, prop, obj in triples
        if subj.value in schema and prop != RDF_TYPE and obj.typ == "uri"
    }
    return schema | related


def source_profile(triples: list[Triple]) -> Source:
    # values and links of the instances, without the triples about classes
    # and properties themselves
    schema = schema_nodes(triples)
    classes: Counter[str] = Counter()
    properties: dict[str, SourceProperty] = {}
    for subj, prop, obj in triples:
        if subj.value in schema:
            continue
        elif prop == RDF_TYPE:
            classes[obj.value] += 1
            continue

        source = properties.setdefault(prop, SourceProperty(iri=prop))
        if obj.typ == "literal":
            source.values.add(obj.value)
        else:
            source.links += 1

    return Source(classes=dict(classes), properties=properties)


def value_keys(value: str) -> set[str]:
    # the value itself and its number, so 1 matches 1.0
    value = value.strip()
    keys = {value}
    try:
        number = float(value)
    except ValueError:
        return keys
    if math.isfinite(number):
        keys.add(f"number {number!r}")
    return keys


def output_keys(triples: list[Triple]) -> set[str]:
    # literal values and the last part of IRIs, values can end up in both
    keys = set()
    for subj, _, obj in triples:
        for term in (subj, obj):
            if term.typ == "literal":
                keys.update(value_keys(term.value))
            elif term.typ == "uri":
                local = term.value.replace("#", "/").rsplit("/", 1)[-1]
                keys.update({local, unquote(local)})
    return keys


def coverage(
    source: Source,
    triples: list[Triple],
    effects: dict[str, int],
    explanations: dict[str, str],
) -> list[Coverage]:
    keys = output_keys(triples)
    result = []
    for prop, source_prop in sorted(source.properties.items()):
        missing = sorted(
            value for value in source_prop.values if not value_keys(value) & keys
        )
        result.append(
            Coverage(
                iri=prop,
                values=len(source_prop.values),
                matched=len(source_prop.values) - len(missing),
                missing=missing[:MAX_EXAMPLES],
                links=source_prop.links,
                used=prop in effects,
                dependent=effects.get(prop, 0),
                explanation=explanations.get(prop),
            )
        )
    return result


def object_kind(term: Binding) -> str:
    if term.typ == "literal":
        return term.datatype or ("language string" if term.lang else "string")
    return "IRI" if term.typ == "uri" else "blank node"


def output_properties(triples: list[Triple]) -> list[OutputProperty]:
    types = defaultdict(set)
    for subj, prop, obj in triples:
        if prop == RDF_TYPE:
            types[subj.value].add(obj.value)

    counts: Counter[str] = Counter()
    subjects: dict[str, Counter[str]] = defaultdict(Counter)
    objects: dict[str, Counter[str]] = defaultdict(Counter)
    for subj, prop, obj in triples:
        if prop == RDF_TYPE:
            continue
        counts[prop] += 1
        subjects[prop].update(types.get(subj.value) or ["untyped"])
        if obj.typ == "uri":
            objects[prop].update(types.get(obj.value) or ["IRI"])
        else:
            objects[prop][object_kind(obj)] += 1

    return [
        OutputProperty(
            iri=prop,
            triples=count,
            subjects=[name for name, _ in subjects[prop].most_common()],
            objects=[name for name, _ in objects[prop].most_common()],
        )
        for prop, count in sorted(counts.items())
    ]


def examples(items: set[str]) -> list[str]:
    return sorted(items)[:MAX_EXAMPLES]


def short(datatype: str) -> str:
    return "xsd:" + datatype[len(XSD) :] if datatype.startswith(XSD) else datatype


def value_findings(triples: list[Triple]) -> list[Finding]:
    # literals whose datatype or use as a string does not fit their values
    labels: dict[str, set[str]] = defaultdict(set)
    literals: dict[str, list[tuple[str, Binding]]] = defaultdict(list)
    for subj, prop, obj in triples:
        if obj.typ != "literal":
            continue
        elif prop == RDFS_LABEL:
            labels[obj.value].add(subj.value)
        else:
            literals[prop].append((subj.value, obj))

    findings = []

    def warn(message: str, values: list[str]) -> None:
        distinct = sorted(set(values))[:MAX_EXAMPLES]
        findings.append(
            Finding(
                level="warning",
                message=message,
                examples=[f'"{clip(value, 40)}"' for value in distinct],
            )
        )

    for prop, objects in sorted(literals.items()):
        by_type: dict[str, list[str]] = defaultdict(list)
        for _, obj in objects:
            by_type[obj.datatype or ""].append(obj.value)

        for datatype, values in sorted(by_type.items()):
            if not datatype.startswith(XSD):
                continue
            pattern = LEXICAL.get(datatype[len(XSD) :])
            if pattern is None:
                continue

            invalid = [value for value in values if not pattern.fullmatch(value)]
            if invalid:
                warn(
                    f"{len(invalid):,} values of <{prop}> are not valid for "
                    f"their datatype {short(datatype)}",
                    invalid,
                )
            elif (
                datatype in FRACTIONAL
                and len(values) >= MIN_VALUES
                and all(float(value).is_integer() for value in values)
            ):
                warn(
                    f"All {len(values):,} values of <{prop}> are whole numbers "
                    f"with the datatype {short(datatype)}, check whether they "
                    "are fractional by nature or need another datatype",
                    values,
                )

        strings = [
            (subj, obj.value)
            for subj, obj in objects
            if obj.datatype in (None, XSD + "string")
        ]
        values = [value.strip() for _, value in strings]
        if len(values) < MIN_VALUES:
            continue

        kinds = [("numbers", DOUBLE), ("dates", DATE), ("booleans", BOOLEAN)]
        kind = next(
            (
                kind
                for kind, pattern in kinds
                if all(pattern.fullmatch(value) for value in values)
            ),
            None,
        )
        if kind is not None:
            warn(
                f"All {len(values):,} values of <{prop}> are strings that look "
                f"like {kind}",
                values,
            )
            continue

        # labels of other entities than the subject
        references = [
            value for subj, value in strings if labels.get(value, set()) - {subj}
        ]
        distinct = len(set(values))
        if 2 * len(references) >= len(values):
            warn(
                f"{len(references):,} of {len(values):,} strings of <{prop}> are "
                "labels of other entities in the output, check whether they "
                "refer to them and should be links",
                references,
            )
        elif (
            len(values) >= 2 * MIN_VALUES
            and 2 * distinct <= len(values)
            and sum(map(len, values)) <= MAX_NAME_LENGTH * len(values)
        ):
            warn(
                f"<{prop}> has only {distinct:,} distinct strings in "
                f"{len(values):,} triples, check whether they name things that "
                "should be entities of their own",
                values,
            )

    return findings


def output_findings(triples: list[Triple], vocabulary: Vocabulary) -> list[Finding]:
    schema = schema_nodes(triples)
    subjects = {subj.value for subj, _, _ in triples}
    typed = {subj.value for subj, prop, _ in triples if prop == RDF_TYPE}
    labeled = {subj.value for subj, prop, _ in triples if prop == RDFS_LABEL}
    entities = subjects - schema

    findings = []

    def warn(message: str, items: set[str]) -> None:
        if items:
            findings.append(
                Finding(
                    level="warning",
                    message=f"{len(items):,} {message}",
                    examples=examples(items),
                )
            )

    warn("entities without rdf:type", entities - typed)
    warn("entities without rdfs:label", entities - labeled)
    custom = {
        iri
        for iri in schema
        if not iri.startswith(W3C) and iri not in vocabulary.defined
    }
    warn("classes and properties without rdfs:label", custom - labeled)

    blank = {
        term.value
        for subj, _, obj in triples
        for term in (subj, obj)
        if term.typ == "bnode"
    }
    warn("blank nodes, they have no stable identity across queries", blank)

    for kg in sorted(set(vocabulary.failed)):
        findings.append(
            Finding(
                level="warning",
                message=f"Failed to look up the classes and properties in {kg}",
            )
        )
    for kg in sorted(set(vocabulary.undefined.values())):
        undefined = {iri for iri, other in vocabulary.undefined.items() if other == kg}
        warn(
            f"classes and properties in a namespace of {kg} that it does not define",
            undefined,
        )

    dangling: dict[str, set[str]] = defaultdict(set)
    kinds: dict[str, set[str]] = defaultdict(set)
    for _, prop, obj in triples:
        if prop == RDF_TYPE:
            continue
        kinds[prop].add(object_kind(obj))
        if obj.typ != "uri" or obj.value in subjects:
            continue
        elif obj.value.startswith(W3C) or obj.value in vocabulary.defined:
            # described elsewhere
            continue
        dangling[prop].add(obj.value)

    for prop, targets in sorted(dangling.items()):
        warn(
            f"IRIs linked by <{prop}> that no query describes (targets outside "
            "of the sample can be a reason)",
            targets,
        )

    for prop, prop_kinds in sorted(kinds.items()):
        if len(prop_kinds) > 1:
            findings.append(
                Finding(
                    level="warning",
                    message=f"<{prop}> has objects of different kinds",
                    examples=sorted(prop_kinds),
                )
            )

    return findings + value_findings(triples)


def check_output(
    source: Source,
    triples: list[Triple],
    effects: dict[str, int],
    explanations: dict[str, str],
    empty: dict[str, list[str]],
    vocabulary: Vocabulary,
) -> Report:
    # facts about the output of the queries on the sample; properties of the
    # sample without any trace in the output are errors unless explained.
    # effects has the link properties with an effect on the output of a
    # query and how many triples of its output depend on them at most,
    # empty the UNION branches without matches by query, and vocabulary says
    # which classes and properties are from other knowledge graphs
    findings = []
    unknown = sorted(set(explanations) - set(source.properties))
    if unknown:
        findings.append(
            Finding(
                level="error",
                message="Explanations for properties that are not in the sample "
                "of the knowledge graph, use their full IRIs",
                examples=unknown,
            )
        )

    covered = coverage(source, triples, effects, explanations)
    unexplained = [
        cov.iri for cov in covered if cov.unmatched and cov.explanation is None
    ]
    if unexplained:
        findings.append(
            Finding(
                level="error",
                message=f"{len(unexplained):,} properties of the knowledge graph "
                "have no value in the output and their links have no effect on "
                "the output of any query; map them, or explain each of them when "
                "submitting",
                examples=unexplained,
            )
        )

    partial = [cov for cov in covered if cov.partial]
    if partial:
        findings.append(
            Finding(
                level="warning",
                message=f"{len(partial):,} properties of the knowledge graph have "
                "more links in the sample than triples in the output of a query "
                "that depend on them, check that no links are dropped",
                examples=[
                    f"<{cov.iri}> ({cov.dependent:,} triples for {cov.links:,} links)"
                    for cov in partial
                ],
            )
        )

    for name, branches in empty.items():
        findings.append(
            Finding(
                level="warning",
                message=f"{len(branches):,} UNION branches of query {name} match "
                "nothing on the sample",
                examples=branches,
            )
        )

    classes: Counter[str] = Counter(
        obj.value for _, prop, obj in triples if prop == RDF_TYPE
    )
    return Report(
        source_classes=dict(sorted(source.classes.items())),
        output_classes=dict(sorted(classes.items())),
        output_properties=output_properties(triples),
        coverage=covered,
        findings=findings + output_findings(triples, vocabulary),
        vocabulary=vocabulary,
    )


def capped(lines: list[str]) -> str:
    if not lines:
        return "None"
    more = len(lines) - MAX_LINES
    if more > 0:
        lines = lines[:MAX_LINES] + [f"... and {more:,} more"]
    return format_list(lines)


def format_coverage(cov: Coverage) -> str:
    parts = []
    if cov.values:
        part = f"{cov.matched:,} of {cov.values:,} values in the output"
        if cov.missing:
            part += ", missing e.g. " + ", ".join(
                f'"{clip(value, 40)}"' for value in cov.missing
            )
        parts.append(part)
    if cov.links:
        if cov.partial:
            used = f"{cov.dependent:,} triples of the output depend on them"
        else:
            used = "used" if cov.used else "without effect on the output"
        parts.append(f"{cov.links:,} links, {used}")
    fmt = f"<{cov.iri}>: {'; '.join(parts)}"
    if cov.explanation is not None:
        fmt += f"; explanation: {cov.explanation}"
    return fmt


def format_report(report: Report) -> str:
    # full coverage is the common case and only counted
    partial = [
        cov
        for cov in report.coverage
        if cov.matched < cov.values
        or cov.unmatched
        or cov.partial
        or cov.explanation is not None
    ]
    partial.sort(key=lambda cov: (cov.matched / max(cov.values, 1), cov.iri))
    full = len(report.coverage) - len(partial)

    def findings(level: str) -> str:
        return capped(
            [
                finding.message
                + (f": {', '.join(finding.examples)}" if finding.examples else "")
                for finding in report.findings
                if finding.level == level
            ]
        )

    def origin(iri: str) -> str:
        kg = report.vocabulary.defined.get(iri)
        return "" if kg is None else f", defined in {kg}"

    def counts(classes: dict[str, int]) -> str:
        return capped(
            [f"<{cls}>: {count:,}{origin(cls)}" for cls, count in classes.items()]
        )

    properties = capped(
        [
            f"<{prop.iri}>: {prop.triples:,} triples, "
            f"from {', '.join(prop.subjects[:MAX_EXAMPLES])} "
            f"to {', '.join(prop.objects[:MAX_EXAMPLES])}{origin(prop.iri)}"
            for prop in report.output_properties
        ]
    )
    return f"""\
Errors, the queries cannot be submitted with them:
{findings("error")}

Warnings:
{findings("warning")}

Properties of the sample whose values are not all in the output \
({full:,} other properties have all values in the output or links that \
are used by a query):
{capped([format_coverage(cov) for cov in partial])}

Instances per class in the sample:
{counts(report.source_classes)}

Instances per class in the output:
{counts(report.output_classes)}

Properties in the output:
{properties}"""
