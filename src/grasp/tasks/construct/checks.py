import math
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

MAX_EXAMPLES = 3
MAX_LINES = 50


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
    # whether a query mentions the property
    mentioned: bool = False
    explanation: str | None = None

    @property
    def unmatched(self) -> bool:
        if self.values:
            return self.matched == 0
        return not self.mentioned


class Finding(BaseModel):
    level: Literal["error", "warning"]
    message: str
    examples: list[str] = []


class OutputProperty(BaseModel):
    iri: str
    triples: int
    subjects: list[str]
    objects: list[str]


class Report(BaseModel):
    source_classes: dict[str, int]
    output_classes: dict[str, int]
    output_properties: list[OutputProperty]
    coverage: list[Coverage]
    findings: list[Finding]

    @property
    def errors(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "error"]


def schema_nodes(triples: list[Triple]) -> set[str]:
    # classes and properties
    nodes = {prop for _, prop, _ in triples}
    nodes.update(obj.value for _, prop, obj in triples if prop == RDF_TYPE)
    return nodes


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
    mentioned: set[str],
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
                mentioned=prop in mentioned,
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


def output_findings(triples: list[Triple]) -> list[Finding]:
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
    custom = {iri for iri in schema if not iri.startswith(W3C)}
    warn("classes and properties without rdfs:label", custom - labeled)

    blank = {
        term.value
        for subj, _, obj in triples
        for term in (subj, obj)
        if term.typ == "bnode"
    }
    warn("blank nodes, they have no stable identity across queries", blank)

    dangling: dict[str, set[str]] = defaultdict(set)
    kinds: dict[str, set[str]] = defaultdict(set)
    for _, prop, obj in triples:
        if prop == RDF_TYPE:
            continue
        kinds[prop].add(object_kind(obj))
        if obj.typ == "uri" and obj.value not in subjects:
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

    return findings


def check_output(
    source: Source,
    triples: list[Triple],
    mentioned: set[str],
    explanations: dict[str, str],
) -> Report:
    # facts about the output of the queries on the sample; properties of the
    # sample without any trace in the output are errors unless explained
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

    covered = coverage(source, triples, mentioned, explanations)
    unexplained = [
        cov.iri for cov in covered if cov.unmatched and cov.explanation is None
    ]
    if unexplained:
        findings.append(
            Finding(
                level="error",
                message=f"{len(unexplained):,} properties of the knowledge graph "
                "have no value in the output and no query mentions their links; "
                "map them, or explain each of them when submitting",
                examples=unexplained,
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
        findings=findings + output_findings(triples),
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
        mentioned = "mentioned" if cov.mentioned else "not mentioned"
        parts.append(f"{cov.links:,} links, {mentioned} by a query")
    fmt = f"<{cov.iri}>: {'; '.join(parts)}"
    if cov.explanation is not None:
        fmt += f"; explanation: {cov.explanation}"
    return fmt


def format_report(report: Report) -> str:
    # full coverage is the common case and only counted
    partial = [
        cov
        for cov in report.coverage
        if cov.matched < cov.values or cov.unmatched or cov.explanation is not None
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

    def counts(classes: dict[str, int]) -> str:
        return capped([f"<{cls}>: {count:,}" for cls, count in classes.items()])

    properties = capped(
        [
            f"<{prop.iri}>: {prop.triples:,} triples, "
            f"from {', '.join(prop.subjects[:MAX_EXAMPLES])} "
            f"to {', '.join(prop.objects[:MAX_EXAMPLES])}"
            for prop in report.output_properties
        ]
    )
    return f"""\
Errors, the queries cannot be submitted with them:
{findings("error")}

Warnings:
{findings("warning")}

Properties of the sample whose values are not all in the output \
({full:,} other properties have all values in the output or their links \
mentioned by a query):
{capped([format_coverage(cov) for cov in partial])}

Instances per class in the sample:
{counts(report.source_classes)}

Instances per class in the output:
{counts(report.output_classes)}

Properties in the output:
{properties}"""
