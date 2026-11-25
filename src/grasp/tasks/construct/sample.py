from logging import Logger
from typing import Iterator

from pydantic import BaseModel

from grasp.manager import KgManager
from grasp.sparql.types import SelectResult
from grasp.tasks.construct.scratch import (
    SAMPLE_GRAPH,
    Scratch,
    construct,
    parse_ntriples,
)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# limited subqueries per query
MAX_BRANCHES = 200


class SampleConfig(BaseModel):
    # changes which instances are picked
    seed: int = 22
    # instances per class, picked in the order of a hash of their IRIs
    instances_per_class: int = 50
    # more instances per class and property, each from the instances with
    # the smallest, largest and longest values and from those in hash order
    witnesses_per_property: int = 2
    # subjects pointing to a picked instance, per instance and property
    incoming_per_property: int = 3
    # hops along the links from the picked instances and subjects
    max_depth: int = 3
    # the sample stops growing along links at this size
    max_triples: int = 500_000
    # subjects per query
    batch_size: int = 200
    # for queries on the knowledge graph, in seconds; QLever rejects
    # timeouts above the one of the server
    timeout: float = 60.0


class SampleStats(BaseModel):
    classes: int
    subjects: int
    triples: int
    # whether max_triples was reached
    truncated: bool


def batches(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def union(branches: list[str]) -> str:
    return "SELECT DISTINCT ?s WHERE { " + " UNION ".join(branches) + " }"


class Sampler:
    def __init__(
        self,
        manager: KgManager,
        scratch: Scratch,
        config: SampleConfig,
        logger: Logger,
    ) -> None:
        self.manager = manager
        self.scratch = scratch
        self.config = config
        self.logger = logger

    def select(self, sparql: str, var: str = "s") -> list[str]:
        result = self.manager.execute_sparql(
            sparql,
            request_timeout=(6.0, self.config.timeout),
            read_timeout=self.config.timeout,
        )
        assert isinstance(result, SelectResult)
        return [
            row[var].value
            for row in result.rows()
            if var in row and row[var].typ == "uri"
        ]

    def in_hash_order(self, pattern: str, limit: int) -> str:
        # a reproducible pick that does not follow the order of the data
        return f"""\
{{ SELECT ?s WHERE {{ {pattern} FILTER(isIRI(?s)) }}
ORDER BY SHA1(CONCAT(STR(?s), "{self.config.seed}")) LIMIT {limit} }}"""

    def instances(self, cls: str) -> list[str]:
        pattern = f"?s <{RDF_TYPE}> <{cls}> ."
        return self.select(
            union([self.in_hash_order(pattern, self.config.instances_per_class)])
        )

    def witnesses(self, cls: str) -> list[str]:
        # instances for rare properties and for the ends of the value ranges
        limit = self.config.witnesses_per_property
        if limit < 1:
            return []

        props = self.select(
            f"SELECT DISTINCT ?p WHERE {{ ?s <{RDF_TYPE}> <{cls}> ; ?p ?o }}",
            "p",
        )
        branches = []
        for prop in props:
            if prop == RDF_TYPE:
                continue
            pattern = f"?s <{RDF_TYPE}> <{cls}> ; <{prop}> ?o ."
            branches.append(self.in_hash_order(pattern, limit))
            branches.extend(
                f"{{ SELECT ?s WHERE {{ {pattern} FILTER(isIRI(?s)) }} "
                f"ORDER BY {order} LIMIT {limit} }}"
                for order in ["?o", "DESC(?o)", "DESC(STRLEN(STR(?o)))"]
            )

        subjects = set()
        for batch in batches(branches, MAX_BRANCHES):
            subjects.update(self.select(union(batch)))
        return sorted(subjects)

    def incoming(self, cls: str, instances: list[str]) -> list[str]:
        # without them, a sampled instance misses what refers to it
        limit = self.config.incoming_per_property
        if limit < 1:
            return []

        props = self.select(
            f"SELECT DISTINCT ?p WHERE {{ ?o <{RDF_TYPE}> <{cls}> . ?s ?p ?o }}",
            "p",
        )
        branches = [
            f"{{ SELECT ?s WHERE {{ ?s <{prop}> <{instance}> FILTER(isIRI(?s)) }} "
            f"LIMIT {limit} }}"
            for instance in instances
            for prop in props
        ]
        subjects = set()
        for batch in batches(branches, MAX_BRANCHES):
            subjects.update(self.select(union(batch)))
        return sorted(subjects)

    def copy(self, sparql: str) -> tuple[int, set[str]]:
        # adds the result to the sample, returns its size and the IRIs its
        # triples link to
        text = construct(self.manager, sparql, self.config.timeout).text
        triples = parse_ntriples(text)
        if triples:
            self.scratch.load(SAMPLE_GRAPH, text, replace=False)

        links = {
            obj.value
            for _, prop, obj in triples
            if obj.typ == "uri" and prop != RDF_TYPE
        }
        return len(triples), links

    def build(self) -> SampleStats:
        config = self.config
        self.scratch.drop(SAMPLE_GRAPH)

        classes = self.select(
            f"SELECT DISTINCT ?c WHERE {{ ?s <{RDF_TYPE}> ?c }} ORDER BY ?c",
            "c",
        )
        self.logger.info(f"Sampling {len(classes):,} classes of {self.manager.kg}")

        frontier: set[str] = set()
        for cls in classes:
            instances = self.instances(cls)
            frontier.update(instances)
            if len(instances) < config.instances_per_class:
                # all instances are in the sample
                continue
            frontier.update(self.witnesses(cls))
            frontier.update(self.incoming(cls, instances))

        num_triples = 0
        subjects: set[str] = set()
        truncated = False
        for depth in range(config.max_depth + 1):
            self.logger.info(
                f"Copying {len(frontier):,} subjects at depth {depth}, "
                f"{num_triples:,} triples so far"
            )
            subjects.update(frontier)
            links: set[str] = set()
            for batch in batches(sorted(frontier), config.batch_size):
                values = " ".join(f"<{iri}>" for iri in batch)
                num_batch, batch_links = self.copy(
                    f"CONSTRUCT {{ ?s ?p ?o }} WHERE {{ VALUES ?s {{ {values} }} ?s ?p ?o }}"
                )
                num_triples += num_batch
                links.update(batch_links)
                truncated = num_triples >= config.max_triples
                if truncated:
                    break

            frontier = links - subjects
            if truncated or not frontier:
                break

        if truncated:
            self.logger.warning(f"Sample reached the limit of {config.max_triples:,}")

        # what the knowledge graph says about its classes and properties
        for pattern in ["?x a ?s", "?x ?s ?y"]:
            num_schema, _ = self.copy(
                "CONSTRUCT { ?s ?p ?o } WHERE { "
                f"{{ SELECT DISTINCT ?s WHERE {{ {pattern} }} }} ?s ?p ?o }}"
            )
            num_triples += num_schema

        return SampleStats(
            classes=len(classes),
            subjects=len(subjects),
            triples=num_triples,
            truncated=truncated,
        )


def build_sample(
    manager: KgManager,
    scratch: Scratch,
    config: SampleConfig,
    logger: Logger,
) -> SampleStats:
    # a connected sample of the knowledge graph in the sample graph of the
    # scratch server, built with queries on the knowledge graph only
    return Sampler(manager, scratch, config, logger).build()
