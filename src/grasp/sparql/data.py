import random
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Iterator, Optional

from search_index import SearchIndex, normalize

from grasp.sparql.constants import Binding, ObjType, Position
from grasp.sparql.manager.base import KgManager
from grasp.sparql.mapping import Mapping
from grasp.sparql.selection import Alternative, Selection
from grasp.sparql.sparql import (
    autocomplete_prefix,
    find_all,
    find_longest_prefix,
    parse_string,
)

__all__ = [
    "Item",
    "get_sparql_items",
    "natural_sparql_from_items",
    "selections_from_items",
]

# common stopwords from NLTK
STOP = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "et",
    "al",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
}


def _get_keywords(s: str, exclude_stopwords: bool = True) -> Iterator[str]:
    for w in normalize(s).split():
        wn = w.strip(string.punctuation)
        if wn and (not exclude_stopwords or wn not in STOP):
            yield wn


@dataclass
class Item:
    parse: dict
    item_span: tuple[int, int]
    prefix: str
    item: str
    suffix: str
    alternative: Alternative
    obj_type: ObjType
    variant: str | None
    invalid: bool = False

    def same_as(self, other: "Item") -> bool:
        return self.alternative.identifier == other.alternative.identifier

    @property
    def full_prefix(self) -> str:
        return self.prefix + self.item

    @property
    def is_other_or_literal(self) -> bool:
        return self.obj_type == ObjType.OTHER or self.obj_type == ObjType.LITERAL

    @property
    def is_entity_or_property(self) -> bool:
        return not self.is_other_or_literal

    @property
    def selection(self) -> Selection:
        return Selection(self.alternative, self.obj_type, self.variant)

    def continuation(self, other: Optional["Item"]) -> str:
        end, _ = self.item_span
        if other is None:
            return self.full_prefix[:end]

        assert other.item_span < self.item_span, "other item must come before this one"
        assert self.prefix.startswith(other.prefix), "prefix mismatch"

        _, start = other.item_span
        return self.full_prefix[start:end]

    def search_query(self, question: str, manager: KgManager) -> str:
        label = self.alternative.label or self.alternative.get_identifier()
        if self.is_other_or_literal:
            return label

        labels = [label]
        aliases = self.alternative.aliases or []
        labels.extend(aliases)

        # make aliases less likely to be selected compared to the label
        counts = [max(1, len(aliases))] + [1] * len(aliases)
        label = random.sample(labels, 1, counts=counts)[0]

        index = _index(manager, self.obj_type)
        index_type = index.get_type()

        if index_type == "prefix" or index_type == "similarity":
            label_keywords = []
            # get unique label keywords
            for k in _get_keywords(label, exclude_stopwords=True):
                if k in label_keywords:
                    continue

                label_keywords.append(k)

            # determine the question keywords that match the label keywords
            matching = defaultdict(list)
            question_keywords = set()
            for k in _get_keywords(question, exclude_stopwords=False):
                if k in question_keywords:
                    continue

                question_keywords.add(k)
                i = next(
                    (i for i, lk in enumerate(label_keywords) if lk.startswith(k)),
                    None,
                )
                if i is not None:
                    # question keyword k matches label keyword lk at position i
                    matching[i].append(k)

            for matches in matching.values():
                # longest first
                matches.sort(key=len, reverse=True)

            non_matching = []
            for i, k in enumerate(label_keywords):
                if i in matching:
                    continue
                elif any(qk.startswith(k) for qk in question_keywords):
                    # also count keywords that are prefixes of question keywords
                    matching[i].append(k)
                else:
                    non_matching.append(k)

            # sort by label position
            matching = [matching[i][0] for i in sorted(matching)]

            query = " ".join(matching or non_matching)

        else:
            raise ValueError(f"Unsupported index type {index_type}")

        return query


def _byte_span(parse: dict, start: int = sys.maxsize, end: int = 0) -> tuple[int, int]:
    if "children" in parse:
        for child in parse["children"]:
            start, end = _byte_span(child, start, end)
        return start, end

    f, t = parse["byte_span"]
    return min(start, f), max(end, t)


def _mapping(manager: KgManager, obj_type: ObjType) -> Mapping:
    if obj_type == ObjType.ENTITY:
        return manager.entity_mapping
    elif obj_type == ObjType.PROPERTY:
        return manager.property_mapping
    else:
        raise ValueError(f"Invalid object type: {obj_type}")


def _index(manager: KgManager, obj_type: ObjType) -> SearchIndex:
    if obj_type == ObjType.ENTITY:
        return manager.entity_index
    elif obj_type == ObjType.PROPERTY:
        return manager.property_index
    else:
        raise ValueError(f"Invalid object type: {obj_type}")


def format_literal(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        return s.strip('"')
    elif s.startswith("'") and s.endswith("'"):
        return s.strip("'")
    else:
        return s


def parse_binding(input: str, manager: KgManager) -> Binding | None:
    try:
        parse, _ = parse_string(
            input,
            manager.iri_literal_parser,
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
            if pfx not in manager.prefixes:
                return None

            uri = manager.prefixes[pfx][1:] + name

            # prefixed IRI
            return Binding(
                typ="uri",
                value=uri,
            )

        # not used as of now, but keep for later
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
                    if pfx not in manager.prefixes:
                        return None

                    datatype = manager.prefixes[pfx][1:] + name

                return Binding(
                    typ="literal",
                    value=format_literal(lit["value"]),
                    datatype=datatype,
                )

        case other:
            raise ValueError(
                f"Unexpected type {other} for IRI or literal: {input}",
            )


def _get_item(
    parse: dict,
    manager: KgManager,
    sparql_encoded: bytes,
    indexed_prefixes: dict[str, str] | None = None,
) -> Item | None:
    # return tuple with identifier, variant, label, synonyms
    # and additional info
    (byte_start, byte_end) = _byte_span(parse)
    prefix = sparql_encoded[:byte_start].decode()
    item = sparql_encoded[byte_start:byte_end].decode()
    suffix = sparql_encoded[byte_end:].decode()
    start = len(prefix)
    end = start + len(item)

    infos = {
        "parse": parse,
        "item": item,
        "item_span": (start, end),
        "prefix": prefix,
        "suffix": suffix,
    }

    binding = parse_binding(item, manager)
    if binding is None:
        return None

    if binding.typ == "literal":
        if binding.datatype is not None:
            info = manager.format_iri("<" + binding.datatype + ">")
        elif binding.lang is not None:
            info = binding.lang
        else:
            info = None

        return Item(
            alternative=Alternative(
                identifier=binding.identifier(),
                short_identifier=binding.identifier(),
                label=binding.value,
                infos=[info] if info else None,
            ),
            obj_type=ObjType.LITERAL,
            variant=None,
            **infos,
        )

    # we have an iri
    iri = binding.identifier()

    # check that it is in known prefixes
    if manager.find_longest_prefix(iri) is None:
        return None

    try:
        _, position = autocomplete_prefix(prefix, manager.sparql_parser)
        if position in [Position.SUBJECT, Position.OBJECT]:
            obj_types = [ObjType.ENTITY]
        else:
            obj_types = [ObjType.PROPERTY]
    except Exception:
        obj_types = [ObjType.PROPERTY, ObjType.ENTITY]

    # check whether the iri is a valid entity or property
    for i, obj_type in enumerate(obj_types):
        map = _mapping(manager, obj_type)
        norm = map.normalize(iri)
        if norm is None:
            continue

        norm_iri, variant = norm
        if norm_iri not in map:
            continue

        id = map[norm_iri]

        alternative = manager.build_alternative(
            _index(manager, obj_type).get_row(id),
            {variant} if variant else None,
        )

        return Item(
            alternative=alternative,
            obj_type=obj_type,
            variant=variant,
            **infos,
        )

    # we know that it is an IRI of another known prefix,
    # e.g. rdfs:label or schema:about, or a unknown entity or property
    # of a known prefix, e.g. wd:Q123456789
    invalid = False
    if indexed_prefixes is not None:
        # check whether iri has an indexed prefix (was expected to be indexed)
        invalid = find_longest_prefix(iri, indexed_prefixes) is None

    return Item(
        alternative=Alternative(
            identifier=iri,
            short_identifier=manager.format_iri(iri),
        ),
        obj_type=ObjType.OTHER,
        variant=None,
        invalid=invalid,
        **infos,
    )


def selections_from_items(item: list[Item]) -> list[Selection]:
    return [item.selection for item in item]


def natural_sparql_from_items(
    items: list[Item],
    is_prefix: bool = False,
    full_identifier: bool = False,
) -> str:
    prefix = ""
    for i, item in enumerate(items):
        prev = items[i - 1] if i > 0 else None
        prefix += item.continuation(prev)
        prefix += item.selection.get_natural_sparql_label(full_identifier)
        if i == len(items) - 1 and not is_prefix:
            prefix += item.suffix
    return prefix


def _get_indexed_prefixes(
    index: SearchIndex,
    prefixes: dict[str, str],
) -> dict[str, str]:
    indexed = {}
    for i in range(len(index)):
        iri = index.get_val(i, 3)
        pfx = find_longest_prefix(iri, prefixes)
        if pfx is None:
            continue

        short, long = pfx
        indexed[short] = long

    return indexed


def get_indexed_prefixes(manager: KgManager) -> dict[str, str]:
    entity_indexed = _get_indexed_prefixes(
        manager.entity_index,
        manager.prefixes,
    )
    property_indexed = _get_indexed_prefixes(
        manager.property_index,
        manager.prefixes,
    )
    return {**entity_indexed, **property_indexed}


def get_sparql_items(
    sparql: str,
    manager: KgManager,
    normalized: bool = False,
    is_prefix: bool = False,
    indexed_prefixes: dict[str, str] | None = None,
) -> tuple[str, list[Item]]:
    sparql = manager.fix_prefixes(
        sparql,
        is_prefix=is_prefix,
        remove_known=True,
    )

    if normalized:
        sparql = manager.normalize_sparql(sparql, is_prefix=is_prefix)

    sparql_encoded = sparql.encode()
    parse, _ = parse_string(
        sparql,
        manager.sparql_parser,
        collapse_single=False,
        skip_empty=True,
        is_prefix=is_prefix,
    )

    # get all items in triples
    items = filter(
        lambda item: item is not None,
        chain(
            (
                # get IRIs (excluding prefixes)
                _get_item(iri, manager, sparql_encoded)
                for iri in find_all(parse, name="iri", skip={"Prologue"})
            ),
            (
                # only literals in triples are searchable in addition to IRIs
                # rest should be predicted directly
                _get_item(lit, manager, sparql_encoded)
                for triple in find_all(parse, name="TriplesSameSubject")
                for lit in find_all(
                    triple,
                    name={"RDFLiteral", "NumericLiteral", "BooleanLiteral"},
                )
            ),
        ),
    )

    # by occurence position in the query
    return sparql, sorted(items, key=lambda item: item.item_span)


def drop_sparql_items(
    items: list[Item],
    p: float,
    other_or_literal_only: bool = False,
) -> list[Item]:
    # drop items that can be dropped with the given probability
    # dropped items are either:
    # - literals
    # - iris that are not entities or properties, e.g. rdfs:label
    # - entities or properties that occurr earlier in the query
    #   and can therefore be predicted directly

    def matches(item: Item, other: Item) -> bool:
        # if any of the two items is invalid, they should not match
        if item.invalid or other.invalid:
            return False

        # only check for identifiers here, not the variant because
        # we want to allow to directly predict other variants of
        # already seen items, e.g. if there is wdt:P31 earlier in the query
        # allow to predict p:P31 as well
        return item.alternative.identifier == other.alternative.identifier

    drop_mask = []
    for item in items:
        in_prefix = not other_or_literal_only and any(
            not dropped and matches(item, other)
            for dropped, other in zip(drop_mask, items[: len(drop_mask)])
        )

        # only drop valid items, e.g. rdfs:label
        # if they are not valid, it means they should be known
        # but are not covered by the index
        droppable = not item.invalid and (item.is_other_or_literal or in_prefix)

        drop = droppable and random.random() < p

        drop_mask.append(drop)

    return [item for item, drop in zip(items, drop_mask) if not drop]
