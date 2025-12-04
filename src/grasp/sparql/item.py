import random
import sys
from dataclasses import dataclass
from itertools import chain
from typing import Optional

from grasp.manager import KgManager
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.sparql.types import Alternative, ObjType, Position, Selection
from grasp.sparql.utils import (
    autocomplete_prefix,
    find_all,
    find_longest_prefix,
    parse_into_binding,
    parse_string,
)

__all__ = [
    "Item",
    "get_sparql_items",
    "natural_sparql_from_items",
    "selections_from_items",
]


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


def _byte_span(parse: dict, start: int = sys.maxsize, end: int = 0) -> tuple[int, int]:
    if "children" in parse:
        for child in parse["children"]:
            start, end = _byte_span(child, start, end)
        return start, end

    f, t = parse["byte_span"]
    return min(start, f), max(end, t)


COMMON_PREFIXES = get_common_sparql_prefixes()


def _get_item(
    parse: dict,
    manager: KgManager,
    sparql_encoded: bytes,
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

    binding = parse_into_binding(item, manager.iri_literal_parser, manager.prefixes)
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

    try:
        *_, position = autocomplete_prefix(prefix, manager.sparql_parser)
        if position in [Position.SUBJECT, Position.OBJECT]:
            obj_types = [ObjType.ENTITY]
        else:
            obj_types = [ObjType.PROPERTY]
    except Exception:
        obj_types = [ObjType.PROPERTY, ObjType.ENTITY]

    # check whether the iri is a valid entity or property
    for obj_type in obj_types:
        norm = manager.normalize(iri, obj_type)
        if norm is None:
            continue

        norm_iri, variant = norm
        row = manager.get_row(norm_iri, obj_type)
        if row is None:
            continue

        identifier, label, *synonyms = row

        alternative = manager.build_alternative(
            identifier,
            label,
            synonyms,
            variants=[variant] if variant else None,
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
    # if iri has a common prefix, it is not expected to be indexed
    invalid = find_longest_prefix(iri, COMMON_PREFIXES) is None

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


def get_sparql_items(
    sparql: str,
    manager: KgManager,
    is_prefix: bool = False,
) -> tuple[str, list[Item]]:
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
    return sparql, sorted(items, key=lambda item: item.item_span)  # type: ignore


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
