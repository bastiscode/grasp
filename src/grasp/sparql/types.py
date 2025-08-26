import re
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from typing import Any, Iterator

from grasp.utils import clip


class ObjType(str, Enum):
    ENTITY = "entity"
    PROPERTY = "property"
    OTHER = "other"
    LITERAL = "literal"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


def obj_types_before(obj_type: ObjType) -> list[ObjType]:
    values = list(ObjType)
    return values[: values.index(obj_type)]


class Position(str, Enum):
    SUBJECT = "subject"
    PROPERTY = "property"
    OBJECT = "object"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


@dataclass
class AskResult:
    boolean: bool

    def __len__(self) -> int:
        return 1

    @property
    def is_empty(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AskResult):
            return False

        return self.boolean == other.boolean


@dataclass
class Binding:
    typ: str
    value: str
    datatype: str | None = None
    lang: str | None = None

    def __hash__(self) -> int:
        return hash((self.typ, self.value, self.datatype, self.lang))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Binding):
            return False

        return (
            self.typ == other.typ
            and self.value == other.value
            and self.datatype == other.datatype
            and self.lang == other.lang
        )

    @staticmethod
    def from_dict(data: dict) -> "Binding":
        return Binding(
            typ=data["type"],
            value=data["value"],
            datatype=data.get("datatype"),
            lang=data.get("xml:lang"),
        )

    def identifier(self) -> str:
        assert self.typ in ["uri", "literal", "bnode"]
        match self.typ:
            case "uri":
                return f"<{self.value}>"
            case "literal":
                if self.datatype is not None:
                    return f'"{self.value}"^^<{self.datatype}>'
                elif self.lang is not None:
                    return f'"{self.value}"@{self.lang}'
                else:
                    return f'"{self.value}"'
            case "bnode":
                return f"_:{self.value}"
            case _:
                raise ValueError(f"Unknown binding type: {self.typ}")


SelectRow = dict[str, Binding]


@dataclass
class SelectResult:
    variables: list[str]
    data: list[dict | None]

    @staticmethod
    def from_json(data: dict) -> "SelectResult":
        return SelectResult(
            variables=data["head"]["vars"],
            data=data["results"]["bindings"],
        )

    def __len__(self) -> int:
        return len(self.data)

    def bindings(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[tuple[Binding, ...]]:
        for row in self.rows(start, end):
            bindings = tuple(row[var] for var in self.variables if var in row)
            yield bindings

    def rows(self, start: int = 0, end: int | None = None) -> Iterator[SelectRow]:
        start = max(start, 0)

        if end is None:
            end = len(self.data)
        else:
            end = min(end, len(self.data))

        for i in range(start, end):
            data = self.data[i]
            if data is None:
                yield {}
            else:
                yield {
                    var: Binding.from_dict(data[var])
                    for var in self.variables
                    if var in data
                }

    @property
    def num_rows(self) -> int:
        return len(self.data)

    @property
    def num_columns(self) -> int:
        return len(self.variables)

    @property
    def is_empty(self) -> bool:
        return not self.data

    def to_ask_result(self) -> AskResult:
        return AskResult(not self.is_empty)


@dataclass
class Example:
    question: str
    sparql: str


class Alternative:
    def __init__(
        self,
        identifier: str,
        short_identifier: str | None = None,
        label: str | None = None,
        variants: set[str] | None = None,
        aliases: list[str] | None = None,
        infos: list[str] | None = None,
    ) -> None:
        self.identifier = identifier
        self.short_identifier = short_identifier
        self.label = label
        self.aliases = aliases
        self.variants = variants
        self.infos = infos

    def __hash__(self) -> int:
        # hash identifier
        return hash(self.identifier)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Alternative):
            return False

        return self.identifier == other.identifier

    def __repr__(self) -> str:
        return f"Alternative({self.label}, {self.get_identifier()}, {self.variants})"

    def get_identifier(self) -> str:
        return self.short_identifier or self.identifier

    def has_label(self) -> bool:
        return bool(self.label)

    def get_label(self) -> str | None:
        return clip(self.label) if self.label else None

    def has_variants(self) -> bool:
        return bool(self.variants)

    def get_selection_string(
        self,
        max_aliases: int = 5,
        add_infos: bool = True,
        include_variants: set[str] | None = None,
    ) -> str:
        s = self.get_label() or self.get_identifier()

        if add_infos and max_aliases and self.aliases:
            s += ", also known as " + ", ".join(
                clip(a) for a in self.aliases[:max_aliases]
            )

        variants = self.variants if include_variants is None else include_variants
        if self.has_label() and not variants:
            s += f" ({self.get_identifier()})"
        elif not self.has_label() and variants:
            s += f" (as {'/'.join(variants)})"
        elif self.has_label() and variants:
            s += f" ({self.get_identifier()} as {'/'.join(variants)})"

        if add_infos and self.infos:
            s += ": " + " / ".join(clip(info) for info in self.infos)

        return s

    def get_selection_target(self, variant: str | None = None) -> str:
        s = self.get_label() or self.get_identifier()
        if variant:
            s += f" ({variant})"
        return s

    def get_selection_regex(self) -> str:
        # matches format of selection label above
        r = re.escape(self.get_label() or self.get_identifier())
        if self.variants:
            r += (
                re.escape(" (")
                + "(?:"
                + "|".join(map(re.escape, self.variants))
                + ")"
                + re.escape(")")
            )

        return r


class Selection:
    alternative: Alternative
    obj_type: ObjType
    variant: str | None

    def __init__(
        self,
        alternative: Alternative,
        obj_type: ObjType,
        variant: str | None = None,
    ) -> None:
        self.alternative = alternative
        self.obj_type = obj_type
        if variant:
            assert alternative.has_variants() and variant in alternative.variants, (  # type: ignore
                f"Variant {variant} not in {alternative.variants}"
            )
        self.variant = variant

    def __repr__(self) -> str:
        return f"Selection({self.alternative}, {self.obj_type}, {self.variant})"

    def __hash__(self) -> int:
        return hash((self.alternative, self.obj_type, self.variant))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Selection):
            return False

        return (
            self.alternative == other.alternative
            and self.obj_type == other.obj_type
            and self.variant == other.variant
        )

    @property
    def is_entity_or_property(self) -> bool:
        return self.obj_type == ObjType.ENTITY or self.obj_type == ObjType.PROPERTY

    def get_natural_sparql_label(self, full_identifier: bool = False) -> str:
        identifier = self.alternative.get_identifier()
        if not self.alternative.has_label():
            return identifier

        label: str = self.alternative.get_label()  # type: ignore

        if full_identifier:
            label += f" ({identifier})"
        elif self.variant:
            label += f" ({self.variant})"

        if self.is_entity_or_property:
            return f"<{label}>"
        else:
            return label


def group_selections(
    selections: list[Selection],
) -> dict[ObjType, list[tuple[Alternative, set[str]]]]:
    def _key(sel: Selection) -> tuple[str, str]:
        return sel.alternative.identifier, sel.obj_type.name

    grouped = {}
    for _, group in groupby(sorted(selections, key=_key), key=_key):
        selections = list(group)
        obj_type = selections[0].obj_type
        if obj_type not in grouped:
            grouped[obj_type] = []

        variants = {selection.variant for selection in selections if selection.variant}
        alt = selections[0].alternative
        grouped[obj_type].append((alt, variants))

    return grouped
