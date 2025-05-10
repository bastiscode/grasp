from dataclasses import dataclass
from enum import Enum
from typing import Iterator

QLEVER_API = "https://qlever.cs.uni-freiburg.de/api"


def get_endpoint(kg: str) -> str:
    return f"{QLEVER_API}/{kg}"


# default request timeout
# 6 seconds for establishing a connection, 30 seconds for processing query
# and beginning to receive the response
REQUEST_TIMEOUT = (6, 30)

# default read timeout
# 60 seconds for everything (including receiving the response)
READ_TIMEOUT = 60


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


SelectRow = list[Binding | None]


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

    def rows(self, start: int = 0, end: int | None = None) -> Iterator[SelectRow]:
        start = max(start, 0)

        if end is None:
            end = len(self.data)
        else:
            end = min(end, len(self.data))

        for i in range(start, end):
            data = self.data[i]
            if data is None:
                row = [None] * self.num_columns
            else:
                row = [
                    Binding.from_dict(data[var]) if var in data else None
                    for var in self.variables
                ]
            yield row

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
