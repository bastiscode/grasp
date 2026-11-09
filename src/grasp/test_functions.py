from grasp.functions import (
    row_iris,
    update_known_from_index_iris,
    update_known_from_selections,
)
from grasp.manager.normalizer import Normalizer
from grasp.sparql.types import Alternative, Binding, ObjType, Selection


class VariantNormalizer(Normalizer):
    def denormalize(self, iri: str, variant: str | None) -> str | None:
        return f"{iri}/{variant}"


class PropertyNormalizer(Normalizer):
    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        if not iri.startswith("ex:direct/"):
            return None
        return "ex:" + iri.removeprefix("ex:direct/"), "direct"


class Manager:
    def get_normalizer(self, name: str) -> Normalizer:
        if name == "classes":
            return VariantNormalizer()
        elif name == "properties":
            return PropertyNormalizer()
        return Normalizer()


def test_known_from_selections():
    selections = [
        Selection(Alternative("ex:e"), ObjType.ENTITY),
        Selection(Alternative("ex:c", variants=["v"]), ObjType.CLASS),
        Selection(Alternative("ex:p"), ObjType.PROPERTY),
    ]
    known = set()
    update_known_from_selections(known, selections, Manager())  # type: ignore
    assert known == {"ex:e", "ex:c", "ex:c/v", "ex:p"}


def test_known_from_rows_uses_all_indices():
    rows = (
        {"p": Binding("uri", "ex:direct/p"), "v": Binding("literal", "x")}
        for _ in range(2)
    )
    known = set()
    update_known_from_index_iris(known, row_iris(rows), Manager())  # type: ignore
    assert known == {"ex:direct/p", "ex:p"}
