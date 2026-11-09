from grasp.functions import update_known_from_selections
from grasp.manager.normalizer import Normalizer
from grasp.sparql.types import Alternative, ObjType, Selection


class VariantNormalizer(Normalizer):
    def denormalize(self, iri: str, variant: str | None) -> str | None:
        return f"{iri}/{variant}"


class Manager:
    def get_normalizer(self, name: str) -> Normalizer:
        return VariantNormalizer() if name == "classes" else Normalizer()


def test_known_from_selections():
    selections = [
        Selection(Alternative("ex:e"), ObjType.ENTITY),
        Selection(Alternative("ex:c", variants=["v"]), ObjType.CLASS),
        Selection(Alternative("ex:p"), ObjType.PROPERTY),
    ]
    known = set()
    update_known_from_selections(known, selections, Manager())  # type: ignore
    assert known == {"ex:e", "ex:c", "ex:c/v", "ex:p"}
