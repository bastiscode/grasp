from search_index import IndexData
from search_index import Mapping as SearchIndexMapping


class Mapping:
    def __init__(self) -> None:
        self.map: SearchIndexMapping | None = None

    @classmethod
    def load(cls, data: IndexData, mapping_file: str) -> "Mapping":
        mapping = cls()
        mapping.map = SearchIndexMapping.load(data, mapping_file)
        return mapping

    def __getitem__(self, iri: str) -> int:
        assert self.map is not None, "mapping not loaded"
        item = self.map.get(iri)
        assert item is not None, f"{iri} not in mapping"
        return item

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        return iri, None

    def denormalize(self, iri: str, variant: str | None) -> str | None:
        return iri

    def default_variants(self) -> set[str] | None:
        return None

    def __contains__(self, iri: str) -> bool:
        assert self.map is not None, "mapping not loaded"
        return self.map.get(iri) is not None
