from search_index import IndexData
from search_index import Mapping as SearchIndexMapping

from grasp.sparql.utils import find_longest_prefix

WIKIDATA_PROPERTY_VARIANTS = {
    # ordered by most frequently used, dict keeps insertion order
    "wdt": "<http://www.wikidata.org/prop/direct/",
    "p": "<http://www.wikidata.org/prop/",
    "ps": "<http://www.wikidata.org/prop/statement/",
    "pq": "<http://www.wikidata.org/prop/qualifier/",
    "pr": "<http://www.wikidata.org/prop/reference/",
    "wdtn": "<http://www.wikidata.org/prop/direct-normalized/",
    "psn": "<http://www.wikidata.org/prop/statement/value-normalized/",
    "pqn": "<http://www.wikidata.org/prop/qualifier/value-normalized/",
    "prn": "<http://www.wikidata.org/prop/reference/value-normalized/",
    "psv": "<http://www.wikidata.org/prop/statement/value/",
    "pqv": "<http://www.wikidata.org/prop/qualifier/value/",
    "prv": "<http://www.wikidata.org/prop/reference/value/",
}


class Mapping:
    def __init__(self) -> None:
        self.map: SearchIndexMapping | None = None

    @classmethod
    def load(cls, data: IndexData, mapping_file: str) -> "Mapping":
        mapping = cls()
        mapping.map = SearchIndexMapping.load(data, mapping_file)  # type: ignore
        return mapping

    def get(self, iri: str) -> int | None:
        assert self.map is not None, "mapping not loaded"
        return self.map.get(iri)

    def __getitem__(self, iri: str) -> int:
        item = self.get(iri)
        assert item is not None, f"{iri} not in mapping"
        return item

    def __len__(self) -> int:
        assert self.map is not None, "mapping not loaded"
        return len(self.map)  # type: ignore

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        return iri, None

    def denormalize(self, iri: str, variant: str | None) -> str | None:
        return iri

    def default_variants(self) -> list[str] | None:
        return None

    def __contains__(self, iri: str) -> bool:
        return self.get(iri) is not None


class WikidataPropertyMapping(Mapping):
    NORM_PREFIX = "<http://www.wikidata.org/entity/"

    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        longest = find_longest_prefix(iri, WIKIDATA_PROPERTY_VARIANTS)
        if longest is None:
            return None

        short, long = longest
        iri = self.NORM_PREFIX + iri[len(long) :]
        return iri, short

    def denormalize(self, iri: str, variant: str | None) -> str | None:
        if variant is None:
            return iri
        elif variant not in WIKIDATA_PROPERTY_VARIANTS:
            return None
        elif not iri.startswith(self.NORM_PREFIX):
            return None
        pfx = WIKIDATA_PROPERTY_VARIANTS[variant]
        return pfx + iri[len(self.NORM_PREFIX) :]

    def default_variants(self) -> list[str] | None:
        return list(WIKIDATA_PROPERTY_VARIANTS.keys())
