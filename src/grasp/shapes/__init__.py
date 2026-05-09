import json
import os
import time
from dataclasses import dataclass

from safetensors.numpy import save_file
from search_rdf import Data, EmbeddingIndex
from search_rdf.model import SentenceTransformerModel
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import flatten


class ShapeSample:
    def __init__(self, iri: str, short_iri: str, shex: str, **_: object) -> None:
        self.iri = iri
        self.short_iri = short_iri
        self.shex = shex

    def queries(self) -> list[str]:
        return [self.shex, self.short_iri]

    def model_dump(self) -> dict:
        return {
            "iri": self.iri,
            "short_iri": self.short_iri,
            "shex": self.shex,
        }


class ShapeIndex:
    def __init__(
        self,
        data: Data,
        index: EmbeddingIndex,
        model: SentenceTransformerModel,
        samples: list[ShapeSample],
    ) -> None:
        self.data = data
        self.index = index
        self.model = model
        self.samples = samples
        self._iri_map: dict[str, ShapeSample] = {s.iri: s for s in samples}

    def search(self, query: str, k: int = 10) -> list[ShapeSample]:
        embedding = self.model.embed([query])[0]
        matches = self.index.search(embedding, k)
        return [self.samples[id] for id, *_ in matches]

    def get_by_iri(self, iri: str) -> "ShapeSample | None":
        return self._iri_map.get(iri)

    @classmethod
    def load(cls, dir: str, model: SentenceTransformerModel) -> "ShapeIndex":
        data = Data.load(os.path.join(dir, "data"))
        embedding_path = os.path.join(dir, "data", "embedding.safetensors")
        index_dir = os.path.join(dir, "index")

        index = EmbeddingIndex.load(data, embedding_path, index_dir)
        assert index.model == model.model, (
            f"Embedding model mismatch: index model {index.model}, "
            f"provided model {model.model}"
        )

        samples = [
            ShapeSample(**s) for s in load_jsonl(os.path.join(dir, "samples.jsonl"))
        ]

        return cls(data, index, model, samples)

    @classmethod
    def build(
        cls,
        samples: list[ShapeSample],
        output_dir: str,
        model: SentenceTransformerModel,
        batch_size: int = 256,
        overwrite: bool = False,
        log_level: str | int | None = None,
    ) -> None:
        logger = get_logger("SHAPE INDEX BUILD", log_level)

        if os.path.exists(output_dir) and not overwrite:
            logger.info(f"Index directory {output_dir} already exists, skipping build")
            return

        start = time.monotonic()
        logger.info(
            f"Building shape index at {output_dir} from {len(samples):,} samples"
        )

        data_dir = os.path.join(output_dir, "data")
        index_dir = os.path.join(output_dir, "index")

        samples_file = os.path.join(output_dir, "samples.jsonl")
        dump_jsonl((s.model_dump() for s in samples), samples_file)

        items = []
        for i, sample in enumerate(samples):
            identifier = f"sample-{i}"
            fields = [{"type": "text", "value": q} for q in sample.queries()]
            items.append({"identifier": identifier, "fields": fields})

        Data.build_from_items(items, data_dir)
        data = Data.load(data_dir)

        texts = list(flatten(fields for _, fields in data))
        embedding = model.embed(texts, batch_size=batch_size, show_progress=True)

        embedding_path = os.path.join(data_dir, "embedding.safetensors")
        save_file(
            {"embedding": embedding},
            filename=embedding_path,
            metadata={"model": model.model},
        )

        EmbeddingIndex.build(data, embedding_path, index_dir)

        with open(os.path.join(output_dir, "info.json"), "w") as f:
            json.dump({}, f)

        end = time.monotonic()
        logger.info(f"Shape index built in {end - start:.2f} seconds")


@dataclass
class Shapes:
    pattern: str | None = None
    index: ShapeIndex | None = None


def load_shapes(shapes_dir: str, model: SentenceTransformerModel) -> "Shapes | None":
    pattern = None
    pattern_file = os.path.join(shapes_dir, "pattern.sparql")
    if os.path.exists(pattern_file):
        with open(pattern_file) as f:
            pattern = f.read().strip() or None

    index = None
    index_dir = os.path.join(shapes_dir, "index")
    if os.path.exists(index_dir):
        index = ShapeIndex.load(index_dir, model)

    if pattern is None and index is None:
        return None
    return Shapes(pattern=pattern, index=index)
