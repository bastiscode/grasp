import os
import time
from logging import Logger

from safetensors.numpy import save_file
from search_rdf import Data, EmbeddingIndex, FuzzyIndex, KeywordIndex
from search_rdf.model import SentenceTransformerModel
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import flatten

from grasp.manager.utils import load_data
from grasp.utils import get_index_dir


def build_index(
    index_dir: str,
    index_type: str,
    logger: Logger,
    overwrite: bool = False,
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int = 256,
    embedding_dim: int | None = None,
) -> None:
    data = load_data(index_dir)

    index_dir = os.path.join(index_dir, index_type)
    if os.path.exists(index_dir) and not overwrite:
        logger.info(
            f"Index of type {index_type} already exists at {index_dir}. Skipping build."
        )
        return

    os.makedirs(index_dir, exist_ok=True)
    start = time.perf_counter()
    logger.info(f"Building {index_type} index at {index_dir} from {len(data):,} items")

    if index_type == "keyword":
        KeywordIndex.build(data, index_dir)

    elif index_type == "fuzzy":
        FuzzyIndex.build(data, index_dir)

    elif index_type == "embedding":
        assert embedding_model is not None, (
            "Embedding model must be specified for embedding index"
        )
        embedding_path = os.path.join(index_dir, "embedding.safetensors")

        generate_embeddings(
            data,
            embedding_path,
            model_name=embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
            dim=embedding_dim,
        )

        EmbeddingIndex.build(data, embedding_path, index_dir)

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    end = time.perf_counter()
    logger.info(f"Index build took {end - start:.2f} seconds")


def generate_embeddings(
    data: Data,
    embedding_path: str,
    model_name: str,
    device: str | None = None,
    batch_size: int = 256,
    dim: int | None = None,
) -> None:
    model = SentenceTransformerModel(model_name, device)

    texts = list(flatten(fields for _, fields in data))
    embedding = model.embed(texts, dim, batch_size, show_progress=True)

    save_file(
        {"embedding": embedding},
        filename=embedding_path,
        metadata={"model": model_name},
    )


def build_indices(
    kg: str,
    entities_type: str,
    properties_type: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int = 256,
    embedding_dim: int | None = None,
) -> None:
    logger = get_logger("GRASP INDEX", log_level)

    index_dir = get_index_dir(kg)

    # entities
    entities_dir = os.path.join(index_dir, "entities")
    build_index(
        entities_dir,
        entities_type,
        logger,
        overwrite,
        embedding_model,
        embedding_device,
        embedding_batch_size,
        embedding_dim,
    )

    # properties
    properties_dir = os.path.join(index_dir, "properties")
    build_index(
        properties_dir,
        properties_type,
        logger,
        overwrite,
        embedding_model,
        embedding_device,
        embedding_batch_size,
        embedding_dim,
    )
