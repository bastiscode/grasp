import os
import time
from typing import Any, Type

from search_index import IndexData, SimilarityIndex
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.configs import GraspConfig
from grasp.tasks.utils import Sample


class ExampleIndex:
    sample_cls: Type[Sample]

    def __init__(
        self,
        data: IndexData,
        index: SimilarityIndex,
        samples: list[Sample],
    ) -> None:
        self.data = data
        self.index = index
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def find_matches(
        self,
        question: str,
        k: int = 3,
        **kwargs: Any,
    ) -> list:
        """
        Find the top-k matching samples for a given question.
        """
        matches = self.index.find_matches(question, k, **kwargs)
        return [self.samples[id] for id, _ in matches]

    @classmethod
    def load(
        cls,
        dir: str,
        load_kwargs: dict[str, Any] | None = None,
    ) -> "ExampleIndex":
        data = IndexData.load(
            os.path.join(dir, "data.tsv"),
            os.path.join(dir, "offsets.bin"),
        )

        if load_kwargs is None:
            load_kwargs = {}
        index = SimilarityIndex.load(
            data,
            os.path.join(dir, "index"),
            **load_kwargs,
        )

        samples = [
            cls.sample_cls(**sample)
            for sample in load_jsonl(os.path.join(dir, "samples.jsonl"))
        ]
        return ExampleIndex(data, index, samples)

    @classmethod
    def build(
        cls,
        examples_file: str,
        output_dir: str,
        batch_size: int = 256,
        overwrite: bool = False,
        log_level: str | int | None = None,
    ) -> None:
        logger = get_logger("EXAMPLE INDEX BUILD", log_level)

        samples = [cls.sample_cls(**sample) for sample in load_jsonl(examples_file)]

        if os.path.exists(output_dir) and not overwrite:
            logger.info(f"Index directory {output_dir} already exists, skipping build")
            return

        start = time.perf_counter()
        logger.info(
            f"Building example index at {output_dir} from {len(samples):,} samples"
        )
        data_file = os.path.join(output_dir, "data.tsv")
        offsets_file = os.path.join(output_dir, "offsets.bin")
        index_dir = os.path.join(output_dir, "index")
        os.makedirs(index_dir, exist_ok=True)

        # save samples in index directory
        samples_file = os.path.join(output_dir, "samples.jsonl")
        dump_jsonl((sample.model_dump() for sample in samples), samples_file)

        with open(data_file, "w") as of:
            # write header
            of.write("id\tinputs\n")
            for i, sample in enumerate(samples):
                queries = sample.queries()
                if not queries:
                    continue
                of.write(f"{i}\t" + "\t".join(queries) + "\n")

        IndexData.build(data_file, offsets_file)
        data = IndexData.load(data_file, offsets_file)

        SimilarityIndex.build(
            data,
            index_dir,
            batch_size=batch_size,
            show_progress=True,
        )
        end = time.perf_counter()
        logger.info(f"Example index built in {end - start:.2f} seconds")


def task_to_index(task: str) -> Type[ExampleIndex]:
    if task == "sparql-qa":
        from grasp.tasks.sparql_qa.examples import SparqlQaExampleIndex

        return SparqlQaExampleIndex

    else:
        raise ValueError(f"Unknown task {task}")


def load_example_indices(
    task: str,
    config: GraspConfig,
    **kwargs: Any,
) -> dict[str, ExampleIndex]:
    try:
        index_cls = task_to_index(task)
    except ValueError:
        # unsupported task
        return {}

    indices = {}
    for kg in config.knowledge_graphs:
        if kg.example_index is None:
            continue

        indices[kg] = index_cls.load(kg.example_index, **kwargs)
    return indices
