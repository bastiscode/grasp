import os
import time
from typing import Any

from search_index import IndexData, SimilarityIndex
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.utils import Sample


class ExampleIndex:
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
    ) -> list[Sample]:
        """
        Find the top-k matching samples for a given question.
        """
        matches = self.index.find_matches(question, k, **kwargs)
        return [self.samples[id] for id, _ in matches]

    @staticmethod
    def load(dir: str, load_kwargs: dict[str, Any] | None = None) -> "ExampleIndex":
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
            Sample(**sample)
            for sample in load_jsonl(os.path.join(dir, "samples.jsonl"))
        ]
        return ExampleIndex(data, index, samples)

    @staticmethod
    def build(
        examples_file: str,
        output_dir: str,
        batch_size: int = 256,
        overwrite: bool = False,
        log_level: str | int | None = None,
    ) -> None:
        logger = get_logger("EXAMPLE INDEX BUILD", log_level)

        samples = [Sample(**sample) for sample in load_jsonl(examples_file)]

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
            of.write("id\tquestions\n")
            for i, sample in enumerate(samples):
                questions = [sample.question] + sample.paraphrases
                if not questions:
                    continue
                of.write(f"{i}\t" + "\t".join(questions) + "\n")

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
