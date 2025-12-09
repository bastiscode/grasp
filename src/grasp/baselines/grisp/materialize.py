import argparse
import os
import random

from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl
from universal_ml_utils.logging import setup_logging

from grasp.baselines.grisp.data import (
    GRISPMaterializedSample,
    GRISPSample,
    load_samples,
    prepare_selection,
    prepare_skeleton,
)
from grasp.configs import KgConfig
from grasp.manager import load_kg_manager
from grasp.utils import get_available_knowledge_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize GRISP data for a given number of materializations per sample"
    )
    parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=get_available_knowledge_graphs(),
        help="Knowledge graph to use",
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input file containing data samples"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file to save materialized data",
    )
    parser.add_argument(
        "num_materializations",
        type=int,
        help="Number of materializations per sample",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Custom SPARQL endpoint for the knowledge graph",
    )
    parser.add_argument(
        "--skeleton-p",
        type=float,
        default=0.2,
        help="Augmentation probability for skeletons",
    )
    parser.add_argument(
        "--selection-p",
        type=float,
        default=0.2,
        help="Augmentation probability for selections",
    )
    parser.add_argument(
        "--is-val",
        action="store_true",
        help="Whether the input file is a validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # to show info from kg manager
    setup_logging("INFO")

    if os.path.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output_file} already exists. "
            "Use --overwrite to overwrite."
        )

    samples = load_samples([args.input_file])

    kg_manager = load_kg_manager(
        KgConfig(kg=args.knowledge_graph, endpoint=args.endpoint)
    )

    random.seed(args.seed)
    n = args.num_materializations if not args.is_val else 1

    materialized = []
    for sample in tqdm(samples, desc="Materializing samples"):
        assert isinstance(sample, GRISPSample), "Expected non-materialized GRISP sample"

        skeletons = [
            prepare_skeleton(
                sample,
                args.is_val,
                args.skeleton_p,
            )
            for _ in range(n)
        ]

        if sample.has_placeholders:
            selections = [
                prepare_selection(
                    sample,
                    kg_manager,
                    args.is_val,
                    args.skeleton_p,
                    args.selection_p,
                )
                for _ in range(n)
            ]
        else:
            selections = []

        materialized_sample = GRISPMaterializedSample(
            skeletons=skeletons,
            selections=selections,
        )
        materialized.append(materialized_sample.model_dump())

    dump_jsonl(materialized, args.output_file)


if __name__ == "__main__":
    main(parse_args())
