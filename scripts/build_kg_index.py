import argparse
import os

from search_index import IndexData, PrefixIndex, SimilarityIndex
from universal_ml_utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input dir")
    parser.add_argument("output", type=str, help="Output dir")
    parser.add_argument(
        "--type",
        type=str,
        default="prefix",
        choices=["prefix", "similarity"],
    )
    parser.add_argument(
        "--no-syns",
        action="store_true",
        help="Whether to remove synonyms",
    )
    parser.add_argument(
        "--sim-precision",
        type=str,
        choices=[None, "float32", "ubinary"],
        default=None,
    )
    parser.add_argument("--sim-batch-size", type=int, default=256)
    parser.add_argument(
        "--sim-use-columns",
        type=int,
        nargs="+",
        default=None,
        help="Columns to use for similarity index",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing index",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for the build process",
    )
    return parser.parse_args()


def build(args: argparse.Namespace):
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Index already exists at {args.output}")
        return

    setup_logging(args.log_level)

    print(f"Building {args.type} index at {args.output}")
    os.makedirs(args.output, exist_ok=True)

    data = IndexData.load(
        os.path.join(args.input, "data.tsv"),
        os.path.join(args.input, "offsets.bin"),
    )

    if args.type == "prefix":
        PrefixIndex.build(
            data,
            args.output,
            use_synonyms=not args.no_syns,
        )
    else:
        SimilarityIndex.build(
            data,
            args.output,
            precision=args.sim_precision,
            batch_size=args.sim_batch_size,
            use_synonyms=not args.no_syns,
            use_columns=args.sim_use_columns,
            show_progress=True,
        )


if __name__ == "__main__":
    build(parse_args())
