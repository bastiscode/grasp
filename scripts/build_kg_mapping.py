import argparse
import os

from search_index import IndexData, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing file",
    )
    return parser.parse_args()


def build(args: argparse.Namespace):
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Mapping already exists at {args.output}")
        return

    # 3 is expected column index for unique line identifier
    Mapping.build(IndexData(args.input), args.output, 3)


if __name__ == "__main__":
    build(parse_args())
