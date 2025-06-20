import argparse
import os

from search_index import IndexData, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help="Input data file")
    parser.add_argument("offsets_file", type=str, help="Output offsets file")
    parser.add_argument("mapping_file", type=str, help="Output mapping file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files",
    )
    return parser.parse_args()


def build(args: argparse.Namespace):
    if not os.path.exists(args.offsets_file) or args.overwrite:
        # build index data
        IndexData.build(args.data_file, args.offsets_file)

    data = IndexData.load(args.data_file, args.offsets_file)

    if not os.path.exists(args.mapping_file) or args.overwrite:
        # build mapping
        Mapping.build(data, args.mapping_file, 3)


if __name__ == "__main__":
    build(parse_args())
