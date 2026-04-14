import argparse

from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.manager import (
    get_common_sparql_prefixes,
    load_iri_and_literal_parser,
    load_kg_info,
    load_sparql_parser,
)
from grasp.manager.utils import merge_prefixes
from grasp.sparql.utils import fix_prefixes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="Path to the JSONL file, whose 'sparql' field will be fixed in-place.",
    )
    return parser.parse_args()


def fix(args: argparse.Namespace) -> None:
    logger = get_logger("FIX PREFIXES")
    data = load_jsonl(args.file)
    prefixes = get_common_sparql_prefixes()
    kg_prefixes, _ = load_kg_info("wikidata")
    prefixes, *_ = merge_prefixes(prefixes, kg_prefixes, logger)

    sparql_parser = load_sparql_parser()
    iri_parser = load_iri_and_literal_parser()

    changed = 0
    for item in data:
        try:
            sparql = fix_prefixes(item["sparql"], sparql_parser, iri_parser, prefixes)
            changed += int(sparql != item["sparql"])
            item["sparql"] = sparql
        except Exception:
            pass

    logger.info(f"Fixed prefixes in {changed:,} SPARQL queries.")
    dump_jsonl(data, args.file)


if __name__ == "__main__":
    fix(parse_args())
