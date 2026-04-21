import argparse

from tqdm import tqdm
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
        help="Path to the JSONL file, whose 'sparql' field will be fixed in-place",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def fix(args: argparse.Namespace) -> None:
    logger = get_logger("FIX PREFIXES", "INFO")
    data = load_jsonl(args.file)

    info = load_kg_info("wikidata")
    prefixes, *_ = merge_prefixes(
        get_common_sparql_prefixes(),
        info.prefixes or {},
        logger,
    )

    sparql_parser = load_sparql_parser()
    iri_parser = load_iri_and_literal_parser()

    changed = 0
    failed = 0
    for item in tqdm(data, desc="Fixing prefixes in SPARQL queries"):
        try:
            sparql = fix_prefixes(
                item["sparql"],
                sparql_parser,
                iri_parser,
                prefixes,
                sort=True,
            ).strip()
            changed += int(sparql != item["sparql"].strip())
            if changed and args.verbose:
                logger.info(f"Original SPARQL:\n{item['sparql']}")
                logger.info(f"Fixed SPARQL:\n{sparql}")
            item["sparql"] = sparql
        except Exception:
            failed += 1

    logger.info(f"Fixed prefixes in {changed:,} SPARQL queries.")
    logger.info(f"Failed to fix prefixes in {failed:,} SPARQL queries.")
    dump_jsonl(data, args.file)


if __name__ == "__main__":
    fix(parse_args())
