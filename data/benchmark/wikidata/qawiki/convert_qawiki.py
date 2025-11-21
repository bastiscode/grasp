import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

from grasp.sparql.utils import load_sparql_parser, fix_prefixes
from grasp.manager.utils import load_kg_prefixes

DEFAULT_LANGUAGE = "en"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a QA Wiki TSV file into JSON Lines formatted like sample.jsonl."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the TSV source file (qawiki-v1-simple-2025-09-09.tsv).",
    )
    parser.add_argument(
        "output_file",
        help="Destination JSONL path that will be created from the TSV contents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def extract_lang_strings(raw_field: str | None, lang: str) -> list[str]:
    """Split a pipe-delimited field and keep entries tagged with the language."""
    if not raw_field:
        return []

    lang = lang.lower()
    values: list[str] = []
    for chunk in raw_field.split("|"):
        chunk = chunk.strip()
        if not chunk or "@" not in chunk:
            continue
        text, tag = chunk.rsplit("@", 1)
        if tag.strip().lower() != lang:
            continue
        cleaned = text.strip()
        if cleaned:
            values.append(cleaned)
    return values


def unique_in_order(items: Iterable[str]) -> list[str]:
    """Deduplicate while preserving the original order."""
    seen = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def convert(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    parser = load_sparql_parser()
    prefixes = load_kg_prefixes("wikidata")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_lang = 0
    skipped_no_sparql = 0

    with (
        input_path.open("r", encoding="utf-8", newline="") as tsv_file,
        output_path.open("w", encoding="utf-8") as jsonl_file,
    ):
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            qid = row.get("qId") or row.get("questionId") or f"row_{idx}"
            questions = extract_lang_strings(row.get("questions"), DEFAULT_LANGUAGE)
            if not questions:
                skipped_lang += 1
                print(
                    f"Skipping {qid}: no @{DEFAULT_LANGUAGE} question available",
                    file=sys.stderr,
                )
                continue

            sparql_query = (row.get("sparql") or "").strip()
            if not sparql_query:
                skipped_no_sparql += 1
                print(f"Skipping {qid}: missing SPARQL query", file=sys.stderr)
                continue

            try:
                sparql_query = fix_prefixes(sparql_query, parser, prefixes)
            except Exception as e:
                print(
                    f"Warning: could not fix prefixes for {qid}: {e}",
                    file=sys.stderr,
                )

            paraphrases = unique_in_order(
                extract_lang_strings(row.get("paraphrases"), DEFAULT_LANGUAGE)
            )

            record = {
                "id": str(qid),
                "question": questions[0],
                "sparql": sparql_query,
                "paraphrases": paraphrases,
                "info": {},
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    summary = (
        f"Wrote {converted} examples to {output_path} "
        f"(skipped {skipped_lang} rows without @{DEFAULT_LANGUAGE} questions, "
        f"{skipped_no_sparql} rows without SPARQL)."
    )
    print(summary, file=sys.stderr)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    convert(input_path, output_path, args.overwrite)


if __name__ == "__main__":
    main()
