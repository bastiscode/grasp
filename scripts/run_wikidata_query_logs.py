import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm
from universal_ml_utils.io import load_jsonl


def send_post(
    url: str,
    data: Any,
    timeout: int = 300,
) -> dict | None:
    try:
        response = requests.post(
            url,
            json={
                "knowledge_graphs": ["wikidata"],
                "input": data,
                "task": "wikidata-query-logs",
            },
            timeout=timeout,
        )
        return response.json()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send JSONL POST requests to GRASP in parallel."
    )
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    parser.add_argument("endpoint", help="GRASP endpoint")
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max parallel requests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    lines = load_jsonl(args.input_file)
    random.seed(args.seed)
    random.shuffle(lines)

    skip = 0
    if os.path.exists(args.output_file) and not args.overwrite:
        skip = len(load_jsonl(args.output_file))
        print(f"Resuming from existing output file, skipping {skip} lines")

    # Use a single session for all requests
    with (
        ThreadPoolExecutor(max_workers=args.parallel) as executor,
        open(args.output_file, "a" if skip else "w") as f,
    ):
        futures = (
            executor.submit(send_post, args.endpoint, line, args.timeout)
            for line in lines[skip:]
        )

        for future in tqdm(
            as_completed(futures),
            total=len(lines) - skip,
            desc="Processing",
        ):
            result = future.result()
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    run(parse_args())
