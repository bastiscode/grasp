import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm
from universal_ml_utils.io import dump_json, load_json, load_jsonl


def send_post(
    url: str,
    idx: int,
    data: Any,
    timeout: int = 300,
) -> tuple[int, dict] | None:
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
        return idx, response.json()
    except Exception:
        return None


def path(idx: int, dir: str) -> str:
    return os.path.join(dir, f"{idx}.json")


def exists(idx: int, dir: str) -> bool:
    return os.path.exists(path(idx, dir))


def failed_output(output: dict | None) -> bool:
    if output is None:
        return True
    elif output["type"] == "cancel":
        # explicit cancel is not a failure
        return False

    result = output.get("sparql_result", output.get("formatted", ""))
    return (
        # comment this out for now, since it is actually not a failure
        # in terms of getting a valid response
        # "Got no rows" in result or
        "SPARQL execution failed" in result
        or "Error executing SPARQL query over" in result
    )


def failed_result(result: dict | None) -> bool:
    return result is None or "output" not in result


def failed(idx: int, dir: str) -> bool:
    try:
        result = load_json(path(idx, dir))
        return failed_result(result) or failed_output(result["output"])
    except Exception:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send JSONL POST requests to GRASP in parallel."
    )
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_dir", help="Path to output dir")
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
        "--retry-failed",
        action="store_true",
        help="Whether to retry failed samples",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output files if it exists",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    lines = list(enumerate(load_jsonl(args.input_file)))
    random.seed(args.seed)
    random.shuffle(lines)

    # filter out existing files
    lines = [
        (idx, data)
        for idx, data in lines
        if not exists(idx, args.output_dir)
        or args.overwrite
        or (args.retry_failed and failed(idx, args.output_dir))
    ]

    # Use a single session for all requests
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = (
            executor.submit(send_post, args.endpoint, idx, data, args.timeout)
            for idx, data in lines
        )

        for future in tqdm(
            as_completed(futures),
            total=len(lines),
            desc="Processing",
        ):
            result = future.result()
            if result is None:
                continue

            idx, result = result
            dump_json(result, path(idx, args.output_dir))


if __name__ == "__main__":
    run(parse_args())
