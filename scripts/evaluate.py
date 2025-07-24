import argparse
import json
import os

from tqdm import tqdm
from universal_ml_utils.io import load_jsonl

from grasp.sparql.constants import get_endpoint
from grasp.sparql.manager import load_kg_manager
from grasp.sparql.metrics import (
    f1_score,
    get_result_or_error,
    get_result_size,
)
from grasp.utils import (
    Sample,
    get_answer_or_cancel,
    is_invalid_evaluation,
    is_invalid_model_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("prediction", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "-kg",
        "--knowledge_graph",
        type=str,
        default="wikidata",
    )
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--exact-after", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--reparse-answer",
        action="store_true",
    )
    return parser.parse_args()


def create_dir(path: str):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def load_json(file: str) -> dict:
    with open(file) as f:
        return json.load(f)


def dump_json(obj: dict, file: str):
    create_dir(file)
    with open(file, "w") as f:
        json.dump(obj, f, indent=2)


def evaluate(args: argparse.Namespace):
    base = os.path.splitext(args.prediction)[0]
    evaluation_file = f"{base}.evaluation.json"

    exists = os.path.exists(evaluation_file)

    if exists and not args.overwrite:
        evaluations = load_json(evaluation_file)
    else:
        evaluations = {}

    predictions = load_jsonl(args.prediction)

    if args.reparse_answer:
        manager = load_kg_manager(args.knowledge_graph)
    else:
        manager = None

    inputs: dict[str, Sample] = {}
    for sample in load_jsonl(args.input):
        sample = Sample(**sample)
        assert sample.id not in inputs, f"Duplicate id {sample.id}"
        assert sample.id is not None, "Sample id must not be None"
        inputs[sample.id] = sample

    exact = args.exact or args.exact_after
    if args.endpoint is None:
        args.endpoint = get_endpoint(args.knowledge_graph)

    for pred in tqdm(
        predictions,
        desc="Evaluating",
        leave=False,
    ):
        if is_invalid_model_output(pred):
            continue

        id = pred["id"]
        if id in evaluations:
            evaluation = evaluations[id]
            if not args.retry_failed or not is_invalid_evaluation(evaluation):
                continue

        target = inputs[id].sparql
        target_set, target_err = get_result_or_error(
            target,
            args.endpoint,
            request_timeout=args.timeout,
            read_timeout=args.timeout,
        )
        evaluations[id] = {
            "target": {
                "err": target_err,
                "size": get_result_size(target_set),
            },
        }

        sparql = pred["sparql"]
        if args.reparse_answer and "messages" in pred:
            answer, cancel = get_answer_or_cancel(pred["messages"])
            if answer is not None:
                sparql = answer["sparql"]
            elif cancel is not None:
                best_attempt = cancel.get("best_attempt")
                if best_attempt:
                    sparql = best_attempt.get("sparql")

        if args.reparse_answer and sparql is not None:
            try:
                sparql = manager.fix_prefixes(sparql)
                sparql = manager.prettify(sparql)
            except Exception:
                pass

        if target_set is None or sparql is None:
            dump_json(evaluations, evaluation_file)
            continue

        pred_set, pred_err = get_result_or_error(
            sparql,
            args.endpoint,
            request_timeout=args.timeout,
            read_timeout=args.timeout,
        )
        if pred_set is not None:
            score = f1_score(pred_set, target_set, exact)
        else:
            score = 0.0

        evaluations[id]["prediction"] = {
            "sparql": sparql,
            "err": pred_err,
            "size": get_result_size(pred_set),
            "score": score,
            "elapsed": pred["elapsed"],
        }
        dump_json(evaluations, evaluation_file)

    dump_json(evaluations, evaluation_file)


if __name__ == "__main__":
    evaluate(parse_args())
