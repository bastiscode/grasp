import argparse
import json
import random

from grasp.sparql.manager import load_kg_manager


def load_input_mapping(filepath: str) -> dict[str, str]:
    mapping = {}
    with open(filepath) as f:
        for line in f:
            obj = json.loads(line)
            question = obj["question"]
            assert question not in mapping, f"Duplicate question {question}"
            mapping[question] = obj["id"]
    return mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spinach", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace):
    input_mapping = load_input_mapping(args.input)

    manager = load_kg_manager("wikidata")

    with open(args.input) as f:
        input = [json.loads(line) for line in f]

        if args.shuffle:
            random.seed(args.seed)
            random.shuffle(input)

        ref_order = {item["id"]: i for i, item in enumerate(input)}

    with open(args.spinach) as f:
        spinach = json.load(f)

    outputs = [None] * len(input)
    for item in spinach:
        record_id = input_mapping[item["question"]]
        sparql = item["predicted_sparql"]
        try:
            sparql = manager.fix_prefixes(sparql)
        except Exception as e:
            print(f"Error fixing prefixes for {sparql}: {e}")

        idx = ref_order[record_id]

        outputs[idx] = {
            "typ": "output",
            "sparql": sparql,
            "elapsed": 0.0,
            "id": record_id,
        }

    with open(args.output, "w") as outf:
        for output in outputs:
            outf.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    main(parse_args())
