import argparse
import csv
import glob
import json
import os
import sys

# CompoST (ag-sc, ISWC 2025) to GRASP benchmark files, one directory per level;
# "max" in their code is "hard" in the paper
LEVELS = {"easy": "easy", "medium": "medium", "hard": "max"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("compost_dir", type=str, help="Clone of ag-sc/compost")
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()


def gold_sizes(compost_dir: str, level: str) -> dict[tuple[str, str], tuple]:
    # their evaluated zero-shot run has the test (id, subid) and the gold size
    # on their Virtuoso endpoint (TP + FN)
    csv.field_size_limit(sys.maxsize)
    pattern = os.path.join(compost_dir, "results", level, "evaluated", "zeroshot_*")
    with open(sorted(glob.glob(pattern))[0]) as inf:
        rows = [r for r in csv.DictReader(inf) if r["split"] == "test"]
    return {
        (r["question"].strip(), " ".join(r["sparql"].split())): (
            r["id"],
            r["subid"],
            int(r["TP"]) + int(r["FN"]),
        )
        for r in rows
    }


def convert(args: argparse.Namespace) -> None:
    resources = os.path.join(args.compost_dir, "src", "lemon", "resources")
    for level, name in LEVELS.items():
        path = os.path.join(resources, f"compositionality_subgraph_{name}_10_fixed.json")
        with open(path) as inf:
            data = json.load(inf)

        sizes = gold_sizes(args.compost_dir, level)
        out_dir = os.path.join(args.output_dir, f"compost-{level}")
        os.makedirs(out_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as outf:
                for i, item in enumerate(data[split]):
                    info = {
                        "gid": item["gid"],
                        "compost_id": item["id"],
                        "depth": item["depth"],
                        "breadth": item["breadth"],
                        "num_edges": item["num_edges"],
                    }
                    if split == "test":
                        key = (item["question"].strip(), " ".join(item["query"].split()))
                        _, subid, size = sizes[key]
                        info["compost_subid"] = subid
                        info["virtuoso_gold_size"] = size

                    sample = {
                        "id": f"{split}_{i}",
                        "question": item["question"],
                        "sparql": item["query"],
                        "paraphrases": [],
                        "info": info,
                    }
                    outf.write(json.dumps(sample) + "\n")

            print(f"{level} {split}: {len(data[split])}")


if __name__ == "__main__":
    convert(parse_args())
