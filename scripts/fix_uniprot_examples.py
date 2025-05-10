import argparse
import json

from litellm import completion
from pydantic import BaseModel
from tqdm import tqdm

from grasp.utils import Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Paths to the input JSON files")
    parser.add_argument("output", help="Path to the output JSONL file")
    return parser.parse_args()


class Pair(BaseModel):
    question: str
    sparql: str


class Response(BaseModel):
    pair: Pair | None = None


def fix(args: argparse.Namespace):
    # Read the input JSON files
    all_data = []
    seen_ids = set()
    seen_questions = set()
    seen_sparqls = set()
    for input_file in args.inputs:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if entry["id"] in seen_ids:
                    continue
                elif entry["question"] in seen_questions:
                    continue
                elif entry["sparql"] in seen_sparqls:
                    continue
                seen_questions.add(entry["question"])
                seen_sparqls.add(entry["sparql"])
                all_data.append(entry)

    print(f"Loaded {len(all_data):,} unique entries from {len(args.inputs)} files.")

    # Process each entry
    samples: list[Sample] = []
    for entry in tqdm(all_data, desc="Processing entries"):
        comp = completion(
            "openai/gpt-4.1-mini",
            [
                {
                    "role": "user",
                    "content": f"""\
Fix the following question and SPARQL query pair over UniProt. Remove \
any artifacts from the question to make it more natural. \
Return null if the SPARQL query is not an ASK or SELECT query.

Question:
{entry["question"]}

SPARQL:
{entry["sparql"]}""",
                }
            ],
            top_p=0.9,
            temperature=0.2,
            response_format=Response,
        )
        resp = Response(**json.loads(comp.choices[0].message.content))
        if resp.pair is None:
            print(f"Skipping entry:\n{json.dumps(entry, indent=2)}")
            continue

        samples.append(
            Sample(
                id=entry["id"],
                question=resp.pair.question,
                sparql=resp.pair.sparql,
            )
        )

    with open(args.output, "w") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")


if __name__ == "__main__":
    fix(parse_args())
