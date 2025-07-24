import argparse

import fastapi
import uvicorn
from universal_ml_utils.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_file",
        type=str,
        default="Path to output file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the FastAPI server on",
    )
    return parser.parse_args()


app = fastapi.FastAPI(title="TEXT2SPARQL API")

KNOWN_DATASETS = [
    "https://text2sparql.aksw.org/2025/dbpedia/",
    "https://text2sparql.aksw.org/2025/corporate/",
]


def run(args: argparse.Namespace):
    data = load_jsonl(args.output_file)

    @app.get("/")
    async def _answer(question: str, dataset: str):
        if dataset not in KNOWN_DATASETS:
            raise fastapi.HTTPException(404, "Unknown dataset ...")

        samples = [s for s in data if s["questions"][0] == question]
        if not samples:
            raise fastapi.HTTPException(404, "Question not found ...")

        elif len(samples) > 1:
            raise fastapi.HTTPException(500, "Multiple samples found for question ...")

        sample = samples[0]
        # can be None if no SPARQL query was generated
        sparql = sample["sparql"] or ""

        return {"dataset": dataset, "question": question, "query": sparql}

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    run(parse_args())
