import argparse
import json
import re

from bs4 import BeautifulSoup


def extract_question_sparql(html):
    """
    Parses the given HTML string and returns a list of
    (question_text, sparql_query) tuples, where the SPARQL
    is taken from the first <span id="..."> in each <div>.
    """
    soup = BeautifulSoup(html, "html.parser")
    pairs = []

    # For each <div> containing both a <p> and a <span id="...">
    for div in soup.find_all("div"):
        p = div.find("p")
        span = div.find("span", id=re.compile(r"example\d+"))
        if not (p and span):
            continue

        question = p.get_text(strip=True)
        id = span["id"]
        sparql = span.get_text(strip=True)

        # Skip empty spans
        if not sparql:
            continue

        pairs.append((id, question, sparql))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract question/SPARQL pairs from UniProt HTML (SPARQL in <span>)."
    )
    parser.add_argument(
        "input",
        help="Path to the HTML file to parse.",
    )
    parser.add_argument(
        "output",
        help="Write the result as JSON to this file.",
    )
    args = parser.parse_args()

    # Read the HTML
    with open(args.input, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract
    pairs = extract_question_sparql(html)

    # Output
    out_list = [{"id": id, "question": q, "sparql": s} for id, q, s in pairs]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(pairs):,} examples to {args.output}")


if __name__ == "__main__":
    main()
