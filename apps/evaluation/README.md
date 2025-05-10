# SPARQL QA Evaluation

This Streamlit application allows you to compare different SPARQL QA models across various knowledge graphs and benchmarks.

## Usage

1. Set `KG_BENCHMARK_DIR` env variable to `data/benchmark`: `export KG_BENCHMARK_DIR=$(pwd)/data/benchmark`
2. Navigate to the `apps/evaluation` directory
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run app.py`

## Directory structure

The app expects data to be organized in the following format:

```
[KG_BENCHMARK_DIR]/
  [knowledge-graph]/            # Name of the knowledge graph
    [benchmark]/                # Name of the benchmark
      test.jsonl                # Benchmark input and ground truth
      outputs/
        [model].jsonl           # Model output
        [model].config.json     # Model config
        [model].evaluation.json # Evaluation against ground truth
        ...
```

For example:

```
data/benchmark/
  wikidata/
    qald10/
      test.jsonl
      outputs/
        qwen-72b.search_extended.jsonl
        qwen-72b.search_extended.config.json
        qwen-72b.search_extended.evaluation.json
```

## Expected file formats

### Ground truth (test.jsonl)

Each line contains a JSON object like this, where `info` contains arbitrary additional
information:

```jsonc
{
  "id": "example_0",
  "question": "Select everything",
  "sparql": "SELECT * WHERE { ?s ?p ?o }",
  "paraphrases": ["Select all triples", "Get all triples"],
  "info": {
    "some_key": "some_value",
    ...
  }
}
```

### Model output (outputs/*.jsonl)

Each line contains a JSON object like this, with optional additional fields:

```jsonc
{
  "id": "example_0",
  "sparql": "SELECT * WHERE { ?s ?p ?o }",
  ...
}
```

### Evaluation (outputs/*.evaluation.json)

Model output evaluated against ground truth, as
produced by the `scripts/evaluate.py` script, in the format:

```jsonc
{
  "example_0": {
    // Ground truth SPARQL query
    "target": {
      "size": 12345, // Size of ground truth SPARQL results
      "err": null, // Error message if error occurred during ground truth SPARQL execution
    },
    // Model output, can be null if no output was produced
    "prediction": {
      "sparql": "SELECT * WHERE { ?s ?p ?o }",
      "err": null, // Error message if error occurred during predicted SPARQL execution
      "size": 12345, // Size of predicted SPARQL results
      "score": 0.8, // F1-score between predicted and ground truth SPARQL results
      "elapsed": 10.3 // Time taken to predict the SPARQL query by the model
  },
  ...
}
```
