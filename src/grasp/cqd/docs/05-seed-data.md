# Seed data: competency questions

CQD needs a small set of high-quality competency questions (CQs) per KG as
seeds for the teacher. **Seeds are provided as input, not built by CQD**: we
expect a simple JSONL file with SPARQL queries and optionally corresponding
NL questions. How that file is produced (hand-written, exported from QLever
examples, benchmark train splits, ...) is up to the user.

## Format

One seed per line:

```jsonc
{
  "sparql": "PREFIX wd: ... SELECT ...",   // required: reference SPARQL query
  "question": "What is the highest peak per country?",  // optional NL question
  "id": "wikidata-cq-042",                  // optional, auto-assigned if missing
  "info": {"tags": ["aggregation"], ...}    // optional free-form metadata
}
```

- `sparql` is required — it grounds the seed in the KG and lets CQD verify it
  by execution.
- `question` is optional; for seeds without one, the teacher phrases an NL
  question itself as part of the cq-distillation task.
- `info` is free-form; tags feed fan-out/coverage tracking if present.

`cqd/seeds.py` loads such a file, validates the schema, and verifies each
seed by executing its query (parseable, executes without error, non-empty and
non-degenerate result). Verification results (result size, errors) are
recorded per seed; failing seeds are reported, not silently dropped.

## Possible sources for seed files

- **QLever example queries**: `https://qlever.dev/api/examples/<kg>` returns
  TSV (`title \t sparql`). Availability (checked 2026-07-17): wikidata 71,
  osm-planet 58, dblp 19, uniprot 18, imdb 13 (freebase/dbpedia/orkg too
  sparse). Good CQ material — the Wikidata ones deliberately exercise KG
  peculiarities (statement nodes `p:`/`ps:`/`psn:`, qualifiers, normalized
  quantities, `wdt:P31/wdt:P279*`, language filters, subqueries). Needs
  manual filtering before use: drop endpoint diagnostics (meta-queries over
  `?s ?p ?o`, unicode/stress tests), dedupe near-identical variants; note
  QLever-specific extensions (spatial functions) if portability matters.
  Titles are noun phrases, not questions — fine, since `question` is
  optional.
- Benchmark train splits (QALD-10, WWQ, ...) — never test splits of
  benchmarks used for evaluation.
- WDQL (Wikidata query logs, SIGIR'26) — real user queries; noisy but huge.
- Hand-written CQs for quirks not covered elsewhere (e.g. Wikidata ranks /
  deprecated statements, `wikibase:timePrecision`).
