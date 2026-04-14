# GRASP - Generic Reasoning and SPARQL Generation across Knowledge Graphs

## News

- October 20th 2025:
  - GRASP can now be used for entity linking, in particular for
cell entity annotation
  - Workshop paper published for [SemTab 2025 challenge](https://sem-tab-challenge.github.io/2025/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/SEMTAB_entity_linking_grasp_WB_2025.pdf)

- August 28th 2025:
  - Demo paper of GRASP has also been accepted to [ISWC 2025](https://iswc2025.semanticweb.org/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_demo_WB_2025.pdf)

- July 31st 2025:
  - GRASP has been accepted to [ISWC 2025](https://iswc2025.semanticweb.org/)
  - Preview of camera-ready version available [here](https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_WB_2025.pdf)

- July 14th 2025:
  - arXiv preprint available at [arxiv.org/abs/2507.08107](https://arxiv.org/abs/2507.08107)

- July 10th 2025:
  - Code release
  - Data release

## Overview and directory structure

Links:

- Public demo available at [grasp.cs.uni-freiburg.de](https://grasp.cs.uni-freiburg.de)
- Data available at [ad-publications.cs.uni-freiburg.de/grasp](https://ad-publications.cs.uni-freiburg.de/grasp)

```
apps/
  evaluation/                     # Streamlit app for evaluation
  grasp/                          # Svelte web app compatible with GRASP server
  grisp/                          # Svelte web app compatible with GRISP server
bash/                             # Bash scripts to run and evaluate GRASP
configs/
  run.yaml                        # Config to run GRASP with a single KG
  serve.yaml                      # Config to run GRASP with all available KGs
  grisp/                          # Configs for the GRISP baseline
  notes/                          # Configs for note-taking
queries/                          # Custom index data and info SPARQL queries
                                    for various knowledge graphs
scripts/                          # Various helper scripts
data/                          
  benchmark/                      # Benchmarks grouped by knowledge graph
    [knowledge-graph]/
      [benchmark]/                   
        test.jsonl                # Test set with input and ground truth
        train-example-index/      # Index based on train set for few-shot learning
                                    (needs to be downloaded)
        outputs/
          [model].jsonl           # Model output
          [model].config.json     # Model config
          [model].evaluation.json # Evaluation against ground truth
  kg-index/                       # KG indices (need to be downloaded)
    wikidata/
    freebase/
    ...
src/                              # Source code for GRASP
Makefile                          # Makefile for building benchmarks
```

## Quickstart

Follow these steps to run GRASP. If you want to use Docker, see section
[Run GRASP with Docker](#run-grasp-with-docker) below.

### Run GRASP

1. Install GRASP

```bash
# Via git (recommended, up-to-date version)
pip install git+https://github.com/ad-freiburg/grasp.git@main

# From PyPI (not recommended as of now, at least as long
# as GRASP is under heavy development)
pip install grasp-rdf
```

2. Set the `GRASP_INDEX_DIR` env variable. Defaults to `$HOME/.grasp/index` if not
set. We set it to `$PWD/data/kg-index`, but you can choose any directory you like.

3. Get indices for the knowledge graphs you want to use. All indices are available
[publicly](https://ad-publications.cs.uni-freiburg.de/grasp/kg-index).
For example, to get the indices for Wikidata:

```bash
# Change to index directory
cd $GRASP_INDEX_DIR
# Download Wikidata index
wget https://ad-publications.cs.uni-freiburg.de/grasp/kg-index/wikidata.tar.gz
# Extract index
tar -xzf wikidata.tar.gz
```

Optionally, you can also download example indices for few-shot learning.
Example indices are always built from the train set of a benchmark
and called `train-example-index`.
For example, to get the example index for QALD-10 on Wikidata:

```bash
# Change to benchmark directory
cd data/benchmark/wikidata/qald10
# Download example index
wget https://ad-publications.cs.uni-freiburg.de/grasp/benchmark/wikidata/qald10/train-example-index.tar.gz
# Extract example index
tar -xzf train-example-index.tar.gz
```

4. Run GRASP:

```bash
# Note, that if you e.g. run OpenAI models, you also need to set the
# OPENAI_API_KEY env variable (see section about supported models below).

# Tip: Set --log-level DEBUG to show the individual steps of GRASP
# (reasoning and function calls) in a nicely formatted way.

# Run GRASP on an input and output the result to stdout as JSON with metadata.
# Actual output for the task is in the "output" field of that JSON object.

# Input from stdin:
echo "Where was Angela Merkel born?" | grasp run configs/run.yaml

# Input via CLI argument:
grasp run configs/run.yaml --input "Where was Angela Merkel born?"

# You can run different tasks with GRASP (default is sparql-qa).
# Depending on the task, the expected input format and output format
# will differ. For general-qa, the input is also a natural language
# question, same as for sparql-qa, but the output will be just a natural
# language answer instead of a SPARQL query.
echo "Where was Angela Merkel born?" | grasp run configs/run.yaml --task general-qa

# For cell entity annotation (cea), the input is a JSON object with a "table"
# field containing "header" and "data". The task links table cells to entities
# in the knowledge graph.
grasp run configs/run.yaml --task cea --input-format json \
  --input '{"table": {"header": ["Country", "Capital"], "data": [["France", "Paris"]]}}'

# Show all available options:
grasp run -h

# You can also run GRASP on multiple inputs (in JSONL format).
# In the following, we show an example to run GRASP on the QALD-10 
# test set over Wikidata.

# Input from stdin:
cat data/benchmark/wikidata/qald10/test.jsonl | grasp file configs/run.yaml

# Input via CLI argument:
grasp file configs/run.yaml --input-file data/benchmark/wikidata/qald10/test.jsonl

# Save output to a file instead of stdout and show progress bar:
grasp file configs/run.yaml \
  --input-file data/benchmark/wikidata/qald10/test.jsonl \
  --output-file data/benchmark/wikidata/qald10/outputs/test.jsonl \
  --progress

# Show all available options:
grasp file -h

# You can also run GRASP in a client-server setup. This is also the server
# that powers the corresponding web app.
# To start a GRASP server, by default on port 8000, just run:
grasp serve configs/run.yaml

# For convenience, we also provide a config to run the server with all
# available knowledge graphs (make sure to download all indices first):
grasp serve configs/serve.yaml

# Show all available options:
grasp serve -h

# Evaluate GRASP output with F1 score (for sparql-qa task) by executing
# predicted and ground truth SPARQL queries and comparing results:
grasp evaluate f1 wikidata \
  data/benchmark/wikidata/qald10/test.jsonl \
  data/benchmark/wikidata/qald10/outputs/gpt-41.search_extended.jsonl

# Use a judge model to pick the best output from multiple prediction files.
# The last argument is the output evaluation file:
grasp evaluate judge configs/run.yaml \
  data/benchmark/wikidata/qald10/test.jsonl \
  data/benchmark/wikidata/qald10/outputs/model1.jsonl \
  data/benchmark/wikidata/qald10/outputs/model2.jsonl \
  data/benchmark/wikidata/qald10/outputs/judge.evaluation.json

# Show all available options:
grasp evaluate f1 -h
grasp evaluate judge -h

# Build an example index for few-shot learning from a JSONL file of examples:
grasp examples data/benchmark/wikidata/qald10/train.jsonl \
  data/benchmark/wikidata/qald10/train-example-index

# Merge data from multiple knowledge graphs into a single combined KG.
# The first KG is the primary one; entities/properties from subsequent KGs
# are added to it. For example used to combine language-specific indices
# of the same knowledge graph:
grasp merge wikidata-en wikidata-de wikidata-fr wikidata-multilingual

# Note-taking: run GRASP on a knowledge graph to produce notes that
# can be included in the config to improve performance.
# Take notes by running GRASP on exemplary task samples:
grasp notes samples configs/notes/samples.yaml notes/
# Take notes from existing GRASP output files:
grasp notes outputs configs/notes/outputs.yaml notes/
# Take notes by freely exploring a knowledge graph (no task samples needed):
grasp notes explore configs/notes/explore.yaml notes/
```

### Server API

When running `grasp serve`, the server exposes the following HTTP endpoints (by default on port 8000).

| Endpoint | Method | Description |
|---|---|---|
| `/knowledge_graphs` | GET | Returns list of available KG names |
| `/config` | GET | Returns the server configuration |
| `/run` | POST | Run GRASP on a single input |

#### `POST /run`

**Request body:**

```json
{
  "task": "sparql-qa",
  "input": "Where was Angela Merkel born?",
  "knowledge_graphs": ["wikidata"],
  "past": null
}
```

- `task`: one of `"sparql-qa"`, `"general-qa"`, `"cea"`, `"wikidata-query-logs"`
- `input`: the task input (string for most tasks, JSON object for `"cea"`)
- `knowledge_graphs`: list of one or more KG names available on the server
- `past` *(optional)*: conversation history for multi-turn interactions, with fields `messages` (list of previous messages) and `known` (set of known entity/property IRIs)

**Response:** GRASP output as a JSON object

**Error codes:** `400` invalid KG selection, `429` rate limit exceeded, `503` server busy, `504` generation timeout

### Run GRASP with Docker

Build the Docker image:

```bash
docker build -t grasp .
```

The entrypoint for the Docker image is the `grasp` CLI. To run it with
Docker, make sure that your `GRASP_INDEX_DIR` is mounted to `/opt/grasp`
and your API keys (e.g. `OPENAI_API_KEY`) are set as env variables.

Some example commands are shown below.

```bash
# Answer a single question from stdin
echo "Where was Angela Merkel born?" | \
  docker run -i --rm \
  --user $(id -u):$(id -g) \
  -e OPENAI_API_KEY \
  -v $GRASP_INDEX_DIR:/data/index \
  -e HF_HOME=/hf \
  -v $HF_HOME:/hf \
  grasp run configs/run.yaml

# If you want to run a server with your own config,
# just mount it into the container
docker run --rm \
  --user $(id -u):$(id -g) \
  -e OPENAI_API_KEY \
  -v $GRASP_INDEX_DIR:/data/index \
  -e HF_HOME=/hf \
  -v $HF_HOME:/hf \
  -v $PWD/my_config.yaml:/grasp/server.yaml \
  grasp serve server.yaml
```

### Configure GRASP

GRASP can be configured via a single YAML config file, which is passed
to `grasp run`, `grasp file`, or `grasp serve` as first argument (see above).
You can use env variable placeholders in the config file of the form
`env(VAR_NAME:default_value)`, which will be replaced at runtime by the value of
the env variable `VAR_NAME` if it is set, or by `default_value` otherwise.
If no default value is given and the env variable is not set, an error
is raised. If you omit an entire config option, we also use a default value
as specified in the config code.

The configuration options and the use of env variable placeholders are
mostly self-explanatory, so we refer you to the [example config files](configs)
and the [config code](src/grasp/configs.py) for details.

### Build your own knowledge graph indices

Using GRASP with your own knowledge graph requires two steps:

- Getting the index data from a SPARQL endpoint for the knowledge graph
- Building the indices

#### Get index data

We get the index data by issuing two SPARQL queries to a SPARQL endpoint,
one for entities and one for properties. Both queries are expected to
return three columns in their results:

1. The IRI of the entity/property (required, must be unique)
2. The main label of the entity/property (optional)
3. All other labels/aliases of the entity/property, separated by `;;;` (optional)

A typical SPARQL query for that looks like this:

```sparql
SELECT
  # unique identifier of the entity/property
  ?id
  # main label of the entity/property, typically in English via rdfs:label
  (SAMPLE(?label) AS ?main_label)
  # all other labels/aliases, separated by ;;;
  (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=";;;") AS ?aliases)
WHERE {
  ...
}
# group by the identifier to ensure uniqueness
GROUP BY ?id
```

The query body will determine which entities/properties are included included
in the index, and how their labels and aliases are retrieved.

> Notes:
>
> - If you do not provide custom index data SPARQL queries, we use the generic
> default queries from [here](src/grasp/sparql/queries)
> - Our custom index data queries for various knowledge graphs
> are [here](queries)
> - If there is neither a label nor an alias for an entity/property, we use
> its IRI as fallback label
> - For properties, we always add the IRI as alias, to make them searchable by
> their IRI as well

With the CLI, you can use the `grasp data` command as follows:

```bash
# By default, if you just specify the knowledge graph name,
# we use https://qlever.cs.uni-freiburg.de/api/<kg_name> as SPARQL endpoint.
# The data will be saved to $GRASP_INDEX_DIR/<kg_name>/entities/data.tsv
# and $GRASP_INDEX_DIR/<kg_name>/properties/data.tsv.
# For example, to get the index data for IMDB:
grasp data imdb

# You can also set a custom SPARQL endpoint:
grasp data my-imdb --endpoint https://my-imdb-sparql-endpoint.com/sparql

# To download the index data, we use generic queries for both
# entities and properties by default. You can also provide your own queries,
# which is recommended, especially for larger knowledge graphs or
# knowledge graph with unusual schema.
grasp data imdb \
  --entity-sparql <path/to/entity.sparql> \
  --property-sparql <path/to/property.sparql>

# Show all available options:
grasp data -h
```

#### Build indices

After getting the index data, you can build the indices for the knowledge graph.
You probably do not need to change any parameters here.

With the CLI, you can use the `grasp index` command as follows:

```bash
# The indices will be saved to $GRASP_INDEX_DIR/<kg_name>/entities/<index_type>
# and $GRASP_INDEX_DIR/<kg_name>/properties/<index_type>.
# For example, to build the indices for IMDB:
grasp index imdb

# You can also change the types of indices that are built. By default, we build a
# fuzzy index for entities and an embedding index for properties.
grasp index imdb \
  --entities-type <keyword|fuzzy|embedding> \
  --properties-type <keyword|fuzzy|embedding>

# Show all available options:
grasp index -h
```

After this step is done, you can use the knowledge graph with GRASP by
including it in your config file (see above).

#### Customizing prefixes and info SPARQL queries

There are two more optional steps you can perform to customize the behavior
of GRASP related to your knowledge graph.

**KG info (prefixes and description)**

First, you can provide metadata about a knowledge graph via an
`info.json` file. This file can contain a description of the knowledge graph
(shown to the model in the system prompt) and prefix mappings used at build
time and runtime.
Create a file `$GRASP_INDEX_DIR/<kg_name>/info.json`
in the following format (example for Wikidata):

```jsonc
{
  "description": "A free, collaborative, multilingual knowledge base maintained by the Wikimedia Foundation.",
  "prefixes": {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    // other prefixes ...
  }
}
```

Both fields are optional. The description is shown to the model alongside the
knowledge graph name and endpoint. The prefixes are used during build time for
fallback label generation if an entity/property has neither a label nor an
alias. During runtime, the prefixes are used to shorten IRIs in function call
results, and allow GRASP to use prefixed instead of full IRIs in function call
arguments.

> Note: For QLever endpoints, we automatically retrieve prefixes via the API at
> `https://qlever.cs.uni-freiburg.de/api/prefixes/<kg_name>`, so you do not
> need to create an `info.json` file just for prefixes in that case

**Info SPARQL queries**

Second, you can customize the SPARQL queries that GRASP uses to fetch additional
information about entities and properties for enriching search results.
For that, create a file `$GRASP_INDEX_DIR/<kg_name>/entities/info.sparql`
for entities or `$GRASP_INDEX_DIR/<kg_name>/properties/info.sparql` for properties.
The file should contain a SPARQL query that returns three columns in its results:

1. `?id`: the IRI of the entity/property (required)
2. `?value`: a single piece of additional information (e.g. a label, alias, or description)
3. `?type`: the type of information, one of `"label"`, `"alias"`, or `"info"`

The query returns one row per piece of information (not one row per entity).
A typical SPARQL query for that looks like this:

```sparql
SELECT DISTINCT ?id ?value ?type WHERE {
  {
    VALUES ?id { {IDS} }
    ?id rdfs:label ?value
    BIND("label" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id skos:altLabel ?value
    BIND("alias" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id rdfs:comment ?value
    BIND("info" AS ?type)
  }
  FILTER(LANG(?value) = "en")
}
ORDER BY ?id ?type ?value
```

At runtime, all places where `{IDS}` appears in the query will be
replaced by the list of entity/property IRIs to get information for.
Typically, this will be within a `VALUES ?id { ... }` clause as
shown above.

See our [info SPARQL query for Wikidata entities](queries/wikidata.entity.info.sparql) as an example.

> Note: If no custom info SPARQL query is found, we use the
> default ones from [here](src/grasp/sparql/queries)

## Run GRASP webapp

Make sure to start a GRASP server first (see above).
Then follow [these instructions](apps/grasp/README.md) to run the GRASP web app.

## Run GRISP baseline

GRISP (Guided Recurrent IRI Selection over SPARQL Skeletons) is an alternative
question-answering baseline included in this repository. It works by fine-tuning
a small language model to generate SPARQL skeletons, then iteratively retrieving
and re-ranking entities using the GRASP search indices.

Follow [these instructions](src/grasp/baselines/grisp/README.md) to train,
run, and evaluate GRISP. To run the GRISP web app, follow
[these instructions](apps/grisp/README.md).

## Run evaluation app

Follow [these instructions](apps/evaluation/README.md) to run the
evaluation app for the SPARQL QA task.

## Supported models

GRASP supports both commercial and open-source models.

### OpenAI

1. Set `OPENAI_API_KEY` env variable
2. Set model to `openai/<model_name>` in the config file or with
`MODEL` env variable, we tested:

- `openai/gpt-4.1`
- `openai/gpt-4.1-mini`
- `openai/o4-mini`
- `openai/gpt-5-mini`
- `openai/gpt-5`

### Google Gemini

1. Set `GEMINI_API_KEY`
2. Set model to `gemini/<model_name>` in the config file or with
`MODEL` env variable, we tested:

- `gemini/gemini-2.0-flash`
- `gemini/gemini-2.5-flash-preview-04-17`

### Local server with vLLM

1. Install vLLM with `pip install vllm`
2. Run vLLM server with a model of your choice, see below
3. Set model to `hosted_vllm/<model_name>` in the config file or with
`MODEL` env variable, we tested:

- `hosted_vllm/Qwen/Qwen2.5-72B-Instruct` (and other sizes)
- `hosted_vllm/Qwen/Qwen3-32B` (and other sizes)

1. Set model_endpoint in the config file or with `MODEL_ENDPOINT` env variable
to your vLLM server endpoint, by default this will be `http://localhost:8000/v1`

#### Run Qwen2.5

Change 72B to 7B, 14B, or 32B to run other sizes. Adapt the tensor parallel size
to your GPU setup, we used two H100 GPUs for Qwen2.5 72B.

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --tool-call-parser hermes \
--enable-auto-tool-choice --tensor-parallel-size 2
```

#### Run Qwen3

Change 32B to 4B, 8B, or 14B to run other sizes.

```bash
vllm serve Qwen/Qwen3-32B --reasoning-parser qwen3 \
--tool-call-parser hermes --enable-auto-tool-choice
```

## Misc

To prepare some benchmark datasets with the [Makefile](Makefile),
e.g. using `make wikidata-benchmarks`, you first need to clone
[github.com/KGQA/KGQA-datasets](https://github.com/KGQA/KGQA-datasets) into `third_party`:

```bash
mkdir -p third_party
git clone https://github.com/KGQA/KGQA-datasets.git third_party/KGQA-datasets
```

## Citation

If you use this project, please consider citing the following works:

```bibtex
@inproceedings{DBLP:conf/semweb/WalterB25,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {{GRASP:} Generic Reasoning And {SPARQL} Generation Across Knowledge
                  Graphs},
  booktitle    = {{ISWC} {(1)}},
  series       = {Lecture Notes in Computer Science},
  volume       = {16140},
  pages        = {271--289},
  publisher    = {Springer},
  year         = {2025}
}

@inproceedings{DBLP:conf/semweb/WalterB25a,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {{GRASP:} Generic Reasoning And {SPARQL} Generation across Knowledge
                  Graphs - Demo System},
  booktitle    = {{ISWC} (Industry/Doctoral Consortium/Posters/Demos)},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {4085},
  pages        = {475--482},
  publisher    = {CEUR-WS.org},
  year         = {2025}
}

@inproceedings{GRASP_EntityLinking_WalterB25,
  author       = {Sebastian Walter and
                  Hannah Bast},
  title        = {Knowledge Graph Entity Linking via Interactive Reasoning and
Exploration with {GRASP}},
  booktitle    = {{OM} 2025 (Ontology Matching Workshop)},
  note         = {To appear},
  series       = {{CEUR} Workshop Proceedings},
  publisher    = {CEUR-WS.org},
  year         = {2025}
}
```
