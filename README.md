# GRASP - Generic Reasoning and SPARQL generation across Knowledge Graphs

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
  grasp/                          # Web app compatible with GRASP server
bash/                             # Bash scripts to run and evaluate GRASP
configs/
  run.yaml                        # Config to run GRASP with a single KG
  serve.yaml                      # Config to run GRASP with all available KGs
queries/                          # Custom index data and info SPARQL queries
                                    for various knowledge graphs
scripts/                          # Various helper scripts
data/                          
  benchmark/                      # Benchmarks grouped by knowledge graph
    [knowledge-graph]/
      [benchmark]/                   
        test.jsonl                # Test set with input and ground truth
        train.example_index/      # Index based on train set for few-shot learning
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

Follow these steps to run GRASP.

### Run GRASP

> Note: We recommend to use conda for ease of installation of Faiss and to avoid
> dependency issues.

1. Create and activate conda environment:
`conda create -n grasp python=3.12 && conda activate grasp`

2. Install Faiss (not supported to be installed with pip):
`conda install -c pytorch -c nvidia faiss-gpu=1.11.0`

> You might have to install the CPU version of Faiss, since
> the GPU version leads to issues on some systems.

3. Install GRASP

```bash
# From PyPI (recommended)
pip install grasp-rdf

# Or via git
pip install git+https://github.com/ad-freiburg/grasp.git@main
```

4. Set the `GRASP_INDEX_DIR` env variable. Defaults to `$HOME/.grasp/index` if not
set. We set it to `$PWD/data/kg-index`, but you can choose any directory you like.

> We recommend to set it with conda, such that it is set automatically when you activate
> the conda environment: `conda env config vars set GRASP_INDEX_DIR=/path/to/dir`

5. Get indices for the knowledge graphs you want to use. All indices are available
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
and called `train.example-index`.
For example, to get the example index for QALD-10 on Wikidata:

```bash
# Change to benchmark directory
cd data/benchmark/wikidata/qald10
# Download example index
wget https://ad-publications.cs.uni-freiburg.de/grasp/benchmark/wikidata/qald10/train.example-index.tar.gz
# Extract example index
tar -xzf train.example-index.tar.gz
```

6. Run GRASP:

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
# prefix index for entities and a similarity index for properties.
grasp index imdb \
  --entities-type <prefix|similarity> \
  --properties-type <prefix|similarity>

# Show all available options:
grasp index -h
```

After this step is done, you can use the knowledge graph with GRASP by
including it in your config file (see above).

#### Customizing prefixes and info SPARQL queries

There are two more optional steps you can perform to customize the behavior
of GRASP related to your knowledge graph.

**Prefixes**

First, you can customize the prefixes that GRASP uses for a
knowledge graph at build time and runtime.
For that, create a file `$GRASP_INDEX_DIR/<kg_name>/prefixes.json`
in the following format (example for Wikidata):

```jsonc
{
  "wd": "<http://www.wikidata.org/entity/",
  "wdt": "<http://www.wikidata.org/prop/direct/",
  // other prefixes ...
}
```

During build time, the prefixes are used for the fallback label generation
if an entity/property has neither a label nor an alias. During runtime, the
prefixes are used to shorten IRIs in function call results, and allows GRASP
to use prefixed instead of full IRIs in function call arguments.

> Note: For QLever endpoints, we automatically retrieve prefixes via the API at
> `https://qlever.cs.uni-freiburg.de/api/prefixes/<kg_name>`, so you do not
> need to create a `prefixes.json` file in that case

**Info SPARQL queries**

Second, you can customize the SPARQL queries that GRASP uses to fetch additional
information about entities and properties for enriching search results.
For that, create a file `$GRASP_INDEX_DIR/<kg_name>/entities/info.sparql`
for entities or `$GRASP_INDEX_DIR/<kg_name>/properties/info.sparql` for properties.
The file should contain a SPARQL query, that returns two columns in its results:

1. The IRI of the entity/property (required, must be unique)
2. All additional information about the entity/property, separated by `;;;` (optional)

A typical SPARQL query for that looks like this:

```sparql
SELECT
  # unique identifier of the entity/property
  ?id
  # all additional information, separated by ;;;
  (GROUP_CONCAT(DISTINCT ?info; SEPARATOR=";;;") AS ?infos)
} WHERE {
  {
    VALUES ?id { {IDS} }
    ...
  } UNION {
    VALUES ?id { {IDS} }
    ...
  }
  ...
}
# group by the identifier to ensure uniqueness
GROUP BY ?id
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

4. Set model_endpoint in the config file or with `MODEL_ENDPOINT` env variable
to your vLLM server endpoint, by default this will be `http://localhost:8000/v1`

#### Run Qwen2.5

Change 72B to 7B, 14B, or 32B to run other sizes. Adapt the tensor parallel size
to your GPU setup, we used two H100 GPUs for Qwen2.7 72B.

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
