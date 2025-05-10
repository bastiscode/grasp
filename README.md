# GRASP - Generic Reasoning and SPARQL generation across Knowledge Graphs

Code and data for the corresponding ISWC 2025 paper submission.

# Overview and directory structure

```
Makefile                          # Makefile for getting data and building indices
src/                              # Source code for GRASP
bash/                             # Bash scripts to run and evaluate GRASP
scripts/                          # Various helper scripts
app/
  evaluation/                     # Streamlit app for evaluation
data/                          
  benchmark/                      # Benchmarks grouped by knowledge graph
    [knowledge-graph]/
      [benchmark]/                   
        test.jsonl                # Test set with input and ground truth
        train.example_index/      # Index based on train set for few-shot learning (needs to be downloaded)
        outputs/
          [model].jsonl           # Model output
          [model].config.json     # Model config
          [model].evaluation.json # Evaluation against ground truth
  kg-index/                       # KG indices (need to be downloaded)
    wikidata/
    freebase/
    ...
configs/
  single_kg.yaml                  # Config to run GRASP with a single KG
  serve.yaml                      # Config to run GRASP with all available KGs
```

# Quickstart

Follow these steps to run GRASP and the evaluation app.

## Run GRASP

> Note: We recommend to use conda for ease of installation of FAISS and to avoid
> dependency issues.

1. Create and activate conda environment: `conda create -n grasp python=3.12 && conda activate grasp`

2. Install FAISS (not supported to be installed with pip): `conda install -c pytorch -c nvidia faiss-gpu=1.9.0`

3. Clone the repository: `git clone https://github.com/ad-freiburg/grasp`

4. Go to directory and install with pip: `cd grasp && pip install -e .`

5. Get indices for the knowledge graphs you want to use. All indices are available
at [](https://iswc25-250.hopto.org/kg-index). For example, to get the indices for Wikidata:
```bash
# create index directory
mkdir -p data/kg-index
# download Wikidata index
wget -P data/kg-index https://iswc25-250.hopto.org/kg-index/wikidata.tar.gz
# extract index
tar -xzf data/kg-index/wikidata.tar.gz -C data/kg-index
```
Optionally, you can also download example indices for few-shot learning. Example indices are always built from the train set of a benchmark and called `train.example-index`. For example, to get the example index for QALD-10 on Wikidata:
```bash
# create benchmark directory
mkdir -p data/benchmark/wikidata/qald10
# download example index
wget -P data/benchmark/wikidata/qald10 https://iswc25-250.hopto.org/benchmark/wikidata/qald10/train.example-index.tar.gz
# extract example index
tar -xzf data/benchmark/wikidata/qald10/train.example-index.tar.gz -C data/benchmark/wikidata/qald10
```

6. Set `KG_INDEX_DIR` env variable: `export KG_INDEX_DIR=$(pwd)/data/kg-index`
> We recommend to set it with conda, such that it is set automatically when you activate
> the conda environment: `conda env config vars set KG_INDEX_DIR=$(pwd)/data/kg-index`

7. Run GRASP:
```bash
# MODEL, FN_SET, and KG need to be specified via env variables or
# can be set directly in the config file. An example index for few-shot learning 
# can be specified via the KG_EXAMPLES env variable or also in the config file.
# See the config files for more details and other options.

# Note, that if you e.g. run OpenAI models, you also need to set the
# OPENAI_API_KEY or API_KEY env variable or the api_key field in the config file
# (see section about supported models below).

# --log-level DEBUG is recommended for more verbose output showing
# intermediate steps.

# Run GRASP on a question:
# By default, GRASP outputs the answer to stdout as JSON with some extra metadata.
# To avoid this we redirect it to /dev/null here, and set --log-level to DEBUG which
# shows all steps in a nicely formatted way.
MODEL=openai/gpt-4.1 FN_SET=search_extended KG=wikidata grasp \
--config configs/single_kg.yaml \
--question "Where was Albert Einstein born?" \
--log-level DEBUG > /dev/null

# Run GRASP on a benchmark and save the output to a file, in this case QALD-10:
MODEL=openai/gpt-4.1 FN_SET=search_extended KG=wikidata grasp \
--config config/single_kg.yaml \
--file data/benchmark/wikidata/qald10/test.jsonl \
--output-file data/benchmark/wikidata/qald10/outputs/grasp-example.jsonl \
--log-level DEBUG

# Start a GRASP server with a Websocket endpoint at /live, in this case on port 8000:
MODEL=openai/gpt-4.1 FN_SET=search_extended KG=wikidata grasp \
--config configs/single_kg.yaml \
--serve 8000 \
--log-level DEBUG

# For convenience, we also provide a config to run the server with all available KGs,
# and model and function set already specified:
grasp --config configs/serve.yaml --serve 8000 --log-level DEBUG
```

## Run evaluation app

Follow steps [here](apps/evaluation/README.md) to run the evaluation app.

# Supported models

GRASP supports both commercial and open-source models.

## OpenAI

1. Set `OPENAI_API_KEY` or `API_KEY` env variable or `api_key` in the config file
2. Set model to `openai/<model_name>` in the config file or with `MODEL` env variable, we used:
- `openai/gpt-4.1`
- `openai/gpt-4.1-mini`
- `openai/o4-mini`

## Google Gemini

1. Set `GEMINI_API_KEY` or `API_KEY` env variable or `api_key` in the config file
2. Set model to `gemini/<model_name>` in the config file or with `MODEL` env variable, we used:
- `gemini/gemini-2.0-flash`
- `gemini/gemini-2.5-flash-preview-04-17`

## Local server with vLLM

1. Install vLLM with `pip install vllm`
2. Run vLLM server with a model of your choice, see below
3. Set model to `hosted_vllm/<model_name>` in the config file or with `MODEL` env variable, we used:
- `hosted_vllm/Qwen/Qwen2.5-72B-Instruct` (and other sizes)
- `hosted_vllm/Qwen/Qwen3-32B` (and other sizes)
4. Set model_endpoint in the config file or with `MODEL_ENDPOINT` env variable to your vLLM server endpoint, by default this will be `http://localhost:8000/v1`

### Run Qwen2.5

Change 72B to 7B, 14B, or 32B to run other sizes. Adapt the tensor parallel size to your GPU setup, we used two H100 GPUs for Qwen2.7 72B.

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --tool-call-parser hermes --enable-auto-tool-choice --tensor-parallel-size 2
```

### Run Qwen3

Change 32B to 4B, 8B, or 14B to run other sizes.

```bash
vllm serve Qwen/Qwen3-32B --enable-reasoning --reasoning-parser deepseek_r1 --tool-call-parser hermes --enable-auto-tool-choice
```

# Misc

To prepare some benchmark datasets with the [Makefile](Makefile), e.g. using `make wikidata-benchmarks`, you first need to clone [github.com/KGQA/KGQA-datasets](github.com/KGQA/KGQA-datasets) into `third_party`:
```bash
mkdir -p third_party
git clone https://github.com/KGQA/KGQA-datasets.git third_party/KGQA-datasets
```
