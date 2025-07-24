#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cat "$SCRIPT_DIR"/questions.yml | yq -o=json -I=0 '.questions | map({"id": .id|tostring, "question": .question.en, "sparql": .query.sparql, "paraphrases": [], "info": {}}) | .[]' >"$SCRIPT_DIR"/test.jsonl
