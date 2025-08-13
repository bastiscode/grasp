# bash script to make it easier to run models
# on multiple benchmarks

# error if knowledge graph is not set
kg=${KG:?"Set KG to the name of the knowledge graph"}
# error if name is not set
name=${NAME:?"Set NAME to the name of the model"}

# check if BENCHMARKS is not set
if [ -z "$BENCHMARKS" ]; then
  echo "BENCHMARKS not set"
  return 1
fi

IFS=" " read -ra benchmarks <<<"$BENCHMARKS"

args=${ARGS:-""}

for benchmark in "${benchmarks[@]}"; do
  dir="data/benchmark/$kg/$benchmark"
  file="$dir/test.jsonl"

  if [ ! -f "$file" ]; then
    continue
  fi

  # if EXAMPLES is set
  if [ -n "$EXAMPLES" ]; then
    export KG_EXAMPLES=$dir/train.example-index
  fi

  mkdir -p $dir/outputs

  grasp \
    --config configs/run.yaml \
    --question-file $file \
    --output-file $dir/outputs/$name.jsonl \
    $args \
    --shuffle \
    --seed 22

done
