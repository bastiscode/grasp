# bash script to make it easier to evaluate multiple models
# on multiple benchmarks

# by default use wikidata
kg=${KG:-"wikidata"}
# by default evaluate all benchmarks
benchmark=${BENCHMARK:-"*"}
# by default evaluate all models
name=${NAME:-"*"}

glob="data/benchmark/$kg/$benchmark/outputs/$name.jsonl"

args=${ARGS:-""}

for file in $glob; do
    if [[ ! -f $file ]]; then
        continue
    fi

    dir=$(dirname $(dirname $file))
    echo "$(basename $dir): $(basename $file)"

    python scripts/evaluate.py \
        $dir/test.jsonl \
        $file \
        $args \
        -kg $kg

    echo
done

