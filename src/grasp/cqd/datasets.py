import random
from logging import Logger

from tqdm import tqdm
from universal_ml_utils.io import load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.cqd.pool import PoolItem, PoolItemInfo, TaskPool, item_id
from grasp.evaluate import get_result_or_error, get_result_size
from grasp.sparql.types import AskResult
from grasp.tasks.sparql_qa.examples import SparqlQaSample

# Load a fixed benchmark split into pool items so train_rl can run RL over a
# frozen pool (curriculum_interval=0), isolating GRPO-on-execution-f1 +
# success-rate sampling from the auto-curriculum. All benchmarks share the
# layout data/benchmark/<kg>/<dataset>/{train,test}.jsonl, one row per line in
# the SparqlQaSample schema ({question, sparql, paraphrases, info}). The kg
# comes from the path (passed in), not the row.


def load_samples(file: str) -> list[SparqlQaSample]:
    return [SparqlQaSample(**row) for row in load_jsonl(file)]


# build pool items from benchmark samples for one kg, dropping samples missing
# a question/sparql or flagged invalid in info. ids are content derived
# (item_id), so duplicates within the dataset collapse onto one item.
def pool_items(
    samples: list[SparqlQaSample],
    kg: str,
    tag: str | None = None,
) -> list[PoolItem]:
    items: dict[str, PoolItem] = {}
    for sample in samples:
        question = sample.question.strip()
        sparql = sample.sparql.strip()
        if not question or not sparql or sample.info.get("invalid"):
            continue
        id = item_id(kg, question, sparql)
        if id in items:
            continue
        items[id] = PoolItem(
            id=id,
            question=question,
            sparql=sparql,
            info=PoolItemInfo(kg=kg, tags=[tag] if tag else []),
        )
    return list(items.values())


# keep only items whose reference query executes with a non-empty SELECT
# result on the endpoint. Drops broken/empty references (RL would waste
# rollouts on them, scored as reference-error) AND boolean (ASK) references:
# an ASK answer is one bit, so it is trivially gameable (a constant-true query
# scores against every true-answer question) and a poor execution-f1 reward
# signal -- see the f1 ask-hack finding.
def verify_items(
    items: list[PoolItem],
    endpoint: str,
    timeout: float = 300.0,
    max_rows: int | None = 100_000,
    progress: bool = False,
    logger: Logger | None = None,
) -> list[PoolItem]:
    logger = logger or get_logger("CQD DATASETS")
    kept = []
    ask = empty = broken = 0
    for item in tqdm(items, desc="Verifying references", disable=not progress):
        result, error = get_result_or_error(item.sparql, endpoint, timeout, max_rows)
        if error is not None or result is None:
            broken += 1
        elif isinstance(result, AskResult):
            ask += 1
        elif get_result_size(result) == 0:
            empty += 1
        else:
            kept.append(item)
    logger.info(
        f"{len(kept)}/{len(items)} references kept "
        f"(dropped {ask} ASK/boolean, {empty} empty, {broken} broken)"
    )
    return kept


# benchmark split -> ready-to-train pool file: load, convert, verify the
# references against the endpoint and save. limit subsamples the VERIFIED
# items, so the saved pool has exactly that many (e.g. a small holdout);
# without it the whole split is kept. The result doubles as an RlConfig
# val_file, which reads the same question/sparql rows.
def build_pool(
    file: str,
    kg: str,
    endpoint: str,
    out: str,
    tag: str | None = None,
    limit: int | None = None,
    seed: int = 22,
    timeout: float = 300.0,
    max_rows: int | None = 100_000,
    progress: bool = False,
    logger: Logger | None = None,
) -> TaskPool:
    logger = logger or get_logger("CQD DATASETS")
    items = pool_items(load_samples(file), kg, tag)
    logger.info(f"{len(items)} items from {file}")

    items = verify_items(items, endpoint, timeout, max_rows, progress, logger)
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)

    pool = TaskPool(items)
    pool.save(out)
    logger.info(f"Saved {len(pool)} items to {out}")
    return pool
