import os
import re
from logging import Logger

from tqdm import tqdm
from universal_ml_utils.io import dump_json
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import consume_generator

from grasp.core import generate, load_notes, setup
from grasp.cqd.configs import CqdConfig
from grasp.cqd.pool import TaskPool
from grasp.cqd.seeds import Seed, load_seeds, verify_seed
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.tasks.cq_distillation import (
    CompetencyQuestion,
    CqDistillationState,
    DistilledPair,
)
from grasp.tasks.cq_distillation.functions import Proposal


def seed_to_cq(seed: Seed) -> CompetencyQuestion:
    return CompetencyQuestion(
        id=seed.id,
        sparql=seed.sparql,
        question=seed.question,
        kg=seed.info.get("kg"),
    )


def context_proposals(
    pool: TaskPool,
    cq_id: str | None,
    recent: int,
) -> list[Proposal]:
    # all proposals of the current competency question plus the most
    # recent ones of others, in pool (insertion) order
    own = []
    other = []
    for item in pool.items.values():
        if cq_id is not None and item.info.cq_id == cq_id:
            own.append(item)
        else:
            other.append(item)

    items = other[len(other) - recent :] + own
    return [item.to_proposal() for item in items]


# build the cq-distillation state, snapshotting the pool context now (before
# any concurrent run mutates the pool)
def distillation_state(
    cq: CompetencyQuestion,
    config: CqdConfig,
    pool: TaskPool,
    pair: DistilledPair | None = None,
    traces: list[str] | None = None,
) -> CqDistillationState:
    return CqDistillationState(
        cq=cq,
        pair=pair,
        traces=traces or [],
        proposals=context_proposals(pool, cq.id, config.recent_proposals),
    )


# run the cq-distillation agent for a prebuilt state; no pool mutation, so
# this is safe to call concurrently (ingest the output serially afterwards)
def run_distillation_agent(
    state: CqDistillationState,
    config: CqdConfig,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    logger: Logger | None = None,
) -> dict:
    return consume_generator(
        generate(
            "cq-distillation",
            state,
            config,
            managers,
            kg_notes,
            notes,
            logger=logger or get_logger("GRASP AGENT"),
        )
    )


# ingest an agent run's accepted proposals into the pool; accepted proposals
# are execute-verified, so ingest them even if the run ended with an error
def ingest_proposals(
    pool: TaskPool,
    output: dict,
    teacher_model: str,
) -> list[Proposal]:
    proposals = [
        Proposal(**p) for p in (output.get("output") or {}).get("proposals", [])
    ]
    added = pool.add_proposals(proposals, teacher_model=teacher_model)
    return [item.to_proposal() for item in added]


# run the cq-distillation task once for the given competency question and
# ingest accepted proposals; returns the added proposals and agent output
def run_distillation(
    cq: CompetencyQuestion,
    config: CqdConfig,
    managers: list[KgManager],
    pool: TaskPool,
    pair: DistilledPair | None = None,
    traces: list[str] | None = None,
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    logger: Logger | None = None,
) -> tuple[list[Proposal], dict]:
    state = distillation_state(cq, config, pool, pair, traces)
    output = run_distillation_agent(state, config, managers, kg_notes, notes, logger)
    added = ingest_proposals(pool, output, config.model)
    return added, output


# phase 1 entry point: run cq-distillation over all seeds and grow the task
# pool, saving it after every run
def generate_pool(
    config: CqdConfig,
    log_level: str | int | None = None,
) -> TaskPool:
    logger = get_logger("CQD PROPOSER", log_level)
    agent_logger = get_logger("GRASP AGENT", log_level)

    seeds = load_seeds(config.seeds_file)
    logger.info(f"Loaded {len(seeds)} seeds from {config.seeds_file}")

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    if os.path.exists(config.pool_file):
        pool = TaskPool.load(config.pool_file)
        logger.info(f"Loaded pool with {len(pool)} items from {config.pool_file}")
    else:
        pool = TaskPool()

    if config.verify_seeds:
        verified = []
        for seed in seeds:
            kg = seed.info.get("kg")
            manager, _ = find_manager(managers, kg) if kg else (managers[0], None)
            verification = verify_seed(
                seed,
                manager.endpoint,
                config.seed_verification_timeout,
                config.seed_verification_max_rows,
            )
            if verification.ok:
                verified.append(seed)
            else:
                logger.warning(
                    f"Skipping seed {seed.id} ({verification.status}): "
                    f"{verification.error or 'empty result'}"
                )
        seeds = verified
        logger.info(f"{len(seeds)} seeds passed verification")

    if config.trace_dir is not None:
        os.makedirs(config.trace_dir, exist_ok=True)

    for r in range(config.rounds):
        for seed in tqdm(seeds, desc=f"Distilling round {r + 1}/{config.rounds}"):
            try:
                added, output = run_distillation(
                    seed_to_cq(seed),
                    config,
                    managers,
                    pool,
                    kg_notes=kg_notes,
                    notes=notes,
                    logger=agent_logger,
                )
            except Exception as e:
                logger.exception(f"Distillation for seed {seed.id} failed: {e}")
                continue

            if output.get("error") is not None:
                logger.warning(
                    f"Distillation for seed {seed.id} ended with error: "
                    f"{output['error']}"
                )

            logger.info(f"Added {len(added)} proposals for seed {seed.id}")
            pool.save(config.pool_file)

            if config.trace_dir is not None:
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", seed.id or "unknown")
                dump_json(
                    output,
                    os.path.join(
                        config.trace_dir,
                        f"cq-distillation.{slug}.round_{r}.json",
                    ),
                )

    stats = pool.stats()
    logger.info(
        f"Pool now has {stats['items']} items across "
        f"{len(stats['per_cq'])} competency questions"
    )
    return pool
