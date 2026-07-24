from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger

from universal_ml_utils.logging import get_logger

from grasp.core import load_notes, setup
from grasp.cqd.configs import CqdConfig, RolloutConfig
from grasp.cqd.pool import PoolItem, TaskPool
from grasp.cqd.proposer import (
    distillation_state,
    ingest_proposals,
    run_distillation_agent,
    seed_to_cq,
)
from grasp.cqd.reward import EpisodeReward, RewardConfig, reward_episodes
from grasp.cqd.rollout import Episode, collect_pool_rollouts
from grasp.cqd.seeds import load_seeds
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.cq_distillation import DistilledPair
from grasp.tasks.cq_distillation.functions import Proposal
from grasp.tasks.notes_from_traces import format_output


# rank the visited items the teacher should reformulate, least learnable
# first: low learnability means little within-group reward spread left to
# train on (consistently failed -> the teacher proposes an easier variant,
# consistently solved -> a harder one). The caller caps the count
# (max_revisits) to fix the per-round growth size, so this just orders the
# candidates. item_ids optionally restricts to the current round's
# rolled-out items, for which traces are available; never-attempted items
# have no learnability signal yet and are excluded.
def select_revisits(
    pool: TaskPool,
    item_ids: list[str] | None = None,
    learnability_half_life: float = 3.0,
) -> list[PoolItem]:
    if item_ids is None:
        items = list(pool.items.values())
    else:
        items = [pool[id] for id in item_ids if id in pool]

    visited = [item for item in items if item.n_groups() > 0]
    visited.sort(key=lambda item: item.learnability(learnability_half_life) or 0.0)
    return visited


# render a student episode for the teacher, with the scored outcome up
# front if available
def render_trace(
    episode: Episode,
    reward: EpisodeReward | None = None,
) -> str:
    messages = [Message(**m) for m in episode.messages]
    trace = format_output(None, messages)

    if reward is not None and reward.reward is not None:
        if reward.f1 is not None:
            outcome = (
                "execution F1 of the final query against the reference: "
                f"{reward.f1:.3f}"
            )
        elif reward.invalid:
            outcome = "no executable final query"
        else:
            outcome = "the student cancelled without a working best attempt"
        trace = f"Outcome: {outcome}, reward: {reward.reward:.3f}\n\n{trace}"

    return trace


# run the teacher on items without learning signal, passing the previous
# distilled pair and the student's traces, and ingest new proposals.
# item.info.cq_id stays the root CQ across generations (only parent_id
# advances), so revisiting a multi-hop proposal extends its lineage rather
# than restarting from the seed. managers/notes/kg_notes can be passed in
# to reuse an already-loaded KG instead of loading fresh via setup().
# The teacher agent runs (I/O against the teacher and KG endpoints) run up
# to parallelism at once; contexts are snapshotted before launch and
# proposals ingested serially, so the shared pool is never mutated
# concurrently. Keep parallelism within what the KG endpoint tolerates.
def run_revisits(
    config: CqdConfig,
    pool: TaskPool,
    episodes: list[Episode],
    rewards: list[EpisodeReward] | None = None,
    learnability_half_life: float = 3.0,
    max_revisits: int | None = None,
    managers: list[KgManager] | None = None,
    notes: list[str] | None = None,
    kg_notes: dict[str, list[str]] | None = None,
    parallelism: int = 4,
    logger: Logger | None = None,
) -> list[Proposal]:
    logger = logger or get_logger("CQD CURRICULUM")
    agent_logger = get_logger("GRASP AGENT")

    seeds = {seed.id: seed for seed in load_seeds(config.seeds_file)}

    by_item: dict[str, list[tuple[Episode, EpisodeReward | None]]] = {}
    for i, episode in enumerate(episodes):
        reward = rewards[i] if rewards is not None else None
        by_item.setdefault(episode.item_id, []).append((episode, reward))

    revisits = select_revisits(pool, list(by_item), learnability_half_life)
    if max_revisits is not None:
        revisits = revisits[:max_revisits]
    logger.info(f"Revisiting {len(revisits)} of {len(by_item)} rolled-out items")

    if managers is None:
        managers, _ = setup(config)
    if notes is None and kg_notes is None:
        notes, kg_notes = load_notes(config)

    # build the states first, snapshotting the pool context before any
    # concurrent run so no agent reads the pool while another ingests
    tasks = []
    for item in revisits:
        seed = seeds.get(item.info.cq_id)
        if seed is None:
            logger.warning(
                f"Competency question {item.info.cq_id} of item {item.id} "
                "not found in seeds, skipping revisit"
            )
            continue

        assert item.id is not None, "Pool item id must not be None"
        pair = DistilledPair(
            question=item.question,
            sparql=item.sparql,
            id=item.id,
            intent=item.info.intent,
        )
        traces = [render_trace(e, r) for e, r in by_item[item.id]]
        state = distillation_state(
            seed_to_cq(seed), config, pool, pair=pair, traces=traces
        )
        tasks.append((item, state))

    added = []
    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
        futures = {
            executor.submit(
                run_distillation_agent,
                state,
                config,
                managers,
                kg_notes,
                notes,
                agent_logger,
            ): item
            for item, state in tasks
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                output = future.result()
            except Exception as e:
                logger.exception(f"Revisit for item {item.id} failed: {e}")
                continue

            # ingest serially in the main thread (pool mutation)
            proposals = ingest_proposals(pool, output, config.model)
            logger.info(
                f"Added {len(proposals)} proposals revisiting item "
                f"{item.id} (learnability "
                f"{item.learnability(learnability_half_life)})"
            )
            added.extend(proposals)
            pool.save(config.pool_file)

    return added


# phase 5 entry point: one round of the teacher-student loop over an
# existing pool -- roll out the student on sampled items, score and record
# the attempts, then let the teacher revisit the signal-less items.
# Returns the round's summary statistics.
def run_curriculum_round(
    config: CqdConfig,
    rollout_config: RolloutConfig,
    reward_config: RewardConfig | None = None,
    max_revisits: int | None = None,
    log_level: str | int | None = None,
) -> dict:
    logger = get_logger("CQD CURRICULUM", log_level)
    assert config.pool_file == rollout_config.pool_file, (
        "Teacher and rollout config must use the same pool file"
    )

    episodes = collect_pool_rollouts(rollout_config, log_level)

    pool = TaskPool.load(config.pool_file)
    rewards = reward_episodes(
        episodes,
        pool,
        reward_config,
        max_steps=rollout_config.max_steps,
        progress=True,
        logger=logger,
    )
    pool.save(config.pool_file)

    added = run_revisits(
        config,
        pool,
        episodes,
        rewards,
        learnability_half_life=rollout_config.learnability_half_life,
        max_revisits=max_revisits,
        parallelism=rollout_config.parallelism,
        logger=logger,
    )

    scored = [r for r in rewards if r.reward is not None]
    stats = {
        "episodes": len(episodes),
        "scored": len(scored),
        "mean_reward": sum(r.reward or 0.0 for r in scored) / len(scored)
        if scored
        else None,
        "revisit_proposals": len(added),
        "pool": pool.stats(
            rollout_config.competence_half_life,
            rollout_config.learnability_half_life,
        ),
    }
    logger.info(f"Round finished: {stats}")
    return stats
