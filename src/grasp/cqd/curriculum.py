import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger

from universal_ml_utils.logging import get_logger

from grasp.core import load_notes, setup
from grasp.cqd.configs import CqdConfig, RolloutConfig
from grasp.cqd.pool import PoolItem, PoolItemInfo, TaskPool, item_id
from grasp.cqd.proposer import (
    distillation_state,
    ingest_proposals,
    run_distillation_agent,
    seed_to_cq,
)
from grasp.cqd.reward import EpisodeReward, RewardConfig, reward_episodes, reward_stats
from grasp.cqd.rollout import Episode, collect_pool_rollouts, collect_rollouts
from grasp.cqd.seeds import Seed, load_seeds
from grasp.functions import find_manager
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


# probe a set of items with the current student: roll out num_rollouts each and
# score, recording a GroupRecord per item on the pool. Returns (episodes,
# rewards) -- used to measure difficulty and to feed run_revisits.
def probe_items(
    items: list[PoolItem],
    pool: TaskPool,
    rollout_config: RolloutConfig,
    reward_config: RewardConfig,
    num_rollouts: int,
    managers: list[KgManager],
    notes: list[str] | None,
    kg_notes: dict[str, list[str]] | None,
    logger: Logger,
    round: int | None = None,
) -> tuple[list[Episode], list[EpisodeReward]]:
    episodes = collect_rollouts(
        items, rollout_config, managers, num_rollouts=num_rollouts,
        parallelism=rollout_config.parallelism, kg_notes=kg_notes,
        notes=notes, progress=True, logger=logger,
    )
    rewards = reward_episodes(
        episodes, pool, reward_config, max_steps=rollout_config.max_steps,
        logger=logger, round=round,
    )
    return episodes, rewards


# UNIFIED curriculum step: reformulate the bottom-k least-learnable items from
# the student's fresh traces, over a combined candidate set of (a) the round's
# already-probed items (passed as episodes/rewards) and (b) a batch of uncovered
# seed CQs probed here with the CURRENT student. Revisiting and coverage are the
# same operation: a hard just-probed seed has ~0 learnability -> bottom-k -> the
# teacher makes an EASIER variant (BROADENING to a new CQ); a mastered pool item
# -> harder variant (DEEPENING). The raw seeds are added to the pool only
# transiently for the bottom-k lookup and removed afterwards -- never training
# items, and their probe rollouts feed only the teacher, not the gradient. The
# caller must save the (now seed-free) pool. Returns the teacher's proposals.
def cover_and_revisit(
    config: CqdConfig,
    pool: TaskPool,
    episodes: list[Episode],
    rewards: list[EpisodeReward],
    cover_seeds: list[Seed],
    rollout_config: RolloutConfig,
    reward_config: RewardConfig | None = None,
    num_rollouts: int = 4,
    max_revisits: int | None = None,
    managers: list[KgManager] | None = None,
    notes: list[str] | None = None,
    kg_notes: dict[str, list[str]] | None = None,
    round: int | None = None,
    logger: Logger | None = None,
) -> list[Proposal]:
    logger = logger or get_logger("CQD CURRICULUM")
    reward_config = reward_config or RewardConfig()
    if managers is None:
        managers, _ = setup(rollout_config)
    if notes is None and kg_notes is None:
        notes, kg_notes = load_notes(rollout_config)

    seed_items: list[PoolItem] = []
    for seed in cover_seeds:
        if not seed.question:
            continue
        kg = seed.info.get("kg")
        manager, _ = find_manager(managers, kg) if kg else (managers[0], None)
        kg = kg or manager.kg
        id = item_id(kg, seed.question, seed.sparql)
        if id in pool:
            continue
        pool.items[id] = PoolItem(
            id=id,
            question=seed.question,
            sparql=seed.sparql,
            info=PoolItemInfo(kg=kg, cq_id=seed.id, difficulty="similar"),
        )
        seed_items.append(pool.items[id])

    seed_eps: list[Episode] = []
    seed_rw: list[EpisodeReward] = []
    if seed_items:
        logger.info(f"Coverage: probing {len(seed_items)} uncovered seed CQs")
        seed_eps, seed_rw = probe_items(
            seed_items, pool, rollout_config, reward_config, num_rollouts,
            managers, notes, kg_notes, logger, round,
        )
        s = reward_stats(seed_rw)
        logger.info(
            f"Coverage seeds: mean f1 {(s['mean_f1'] or 0.0):.3f}, "
            f"{s['answered']} answered, {s['invalid']} invalid"
        )

    added = run_revisits(
        config,
        pool,
        list(episodes) + seed_eps,
        list(rewards) + seed_rw,
        learnability_half_life=rollout_config.learnability_half_life,
        max_revisits=max_revisits,
        managers=managers,
        notes=notes,
        kg_notes=kg_notes,
        parallelism=rollout_config.parallelism,
        logger=logger,
    )

    # drop the transient raw seeds; only the teacher's variants persist
    for it in seed_items:
        pool.items.pop(it.id, None)
    return added


# log to the ACTIVE wandb run, if there is one. Warm-start runs before
# train_rl, so a caller that starts the run first (see train.rl.init_wandb)
# gets both phases in one run; without a run, or without wandb installed,
# this is a no-op and the curriculum stays usable outside training.
def log_wandb(metrics: dict) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.run.log(metrics)


# one propose -> probe -> re-propose descent over a candidate set, shared by
# warm-start and the training curriculum. The teacher reformulates from the
# student's traces, the student then ATTEMPTS the fresh variants, and only
# variants whose competence exceeds min_competence are accepted; the rest come
# back as rejects for the caller to drop. Descends up to max_rounds levels
# (each level eases the previous level's rejects), stopping as soon as a level
# lands accepted variants. Probing between propose and accept is the point: a
# proposal never enters the pool unmeasured, and the next level's teacher call
# sees how its own last proposal actually fared.
#
# The default min_competence of 0 rejects exactly the items no rollout got any
# credit on. Those carry reward spread only from executable-vs-not, so their
# gradient teaches validity, never correctness -- the old gate's
# `or learnability > 0` admitted them. Anything with even one partially correct
# rollout is kept: it holds a real "more correct than that" comparison, and
# difficulty pacing is the sampler's job (pool.draw_weight), not this gate's.
# Raise it to also drop weak partial solvers, at the cost of discarding items
# that already cost a teacher run and a probe.
def descend(
    config: CqdConfig,
    pool: TaskPool,
    rollout_config: RolloutConfig,
    reward_config: RewardConfig,
    episodes: list[Episode],
    rewards: list[EpisodeReward],
    cover: list[Seed],
    num_rollouts: int = 4,
    max_rounds: int = 2,
    min_competence: float = 0.0,
    max_revisits: int | None = None,
    managers: list[KgManager] | None = None,
    notes: list[str] | None = None,
    kg_notes: dict[str, list[str]] | None = None,
    round: int = 0,
    logger: Logger | None = None,
) -> tuple[list[PoolItem], list[PoolItem]]:
    logger = logger or get_logger("CQD CURRICULUM")
    hl_c = rollout_config.competence_half_life
    accepted: list[PoolItem] = []
    rejected: list[PoolItem] = []

    for level in range(max_rounds):
        proposals = cover_and_revisit(
            config, pool, episodes, rewards, cover, rollout_config,
            reward_config, num_rollouts=num_rollouts, max_revisits=max_revisits,
            managers=managers, notes=notes, kg_notes=kg_notes,
            round=round + level, logger=logger,
        )
        cover = []  # only the first level introduces new seeds
        added = [
            pool[item_id(p.kg, p.question, p.sparql)]
            for p in proposals
            if item_id(p.kg, p.question, p.sparql) in pool
        ]
        if not added:
            break

        episodes, rewards = probe_items(
            added, pool, rollout_config, reward_config, num_rollouts,
            managers, notes, kg_notes, logger, round + level,
        )
        hit = [it for it in added if (it.competence(hl_c) or 0.0) > min_competence]
        hit_ids = {it.id for it in hit}
        accepted.extend(hit)
        rejected.extend(it for it in added if it.id not in hit_ids)
        logger.info(
            f"Descent level {level}: {len(hit)}/{len(added)} variants above "
            f"competence {min_competence}"
        )
        if hit:
            break  # reached the student's level; stop descending

    return accepted, rejected


# phase 1 (warm-start): build a small initial pool (~target solvable variants)
# with the SAME descent step training uses, bounded on both axes. BREADTH:
# cover seed CQs in shuffled batches until the pool reaches `target` accepted
# variants (or seeds run out). DEPTH: within a batch, descend up to max_rounds
# reformulation levels (easier -> probe -> easier ...) until the student gets
# at least partial credit on some of them (see descend's min_competence),
# stopping early once it does -- so a very hard seed still reaches a variant
# without unbounded growth. Only as many seeds as needed are consumed; the rest
# stay in reserve for coverage-paced growth during training. Raw seeds are never
# kept (cover_and_revisit drops them). Returns (pool, covered_seed_ids).
def warmstart_pool(
    config: CqdConfig,
    rollout_config: RolloutConfig,
    reward_config: RewardConfig | None = None,
    num_rollouts: int = 4,
    target: int = 8,
    batch_size: int | None = None,
    max_rounds: int = 3,
    min_competence: float = 0.0,
    max_revisits: int | None = None,
    managers: list[KgManager] | None = None,
    notes: list[str] | None = None,
    kg_notes: dict[str, list[str]] | None = None,
    log_level: str | int | None = None,
) -> tuple[TaskPool, set[str]]:
    logger = get_logger("CQD WARMSTART", log_level)
    reward_config = reward_config or RewardConfig()
    hl_c = rollout_config.competence_half_life
    if managers is None:
        managers, _ = setup(rollout_config)
    if notes is None and kg_notes is None:
        notes, kg_notes = load_notes(rollout_config)

    seeds = [s for s in load_seeds(config.seeds_file) if s.question]
    random.Random(rollout_config.seed).shuffle(seeds)
    batch_size = batch_size or target

    pool = TaskPool()

    kept: list[PoolItem] = []
    missed: list[PoolItem] = []
    covered: set[str] = set()
    batches = 0
    i = 0
    while len(kept) < target and i < len(seeds):
        batch = seeds[i : i + batch_size]
        i += len(batch)
        covered.update(s.id for s in batch if s.id is not None)
        logger.info(
            f"Warm-start: covering {len(batch)} seeds "
            f"(<={max_rounds} descent levels; {len(kept)}/{target} solvable)"
        )
        accepted, rejected = descend(
            config, pool, rollout_config, reward_config, [], [], batch,
            num_rollouts=num_rollouts, max_rounds=max_rounds,
            min_competence=min_competence, max_revisits=max_revisits,
            managers=managers, notes=notes, kg_notes=kg_notes, logger=logger,
        )
        kept.extend(accepted)
        missed.extend(rejected)
        log_wandb(
            {
                "warmstart/batch": batches,
                "warmstart/accepted": len(kept),
                "warmstart/rejected": len(missed),
                "warmstart/seeds_covered": len(covered),
                "warmstart/pool": len(pool),
            }
        )
        batches += 1

    if not kept:
        # nothing cleared the bar: fall back to the most solvable variants we
        # probed, NOT to all of them -- keeping everything would restore the
        # too-hard pool the competence gate exists to prevent
        logger.warning(
            f"Warm-start: no variant got above competence {min_competence}; "
            f"keeping the {target} most solvable of {len(missed)} probed"
        )
        missed.sort(key=lambda it: it.competence(hl_c) or 0.0, reverse=True)
        kept = missed[:target]
    kept.sort(key=lambda it: it.competence(hl_c) or 0.0, reverse=True)
    final = TaskPool(kept)
    final.save(config.pool_file)
    logger.info(
        f"Warm-start done: {len(final)} solvable variants; "
        f"covered {len(covered)}/{len(seeds)} seeds"
    )
    return final, covered


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
