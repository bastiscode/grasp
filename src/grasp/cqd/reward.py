import time
from logging import Logger

from pydantic import BaseModel, Field
from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.cqd.pool import GroupRecord, TaskPool
from grasp.cqd.rollout import Episode, client_tool_call_errors
from grasp.evaluate import get_result_or_error
from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_kg_info,
    merge_prefixes,
)
from grasp.sparql.metrics import f1_score
from grasp.sparql.types import AskResult, SelectResult
from grasp.sparql.utils import (
    fix_prefixes,
    get_qlever_endpoint,
    load_iri_and_literal_parser,
    load_sparql_parser,
)
from grasp.utils import is_server_error

# episode-level error reasons caused by infrastructure (LLM API down or
# unresponsive, feedback judge failed), not by the policy; such episodes
# are dropped entirely: no reward and no pool attempt, since both would be
# noise and corrupt the learnability estimates. "loop" is model behavior
# and stays penalizable.
INFRASTRUCTURE_REASONS = {"api", "timeout", "empty", "feedback"}

# fallback rollout step cap for the length-penalty ramp when scoring outside
# a training loop (standalone / tests); the loop passes rollout_config.max_steps
DEFAULT_MAX_STEPS = 30


class RewardConfig(BaseModel):
    # weight of the dominant term, the execution f1 of the student's final
    # query against the reference query
    w_f1: float = 1.0
    # flat penalty when the episode produced no executable final query
    # (missing output, unparseable or failing SPARQL, give-up -- but NOT a
    # step-limit truncation, see w_step_limit). Set to 1.0 so the reward is
    # symmetric [-1, +1] around the f1 range: giving up is as far below a
    # wrong-but-valid attempt (~0) as a perfect answer (+1) is above it -- a
    # strong anti-abstention gradient in any mixed group. It also anchors the
    # soft penalties below, which are all expressed as factors of it.
    w_invalid: float = 1.0
    # soft length penalty on VALID episodes: a linear ramp from 0 at
    # step_grace to its max at the rollout step cap (passed to compute_reward,
    # the single source of truth for max_steps), to prefer concise solutions
    # without overriding f1. max_step_penalty is a FACTOR in [0, 1]: the max
    # length penalty is max_step_penalty * w_invalid, i.e. a fraction of the
    # give-up penalty, so a factor < 1 guarantees even a max-length
    # valid-but-wrong answer stays above a give-up (no abstention incentive).
    # No function error term HERE. Bad calls are penalized per TURN, on the
    # advantage of the turn that made them (RlConfig.turn_error_penalty): an
    # episode-level sum would spread the same advantage over every turn, so the
    # model learns "episodes with bad calls score lower" without ever learning
    # WHICH call was bad. fn_errors stays on EpisodeReward as a diagnostic.
    max_step_penalty: float = Field(default=0.15, ge=0.0, le=1.0)
    step_grace: int = 20
    # penalty for burning the whole step budget without ever calling answer or
    # cancel. Above w_invalid so running out of steps ranks strictly below a
    # deliberate give-up: both fail, but the truncation also wasted the budget
    # and yields no usable query. Widens the reward range to [-w_step_limit, 1].
    # Only meaningful when the fallback outcome is off
    # (RolloutConfig.require_answer_call) and truncations are scored rather
    # than skipped (filter_step_limit=False).
    w_step_limit: float = 1.5
    # RESERVED, currently unused: reward for a correct abstention once
    # genuinely-unanswerable items exist (reference with an empty, non-ASK
    # result set). Today every pool reference is answerable, so an explicit
    # cancel without a working query is never correct and is scored exactly
    # like a broken answer (-w_invalid); see compute_reward. A cancel WITH a
    # working best attempt is scored like an answer either way: the returned
    # query is the final result.
    cancel_reward: float = 0.1
    # switch from assignment to exact multiset f1 above this result size
    exact_after: int = 1024
    # fix missing prefixes before executing, mirroring the tolerance of
    # execute_sparql during the episode; without this a query that worked
    # in the agent loop could unfairly count as invalid
    fix_prefixes: bool = True
    # sparql execution against the endpoints
    timeout: float = 300.0
    result_max_rows: int | None = 10_000_000
    # retries for transient endpoint failures (rate limiting, gateway
    # errors); observed with bursts of scoring queries on public endpoints
    retries: int = 2
    retry_delay: float = 5.0
    # endpoint overrides per kg; defaults to the kg info / qlever endpoint
    endpoints: dict[str, str] = {}


class EpisodeReward(BaseModel):
    item_id: str
    kg: str
    # None if the episode was skipped (see skip_reason)
    reward: float | None = None
    # f1 of the final query, None if none was produced or the episode
    # was skipped or cancelled
    f1: float | None = None
    type: str | None = None
    steps: int = 0
    # the length ramp charged to this episode (0 for invalid outcomes, which
    # are flat). Recorded so the process estimator can REDISTRIBUTE it into
    # per-turn increments without double-counting: reward already has it
    # subtracted, so adding it back gives the ramp-free base (see
    # rl.turn_returns and docs/11-process-supervision.md S3a)
    step_penalty: float = 0.0
    # client-attributable function call errors only
    fn_errors: int = 0
    # no executable final query
    invalid: bool = False
    # set if the episode is excluded from training and pool statistics:
    # an infrastructure error reason, a failing reference query, or a hard
    # server error on the prediction
    skip_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skip_reason is not None


# the length ramp as a function of steps used: 0 up to step_grace, then linear
# to max_step_penalty * w_invalid at the step cap. Exposed separately so the
# process estimator can redistribute it into per-turn increments that sum to
# exactly this value (docs/11-process-supervision.md S3a)
def length_penalty(steps: int, config: "RewardConfig", max_steps: int) -> float:
    ramp = max(1, max_steps - config.step_grace)
    return (
        config.max_step_penalty
        * config.w_invalid
        * min(1.0, max(0, steps - config.step_grace) / ramp)
    )


# episode total of the policy's own function call errors; the per-turn counts
# that drive the turn-level penalty come from the same primitive
def client_fn_errors(messages: list[dict]) -> int:
    errors = 0
    for message in messages:
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, dict):
            continue
        errors += client_tool_call_errors(content.get("tool_calls"))
    return errors


# executes queries per knowledge graph with optional prefix fixing,
# caching by (kg, sparql) so each reference query runs once per reward batch
class SparqlExecutor:
    def __init__(
        self,
        config: RewardConfig,
        logger: Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or get_logger("CQD REWARD")
        self.cache: dict[
            tuple[str, str],
            tuple[SelectResult | AskResult | None, str | None],
        ] = {}
        self.endpoints: dict[str, str] = dict(config.endpoints)
        self.prefixes: dict[str, dict[str, str]] = {}
        if config.fix_prefixes:
            self.sparql_parser = load_sparql_parser()
            self.iri_literal_parser = load_iri_and_literal_parser()

    def endpoint(self, kg: str) -> str:
        if kg not in self.endpoints:
            info = load_kg_info(kg)
            self.endpoints[kg] = info.endpoint or get_qlever_endpoint(kg)
        return self.endpoints[kg]

    def fix(self, sparql: str, kg: str) -> str:
        if not self.config.fix_prefixes:
            return sparql

        if kg not in self.prefixes:
            prefixes = get_common_sparql_prefixes()
            info = load_kg_info(kg)
            prefixes, _, _ = merge_prefixes(
                prefixes,
                info.prefixes or {},
                self.logger,
            )
            self.prefixes[kg] = prefixes

        try:
            return fix_prefixes(
                sparql,
                self.sparql_parser,
                self.iri_literal_parser,
                self.prefixes[kg],
            )
        except Exception:
            # leave unparseable queries as they are; execution reports
            # the actual error
            return sparql

    def execute(
        self,
        kg: str,
        sparql: str,
    ) -> tuple[SelectResult | AskResult | None, str | None]:
        key = (kg, sparql)
        if key in self.cache:
            return self.cache[key]

        fixed = self.fix(sparql, kg)
        endpoint = self.endpoint(kg)
        result, error = None, None
        attempts = 1 + max(0, self.config.retries)
        for attempt in range(attempts):
            result, error = get_result_or_error(
                fixed,
                endpoint,
                self.config.timeout,
                self.config.result_max_rows,
            )
            transient = error is not None and (
                is_server_error(error) or "<html" in error.lower()
            )
            if not transient or attempt == attempts - 1:
                break
            self.logger.warning(
                f"Transient endpoint failure, retrying in "
                f"{self.config.retry_delay}s: {error[:200]}"  # type: ignore[index]
            )
            time.sleep(self.config.retry_delay)

        self.cache[key] = (result, error)
        return self.cache[key]


def compute_reward(
    episode: Episode,
    config: RewardConfig,
    executor: SparqlExecutor,
    max_steps: int = DEFAULT_MAX_STEPS,
    filter_step_limit: bool = False,
) -> EpisodeReward:
    reward = EpisodeReward(
        item_id=episode.item_id,
        kg=episode.kg,
        type=episode.type,
        steps=episode.steps,
        fn_errors=client_fn_errors(episode.messages),
    )

    reason = (episode.error or {}).get("reason")
    if reason in INFRASTRUCTURE_REASONS:
        reward.skip_reason = reason
        return reward

    # no answer/cancel and no error. The agent loop has four exits: answer/
    # cancel (sets type), an error reason, hitting the step cap, or an
    # assistant message with no parseable tool call (nothing to run -> loop
    # ends). The last two both land here with type=None, error=None; only
    # `steps >= max_steps` distinguishes a genuine BUDGET TRUNCATION from an
    # EARLY DEGENERATE termination (the model produced no executable tool call
    # before the cap -- e.g. it dumped the call as raw text -- a malformed
    # output). An early degenerate exit is a POLICY failure and is never
    # filtered: it falls through to the invalid branch below (sparql is None ->
    # -w_invalid) so the model gets a gradient AWAY from breaking format --
    # filtering it, as an earlier version did, removed that signal and let
    # format collapse.
    truncated = (
        episode.type is None and episode.error is None and episode.steps >= max_steps
    )
    if truncated and filter_step_limit:
        # legacy behaviour, kept for comparison runs: budget truncations are
        # dropped from the gradient and the pool and dynamic sampling refills
        # around them. Off by default now -- see w_step_limit
        reward.skip_reason = "step-limit"
        return reward

    if truncated:
        # ran out of steps without ever answering or cancelling: no usable
        # query AND the whole budget spent, so it ranks below a give-up
        reward.invalid = True
        reward.reward = -config.w_step_limit
        return reward

    # soft length penalty, VALID outcomes only (see the invalid branches): a
    # linear ramp from 0 at step_grace to max_step_penalty*w_invalid at the
    # step cap (max_steps). A give-up scores -w_invalid, so with the factor
    # < 1 even a max-length valid-but-wrong answer stays strictly above it.
    step_penalty = length_penalty(reward.steps, config, max_steps)
    reward.step_penalty = step_penalty

    if episode.sparql is None:
        # No executable final query at all: an explicit cancel without a
        # best attempt, an answer without a query, a step-limit hit, or a
        # loop abort. Every pool reference is answerable, so giving up is
        # never correct. All invalid outcomes score a FLAT -w_invalid with
        # NO length penalty: among failures a short give-up must not score
        # better than a long failing attempt, or the policy learns to quit
        # fast (the give-up bias); the length penalty tie-breaks valid
        # outcomes only. (Correct abstention on a genuinely unanswerable item
        # would be rewarded here instead, but that requires executing the
        # reference first and is not yet wired; see cancel_reward.)
        reward.invalid = True
        reward.reward = -config.w_invalid
        return reward

    target, target_err = executor.execute(episode.kg, episode.reference_sparql)
    if target is None:
        # the reference was verified at proposal time, so a failure here is
        # an endpoint problem (or a stale reference, a curriculum concern);
        # either way not the student's fault
        reward.skip_reason = "reference-error"
        executor.logger.warning(
            f"Reference query for item {episode.item_id} failed: {target_err}"
        )
        return reward

    pred, pred_err = executor.execute(episode.kg, episode.sparql)
    if pred is None:
        if is_server_error(pred_err, hard_only=True):
            # endpoint down or overloaded, cannot score fairly
            reward.skip_reason = "server-error"
            return reward

        # broken or overly expensive query (parse error, timeout,
        # oversized result), whether from an answer or a cancel's best
        # attempt: the policy's fault, no better than a broken answer.
        # Flat -w_invalid (no penalty), like the no-query case above.
        reward.invalid = True
        reward.f1 = 0.0
        reward.reward = -config.w_invalid
        return reward

    reward.f1 = f1_score(pred, target, config.exact_after)
    reward.reward = config.w_f1 * reward.f1 - step_penalty
    return reward


# compute rewards for the given episodes and record one GroupRecord per
# item (its group of rollouts this round) on the pool. Grouping the whole
# round's episodes by item is what makes competence/learnability robust:
# the within-group reward std is captured at scoring time, never
# reconstructed from a flat attempt list. Skipped episodes are excluded.
def reward_episodes(
    episodes: list[Episode],
    pool: TaskPool,
    config: RewardConfig | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    progress: bool = False,
    logger: Logger | None = None,
    round: int | None = None,
    filter_step_limit: bool = False,
) -> list[EpisodeReward]:
    logger = logger or get_logger("CQD REWARD")
    config = config or RewardConfig()
    executor = SparqlExecutor(config, logger)

    rewards = []
    by_item: dict[str, list[EpisodeReward]] = {}
    for episode in tqdm(
        episodes,
        desc="Computing rewards",
        disable=not progress,
    ):
        reward = compute_reward(
            episode, config, executor, max_steps, filter_step_limit
        )
        rewards.append(reward)

        if reward.skipped:
            continue

        if episode.item_id in pool:
            by_item.setdefault(episode.item_id, []).append(reward)
        else:
            logger.warning(
                f"Episode item {episode.item_id} not in pool, attempt not recorded"
            )

    for item_id, group in by_item.items():
        pool.add_group(
            item_id,
            GroupRecord(
                # invalid / no-query rollouts contribute f1 0.0
                f1s=[r.f1 if r.f1 is not None else 0.0 for r in group],
                rewards=[r.reward for r in group if r.reward is not None],
                n_invalid=sum(r.invalid for r in group),
                round=round,
            ),
        )

    return rewards


def reward_stats(rewards: list[EpisodeReward]) -> dict:
    scored = [r for r in rewards if r.reward is not None]
    # honest QA f1: a scored episode with no final query (a give-up) counts
    # as 0, NOT excluded -- otherwise abstention inflates the mean (a model
    # that only answers the easy ones looks better than one that tries all).
    # mean_f1_answered keeps the conditional "when it answers, how good" view.
    f1_all = [r.f1 if r.f1 is not None else 0.0 for r in scored]
    answered_f1 = [r.f1 for r in scored if r.f1 is not None]
    skipped: dict[str, int] = {}
    for r in rewards:
        if r.skip_reason is not None:
            skipped[r.skip_reason] = skipped.get(r.skip_reason, 0) + 1

    return {
        "episodes": len(rewards),
        "scored": len(scored),
        "mean_reward": sum(r.reward or 0.0 for r in scored) / len(scored)
        if scored
        else None,
        "mean_f1": sum(f1_all) / len(f1_all) if f1_all else None,
        "mean_f1_answered": sum(answered_f1) / len(answered_f1)
        if answered_f1
        else None,
        "answered": sum(r.type == "answer" for r in scored),
        "cancelled": sum(r.type == "cancel" for r in scored),
        "invalid": sum(r.invalid for r in scored),
        "skipped": skipped,
    }


# phase 2 entry point: score collected episodes against the pool
# references and write the attempts back into the pool
def reward_episode_file(
    episodes_file: str,
    pool_file: str,
    config: RewardConfig | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    rewards_file: str | None = None,
    save_pool: bool = True,
    filter_step_limit: bool = False,
    log_level: str | int | None = None,
) -> list[EpisodeReward]:
    logger = get_logger("CQD REWARD", log_level)
    config = config or RewardConfig()

    episodes = [Episode(**e) for e in load_jsonl(episodes_file)]
    pool = TaskPool.load(pool_file)
    logger.info(
        f"Loaded {len(episodes)} episodes from {episodes_file} and "
        f"{len(pool)} pool items from {pool_file}"
    )

    rewards = reward_episodes(
        episodes,
        pool,
        config,
        max_steps,
        progress=True,
        logger=logger,
        filter_step_limit=filter_step_limit,
    )

    stats = reward_stats(rewards)
    logger.info(
        f"Scored {stats['scored']} of {stats['episodes']} episodes: "
        f"mean reward {stats['mean_reward']}, mean f1 {stats['mean_f1']}, "
        f"{stats['answered']} answered, {stats['cancelled']} cancelled, "
        f"{stats['invalid']} invalid, skipped {stats['skipped'] or 'none'}"
    )

    if save_pool:
        pool.save(pool_file)
        logger.info(f"Saved pool with recorded attempts to {pool_file}")

    if rewards_file is not None:
        dump_jsonl((r.model_dump() for r in rewards), rewards_file)
        logger.info(f"Saved rewards to {rewards_file}")

    return rewards
