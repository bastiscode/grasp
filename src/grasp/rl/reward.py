import time
from logging import Logger

from pydantic import BaseModel, Field
from universal_ml_utils.logging import get_logger

from grasp.evaluate import get_result_or_error
from grasp.manager.utils import get_common_sparql_prefixes, load_kg_info, merge_prefixes
from grasp.rl.rollout import Episode, RlItem
from grasp.sparql.metrics import f1_score
from grasp.sparql.types import AskResult, SelectResult
from grasp.sparql.utils import (
    fix_prefixes,
    get_qlever_endpoint,
    load_iri_and_literal_parser,
    load_sparql_parser,
)
from grasp.utils import is_server_error

# episode error reasons caused by infrastructure (LLM API down, feedback
# judge failed), not by the policy; such episodes are dropped entirely.
# "loop" is model behavior and stays penalizable.
INFRASTRUCTURE_REASONS = {"api", "timeout", "empty", "feedback"}


class RewardConfig(BaseModel):
    # reward range is [-w_step_limit, 1]: a task score in [0, 1] for valid
    # outputs, a flat -w_invalid for no usable output (give-up, broken query)
    # so giving up is as far below a wrong answer as a perfect one is above,
    # and -w_step_limit for burning the budget without stopping
    w_invalid: float = 1.0
    w_step_limit: float = 1.5
    # per-error penalty for function calls the policy got wrong, capped below
    # w_invalid so a valid-but-wrong output always beats a give-up
    w_fn_error: float = Field(default=0.05, ge=0.0, le=1.0)
    max_fn_penalty: float = Field(default=0.35, ge=0.0, lt=1.0)

    # sparql-qa: switch from assignment to exact multiset f1 above this size
    exact_after: int = 1024
    # sparql-qa: fix missing prefixes before executing, mirroring the
    # tolerance of execute during the episode
    fix_prefixes: bool = True
    timeout: float = 300.0
    result_max_rows: int | None = 10_000_000
    # retries for transient endpoint failures
    retries: int = 2
    retry_delay: float = 5.0
    # endpoint overrides per kg; defaults to the kg info / qlever endpoint
    endpoints: dict[str, str] = {}


class EpisodeReward(BaseModel):
    item_id: str
    # None if the episode was skipped (see skip_reason)
    reward: float | None = None
    # task score in [0, 1], None if there was no usable output
    score: float | None = None
    steps: int = 0
    fn_errors: int = 0
    invalid: bool = False
    truncated: bool = False
    # set if the episode is excluded from training: infrastructure errors,
    # failing reference queries, or hard server errors on the prediction
    skip_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skip_reason is not None


# executes queries per kg with optional prefix fixing, caching by
# (kg, sparql) so each reference runs once per batch
class SparqlExecutor:
    def __init__(self, config: RewardConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.cache: dict[
            tuple[str, str], tuple[SelectResult | AskResult | None, str | None]
        ] = {}
        self.endpoints = dict(config.endpoints)
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
            prefixes, _, _ = merge_prefixes(
                get_common_sparql_prefixes(),
                load_kg_info(kg).prefixes or {},
                self.logger,
            )
            self.prefixes[kg] = prefixes

        try:
            return fix_prefixes(
                sparql, self.sparql_parser, self.iri_literal_parser, self.prefixes[kg]
            )
        except Exception:
            # leave unparseable queries as they are, execution reports the error
            return sparql

    def execute(
        self, kg: str, sparql: str
    ) -> tuple[SelectResult | AskResult | None, str | None]:
        key = (kg, sparql)
        if key in self.cache:
            return self.cache[key]

        fixed = self.fix(sparql, kg)
        result, error = None, None
        attempts = 1 + max(0, self.config.retries)
        for attempt in range(attempts):
            result, error = get_result_or_error(
                fixed,
                self.endpoint(kg),
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


# a task scorer maps (item, output) to (score, skip_reason): score in [0, 1],
# None for no usable output (invalid); skip_reason set if the episode cannot
# be scored fairly
class SparqlQaScorer:
    def __init__(self, config: RewardConfig, logger: Logger) -> None:
        self.config = config
        self.executor = SparqlExecutor(config, logger)

    def __call__(self, item: RlItem, output: dict) -> tuple[float | None, str | None]:
        sparql = output.get("sparql")
        if not sparql:
            # cancel without a best attempt or answer without a query; every
            # reference is answerable, so giving up is never correct
            return None, None

        assert item.kg is not None, f"sparql-qa item {item.id} has no kg"
        target, target_err = self.executor.execute(item.kg, item.sample.sparql)  # type: ignore[attr-defined]
        if target is None:
            self.executor.logger.warning(
                f"Reference of item {item.id} failed: {target_err}"
            )
            return None, "reference-error"

        # execute on the kg the agent picked, a wrong pick scores ~0
        pred, pred_err = self.executor.execute(output.get("kg") or item.kg, sparql)
        if pred is None:
            if is_server_error(pred_err, hard_only=True):
                return None, "server-error"
            # broken or too expensive query, the policy's fault
            return None, None

        return f1_score(pred, target, self.config.exact_after), None


class CeaScorer:
    def __init__(self, config: RewardConfig, logger: Logger) -> None:
        pass

    def __call__(self, item: RlItem, output: dict) -> tuple[float | None, str | None]:
        gt = {(a.row, a.column): a.identifier for a in item.sample.annotations}  # type: ignore[attr-defined]
        pred = {
            (a["row"], a["column"]): a["identifier"]
            for a in output.get("annotations", [])
        }
        if not gt:
            return float(not pred), None
        if not pred:
            return 0.0, None

        tp = sum(gt.get(key) == identifier for key, identifier in pred.items())
        if tp == 0:
            return 0.0, None
        precision = tp / len(pred)
        recall = tp / len(gt)
        return 2 * precision * recall / (precision + recall), None


SCORERS = {"sparql-qa": SparqlQaScorer, "cea": CeaScorer}


def compute_reward(
    episode: Episode,
    item: RlItem,
    scorer,
    config: RewardConfig,
    max_steps: int,
) -> EpisodeReward:
    reward = EpisodeReward(
        item_id=episode.item_id,
        steps=episode.steps,
        fn_errors=episode.fn_errors,
    )

    reason = (episode.error or {}).get("reason")
    if reason in INFRASTRUCTURE_REASONS:
        reward.skip_reason = reason
        return reward

    # budget truncation: no stop and no error at the step cap. An earlier
    # exit without output (e.g. a tool call dumped as raw text) is a format
    # failure and falls through to invalid below, never filtered.
    if episode.output is None and episode.error is None and episode.steps >= max_steps:
        reward.invalid = True
        reward.truncated = True
        reward.reward = -config.w_step_limit
        return reward

    score, skip_reason = (None, None)
    if episode.output is not None:
        score, skip_reason = scorer(item, episode.output)
    if skip_reason is not None:
        reward.skip_reason = skip_reason
        return reward

    if score is None:
        # invalid outcomes are flat: among failures a short give-up must not
        # beat a long attempt, or the policy learns to quit early
        reward.invalid = True
        reward.reward = -config.w_invalid
        return reward

    fn_penalty = min(config.w_fn_error * episode.fn_errors, config.max_fn_penalty)
    reward.score = score
    reward.reward = score - fn_penalty * config.w_invalid
    return reward


def reward_episodes(
    task: str,
    episodes: list[Episode],
    items: dict[str, RlItem],
    config: RewardConfig,
    max_steps: int,
    logger: Logger | None = None,
) -> list[EpisodeReward]:
    logger = logger or get_logger("RL REWARD")
    scorer = SCORERS[task](config, logger)
    return [
        compute_reward(episode, items[episode.item_id], scorer, config, max_steps)
        for episode in episodes
    ]
