from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger

from pydantic import BaseModel
from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl
from universal_ml_utils.logging import get_logger

from grasp.configs import GraspConfig
from grasp.core import generate, load_notes, setup
from grasp.cqd.configs import RolloutConfig
from grasp.cqd.pool import PoolItem, TaskPool
from grasp.manager import KgManager
from grasp.utils import is_server_error


class Episode(BaseModel):
    item_id: str
    kg: str
    question: str
    reference_sparql: str
    # sparql-qa output type: answer or cancel, None if the run produced
    # no parseable output (e.g. step limit or api error)
    type: str | None = None
    # the student's final sparql query
    sparql: str | None = None
    steps: int = 0
    fn_errors: int = 0
    elapsed: float = 0.0
    error: dict | None = None
    # full trace incl. per-token logprobs and ids on assistant messages
    messages: list[dict] = []
    # function definitions the agent ran with, needed to re-render the
    # trace with a chat template for training
    functions: list[dict] | None = None


# did the episode end by explicitly calling answer or cancel? Everything else
# -- exhausting the step budget, an unparseable tool call, a query pasted as
# raw text -- is not an outcome, even where GRASP's fallback still derives a
# final query from it (see RolloutConfig.require_answer_call).
def called_answer_or_cancel(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        for tool_call in content.get("tool_calls") or []:
            if tool_call.get("name") in ("answer", "cancel"):
                return True
    return False


# function call errors in ONE assistant turn that are the policy's fault:
# unknown IRIs under know_before_use, unparseable SPARQL, bad arguments.
# Endpoint/server failures are excluded -- the model cannot help those, and
# charging for them would make the reward depend on KG uptime. Per-turn (not
# per-episode) so the penalty can be credited to the turn that earned it.
def client_tool_call_errors(tool_calls: list[dict] | None) -> int:
    errors = 0
    for tool_call in tool_calls or []:
        error = tool_call.get("error")
        if error is not None and not is_server_error(error):
            errors += 1
    return errors


def trace_stats(messages: list[dict]) -> tuple[int, int]:
    steps = 0
    fn_errors = 0
    for message in messages:
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, dict):
            continue
        steps += 1
        for tool_call in content.get("tool_calls") or []:
            fn_errors += tool_call.get("error") is not None
    return steps, fn_errors


# request per-token logprobs and ids, required for RL training
def enable_token_data(config: GraspConfig) -> None:
    assert config.model_provider == "openai/completions", (
        "Token data requires the openai/completions model provider "
        "(e.g. a vLLM endpoint)"
    )
    config.model_kwargs["logprobs"] = True
    extra_body = config.model_kwargs.setdefault("extra_body", {})
    extra_body["return_token_ids"] = True


def run_episode(
    item: PoolItem,
    config: RolloutConfig,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    logger: Logger | None = None,
) -> Episode:
    generator = generate(
        "sparql-qa",
        item.question,
        config,
        managers,
        kg_notes,
        notes,
        logger=logger or get_logger("GRASP AGENT"),
    )

    # consume the generator, capturing the function definitions from the
    # system event on the way
    functions = None
    while True:
        try:
            event = next(generator)
        except StopIteration as stop:
            output = stop.value
            break
        if event.get("type") == "system":
            functions = event.get("functions")

    task_output = output.get("output") or {}
    messages = output.get("messages") or []
    steps, fn_errors = trace_stats(messages)

    # drop a fallback-derived outcome, leaving type and sparql None so the
    # reward sees the episode for the non-answer it was
    if config.require_answer_call and not called_answer_or_cancel(messages):
        task_output = {}

    assert item.id is not None, "Pool item id must not be None"
    return Episode(
        item_id=item.id,
        kg=item.info.kg,
        question=item.question,
        reference_sparql=item.sparql,
        type=task_output.get("type"),
        sparql=task_output.get("sparql"),
        steps=steps,
        fn_errors=fn_errors,
        elapsed=output.get("elapsed", 0.0),
        error=output.get("error"),
        messages=messages,
        functions=functions,
    )


# run num_rollouts sparql-qa episodes per item, at most parallelism
# concurrently (each queries the KG endpoints, bounding endpoint load)
def collect_rollouts(
    items: list[PoolItem],
    config: RolloutConfig,
    managers: list[KgManager],
    num_rollouts: int = 1,
    parallelism: int = 4,
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    progress: bool = False,
    logger: Logger | None = None,
) -> list[Episode]:
    logger = logger or get_logger("CQD ROLLOUT")
    agent_logger = get_logger("GRASP AGENT")

    jobs = [item for item in items for _ in range(num_rollouts)]

    episodes = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(
                run_episode,
                item,
                config,
                managers,
                kg_notes,
                notes,
                agent_logger,
            ): item
            for item in jobs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(jobs),
            desc="Collecting rollouts",
            disable=not progress,
        ):
            try:
                episodes.append(future.result())
            except Exception as e:
                item = futures[future]
                logger.exception(f"Episode for item {item.id} failed: {e}")

    return episodes


# phase 2 entry point: sample pool items and collect student episodes
def collect_pool_rollouts(
    config: RolloutConfig,
    log_level: str | int | None = None,
) -> list[Episode]:
    logger = get_logger("CQD ROLLOUT", log_level)

    pool = TaskPool.load(config.pool_file)
    items = pool.sample(
        config.num_items if config.num_items is not None else len(pool),
        competence_half_life=config.competence_half_life,
        learnability_half_life=config.learnability_half_life,
        target_competence=config.sample_target_competence,
        new_fraction=config.new_fraction,
        seed=config.sample_seed,
    )
    logger.info(f"Sampled {len(items)} of {len(pool)} pool items")

    if config.token_data:
        enable_token_data(config)

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    episodes = collect_rollouts(
        items,
        config,
        managers,
        config.num_rollouts,
        config.parallelism,
        kg_notes,
        notes,
        progress=True,
        logger=logger,
    )

    answered = sum(e.type == "answer" for e in episodes)
    cancelled = sum(e.type == "cancel" for e in episodes)
    failed = sum(e.error is not None for e in episodes)
    mean_steps = sum(e.steps for e in episodes) / max(1, len(episodes))
    logger.info(
        f"Collected {len(episodes)} episodes: {answered} answered, "
        f"{cancelled} cancelled, {failed} with errors, "
        f"{mean_steps:.1f} mean steps"
    )

    if config.episodes_file is not None:
        dump_jsonl((e.model_dump() for e in episodes), config.episodes_file)
        logger.info(f"Saved episodes to {config.episodes_file}")

    return episodes
