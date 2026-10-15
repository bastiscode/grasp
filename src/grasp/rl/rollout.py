from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging import Logger
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm
from universal_ml_utils.logging import get_logger

from grasp.configs import GraspConfig
from grasp.core import generate
from grasp.examples import Sample
from grasp.manager import KgManager
from grasp.tasks import get_task
from grasp.utils import is_server_error


# a training or validation sample with a stable id (groups rollouts) and the
# kg its reference belongs to (needed to score sparql-qa, None for cea)
@dataclass
class RlItem:
    id: str
    sample: Sample
    kg: str | None = None


class Episode(BaseModel):
    item_id: str
    # task output, None if the episode produced none or did not end with a
    # successful call to a stopping function (see require_done_call)
    output: dict | None = None
    # ended by a successful (non-rejected) call to a stopping function
    done: bool = False
    steps: int = 0
    # function call errors that are the policy's fault (no server errors)
    fn_errors: int = 0
    elapsed: float = 0.0
    error: dict | None = None
    # full trace incl. per-token logprobs and ids on assistant messages
    messages: list[dict] = []


# function call errors in ONE assistant turn that are the policy's fault:
# unknown IRIs under know_before_use, unparseable SPARQL, bad arguments.
# Endpoint failures are excluded, the model cannot help those.
def client_tool_call_errors(tool_calls: list[dict] | None) -> int:
    errors = 0
    for tool_call in tool_calls or []:
        error = tool_call.get("error")
        if error is not None and not is_server_error(error):
            errors += 1
    return errors


def assistant_turns(messages: list[dict]) -> list[dict]:
    return [
        message["content"]
        for message in messages
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), dict)
    ]


def tool_calls(messages: list[dict]) -> list[dict]:
    return [
        tool_call
        for content in assistant_turns(messages)
        for tool_call in content.get("tool_calls") or []
    ]


# request per-token logprobs and ids from vLLM, required for RL training
def enable_token_data(config: GraspConfig) -> None:
    assert config.model_provider == "openai/completions", (
        "Token data requires the openai/completions model provider "
        "(e.g. a vLLM endpoint)"
    )
    config.model_kwargs["logprobs"] = True
    extra_body = config.model_kwargs.setdefault("extra_body", {})
    extra_body["return_token_ids"] = True


def run_episode(
    task: str,
    item: RlItem,
    config: GraspConfig,
    managers: list[KgManager],
    require_done_call: bool,
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    logger: Logger | None = None,
) -> Episode:
    generator = generate(
        task,
        item.sample.input(),
        config,
        managers,
        kg_notes,
        notes,
        logger=logger or get_logger("GRASP AGENT"),
    )
    output: dict[str, Any] = {}
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            output = stop.value
            break

    messages = output.get("messages") or []
    stop_fns = get_task(task, managers, config)
    done = any(
        stop_fns.done(call["name"]) and call.get("error") is None
        for call in tool_calls(messages)
    )
    task_output = output.get("output")
    # GRASP falls back to deriving an output from an unfinished trace (for
    # sparql-qa e.g. the last executed query). Training drops it so a step
    # limit hit is scored as the failure it is; validation keeps it to
    # measure the agent as deployed.
    if require_done_call and not done:
        task_output = None

    return Episode(
        item_id=item.id,
        output=task_output,
        done=done,
        steps=len(assistant_turns(messages)),
        fn_errors=sum(
            client_tool_call_errors(content.get("tool_calls"))
            for content in assistant_turns(messages)
        ),
        elapsed=output.get("elapsed", 0.0),
        error=output.get("error"),
        messages=messages,
    )


# run num_rollouts episodes per item, at most parallelism concurrently
# (bounds both the vLLM server and the KG endpoint load)
def collect_rollouts(
    task: str,
    items: list[RlItem],
    config: GraspConfig,
    managers: list[KgManager],
    num_rollouts: int,
    parallelism: int,
    require_done_call: bool,
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    logger: Logger | None = None,
) -> list[Episode]:
    logger = logger or get_logger("RL ROLLOUT")
    agent_logger = get_logger("GRASP AGENT")

    jobs = [item for item in items for _ in range(num_rollouts)]
    episodes = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(
                run_episode,
                task,
                item,
                config,
                managers,
                require_done_call,
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
            leave=False,
        ):
            try:
                episodes.append(future.result())
            except Exception as e:
                logger.exception(f"Episode for item {futures[future].id} failed: {e}")

    return episodes
