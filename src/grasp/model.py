from litellm import completion
from litellm.types.utils import ModelResponse

from grasp.configs import Config


def call_model(
    messages: list[dict],
    functions: list[dict],
    config: Config,
) -> ModelResponse:
    return completion(
        model=config.model,
        messages=messages,
        tools=[{"type": "function", "function": fn} for fn in functions],
        parallel_tool_calls=False,
        tool_choice="auto",
        # decoding parameters
        temperature=config.temperature,
        top_p=config.top_p,
        reasoning_effort=config.reasoning_effort,  # type: ignore
        # should be set to more than enough until the next function call
        max_completion_tokens=config.max_completion_tokens,
        base_url=config.model_endpoint,
        timeout=config.completion_timeout,
        seed=config.seed,
        # drop unsupported parameters
        drop_params=True,
    )
