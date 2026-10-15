# Episodes -> RL training data, straight from the token ids and logprobs vLLM
# returned per turn. Never re-render the trace with a chat template: GRASP
# strips response content on parsing and tokens merge across message
# boundaries, so a re-rendered trace is not what the policy generated.

from pydantic import BaseModel

from grasp.rl.rollout import Episode, assistant_turns


class RlTurn(BaseModel):
    # full context as the model saw it, rendered server-side by vLLM
    prompt_ids: list[int]
    # tokens the policy generated in this turn
    completion_ids: list[int]
    # logprobs of the completion tokens under the rollout policy
    logprobs: list[float]


def episode_turns(episode: Episode) -> list[RlTurn]:
    turns = []
    for content in assistant_turns(episode.messages):
        prompt_ids = content.get("prompt_token_ids")
        completion_ids = content.get("token_ids")
        logprobs = content.get("token_logprobs")
        if completion_ids is None:
            # synthetic turn the policy did not generate (forced examples)
            continue
        assert prompt_ids and logprobs, (
            "Assistant turn without token data; collect rollouts with "
            "token data enabled (see rollout.enable_token_data)"
        )
        assert len(completion_ids) == len(logprobs), (
            "Token ids and logprobs are misaligned"
        )
        turns.append(
            RlTurn(
                prompt_ids=prompt_ids, completion_ids=completion_ids, logprobs=logprobs
            )
        )
    return turns
