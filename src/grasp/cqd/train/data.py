# Episodes -> training data. Two paths:
# - SFT (teacher episodes): re-render the trace with the student's template
#   and tokenizer (teacher tokens are from a different model); loss on
#   assistant tokens only. Re-rendering is template-canonical, not
#   byte-identical (GRASP strips response content on parsing), which is fine
#   for SFT and is why the RL path never re-renders.
# - RL (student episodes): vLLM returned exact prompt/completion token ids
#   and logprobs per turn (rollout.enable_token_data), used as-is.

import json

from pydantic import BaseModel

from grasp.cqd.rollout import Episode
from grasp.model import Message

IGNORE_INDEX = -100


# Episode messages in OpenAI chat format for chat templates. Unlike the
# inference-path conversion, reasoning is folded into the assistant text:
# thinking teachers put almost all text there, and a student distilled
# without it degenerates into bare tool-call loops (observed).
def episode_chat(episode: Episode) -> list[dict]:
    chat: list[dict] = []
    for message in episode.messages:
        msg = Message(**message)
        if isinstance(msg.content, str):
            chat.append({"role": msg.role, "content": msg.content})
            continue

        response = msg.content
        # reasoning first, then the regular message content
        text = "\n\n".join(response.get_content().values())
        entry: dict = {"role": msg.role, "content": text}
        if response.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args),
                    },
                }
                for tool_call in response.tool_calls
            ]
        chat.append(entry)

        for tool_call in response.tool_calls:
            assert tool_call.result is not None, "Expected tool call result"
            chat.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": tool_call.result,
                }
            )

    return chat


def chat_tools(episode: Episode) -> list[dict]:
    assert episode.functions is not None, (
        "Episode has no function definitions; it was collected before they "
        "were recorded (see Episode.functions)"
    )
    return [{"type": "function", "function": fn} for fn in episode.functions]


class SftSample(BaseModel):
    input_ids: list[int]
    # loss labels, IGNORE_INDEX on everything but assistant tokens
    labels: list[int]


# Tokenize with the chat template and mask everything but assistant tokens.
# Assistant spans are located at the character level via incremental prefix
# rendering (asserted to hold for ChatML-style templates), then mapped onto
# the full tokenization; char level is needed because tokens can merge
# across message boundaries.
def episode_sft_sample(episode: Episode, tokenizer) -> SftSample:
    chat = episode_chat(episode)
    tools = chat_tools(episode)

    def render(msgs: list[dict], generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            tokenize=False,
            add_generation_prompt=generation_prompt,
        )

    text = render(chat, False)

    spans = []
    for k, msg in enumerate(chat):
        if msg["role"] != "assistant":
            continue

        start = len(render(chat[:k], True))
        end = len(render(chat[: k + 1], False))
        assert text[:start] == render(chat[:k], True)[:start] and end <= len(
            text
        ), "Chat template does not render incrementally"
        # trailing newlines after the end-of-message token are template
        # glue for the next message, not something the model generates
        while end > start and text[end - 1] == "\n":
            end -= 1
        spans.append((start, end))

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    labels = [IGNORE_INDEX] * len(input_ids)

    for i, (token_start, token_end) in enumerate(encoded["offset_mapping"]):
        if any(token_end > start and token_start < end for start, end in spans):
            labels[i] = input_ids[i]

    return SftSample(input_ids=input_ids, labels=labels)


class RlTurn(BaseModel):
    # full context as the model saw it, rendered server-side by vLLM
    prompt_ids: list[int]
    # tokens the policy generated in this turn
    completion_ids: list[int]
    # logprobs of the completion tokens under the rollout policy
    logprobs: list[float]


# One sample per assistant turn, straight from the vLLM token data.
def episode_rl_turns(episode: Episode) -> list[RlTurn]:
    turns = []
    for message in episode.messages:
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, dict):
            continue

        prompt_ids = content.get("prompt_token_ids")
        completion_ids = content.get("token_ids")
        logprobs = content.get("token_logprobs")
        assert prompt_ids and completion_ids and logprobs, (
            "Assistant turn without token data; collect rollouts with "
            "token_data enabled (see rollout.enable_token_data)"
        )
        assert len(completion_ids) == len(logprobs), (
            "Token ids and logprobs are misaligned"
        )

        turns.append(
            RlTurn(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                logprobs=logprobs,
            )
        )

    return turns
