# Per-rollout hindsight for self-distillation advantage modulation.
#
# The teacher sees ONE rollout and its outcome and says what should have been
# done differently. Re-scoring the student's own tokens with that sentence
# prepended gives a per-turn gap: the turns whose likelihood moves most are the
# ones the hindsight is about, and their advantage is scaled up. The scale is
# bounded and never flips a sign, and sdpo_lambda 0 reproduces plain GRPO.
#
# Both sides of the gap are LOCAL forwards through the SAME renderer. data.py
# notes that re-rendering is template-canonical rather than byte-identical, and
# that this is why the RL path never re-renders; differencing a local hinted
# forward against vLLM's stored turn.logprobs would put that mismatch into
# every gap as a per-turn offset.
import math
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from logging import Logger

import torch
from universal_ml_utils.logging import get_logger

from grasp.cqd.configs import CqdConfig
from grasp.cqd.curriculum import render_trace
from grasp.cqd.reward import EpisodeReward
from grasp.cqd.rollout import Episode
from grasp.cqd.train.data import chat_tools, episode_chat
from openai import OpenAI

HINT_HEADER = "\n\nHindsight on the attempt below:\n"

HINT_SYSTEM = """You review one failed attempt at writing a SPARQL query for a question and say, in at most two sentences, what the agent should have done differently.

Rules:
- Describe STRATEGY, never content. Do not name any entity id, property id, IRI, or prefixed name that does not already appear in the trace.
- Point at the specific step that went wrong, not at the outcome.
- No preamble, no restating the question. Just the correction."""

# a hint naming an identifier the trace never showed took it from the
# reference. Such a hint moves the logprobs of the final query a lot while
# teaching nothing the student could have derived, so it is dropped.
IRI = re.compile(r"<[^>\s]+>|\b[A-Za-z][\w-]*:[A-Za-z][\w.-]*\b")


def leaked(hint: str, trace: str) -> list[str]:
    return sorted({m for m in IRI.findall(hint) if m not in trace})


# The OpenAI client is used directly rather than through grasp.model: Model.call
# always sends `tools=[...]`, and an empty tools array is rejected outright
# ("`tools` must not be an empty array"). Hindsight is one plain completion with
# no tools at all.
def hint_for_episode(
    config: CqdConfig, episode: Episode, reward, attempts: int = 5
) -> str:
    trace = render_trace(episode, reward)
    prompt = (
        f"Question: {episode.question}\n\n"
        "Reference query, for your eyes only -- do not reveal any identifier "
        f"from it:\n{episode.reference_sparql}\n\n"
        f"What the agent did:\n{trace}"
    )
    # max_retries=0: the SDK's own retries are silent and multiply a burst by 3
    # against a rate limit, which is the opposite of what helps. Backoff is
    # handled below, where it can be logged and paced.
    client = OpenAI(
        base_url=config.model_endpoint,
        api_key=config.model_api_key,
        timeout=config.model_timeout,
        max_retries=0,
    )
    response = None
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": HINT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                **config.model_kwargs,
            )
            break
        except Exception as e:
            transient = "429" in str(e) or "budget_exceeded" in str(e) or "503" in str(e)
            if not transient or attempt == attempts - 1:
                raise
            # jittered exponential backoff, so a whole round's worth of threads
            # does not retry in lockstep
            time.sleep((2**attempt) * (1.0 + random.random()))

    if response is None or not response.choices:
        return ""
    hint = (response.choices[0].message.content or "").strip()
    if not hint:
        return ""
    return "" if leaked(hint, trace) else hint


# one teacher call per rollout, run up to parallelism at once. Empty string
# where the teacher failed or leaked, so the caller can fall back to an
# unmodulated advantage for that episode.
def generate_hints(
    config: CqdConfig,
    episodes: list[Episode],
    rewards: list[EpisodeReward],
    parallelism: int = 4,
    logger: Logger | None = None,
    attempts: int = 5,
) -> tuple[list[str], dict]:
    logger = logger or get_logger("CQD HINDSIGHT")
    hints = [""] * len(episodes)

    todo = [
        i for i, reward in enumerate(rewards)
        if reward.reward is not None and (reward.f1 is None or reward.f1 < 1.0)
    ]
    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
        futures = {
            executor.submit(
                hint_for_episode, config, episodes[i], rewards[i], attempts
            ): i
            for i in todo
        }
        for future in futures:
            i = futures[future]
            try:
                hints[i] = future.result()
            except Exception as e:
                logger.warning(f"Hindsight for episode {episodes[i].item_id} failed: {e}")

    got = sum(1 for h in hints if h)
    rate = got / len(todo) if todo else 1.0
    # a collapse here turns the run into plain GRPO without changing anything
    # else, so it is an error rather than a warning
    log = logger.info if rate >= 0.5 else logger.error
    log(f"Hindsight for {got}/{len(todo)} failed rollouts (rate {rate:.2f})")
    return hints, {"sdpo/hint_rate": rate, "sdpo/hints": got}


def with_hint(chat: list[dict], hint: str) -> list[dict]:
    out = [dict(m) for m in chat]
    text = HINT_HEADER + hint
    if out and out[0]["role"] == "system":
        out[0]["content"] = (out[0]["content"] or "") + text
    else:
        out.insert(0, {"role": "system", "content": text.strip()})
    return out


# token ids for the rendered chat plus, per assistant message, the [start, end)
# token range. Char level like episode_sft_sample: tokens can merge across
# message boundaries.
def render_spans(chat: list[dict], tools: list[dict], tokenizer):
    def render(msgs: list[dict], generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=generation_prompt
        )

    text = render(chat, False)
    spans = []
    for k, msg in enumerate(chat):
        if msg["role"] != "assistant":
            continue
        start = len(render(chat[:k], True))
        end = len(render(chat[: k + 1], False))
        while end > start and text[end - 1] == "\n":
            end -= 1
        spans.append((start, end))

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    ranges = []
    for start, end in spans:
        idx = [
            i
            for i, (token_start, token_end) in enumerate(encoded["offset_mapping"])
            if token_end > start and token_start < end
        ]
        ranges.append((idx[0], idx[-1] + 1) if idx else None)
    return ids, ranges, text


# entry i is the logprob of ids[i + 1]. Chunked along the sequence: a full
# float log_softmax over a long trace costs several GB at these vocabularies,
# and it is needed twice per episode.
@torch.no_grad()
def token_logprobs(model, ids: list[int], chunk: int = 512) -> torch.Tensor:
    tokens = torch.tensor([ids], device=model.device)
    logits = model(input_ids=tokens).logits[0, :-1]
    target = tokens[0, 1:]
    out = []
    for i in range(0, logits.shape[0], chunk):
        logprobs = torch.log_softmax(logits[i : i + chunk].float(), dim=-1)
        out.append(logprobs.gather(-1, target[i : i + chunk, None]).squeeze(-1))
        del logprobs
    del logits
    return torch.cat(out)


def span_logprobs(logprobs: torch.Tensor, span) -> torch.Tensor | None:
    start, end = span
    start = max(start, 1)
    if end <= start:
        return None
    return logprobs[start - 1 : end - 1]


# mean per-token gap for each assistant turn, or None if the episode cannot be
# rendered or the hinted rendering does not reproduce the assistant tokens.
# Per token rather than summed so a long turn is not salient by length alone.
def turn_gaps(model, tokenizer, episode: Episode, hint: str) -> list[float] | None:
    if not hint or episode.functions is None:
        return None
    try:
        chat = episode_chat(episode)
        tools = chat_tools(episode)
        ids, spans, _ = render_spans(chat, tools, tokenizer)
        hinted_ids, hinted_spans, _ = render_spans(
            with_hint(chat, hint), tools, tokenizer
        )
    except Exception:
        return None

    if len(spans) != len(hinted_spans):
        return None

    base = token_logprobs(model, ids)
    hinted = token_logprobs(model, hinted_ids)

    gaps: list[float] = []
    for span, hinted_span in zip(spans, hinted_spans):
        if span is None or hinted_span is None:
            return None
        if ids[span[0] : span[1]] != hinted_ids[hinted_span[0] : hinted_span[1]]:
            return None
        a = span_logprobs(base, span)
        b = span_logprobs(hinted, hinted_span)
        if a is None or b is None or len(a) != len(b):
            return None
        gaps.append(float((b - a).mean()))
    return gaps


# scale each turn's advantage by 1 + lam * tanh(z), where z standardizes the
# turn's |gap| within its item's group. |gap| is a saliency: the turn the
# hindsight moves most gets more of the episode's credit or blame, in whichever
# direction the advantage already points. |lam| <= 1 keeps the factor
# non-negative, so the sign can never flip, and lam 0 is exactly GRPO.
def modulate(
    turn_advantages: list[list[float] | None],
    gaps: list[list[float] | None],
    rewards: list[EpisodeReward],
    lam: float,
) -> tuple[list[list[float] | None], dict]:
    if lam == 0.0:
        return turn_advantages, {}

    by_item: dict[str, list[float]] = {}
    for reward, episode_gaps in zip(rewards, gaps):
        if episode_gaps is None:
            continue
        by_item.setdefault(reward.item_id, []).extend(abs(g) for g in episode_gaps)

    stats = {}
    for item, values in by_item.items():
        mean = sum(values) / len(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[item] = (mean, stdev)

    out: list[list[float] | None] = []
    factors: list[float] = []
    modulated = 0
    for reward, advantages, episode_gaps in zip(rewards, turn_advantages, gaps):
        if advantages is None or episode_gaps is None:
            out.append(advantages)
            continue
        mean, stdev = stats.get(reward.item_id, (0.0, 0.0))
        if stdev <= 0.0 or len(episode_gaps) != len(advantages):
            out.append(advantages)
            continue
        scaled = []
        for advantage, gap in zip(advantages, episode_gaps):
            factor = 1.0 + lam * math.tanh((abs(gap) - mean) / stdev)
            factors.append(factor)
            scaled.append(advantage * factor)
        out.append(scaled)
        modulated += 1

    metrics = {}
    if factors:
        metrics = {
            "sdpo/episodes_modulated": modulated,
            "sdpo/factor_mean": sum(factors) / len(factors),
            "sdpo/factor_min": min(factors),
            "sdpo/factor_max": max(factors),
        }
    return out, metrics
