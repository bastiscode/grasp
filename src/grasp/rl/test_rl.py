from universal_ml_utils.logging import get_logger

from grasp.rl.data import episode_turns
from grasp.rl.grpo import episode_stats, group_advantages
from grasp.rl.reward import (
    CeaScorer,
    EpisodeReward,
    RewardConfig,
    SparqlQaScorer,
    compute_reward,
)
from grasp.rl.rollout import Episode, RlItem
from grasp.sparql.types import SelectResult
from grasp.tasks.cea import CeaSample
from grasp.tasks.sparql_qa.examples import SparqlQaSample

CONFIG = RewardConfig(fix_prefixes=False)
MAX_STEPS = 10


def sparql_item(id: str = "a") -> RlItem:
    return RlItem(id=id, sample=SparqlQaSample(question="q", sparql="REF"), kg="kg")


def assistant(tool_calls: list[dict], token_data: bool = True) -> dict:
    content: dict = {"id": "r", "tool_calls": tool_calls}
    if token_data:
        content.update(
            prompt_token_ids=[1, 2], token_ids=[3, 4], token_logprobs=[-0.1, -0.2]
        )
    return {"role": "assistant", "content": content}


# scorer whose executor returns canned results instead of hitting an endpoint
def fake_sparql_scorer(results: dict) -> SparqlQaScorer:
    scorer = SparqlQaScorer(CONFIG, get_logger("TEST"))
    scorer.executor.execute = lambda kg, sparql: results[sparql]  # type: ignore[method-assign]
    return scorer


def select(*values: str) -> SelectResult:
    return SelectResult(
        variables=["x"], data=[{"x": {"type": "literal", "value": v}} for v in values]
    )


def reward_of(episode: Episode, results: dict | None = None) -> EpisodeReward:
    scorer = fake_sparql_scorer(results or {})
    return compute_reward(episode, sparql_item(), scorer, CONFIG, MAX_STEPS)


def test_correct_answer_scores_one():
    results = {"REF": (select("1"), None), "PRED": (select("1"), None)}
    r = reward_of(Episode(item_id="a", output={"sparql": "PRED"}, done=True), results)
    assert r.score == 1.0 and r.reward == 1.0 and not r.invalid


def test_fn_errors_are_penalized_and_capped():
    results = {"REF": (select("1"), None), "PRED": (select("1"), None)}
    r = reward_of(Episode(item_id="a", output={"sparql": "PRED"}, fn_errors=2), results)
    assert abs(r.reward - 0.9) < 1e-9
    r = reward_of(
        Episode(item_id="a", output={"sparql": "PRED"}, fn_errors=100), results
    )
    # a wrong-but-valid answer with many errors still beats a give-up
    assert abs(r.reward - (1.0 - CONFIG.max_fn_penalty)) < 1e-9
    assert r.reward > -CONFIG.w_invalid


def test_give_up_and_broken_query_are_flat_invalid():
    r = reward_of(Episode(item_id="a", output={"type": "cancel"}, steps=2))
    assert r.invalid and r.reward == -CONFIG.w_invalid
    results = {"REF": (select("1"), None), "PRED": (None, "syntax error")}
    r = reward_of(Episode(item_id="a", output={"sparql": "PRED"}, steps=9), results)
    assert r.invalid and r.reward == -CONFIG.w_invalid


def test_step_limit_ranks_below_give_up():
    r = reward_of(Episode(item_id="a", steps=MAX_STEPS))
    assert r.truncated and r.reward == -CONFIG.w_step_limit


def test_early_exit_without_output_is_invalid_not_truncated():
    r = reward_of(Episode(item_id="a", steps=3))
    assert r.invalid and not r.truncated and r.reward == -CONFIG.w_invalid


def test_infrastructure_errors_and_failing_references_are_skipped():
    r = reward_of(Episode(item_id="a", error={"reason": "api"}))
    assert r.skip_reason == "api" and r.reward is None
    results = {"REF": (None, "timeout")}
    r = reward_of(Episode(item_id="a", output={"sparql": "PRED"}), results)
    assert r.skip_reason == "reference-error"


def test_loop_error_is_penalized_not_skipped():
    r = reward_of(Episode(item_id="a", error={"reason": "loop"}, steps=4))
    assert not r.skipped and r.invalid


def test_cea_f1():
    sample = CeaSample(
        table={"header": ["a"], "data": [["x"], ["y"]]},
        annotations=[
            {"row": 0, "column": 0, "identifier": "Q1", "entity": "x"},
            {"row": 1, "column": 0, "identifier": "Q2", "entity": "y"},
        ],
    )
    item = RlItem(id="t", sample=sample)
    scorer = CeaScorer(CONFIG, get_logger("TEST"))
    output = {"annotations": [{"row": 0, "column": 0, "identifier": "Q1"}]}
    score, skip = scorer(item, output)
    assert skip is None and abs(score - 2 / 3) < 1e-9
    assert scorer(item, {"annotations": []}) == (0.0, None)


def test_group_advantages_skip_flat_groups_and_skipped_episodes():
    rewards = [
        EpisodeReward(item_id="a", reward=1.0),
        EpisodeReward(item_id="a", reward=0.0),
        EpisodeReward(item_id="a", reward=None, skip_reason="api"),
        EpisodeReward(item_id="b", reward=0.5),
        EpisodeReward(item_id="b", reward=0.5),
    ]
    advantages = group_advantages(rewards)
    assert advantages[0] == 1.0 and advantages[1] == -1.0
    assert advantages[2:] == [None, None, None]


def test_episode_turns_skip_synthetic_turns():
    episode = Episode(
        item_id="a",
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            assistant([{"name": "search_example", "args": {}}], token_data=False),
            assistant([{"name": "answer", "args": {}}]),
        ],
    )
    turns = episode_turns(episode)
    assert len(turns) == 1 and turns[0].completion_ids == [3, 4]


def test_episode_stats_count_functions_per_episode():
    episodes = [
        Episode(
            item_id="a",
            output={"type": "answer"},
            steps=2,
            messages=[
                assistant(
                    [{"name": "execute", "args": {}}, {"name": "execute", "args": {}}]
                ),
                assistant([{"name": "answer", "args": {}}]),
            ],
        ),
        Episode(
            item_id="a", steps=1, messages=[assistant([{"name": "answer", "args": {}}])]
        ),
    ]
    rewards = [
        EpisodeReward(item_id="a", reward=1.0, score=1.0),
        EpisodeReward(item_id="a", reward=-1.0, invalid=True),
    ]
    stats = episode_stats(episodes, rewards)
    assert stats["env/fn/execute"] == 0.5 and stats["env/fn/answer"] == 1.0
    assert stats["score"] == 0.5 and stats["score_valid"] == 1.0
    assert stats["env/type/answer"] == 0.5
