from grasp.cqd.reward import EpisodeReward
from grasp.cqd.rollout import Episode
from grasp.cqd.train.hindsight import leaked, modulate, with_hint


def _rewards(item_ids):
    return [EpisodeReward(item_id=i, kg="freebase", reward=0.0) for i in item_ids]


def test_lambda_zero_is_untouched_grpo():
    advantages = [[1.0, -2.0], [0.5]]
    gaps = [[-3.0, -0.1], [-1.0]]
    out, metrics = modulate(advantages, gaps, _rewards(["a", "a"]), 0.0)
    assert out is advantages
    assert metrics == {}


def test_sign_is_never_flipped_and_factor_is_bounded():
    advantages = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    gaps = [[-9.0, 0.0, -0.01], [-4.0, -0.2, 0.0]]
    for lam in (0.25, 0.5, 1.0):
        out, _ = modulate(advantages, gaps, _rewards(["a", "a"]), lam)
        for original, scaled in zip(advantages, out):
            for a, b in zip(original, scaled):
                assert a * b >= 0.0
                assert abs(b) <= abs(a) * (1.0 + lam) + 1e-9


def test_salient_turn_is_scaled_up_relative_to_the_others():
    # turn 0 of the first episode has by far the largest |gap| in the group
    advantages = [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    gaps = [[-5.0, -0.1, -0.1], [-0.1, -0.1, -0.1]]
    out, metrics = modulate(advantages, gaps, _rewards(["a", "a"]), 0.5)
    assert abs(out[0][0]) > abs(out[0][1])
    assert abs(out[0][0]) > abs(out[1][0])
    assert metrics["sdpo/factor_max"] > metrics["sdpo/factor_min"]


def test_episodes_without_a_gap_pass_through():
    advantages = [[1.0, 1.0], None, [2.0]]
    gaps = [[-1.0, -0.1], [-1.0], None]
    out, _ = modulate(advantages, gaps, _rewards(["a", "a", "a"]), 0.5)
    assert out[1] is None
    assert out[2] == [2.0]


def test_length_mismatch_is_left_alone():
    advantages = [[1.0, 1.0]]
    gaps = [[-1.0]]
    out, _ = modulate(advantages, gaps, _rewards(["a"]), 0.5)
    assert out[0] == [1.0, 1.0]


def test_degenerate_group_has_no_spread_to_standardize():
    advantages = [[1.0, 1.0]]
    gaps = [[-2.0, -2.0]]
    out, _ = modulate(advantages, gaps, _rewards(["a"]), 0.5)
    assert out[0] == [1.0, 1.0]


def test_leak_check_allows_strategy_and_catches_identifiers():
    trace = "ns:m.0dr_9 Edward Scissorhands ns:film.film.starring ?x"
    assert leaked("Enumerate the object's properties before using it.", trace) == []
    assert leaked("You had ns:film.film.starring already; inspect it.", trace) == []
    assert leaked("Use ns:film.performance.actor instead.", trace) == [
        "ns:film.performance.actor"
    ]
    assert leaked("The issue is ns:film.performance.actor.", trace) == [
        "ns:film.performance.actor"
    ]
    assert leaked("Note: inspect it. Rules: go inverse.", trace) == []


def test_hint_goes_into_an_existing_system_message():
    chat = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "q"}]
    out = with_hint(chat, "Try the inverse direction.")
    assert out[0]["content"].startswith("You are an agent.")
    assert "Try the inverse direction." in out[0]["content"]
    assert len(out) == len(chat)
    assert chat[0]["content"] == "You are an agent."


def test_hint_creates_a_system_message_when_there_is_none():
    out = with_hint([{"role": "user", "content": "q"}], "Try the inverse direction.")
    assert out[0]["role"] == "system"
    assert len(out) == 2


def _episode(item_id):
    return Episode(
        item_id=item_id,
        kg="freebase",
        question="q",
        reference_sparql="SELECT ?x WHERE { ?x ?p ?o }",
    )


def test_generate_hints_reports_the_success_rate(monkeypatch):
    import grasp.cqd.train.hindsight as h

    calls = []

    def fake(config, episode, reward, attempts):
        calls.append(episode.item_id)
        return "" if episode.item_id == "b" else "a hint"

    monkeypatch.setattr(h, "hint_for_episode", fake)
    episodes = [_episode("a"), _episode("b"), _episode("c")]
    rewards = [
        EpisodeReward(item_id="a", kg="freebase", reward=0.0, f1=0.0),
        EpisodeReward(item_id="b", kg="freebase", reward=0.0, f1=0.5),
        EpisodeReward(item_id="c", kg="freebase", reward=0.0, f1=0.0),
    ]
    hints, metrics = h.generate_hints(None, episodes, rewards, parallelism=1)
    assert sorted(calls) == ["a", "b", "c"]
    assert metrics["sdpo/hints"] == 2
    assert metrics["sdpo/hint_rate"] == 2 / 3
    assert hints[1] == ""


def test_solved_and_skipped_rollouts_get_no_teacher_call(monkeypatch):
    import grasp.cqd.train.hindsight as h

    calls = []

    def fake(config, episode, reward, attempts):
        calls.append(episode.item_id)
        return "a hint"

    monkeypatch.setattr(h, "hint_for_episode", fake)
    episodes = [_episode("solved"), _episode("skipped"), _episode("failed")]
    rewards = [
        EpisodeReward(item_id="solved", kg="freebase", reward=1.0, f1=1.0),
        EpisodeReward(item_id="skipped", kg="freebase", reward=None),
        EpisodeReward(item_id="failed", kg="freebase", reward=0.0, f1=0.2),
    ]
    hints, metrics = h.generate_hints(None, episodes, rewards, parallelism=1)
    assert calls == ["failed"]
    assert metrics["sdpo/hint_rate"] == 1.0
    assert hints[0] == "" and hints[1] == ""


def test_a_failing_teacher_does_not_take_the_round_down(monkeypatch):
    import grasp.cqd.train.hindsight as h

    def boom(config, episode, reward, attempts):
        raise RuntimeError("429 budget_exceeded")

    monkeypatch.setattr(h, "hint_for_episode", boom)
    rewards = [EpisodeReward(item_id="a", kg="freebase", reward=0.0, f1=0.0)]
    hints, metrics = h.generate_hints(None, [_episode("a")], rewards, parallelism=1)
    assert hints == [""]
    assert metrics["sdpo/hint_rate"] == 0.0


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("chat", (), {"completions": completions})()


def _response(text):
    message = type("m", (), {"content": text})()
    return type("r", (), {"choices": [type("c", (), {"message": message})()]})()


def _teacher():
    from grasp.cqd.configs import CqdConfig

    config = CqdConfig(seeds_file="x", pool_file="x")
    config.model_kwargs = {"temperature": 0.7}
    return config


def test_transient_rate_limit_is_retried(monkeypatch):
    import grasp.cqd.train.hindsight as h

    completions = _FakeCompletions(
        [
            RuntimeError("Error code: 429 - budget_exceeded"),
            RuntimeError("Error code: 429 - budget_exceeded"),
            _response("Enumerate the object's properties first."),
        ]
    )
    monkeypatch.setattr(h, "OpenAI", lambda **kw: _FakeClient(completions))
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
    hint = h.hint_for_episode(_teacher(), _episode("a"), None)
    assert completions.calls == 3
    assert hint.startswith("Enumerate")


def test_a_permanent_error_is_not_retried(monkeypatch):
    import grasp.cqd.train.hindsight as h

    completions = _FakeCompletions([RuntimeError("Error code: 400 - bad request")])
    monkeypatch.setattr(h, "OpenAI", lambda **kw: _FakeClient(completions))
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
    try:
        h.hint_for_episode(_teacher(), _episode("a"), None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the 400 to propagate")
    assert completions.calls == 1


def test_retries_are_bounded(monkeypatch):
    import grasp.cqd.train.hindsight as h

    completions = _FakeCompletions([RuntimeError("429 budget_exceeded")] * 10)
    monkeypatch.setattr(h, "OpenAI", lambda **kw: _FakeClient(completions))
    monkeypatch.setattr(h.time, "sleep", lambda s: None)
    try:
        h.hint_for_episode(_teacher(), _episode("a"), None, attempts=3)
    except RuntimeError:
        pass
    assert completions.calls == 3
