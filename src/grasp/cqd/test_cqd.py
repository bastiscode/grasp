import random

import pytest
from universal_ml_utils.logging import get_logger

from grasp.cqd.pool import (
    GroupRecord,
    PoolItem,
    PoolItemInfo,
    TaskPool,
    draw_weight,
    ewma,
    item_id,
    weighted_sample,
    zpd,
)
from grasp.cqd.curriculum import select_revisits
from grasp.cqd.reward import (
    EpisodeReward,
    RewardConfig,
    compute_reward,
    reward_episodes,
    reward_stats,
)
from grasp.cqd.configs import RolloutConfig
from grasp.cqd.rollout import Episode
from grasp.sparql.types import AskResult
from grasp.tasks.cq_distillation.functions import Proposal


def make_item(
    id: str,
    groups: list[GroupRecord] | None = None,
    kg: str = "wikidata",
    question: str | None = None,
    sparql: str = "ASK {}",
    cq_id: str | None = None,
    parent_id: str | None = None,
) -> PoolItem:
    return PoolItem(
        id=id,
        question=question or f"question {id}",
        sparql=sparql,
        info=PoolItemInfo(
            kg=kg, cq_id=cq_id, parent_id=parent_id, groups=groups or []
        ),
    )


def group(f1s: list[float], rewards: list[float], round: int | None = None) -> GroupRecord:
    return GroupRecord(f1s=f1s, rewards=rewards, round=round)


# ---------------------------------------------------------------------------
# ewma
# ---------------------------------------------------------------------------


def test_ewma_empty_is_none():
    assert ewma([], 1.5) is None


def test_ewma_single_value_is_that_value():
    assert ewma([0.7], 1.5) == 0.7


def test_ewma_constant_sequence_is_the_constant():
    assert ewma([0.4, 0.4, 0.4], 2.0) == pytest.approx(0.4)


def test_ewma_half_life_one_gives_alpha_half():
    # half_life=1 -> alpha = 1 - 0.5**1 = 0.5; fold [0, 1]:
    # e=0 -> e=0.5*1 + 0.5*0 = 0.5
    assert ewma([0.0, 1.0], 1.0) == pytest.approx(0.5)


def test_ewma_weights_recent_observations_more():
    # the 1 is newest in the first, oldest in the second
    recent_high = ewma([0.0, 0.0, 1.0], 1.0)
    recent_low = ewma([1.0, 0.0, 0.0], 1.0)
    assert recent_high > recent_low


def test_ewma_longer_half_life_adapts_more_slowly():
    # same jump [0, 1]; the longer half-life keeps more weight on the old value
    fast = ewma([0.0, 1.0], 1.0)
    slow = ewma([0.0, 1.0], 3.0)
    assert slow < fast


# ---------------------------------------------------------------------------
# PoolItem stats
# ---------------------------------------------------------------------------


def test_new_item_has_no_competence_or_learnability():
    item = make_item("cqd-new")
    assert item.n_groups() == 0
    assert item.competence(1.5) is None
    assert item.learnability(3.0) is None


def test_competence_is_ewma_of_per_group_mean_f1():
    item = make_item("cqd-a", [group([0.0, 0.0], [0.0, 0.0]), group([1.0, 1.0], [1.0, 1.0])])
    # per-group means [0.0, 1.0]; ewma with hl=1 -> 0.5
    assert item.competence(1.0) == pytest.approx(0.5)


def test_learnability_is_ewma_of_per_group_reward_std():
    # constant rewards -> zero spread -> zero learnability
    dead = make_item("cqd-dead", [group([0.5, 0.5], [0.5, 0.5])])
    assert dead.learnability(3.0) == pytest.approx(0.0)
    # spread rewards -> positive learnability
    live = make_item("cqd-live", [group([0.0, 1.0], [-0.5, 0.5])])
    assert live.learnability(3.0) > 0.0


def test_competence_skips_empty_f1_groups():
    item = make_item("cqd-b", [group([], []), group([0.8, 0.8], [0.8, 0.8])])
    assert item.competence(1.5) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# zpd / draw_weight
# ---------------------------------------------------------------------------


def test_zpd_peaks_at_target_and_damps_the_extremes():
    assert zpd(0.5, 0.5) == 1.0
    # symmetric around the target, and monotonically falling away from it
    assert zpd(0.3, 0.5) == pytest.approx(zpd(0.7, 0.5))
    assert zpd(0.5, 0.5) > zpd(0.35, 0.5) > zpd(0.1, 0.5)
    # never actually zero: a stuck (or mastered) item stays reachable
    assert 0.0 < zpd(0.0, 0.5) < 0.2
    assert 0.0 < zpd(1.0, 0.5) < 0.2


def test_draw_weight_prefers_the_ready_item_over_an_equally_learnable_hard_one():
    # identical reward spread (learnability), different competence
    ready = make_item("cqd-ready", [group([0.5, 0.5], [0.6, -0.4])])
    hard = make_item("cqd-hard", [group([0.02, 0.02], [0.6, -0.4])])
    w_ready = draw_weight(ready, 1.5, 3.0, 0.5)
    w_hard = draw_weight(hard, 1.5, 3.0, 0.5)
    assert w_ready > w_hard
    # difficulty-blind weighting cannot tell them apart -- the old behaviour
    assert draw_weight(ready, 1.5, 3.0, None) == draw_weight(hard, 1.5, 3.0, None)


def test_draw_weight_damps_mastered_items():
    mastered = make_item("cqd-done", [group([1.0, 1.0], [0.6, -0.4])])
    ready = make_item("cqd-ready", [group([0.5, 0.5], [0.6, -0.4])])
    assert draw_weight(ready, 1.5, 3.0, 0.5) > draw_weight(mastered, 1.5, 3.0, 0.5)


def test_draw_weight_is_zero_without_learnability():
    # no reward spread -> no GRPO gradient, regardless of competence
    flat = make_item("cqd-flat", [group([0.5, 0.5], [0.3, 0.3])])
    assert draw_weight(flat, 1.5, 3.0, 0.5) == 0.0


def test_sample_prefers_ready_items_over_hard_ones(monkeypatch):
    # a crowded band of hard-but-learnable items plus one ready item: the
    # ready item should be drawn far more often than its 1-in-6 share
    pool = TaskPool(
        [make_item(f"cqd-hard{i}", [group([0.02, 0.02], [0.6, -0.4])]) for i in range(5)]
        + [make_item("cqd-ready", [group([0.5, 0.5], [0.6, -0.4])])]
    )
    picks = sum(
        "cqd-ready" in {it.id for it in pool.sample(1, new_fraction=0.0, seed=s)}
        for s in range(60)
    )
    assert picks > 20  # uniform would be ~10


# ---------------------------------------------------------------------------
# weighted_sample
# ---------------------------------------------------------------------------


def test_weighted_sample_returns_distinct_items():
    items = [make_item(f"cqd-{i}") for i in range(5)]
    picked = weighted_sample(random.Random(1), items, [1.0] * 5, 3)
    assert len(picked) == 3
    assert len({it.id for it in picked}) == 3


def test_weighted_sample_caps_at_available():
    items = [make_item(f"cqd-{i}") for i in range(2)]
    picked = weighted_sample(random.Random(1), items, [1.0, 1.0], 5)
    assert len(picked) == 2


def test_weighted_sample_favors_high_weight():
    items = [make_item("cqd-lo"), make_item("cqd-hi")]
    # near-zero weight on the first: single draw should nearly always pick hi
    picks = [
        weighted_sample(random.Random(s), items, [1e-9, 1.0], 1)[0].id
        for s in range(20)
    ]
    assert picks.count("cqd-hi") >= 18


# ---------------------------------------------------------------------------
# item_id
# ---------------------------------------------------------------------------


def test_item_id_is_deterministic_and_content_derived():
    a = item_id("wikidata", "q", "ASK {}")
    assert a == item_id("wikidata", "q", "ASK {}")
    assert a != item_id("wikidata", "q2", "ASK {}")
    assert a != item_id("dbpedia", "q", "ASK {}")
    assert a.startswith("cqd-") and len(a) == len("cqd-") + 16


# ---------------------------------------------------------------------------
# TaskPool bookkeeping
# ---------------------------------------------------------------------------


def proposal(question: str, sparql: str = "ASK {}", cq_id=None, parent_id=None) -> Proposal:
    return Proposal(
        kg="wikidata",
        question=question,
        sparql=sparql,
        intent="i",
        difficulty="similar",
        cq_id=cq_id,
        parent_id=parent_id,
    )


def test_add_proposals_dedupes_identical_content():
    pool = TaskPool()
    added = pool.add_proposals([proposal("q1"), proposal("q1")])
    assert len(added) == 1
    assert len(pool) == 1
    # re-adding the same proposal is a no-op
    assert pool.add_proposals([proposal("q1")]) == []
    assert len(pool) == 1


def test_add_group_appends_in_order():
    pool = TaskPool([make_item("cqd-a")])
    pool.add_group("cqd-a", group([0.1, 0.2], [0.1, 0.2], round=0))
    pool.add_group("cqd-a", group([0.3, 0.4], [0.3, 0.4], round=1))
    rounds = [g.round for g in pool["cqd-a"].info.groups]
    assert rounds == [0, 1]


def test_by_cq_and_children():
    pool = TaskPool(
        [
            make_item("cqd-root", cq_id="cq1"),
            make_item("cqd-child", cq_id="cq1", parent_id="cqd-root"),
            make_item("cqd-other", cq_id="cq2"),
        ]
    )
    assert {it.id for it in pool.by_cq("cq1")} == {"cqd-root", "cqd-child"}
    assert [it.id for it in pool.children("cqd-root")] == ["cqd-child"]


def test_lineage_walks_parents_and_breaks_cycles():
    pool = TaskPool(
        [
            make_item("cqd-1"),
            make_item("cqd-2", parent_id="cqd-1"),
            make_item("cqd-3", parent_id="cqd-2"),
        ]
    )
    assert [it.id for it in pool.lineage("cqd-3")] == ["cqd-3", "cqd-2", "cqd-1"]
    # a self-cycle must terminate rather than loop forever
    pool["cqd-1"].info.parent_id = "cqd-1"
    assert pool.lineage("cqd-1") == [pool["cqd-1"]]


def test_load_save_roundtrip(tmp_path):
    pool = TaskPool([make_item("cqd-a", [group([0.5, 0.5], [0.5, 0.5], round=0)])])
    file = str(tmp_path / "pool.jsonl")
    pool.save(file)
    loaded = TaskPool.load(file)
    assert len(loaded) == 1
    assert loaded["cqd-a"].info.groups[0].rewards == [0.5, 0.5]


# ---------------------------------------------------------------------------
# TaskPool.sample
# ---------------------------------------------------------------------------


def build_mixed_pool() -> TaskPool:
    # 4 never-attempted, 4 learnable (spread across competence bins,
    # nonzero reward spread), 3 dead (zero reward spread -> learnability 0)
    items = [make_item(f"cqd-new{i}") for i in range(4)]
    for i, mean in enumerate([0.1, 0.4, 0.6, 0.9]):
        items.append(
            make_item(f"cqd-live{i}", [group([mean, mean], [mean, -mean])])
        )
    for i in range(3):
        items.append(make_item(f"cqd-dead{i}", [group([0.5, 0.5], [0.3, 0.3])]))
    return TaskPool(items)


def test_sample_returns_all_when_n_at_least_pool_size():
    pool = build_mixed_pool()
    picked = pool.sample(len(pool) + 5, seed=0)
    assert len(picked) == len(pool)


def test_sample_skips_zero_learnability_items():
    pool = build_mixed_pool()
    # no exploration slots, and 4 learnable items exactly cover n=4, so the
    # backfill never fires and dead (zero-variance) items are never drawn
    picked = pool.sample(4, new_fraction=0.0, seed=0)
    ids = {it.id for it in picked}
    assert len(ids) == 4
    assert not any(id.startswith("cqd-dead") for id in ids)
    assert all(id.startswith("cqd-live") for id in ids)


def test_sample_reserves_new_exploration_slots():
    pool = build_mixed_pool()
    picked = pool.sample(8, new_fraction=0.25, seed=0)
    ids = [it.id for it in picked]
    assert len(ids) == len(set(ids)) == 8
    # round(0.25 * 8) = 2 reserved new items (more may arrive via backfill)
    assert sum(id.startswith("cqd-new") for id in ids) >= 2


def test_sample_is_deterministic_for_a_seed():
    pool = build_mixed_pool()
    a = [it.id for it in pool.sample(6, seed=42)]
    b = [it.id for it in pool.sample(6, seed=42)]
    assert a == b


def test_sample_backfills_to_n_when_signal_is_scarce():
    # only 1 learnable item but 6 dead ones; still must return n distinct
    items = [make_item("cqd-live", [group([0.5, 0.5], [0.4, -0.4])])]
    items += [make_item(f"cqd-dead{i}", [group([0.5, 0.5], [0.2, 0.2])]) for i in range(6)]
    pool = TaskPool(items)
    picked = pool.sample(5, new_fraction=0.0, seed=0)
    ids = {it.id for it in picked}
    assert len(ids) == 5
    assert "cqd-live" in ids  # the only learnable item is drawn first


def test_sample_backfill_prefers_learnable_over_dead():
    # only 3 items carry a weight (the dead ones have no reward spread), so a
    # batch of 4 exhausts them and the backfill supplies the 4th. The weighted
    # path must take every learnable item before touching a dead one.
    items = [make_item(f"cqd-live{i}", [group([0.5, 0.5], [0.5, -0.5])]) for i in range(3)]
    items += [make_item(f"cqd-dead{i}", [group([0.5, 0.5], [0.3, 0.3])]) for i in range(3)]
    pool = TaskPool(items)
    picked = pool.sample(4, new_fraction=0.0, seed=0)
    ids = {it.id for it in picked}
    assert len(ids) == 4
    assert sum(id.startswith("cqd-live") for id in ids) == 3
    assert sum(id.startswith("cqd-dead") for id in ids) == 1


def test_sample_backfill_prefers_new_over_dead():
    # never-attempted items have unknown signal, not proven-flat, so in the
    # backfill they must be preferred over proven-dead items (given a small
    # positive weight), not treated as equally uniform.
    items = [make_item("cqd-live", [group([0.5, 0.5], [0.5, -0.5])])]
    items += [make_item(f"cqd-new{i}") for i in range(3)]
    items += [make_item(f"cqd-dead{i}", [group([0.5, 0.5], [0.3, 0.3])]) for i in range(3)]
    pool = TaskPool(items)
    # new_fraction=0 so new items get no reserved slot and only reach the
    # backfill; the 1 learnable fills the quota, then the 3 new fill the rest
    picked = pool.sample(4, new_fraction=0.0, seed=0)
    ids = {it.id for it in picked}
    assert len(ids) == 4
    assert sum(id.startswith("cqd-new") for id in ids) == 3
    assert sum(id.startswith("cqd-dead") for id in ids) == 0


# ---------------------------------------------------------------------------
# reward shaping (compute_reward)
# ---------------------------------------------------------------------------


class FakeExecutor:
    # stub standing in for SparqlExecutor: canned (result, error) per query,
    # and optionally a canned IRI set per query for the dense-shaping tests
    def __init__(self, results: dict[str, tuple], iris: dict[str, set] | None = None):
        self.results = results
        self.iris = iris or {}
        self.logger = get_logger("TEST")

    def execute(self, kg: str, sparql: str):
        return self.results.get(sparql, (None, "unknown query"))

    def iri_set(self, sparql: str) -> set:
        return self.iris.get(sparql, set())


def episode(sparql=None, reference_sparql="REF", steps=5, error=None, type="answer"):
    return Episode(
        item_id="cqd-a",
        kg="wikidata",
        question="q",
        reference_sparql=reference_sparql,
        sparql=sparql,
        type=type,
        steps=steps,
        error=error,
    )


def test_reward_infrastructure_error_is_skipped():
    cfg = RewardConfig()
    reward = compute_reward(
        episode(error={"reason": "api"}), cfg, FakeExecutor({})
    )
    assert reward.skip_reason == "api"
    assert reward.reward is None


def test_reward_give_up_is_flat_regardless_of_length():
    cfg = RewardConfig()
    short = compute_reward(episode(sparql=None, steps=5), cfg, FakeExecutor({}))
    long = compute_reward(episode(sparql=None, steps=100), cfg, FakeExecutor({}))
    assert short.invalid and long.invalid
    # the give-up fix: no step penalty on invalid outcomes -> both equal
    assert short.reward == long.reward == pytest.approx(-cfg.w_invalid)


def test_reward_broken_query_is_flat_invalid():
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor({"REF": (AskResult(True), None), "BAD": (None, "syntax error")})
    short = compute_reward(episode(sparql="BAD", steps=5), cfg, ex)
    long = compute_reward(episode(sparql="BAD", steps=100), cfg, ex)
    assert short.invalid and short.f1 == 0.0
    assert short.reward == long.reward == pytest.approx(-cfg.w_invalid)


def test_reward_reference_error_is_skipped():
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor({"REF": (None, "boom")})
    reward = compute_reward(episode(sparql="Q"), cfg, ex)
    assert reward.skip_reason == "reference-error"
    assert reward.reward is None


def test_reward_hard_server_error_on_prediction_is_skipped():
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor(
        {"REF": (AskResult(True), None), "Q": (None, "502 Server Error: Bad Gateway")}
    )
    reward = compute_reward(episode(sparql="Q"), cfg, ex)
    assert reward.skip_reason == "server-error"
    assert reward.reward is None


def test_reward_valid_answer_scores_f1_minus_step_penalty():
    # grace 20, cap 30, factor 0.15, w_invalid 1.0 -> ramp 20..30 to 0.15
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor({"REF": (AskResult(True), None), "Q": (AskResult(True), None)})
    clean = compute_reward(episode(sparql="Q", steps=20), cfg, ex, max_steps=30)
    mid = compute_reward(episode(sparql="Q", steps=25), cfg, ex, max_steps=30)
    at_cap = compute_reward(episode(sparql="Q", steps=30), cfg, ex, max_steps=30)
    assert clean.f1 == 1.0
    assert clean.reward == pytest.approx(1.0)  # at step_grace: no penalty
    # halfway up the ramp: 0.15 * w_invalid * (25-20)/(30-20) = 0.075
    assert mid.reward == pytest.approx(1.0 - 0.075)
    # at the cap: full max_step_penalty * w_invalid = 0.15
    assert at_cap.reward == pytest.approx(1.0 - 0.15)


def test_reward_valid_beats_any_giveup():
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor({"REF": (AskResult(True), None), "WRONG": (AskResult(False), None)})
    wrong_but_working = compute_reward(episode(sparql="WRONG"), cfg, ex)
    give_up = compute_reward(episode(sparql=None, type="answer"), cfg, ex)
    # anti-abstention: even a working-but-wrong query beats giving up
    assert wrong_but_working.reward > give_up.reward


def test_iri_jaccard_math():
    from grasp.cqd.reward import iri_jaccard

    assert iri_jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert iri_jaccard({"a", "b", "c"}, {"a"}) == pytest.approx(1 / 3)
    assert iri_jaccard({"a"}, {"b"}) == 0.0
    # both empty -> nothing to ground, treated as a match; one empty -> 0
    assert iri_jaccard(set(), set()) == 1.0
    assert iri_jaccard({"a"}, set()) == 0.0


def test_query_iris_extracts_and_canonicalizes_entities_and_properties():
    from grasp.cqd.reward import query_iris
    from grasp.sparql.utils import load_sparql_parser

    parser = load_sparql_parser()
    # prefixed names and a full IRI must resolve to the same canonical form
    iris = query_iris(
        "SELECT ?r WHERE { wd:Q125006 wdt:P177 ?r . "
        "<http://www.wikidata.org/entity/Q1> wdt:P31 ?r }",
        parser,
    )
    assert iris == {
        "http://www.wikidata.org/entity/Q125006",
        "http://www.wikidata.org/prop/direct/P177",
        "http://www.wikidata.org/entity/Q1",
        "http://www.wikidata.org/prop/direct/P31",
    }
    # non-wikidata IRIs are ignored; unparseable input yields an empty set
    assert query_iris("SELECT ?x WHERE { ?x rdfs:label ?l }", parser) == set()
    assert query_iris("NOT SPARQL", parser) == set()


def test_reward_iri_shaping_is_additive_and_preserves_ordering():
    # dense shaping on: a wrong-but-structurally-close query earns w_iri*jaccard
    # on top of its f1=0, but a correct answer still outranks it
    ref_iris = {"wd:Q1", "wdt:P1"}
    cfg = RewardConfig(fix_prefixes=False, w_iri=0.2)
    ex = FakeExecutor(
        {
            "REF": (AskResult(True), None),
            "RIGHT": (AskResult(True), None),
            "CLOSE": (AskResult(False), None),
        },
        iris={
            "REF": ref_iris,
            "RIGHT": ref_iris,          # correct query, full overlap -> jaccard 1
            "CLOSE": {"wd:Q1", "wdt:P9"},  # half overlap -> jaccard 1/3
        },
    )
    right = compute_reward(episode(sparql="RIGHT", steps=20), cfg, ex, max_steps=30)
    close = compute_reward(episode(sparql="CLOSE", steps=20), cfg, ex, max_steps=30)
    # no step penalty at grace; right = 1 + 0.2*1, close = 0 + 0.2*(1/3)
    assert right.reward == pytest.approx(1.0 + 0.2)
    assert close.reward == pytest.approx(0.2 * (1 / 3))
    assert right.reward > close.reward
    assert close.iri_jaccard == pytest.approx(1 / 3)


def test_reward_iri_shaping_off_by_default_leaves_reward_unchanged():
    cfg = RewardConfig(fix_prefixes=False)  # w_iri defaults to 0.0
    ex = FakeExecutor(
        {"REF": (AskResult(True), None), "CLOSE": (AskResult(False), None)},
        iris={"REF": {"wd:Q1"}, "CLOSE": {"wd:Q1"}},
    )
    close = compute_reward(episode(sparql="CLOSE", steps=20), cfg, ex, max_steps=30)
    # f1 0, no shaping, no penalty at grace -> exactly 0, and no jaccard recorded
    assert close.reward == pytest.approx(0.0)
    assert close.iri_jaccard is None


def test_reward_step_limit_truncation_is_filtered():
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor({})

    def ep(type, error, steps):
        return Episode(
            item_id="i", kg="wikidata", question="q", reference_sparql="REF",
            type=type, sparql=None, steps=steps, error=error,
        )

    # ran to the step cap (no answer/cancel, no error) -> filtered, not trained
    trunc = compute_reward(ep(None, None, 30), cfg, ex, max_steps=30)
    assert trunc.skip_reason == "step-limit" and trunc.reward is None
    # ended EARLY with no answer (steps < cap) = a malformed / degenerate
    # tool-call exit, NOT a budget truncation -> penalized and trained on
    degen = compute_reward(ep(None, None, 4), cfg, ex, max_steps=30)
    assert degen.skip_reason is None and degen.invalid
    assert degen.reward == pytest.approx(-cfg.w_invalid)
    # a genuine give-up (cancelled without a query) is NOT filtered -> invalid
    giveup = compute_reward(ep("cancel", None, 5), cfg, ex, max_steps=30)
    assert giveup.skip_reason is None and giveup.invalid
    assert giveup.reward == pytest.approx(-cfg.w_invalid)
    # a loop abort (non-infra error reason) is NOT filtered -> invalid
    loop = compute_reward(ep(None, {"reason": "loop"}, 8), cfg, ex, max_steps=30)
    assert loop.skip_reason is None and loop.invalid


def test_step_limit_scored_as_failure_when_not_filtered():
    # in eval/validation (filter_step_limit=False) a truncated/degenerate
    # episode must count as a failure (invalid, honest f1 0), never be dropped
    cfg = RewardConfig(fix_prefixes=False)
    ex = FakeExecutor({})
    trunc = Episode(
        item_id="i", kg="wikidata", question="q", reference_sparql="REF",
        type=None, sparql=None, steps=30, error=None,
    )
    r = compute_reward(trunc, cfg, ex, max_steps=30, filter_step_limit=False)
    assert r.skip_reason is None and r.invalid
    assert r.reward == pytest.approx(-cfg.w_invalid)
    # honest mean_f1 counts it as 0, not excluded
    stats = reward_stats([r])
    assert stats["mean_f1"] == pytest.approx(0.0)
    assert stats["scored"] == 1


def test_trainable_items_are_those_with_reward_variance():
    from grasp.cqd.train import rl

    rewards = [
        EpisodeReward(item_id="a", kg="wikidata", reward=1.0),
        EpisodeReward(item_id="a", kg="wikidata", reward=-1.0),  # a: variance
        EpisodeReward(item_id="b", kg="wikidata", reward=-1.0),
        EpisodeReward(item_id="b", kg="wikidata", reward=-1.0),  # b: zero variance
        EpisodeReward(item_id="c", kg="wikidata", reward=0.5),   # c: single rollout
    ]
    # only a has a within-group gradient
    assert rl.trainable_items(rewards) == {"a"}


# ---------------------------------------------------------------------------
# reward_episodes grouping
# ---------------------------------------------------------------------------


def test_reward_episodes_records_one_group_per_item():
    pool = TaskPool([make_item("cqd-a"), make_item("cqd-b")])
    cfg = RewardConfig(fix_prefixes=False)
    # give-ups: type=answer with no query -> invalid (type=None+no error would
    # be a step-limit truncation and get filtered out, see the filter test)
    episodes = [
        # item a: two give-ups (both invalid)
        Episode(item_id="cqd-a", kg="wikidata", question="q", reference_sparql="REF", type="answer"),
        Episode(item_id="cqd-a", kg="wikidata", question="q", reference_sparql="REF", type="answer"),
        # item b: one skipped (infra) + one give-up
        Episode(
            item_id="cqd-b", kg="wikidata", question="q", reference_sparql="REF",
            error={"reason": "api"},
        ),
        Episode(item_id="cqd-b", kg="wikidata", question="q", reference_sparql="REF", type="answer"),
    ]
    rewards = reward_episodes(episodes, pool, cfg, round=3)

    a = pool["cqd-a"].info.groups
    b = pool["cqd-b"].info.groups
    assert len(a) == 1 and len(b) == 1
    # both a-rollouts invalid; f1 filled with 0.0
    assert a[0].rewards == [-cfg.w_invalid, -cfg.w_invalid]
    assert a[0].f1s == [0.0, 0.0]
    assert a[0].n_invalid == 2
    assert a[0].round == 3
    # b's skipped episode is excluded from the group
    assert b[0].rewards == [-cfg.w_invalid]
    assert b[0].n_invalid == 1
    assert isinstance(rewards[0], EpisodeReward)


def test_reward_episodes_ignores_episodes_absent_from_pool():
    pool = TaskPool([make_item("cqd-a")])
    cfg = RewardConfig(fix_prefixes=False)
    episodes = [
        Episode(item_id="cqd-a", kg="wikidata", question="q", reference_sparql="REF", type="answer"),
        Episode(item_id="cqd-missing", kg="wikidata", question="q", reference_sparql="REF", type="answer"),
    ]
    reward_episodes(episodes, pool, cfg)
    # only the in-pool item accrues a group; the orphan episode is dropped
    assert pool["cqd-a"].n_groups() == 1
    assert "cqd-missing" not in pool


# ---------------------------------------------------------------------------
# holdout validation metric aggregation (train.rl.run_validation)
# ---------------------------------------------------------------------------


def test_run_validation_aggregates_per_file_and_overall(monkeypatch):
    from grasp.cqd.train import rl

    rows = {
        "qald7.jsonl": [{"question": "q1", "sparql": "S1"}, {"question": "q2", "sparql": "S2"}],
        "lcquad.jsonl": [{"question": "q3", "sparql": "S3"}],
    }
    # per-file canned rewards, in val_files order; reward_stats runs for
    # real. Item c is a give-up: no final query, f1 None (NOT 0) -- the
    # honest metric must count it as 0, not exclude it.
    per_file = iter(
        [
            [
                EpisodeReward(item_id="a", kg="wikidata", reward=0.5, f1=0.5),
                EpisodeReward(item_id="b", kg="wikidata", reward=1.0, f1=1.0),
                EpisodeReward(item_id="c", kg="wikidata", reward=-0.5, f1=None, invalid=True),
            ],
            [EpisodeReward(item_id="d", kg="wikidata", reward=0.0, f1=0.0)],
        ]
    )
    monkeypatch.setattr(rl, "load_jsonl", lambda f: rows[f])
    monkeypatch.setattr(rl, "collect_rollouts", lambda *a, **k: [])
    monkeypatch.setattr(rl, "reward_episodes", lambda *a, **k: next(per_file))

    val_files = [rl.ValFile(kg="wikidata", file=f) for f in rows]
    metrics = rl.run_validation(
        val_files, RolloutConfig(pool_file="x"), RewardConfig(fix_prefixes=False),
        None, None, None, get_logger("TEST"),
    )

    # honest per-file f1: give-up counts as 0 -> (0.5+1.0+0)/3, NOT 0.75
    assert metrics["val/qald7/f1"] == pytest.approx(1.5 / 3)
    # conditional f1 excludes the give-up -> (0.5+1.0)/2
    assert metrics["val/qald7/f1_answered"] == pytest.approx(0.75)
    assert metrics["val/qald7/invalid"] == 1
    assert metrics["val/lcquad/f1"] == pytest.approx(0.0)
    # honest aggregate over all four samples (give-up = 0), NOT 1.5/3
    assert metrics["val/f1"] == pytest.approx(1.5 / 4)
    assert metrics["val/invalid"] == 1


# ---------------------------------------------------------------------------
# curriculum.descend (propose -> probe -> accept gate)
# ---------------------------------------------------------------------------


def run_descend(monkeypatch, levels: list[list[tuple[str, list[float]]]], **kwargs):
    # drive descend with canned teacher levels: each level is a list of
    # (item_id, f1s) the "teacher" proposes and the "student" then scores
    from grasp.cqd import curriculum
    from grasp.tasks.cq_distillation.functions import Proposal

    pool = TaskPool()
    calls = {"n": 0}

    def fake_cover_and_revisit(config, pool_, episodes, rewards, cover, *a, **kw):
        i = calls["n"]
        calls["n"] += 1
        if i >= len(levels):
            return []
        proposals = []
        for id, _ in levels[i]:
            item = make_item(id)
            pool_.items[id] = item
            proposals.append(
                Proposal(kg="wikidata", question=id, sparql="ASK {}", intent="i",
                         difficulty="easier")
            )
        return proposals

    def fake_probe_items(items, pool_, *a, **kw):
        # score each freshly added item with its canned f1s
        f1s = dict(levels[calls["n"] - 1])
        for item in items:
            vals = f1s[item.id]
            item.info.groups.append(GroupRecord(f1s=vals, rewards=vals))
        return [], []

    # item_id must map a proposal back to the item the fake teacher created
    monkeypatch.setattr(curriculum, "cover_and_revisit", fake_cover_and_revisit)
    monkeypatch.setattr(curriculum, "probe_items", fake_probe_items)
    monkeypatch.setattr(curriculum, "item_id", lambda kg, question, sparql: question)

    accepted, rejected = curriculum.descend(
        None, pool, RolloutConfig(pool_file="x"), RewardConfig(fix_prefixes=False),
        [], [], [], **kwargs,
    )
    return pool, accepted, rejected, calls["n"]


def test_descend_accepts_only_variants_the_student_can_solve(monkeypatch):
    # one variant is solved half the time (competence 0.5), one never
    pool, accepted, rejected, _ = run_descend(
        monkeypatch,
        [[("cqd-solvable", [1.0, 0.0]), ("cqd-toohard", [0.0, 0.0])]],
        min_competence=0.25,
        max_rounds=1,
    )
    assert [it.id for it in accepted] == ["cqd-solvable"]
    assert [it.id for it in rejected] == ["cqd-toohard"]


def test_descend_keeps_easing_until_the_student_can_solve(monkeypatch):
    # level 0 is still too hard (all zero) -> descend again; level 1 lands
    pool, accepted, rejected, levels_run = run_descend(
        monkeypatch,
        [[("cqd-hard", [0.0, 0.0])], [("cqd-easier", [0.6, 0.4])]],
        min_competence=0.25,
        max_rounds=3,
    )
    assert [it.id for it in accepted] == ["cqd-easier"]
    assert [it.id for it in rejected] == ["cqd-hard"]
    # stopped as soon as a level landed, rather than using all 3 levels
    assert levels_run == 2


def test_descend_stops_early_and_does_not_descend_when_first_level_lands(monkeypatch):
    _, accepted, rejected, levels_run = run_descend(
        monkeypatch,
        [[("cqd-good", [1.0, 1.0])], [("cqd-unused", [1.0, 1.0])]],
        min_competence=0.25,
        max_rounds=3,
    )
    assert [it.id for it in accepted] == ["cqd-good"]
    assert rejected == [] and levels_run == 1


def test_descend_default_gate_rejects_only_no_correctness_items(monkeypatch):
    # at the default min_competence of 0 the gate is qualitative: an item no
    # rollout got any credit on is dropped (its reward spread is only
    # executable-vs-not, teaching validity), while a weak partial solver is
    # kept and left to the sampler to down-weight
    _, accepted, rejected, _ = run_descend(
        monkeypatch,
        [[("cqd-weak", [0.1, 0.0]), ("cqd-nocredit", [0.0, 0.0])]],
        max_rounds=1,
    )
    assert [it.id for it in accepted] == ["cqd-weak"]
    assert [it.id for it in rejected] == ["cqd-nocredit"]


def test_learnability_alone_no_longer_admits_a_partial_solver(monkeypatch):
    # the old gate was `competence > 0 or learnability > 0`: this item has
    # reward spread (learnable) but a competence of only 0.05, the
    # partial-solving trap the competence floor exists to reject
    _, accepted, rejected, _ = run_descend(
        monkeypatch,
        [[("cqd-partial", [0.1, 0.0])]],
        min_competence=0.25,
        max_rounds=1,
    )
    assert accepted == []
    assert [it.id for it in rejected] == ["cqd-partial"]


def test_log_wandb_is_a_noop_without_an_active_run():
    # the curriculum must stay usable outside training, where no run exists
    from grasp.cqd.curriculum import log_wandb

    log_wandb({"warmstart/accepted": 1})


def test_init_wandb_attaches_to_an_active_run(monkeypatch):
    # warm-start starts the run; train_rl must join it, not open a second one
    from grasp.cqd.train import rl

    class FakeRun:
        def __init__(self):
            self.config = self
            self.updated = None

        def update(self, cfg, allow_val_change=False):
            self.updated = cfg

    import wandb

    active = FakeRun()
    monkeypatch.setattr(wandb, "run", active)
    monkeypatch.setattr(
        wandb, "init", lambda **kw: pytest.fail("must not init a second run")
    )
    got = rl.init_wandb("proj", "name", {"rl": {"x": 1}})
    assert got is active
    assert active.updated == {"rl": {"x": 1}}
    pool = TaskPool([make_item("cqd-a"), make_item("cqd-b")])
    pool.remove("cqd-a")
    assert "cqd-a" not in pool and "cqd-b" in pool
    pool.remove("cqd-missing")  # no-op
    assert len(pool) == 1


# ---------------------------------------------------------------------------
# curriculum.select_revisits
# ---------------------------------------------------------------------------


def build_revisit_pool() -> TaskPool:
    # a failed extreme, a solved extreme (both zero reward spread -> no
    # signal), a learnable middle item, and a never-attempted item
    return TaskPool(
        [
            make_item("cqd-failed", [group([0.0, 0.0], [-0.5, -0.5])]),
            make_item("cqd-solved", [group([1.0, 1.0], [1.0, 1.0])]),
            make_item("cqd-live", [group([0.5, 0.5], [0.6, -0.4])]),
            make_item("cqd-new"),
        ]
    )


def test_select_revisits_orders_visited_by_learnability_ascending():
    revisits = select_revisits(build_revisit_pool())
    # least-learnable first; the two zero-variance extremes precede the
    # learnable middle, so a bottom-k slice revisits the extremes
    assert [it.id for it in revisits] == ["cqd-failed", "cqd-solved", "cqd-live"]
    assert {it.id for it in revisits[:2]} == {"cqd-failed", "cqd-solved"}


def test_select_revisits_excludes_never_attempted_items():
    ids = {it.id for it in select_revisits(build_revisit_pool())}
    assert "cqd-new" not in ids


def test_select_revisits_restricts_to_given_item_ids():
    pool = build_revisit_pool()
    # only cqd-failed is in scope; the missing id is silently ignored
    revisits = select_revisits(pool, item_ids=["cqd-failed", "cqd-absent"])
    assert [it.id for it in revisits] == ["cqd-failed"]


# ---------------------------------------------------------------------------
# datasets (fixed-dataset loader)
# ---------------------------------------------------------------------------


def test_pool_items_filter_dedupe_and_tag_from_samples():
    from grasp.cqd.datasets import pool_items
    from grasp.cqd.pool import item_id
    from grasp.tasks.sparql_qa.examples import SparqlQaSample

    samples = [
        SparqlQaSample(question="q1", sparql="ASK { ?a ?b ?c }"),
        SparqlQaSample(question="q2", sparql="SELECT ?x WHERE {}"),
        SparqlQaSample(question="q1", sparql="ASK { ?a ?b ?c }"),  # dup -> collapses
        SparqlQaSample(question="no query", sparql=""),  # missing sparql -> skip
        SparqlQaSample(question="q3", sparql="ASK {}", info={"invalid": True}),  # flagged
    ]
    items = pool_items(samples, kg="wikidata", tag="qald7-train")
    assert len(items) == 2
    assert items[0].id == item_id("wikidata", "q1", "ASK { ?a ?b ?c }")
    assert all(it.info.kg == "wikidata" for it in items)
    assert all(it.info.tags == ["qald7-train"] for it in items)


def test_verify_items_drops_ask_empty_and_broken(monkeypatch):
    from grasp.cqd import datasets
    from grasp.sparql.types import AskResult, SelectResult

    def fake_result(sparql, endpoint, timeout, max_rows):
        if "ASK" in sparql:
            return AskResult(True), None  # boolean reference -> dropped
        if "EMPTY" in sparql:
            return SelectResult(["x"], []), None  # empty -> dropped
        if "ERR" in sparql:
            return None, "boom"  # broken -> dropped
        return SelectResult(["x"], [{"x": "a"}]), None  # kept

    monkeypatch.setattr(datasets, "get_result_or_error", fake_result)
    items = [
        make_item("keep", sparql="SELECT ?x WHERE {}"),
        make_item("ask", sparql="ASK { ?s ?p ?o }"),
        make_item("empty", sparql="SELECT EMPTY"),
        make_item("err", sparql="ERR"),
    ]
    kept = datasets.verify_items(items, "http://endpoint")
    assert [it.id for it in kept] == ["keep"]


def test_build_pool_verifies_limits_and_saves(monkeypatch, tmp_path):
    from universal_ml_utils.io import dump_jsonl

    from grasp.cqd import datasets
    from grasp.cqd.pool import TaskPool
    from grasp.sparql.types import SelectResult

    split = str(tmp_path / "train.jsonl")
    dump_jsonl(
        (
            {"question": f"q{i}", "sparql": f"SELECT ?x{i} WHERE {{}}", "info": {}}
            for i in range(5)
        ),
        split,
    )

    # the last two references do not execute and must never reach the pool
    def fake_result(sparql, endpoint, timeout, max_rows):
        if "?x3" in sparql or "?x4" in sparql:
            return None, "boom"
        return SelectResult(["x"], [{"x": "a"}]), None

    monkeypatch.setattr(datasets, "get_result_or_error", fake_result)
    out = str(tmp_path / "pool.jsonl")
    pool = datasets.build_pool(
        split, kg="freebase", endpoint="http://endpoint", out=out, tag="wqsp-train",
        limit=2,
    )

    # limit applies AFTER verification, so the pool is exactly that size
    assert len(pool) == 2
    saved = TaskPool.load(out)
    assert len(saved) == 2
    assert all(it.info.kg == "freebase" for it in saved.items.values())
    assert all(it.info.tags == ["wqsp-train"] for it in saved.items.values())
    assert all("?x3" not in it.sparql for it in saved.items.values())


# ---------------------------------------------------------------------------
# chained chunks (resume)
# ---------------------------------------------------------------------------


def test_resume_round_offset_is_zero_for_a_fresh_output_dir(tmp_path):
    from grasp.cqd.train.rl import resume_round_offset

    assert resume_round_offset(str(tmp_path / "missing")) == 0
    assert resume_round_offset(str(tmp_path)) == 0


def test_resume_round_offset_continues_after_the_highest_round(tmp_path):
    from grasp.cqd.train.rl import resume_round_offset

    for name in ["round_1", "round_2", "round_10", "best", "adapter", "round_x"]:
        (tmp_path / name).mkdir()
    # numeric ordering, not lexicographic, and non-round dirs are ignored
    assert resume_round_offset(str(tmp_path)) == 10


def test_load_best_info_carries_the_watermark_across_jobs(tmp_path):
    import json

    from grasp.cqd.train.rl import load_best_info

    assert load_best_info(str(tmp_path)) == (None, None)

    best = tmp_path / "best"
    best.mkdir()
    (best / "best_info.json").write_text(json.dumps({"round": 6, "val_f1": 0.4736}))
    val_f1, round = load_best_info(str(tmp_path))
    assert round == 6
    assert val_f1 == 0.4736
