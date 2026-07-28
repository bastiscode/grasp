import hashlib
import math
import os
import random
from statistics import pstdev

from pydantic import BaseModel
from universal_ml_utils.io import dump_jsonl, load_jsonl

from grasp.tasks.cq_distillation.functions import Proposal
from grasp.tasks.sparql_qa.examples import SparqlQaSample


class GroupRecord(BaseModel):
    # one rollout group (a round's group_size attempts on the same item).
    # f1s drive the competence (difficulty) estimate; rewards drive the
    # learnability estimate (std of reward = the within-group GRPO signal).
    f1s: list[float]
    rewards: list[float]
    n_invalid: int = 0
    round: int | None = None


class PoolItemInfo(BaseModel):
    kg: str
    # backlink to the root competency question this item derives from
    cq_id: str | None = None
    # immediate predecessor item, None for first-generation items
    parent_id: str | None = None
    intent: str | None = None
    difficulty: str | None = None
    tags: list[str] = []
    teacher_model: str | None = None
    result_size: int | None = None
    # student rollout groups, oldest first
    groups: list[GroupRecord] = []


# exponentially-weighted moving average, oldest -> newest, recent weighted
# most; half_life is in groups. None if there is nothing to average.
def ewma(values: list[float], half_life: float) -> float | None:
    if not values:
        return None
    alpha = 1 - 0.5 ** (1 / half_life)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


class PoolItem(SparqlQaSample):
    # narrow the free-form sample info to the typed CQD metadata;
    # serialized JSONL stays loadable as SparqlQaSample
    info: PoolItemInfo  # type: ignore[assignment]

    def to_proposal(self) -> Proposal:
        return Proposal(
            kg=self.info.kg,
            question=self.question,
            sparql=self.sparql,
            intent=self.info.intent or "",
            difficulty=self.info.difficulty or "similar",
            cq_id=self.info.cq_id,
            parent_id=self.info.parent_id,
            result_size=self.info.result_size,
        )

    def n_groups(self) -> int:
        return len(self.info.groups)

    # current ability, in [0, 1]: EWMA of per-group mean f1. None if the
    # item has never been rolled out (a "new" item).
    def competence(self, half_life: float) -> float | None:
        means = [sum(g.f1s) / len(g.f1s) for g in self.info.groups if g.f1s]
        return ewma(means, half_life)

    # training signal: EWMA of per-group reward std (the within-group
    # spread GRPO turns into advantage). ~0 means no gradient -- either
    # mastered or hopeless. None if never rolled out.
    def learnability(self, half_life: float) -> float | None:
        stds = [pstdev(g.rewards) for g in self.info.groups if g.rewards]
        return ewma(stds, half_life)


def item_id(kg: str, question: str, sparql: str) -> str:
    hashed = hashlib.sha256(f"{kg}\n{question}\n{sparql}".encode()).hexdigest()
    return f"cqd-{hashed[:16]}"


# width of the zpd bump below; at the extremes (competence 0 or 1 against a
# target of 0.5) it damps an item to ~0.14 of its learnability weight rather
# than to nothing, so a stuck item stays reachable and can recover
ZPD_WIDTH = 0.25


# how ready the student is for an item, in (0, 1]: a gaussian bump centred on
# the target competence. Items it never solves and items it has mastered are
# both damped, so the draw concentrates on the productive middle -- the zone
# where it succeeds sometimes and fails sometimes.
def zpd(competence: float, target: float) -> float:
    return math.exp(-((competence - target) ** 2) / (2 * ZPD_WIDTH**2))


# how strongly an item should be drawn: its learnability (the within-group
# reward spread GRPO turns into advantage) focused on the difficulty the
# student is ready for. Learnability alone is difficulty-blind and, because
# hard items far outnumber solvable ones, it fills batches with items the
# student can only ever partially solve. target_competence None keeps the
# plain learnability weighting.
def draw_weight(
    item: "PoolItem",
    competence_half_life: float,
    learnability_half_life: float,
    target_competence: float | None,
) -> float:
    learnability = item.learnability(learnability_half_life) or 0.0
    competence = item.competence(competence_half_life)
    if target_competence is None or competence is None:
        return learnability
    return learnability * zpd(competence, target_competence)


# Weighted sample without replacement (repeated re-weighted draws; fine at
# the batch sizes pool sampling deals with).
def weighted_sample(
    rng: random.Random,
    items: list["PoolItem"],
    weights: list[float],
    n: int,
) -> list["PoolItem"]:
    items = list(items)
    weights = list(weights)
    sampled = []
    for _ in range(min(n, len(items))):
        (item,) = rng.choices(items, weights=weights, k=1)
        index = items.index(item)
        items.pop(index)
        weights.pop(index)
        sampled.append(item)
    return sampled


class TaskPool:
    def __init__(self, items: list[PoolItem] | None = None) -> None:
        self.items: dict[str, PoolItem] = {}
        for item in items or []:
            assert item.id is not None, "Pool item id must not be None"
            assert item.id not in self.items, f"Duplicate pool item id {item.id}"
            self.items[item.id] = item

    @staticmethod
    def load(file: str) -> "TaskPool":
        return TaskPool([PoolItem(**item) for item in load_jsonl(file)])

    def save(self, file: str) -> None:
        dir = os.path.dirname(file)
        if dir:
            os.makedirs(dir, exist_ok=True)
        dump_jsonl((item.model_dump() for item in self.items.values()), file)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, id: str) -> bool:
        return id in self.items

    def __getitem__(self, id: str) -> PoolItem:
        return self.items[id]

    # Ingest accepted proposals from a cq-distillation run. Ids are
    # content-derived, so re-ingesting the same proposal is a no-op and
    # duplicates across runs collapse onto one item. Returns the new items.
    def add_proposals(
        self,
        proposals: list[Proposal],
        teacher_model: str | None = None,
        tags: list[str] | None = None,
    ) -> list[PoolItem]:
        added = []
        for proposal in proposals:
            id = item_id(proposal.kg, proposal.question, proposal.sparql)
            if id in self.items:
                continue

            item = PoolItem(
                id=id,
                question=proposal.question,
                sparql=proposal.sparql,
                info=PoolItemInfo(
                    kg=proposal.kg,
                    cq_id=proposal.cq_id,
                    parent_id=proposal.parent_id,
                    intent=proposal.intent,
                    difficulty=proposal.difficulty,
                    tags=tags or [],
                    teacher_model=teacher_model,
                    result_size=proposal.result_size,
                ),
            )
            self.items[id] = item
            added.append(item)

        return added

    def add_group(self, id: str, group: GroupRecord) -> None:
        self.items[id].info.groups.append(group)

    # drop an item, e.g. a freshly probed variant the student cannot solve yet
    # (see curriculum.descend); a no-op if it is not in the pool
    def remove(self, id: str) -> None:
        self.items.pop(id, None)

    def by_cq(self, cq_id: str) -> list[PoolItem]:
        return [item for item in self.items.values() if item.info.cq_id == cq_id]

    def children(self, id: str) -> list[PoolItem]:
        return [item for item in self.items.values() if item.info.parent_id == id]

    # chain from the given item up to its first-generation ancestor
    def lineage(self, id: str) -> list[PoolItem]:
        chain = [self.items[id]]
        seen = {id}
        while (parent_id := chain[-1].info.parent_id) is not None:
            if parent_id in seen or parent_id not in self.items:
                break
            chain.append(self.items[parent_id])
            seen.add(parent_id)
        return chain

    # Sample up to n distinct items for a rollout batch. A fraction is
    # reserved for never-attempted ("new") items so the teacher's proposals
    # get explored; the rest are drawn in one pass weighted by draw_weight --
    # learnability (reward-std) focused on target_competence, so the batch
    # concentrates on items the student is ready for instead of on whichever
    # difficulty band happens to be crowded. Items with zero learnability
    # (mastered or hopeless) are skipped; the backfill only triggers if too
    # few weighted items exist.
    def sample(
        self,
        n: int,
        competence_half_life: float = 1.5,
        learnability_half_life: float = 3.0,
        target_competence: float | None = 0.5,
        new_fraction: float = 0.25,
        seed: int | None = None,
    ) -> list[PoolItem]:
        rng = random.Random(seed)
        items = list(self.items.values())
        if n >= len(items):
            return items

        new_items = [it for it in items if it.n_groups() == 0]
        k_new = min(round(new_fraction * n), len(new_items))
        sampled = rng.sample(new_items, k_new) if k_new else []

        attempted = [it for it in items if it.n_groups() > 0]
        wt = {
            it.id: draw_weight(
                it, competence_half_life, learnability_half_life, target_competence
            )
            for it in attempted
        }
        cand = [it for it in attempted if wt[it.id] > 0.0]
        sampled.extend(
            weighted_sample(
                rng, cand, [wt[it.id] for it in cand], n - len(sampled)
            )
        )

        # backfill if too few weighted items exist to fill n. Draw the
        # shortfall weighted like the main path so it still front-loads
        # gradient-bearing items: never-attempted items get the smallest
        # positive weight seen -- their signal is unknown, so they rank just
        # below proven-learnable but above proven-dead, which may yet turn out
        # learnable once rolled out. Only proven-dead (zero-weight) items fall
        # to a uniform draw, and only if the batch is still not full.
        if len(sampled) < n:
            remaining = [it for it in items if it not in sampled]
            pos = [v for v in wt.values() if v > 0.0]
            new_weight = min(pos) if pos else 1.0
            weights = [
                new_weight if it.n_groups() == 0 else wt.get(it.id, 0.0)
                for it in remaining
            ]
            weighted = [it for it, w in zip(remaining, weights) if w > 0.0]
            wts = [w for w in weights if w > 0.0]
            if weighted:
                sampled.extend(weighted_sample(rng, weighted, wts, n - len(sampled)))
            if len(sampled) < n:
                rest = [it for it in remaining if it not in sampled]
                sampled.extend(rng.sample(rest, min(n - len(sampled), len(rest))))

        return sampled

    # aggregate statistics, e.g. for the curriculum controller
    def stats(
        self,
        competence_half_life: float = 1.5,
        learnability_half_life: float = 3.0,
    ) -> dict:
        per_cq: dict[str, dict] = {}
        for item in self.items.values():
            cq_id = item.info.cq_id or "unknown"
            cq_stats = per_cq.setdefault(
                cq_id,
                {"items": 0, "attempted": 0, "competences": []},
            )
            cq_stats["items"] += 1
            c = item.competence(competence_half_life)
            if c is not None:
                cq_stats["attempted"] += 1
                cq_stats["competences"].append(c)

        for cq_stats in per_cq.values():
            cs = cq_stats.pop("competences")
            cq_stats["competence"] = sum(cs) / len(cs) if cs else None

        comps = [
            c
            for item in self.items.values()
            if (c := item.competence(competence_half_life)) is not None
        ]
        return {
            "items": len(self.items),
            "attempted": len(comps),
            "competence": sum(comps) / len(comps) if comps else None,
            "per_cq": per_cq,
        }
