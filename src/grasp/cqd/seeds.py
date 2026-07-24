from typing import Any

from pydantic import BaseModel
from tqdm import tqdm
from universal_ml_utils.io import load_jsonl

from grasp.evaluate import get_result_or_error, get_result_size

OVERSIZED_MESSAGE = "SPARQL result exceeded"


class Seed(BaseModel):
    sparql: str
    question: str | None = None
    id: str | None = None
    info: dict[str, Any] = {}


class SeedVerification(BaseModel):
    result_size: int = 0
    error: str | None = None

    @property
    def empty(self) -> bool:
        return self.error is None and self.result_size == 0

    @property
    def oversized(self) -> bool:
        return self.error is not None and OVERSIZED_MESSAGE in self.error

    @property
    def ok(self) -> bool:
        return self.error is None and self.result_size > 0

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        elif self.empty:
            return "empty"
        elif self.oversized:
            return "oversized"
        else:
            return "error"


def load_seeds(file: str) -> list[Seed]:
    seeds = []
    ids = set()
    for i, item in enumerate(load_jsonl(file)):
        seed = Seed(**item)
        if seed.id is None:
            seed.id = f"seed-{i}"
        assert seed.id not in ids, f"Duplicate seed id {seed.id}"
        ids.add(seed.id)
        seeds.append(seed)
    return seeds


def verify_seed(
    seed: Seed,
    endpoint: str,
    timeout: float = 300.0,
    max_rows: int | None = 100_000,
) -> SeedVerification:
    result, error = get_result_or_error(
        seed.sparql,
        endpoint,
        timeout,
        max_rows,
    )
    if error is not None:
        return SeedVerification(error=error)

    return SeedVerification(result_size=get_result_size(result))


def verify_seeds(
    seeds: list[Seed],
    endpoint: str,
    timeout: float = 300.0,
    max_rows: int | None = 100_000,
    progress: bool = False,
) -> dict[str, SeedVerification]:
    verifications = {}
    for seed in tqdm(seeds, desc="Verifying seeds", disable=not progress):
        assert seed.id is not None, "Seed id must not be None"
        verifications[seed.id] = verify_seed(seed, endpoint, timeout, max_rows)
    return verifications
