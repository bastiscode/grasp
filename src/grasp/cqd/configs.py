from grasp.configs import GraspConfig


class RolloutConfig(GraspConfig):
    # jsonl file with the task pool to roll out on
    pool_file: str
    # jsonl file to save the collected episodes to
    episodes_file: str | None = None
    # number of pool items to sample (None = all)
    num_items: int | None = None
    sample_seed: int | None = None
    # curriculum sampling (see pool.py:TaskPool.sample). Competence (which
    # difficulty bin) and learnability (reward-std weight) are EWMAs over
    # rollout groups; half-lives are in groups -- short for competence
    # (current ability), longer for learnability (stable signal).
    competence_half_life: float = 1.5
    learnability_half_life: float = 3.0
    # competence bins; None means one bin per sampled item (width 1/n)
    sample_n_bins: int | None = None
    # share of the batch reserved for never-attempted items (exploration)
    new_fraction: float = 0.25
    # max share of the batch from any one competence bin
    per_bin_cap_fraction: float = 0.5
    # rollouts per sampled item (group size for GRPO-style baselines)
    num_rollouts: int = 1
    # max concurrent episodes: the single knob for both LLM and KG load,
    # since each episode drives the vLLM server and issues SPARQL queries.
    # Raise it to keep the server fed (rollouts run while the trainer GPU is
    # idle), but lower it for fragile KG endpoints that stall under many
    # concurrent queries; tune per dataset/KG.
    parallelism: int = 8
    # request per-token logprobs and ids from the model (required for RL
    # training; needs an openai/completions provider, e.g. vLLM)
    token_data: bool = True


class CqdConfig(GraspConfig):
    # jsonl file with competency question seeds (see docs/05-seed-data.md)
    seeds_file: str
    # jsonl file the task pool is loaded from and saved to;
    # created if it does not exist
    pool_file: str
    # verify seeds by execution before proposing and skip failing ones
    verify_seeds: bool = True
    seed_verification_timeout: float = 300.0
    seed_verification_max_rows: int | None = 100_000
    # number of recent proposals from other competency questions shown to
    # the teacher as context (all proposals of the current one are always
    # shown)
    recent_proposals: int = 20
    # passes over the seed list
    rounds: int = 1
    # directory to dump the teacher traces to, e.g. for debugging or
    # SFT distillation later on
    trace_dir: str | None = None
