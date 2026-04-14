import glob
import os
import random
from typing import Callable, Iterator

from grammar_utils.parse import LR1Parser
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase
from universal_ml_utils.io import load_json

from grasp.utils import read_resource


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = read_resource("grasp.baselines.grisp.grammar", "sparql.y")
    sparql_lexer = read_resource("grasp.baselines.grisp.grammar", "sparql.l")
    return sparql_grammar, sparql_lexer


def load_sparql_parser() -> LR1Parser:
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Parser(sparql_grammar, sparql_lexer)


def set_chat_template(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    # set custom chat template for single turn generation
    chat_template = """\
{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] != 'assistant' %}
        {{- message['role'].capitalize() + ' input:\n' }}
        {{- message['content'] + '\n\n' }}
    {%- else %}
        {{- 'Answer:\n' }}
        {% generation %}
          {{- message['content'].strip() + eos_token }}
        {% endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- 'Answer:\n' }}
{%- endif %}"""
    tokenizer.chat_template = chat_template  # type: ignore
    return tokenizer


def find_latest_checkpoint(run_directory: str) -> str | None:
    def latest_ckpt_key(checkpoint_dir: str) -> int:
        path = os.path.join(checkpoint_dir, "trainer_state.json")
        state = load_json(path)
        return -state["global_step"]

    return find_checkpoint(run_directory, latest_ckpt_key)


def find_best_checkpoint(run_directory: str) -> str | None:
    def best_ckpt_key(checkpoint_dir: str) -> int | float:
        path = os.path.join(checkpoint_dir, "trainer_state.json")
        state = load_json(path)
        global_step = state["global_step"]

        log_entry = next(
            (
                entry
                for entry in state["log_history"]
                if entry["step"] == global_step and entry.get("eval_loss") is not None
            ),
        )
        # sort by eval loss
        return log_entry["eval_loss"]

    return find_checkpoint(run_directory, best_ckpt_key)


def find_checkpoint(
    run_directory: str,
    key: Callable[[str], int | float],
) -> str | None:
    # all subdir starting with checkpoint-*
    checkpoints = glob.glob(os.path.join(run_directory, "checkpoint-*"))
    if not checkpoints:
        return None

    checkpoints.sort(key=key)
    return checkpoints[0]


class SeededRandomSampler(Sampler):
    def __init__(self, n: int, seed: int, epoch: int = 0) -> None:
        self.n = n
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rand = random.Random(self.seed + self.epoch)
        permutation = list(range(self.n))
        rand.shuffle(permutation)
        self.epoch += 1
        return iter(permutation)

    def __len__(self) -> int:
        return self.n
