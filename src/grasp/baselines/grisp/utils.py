from importlib import resources

from grammar_utils.constrain import LR1Constraint
from grammar_utils.parse import LR1Parser
from tokenizers import models
from transformers import PreTrainedTokenizerBase


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = resources.read_text("grasp.baselines.grisp.grammar", "sparql.y")
    sparql_lexer = resources.read_text("grasp.baselines.grisp.grammar", "sparql.l")
    return sparql_grammar, sparql_lexer


def load_sparql_parser() -> LR1Parser:
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Parser(sparql_grammar, sparql_lexer)


def load_sparql_constraint(continuations: list[list[int]]) -> LR1Constraint:
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Constraint(sparql_grammar, sparql_lexer, continuations)


def get_continuations(
    tokenizer: PreTrainedTokenizerBase,
    initial: bool = False,
) -> list[list[int]]:
    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab, key=lambda token: vocab[token])
    decoder = tokenizer.decoder
    model = tokenizer.backend_tokenizer.model
    byte_fallback_enabled = isinstance(model, models.BPE) and model.byte_fallback  # type: ignore

    special_token: str = tokenizer.convert_ids_to_tokens(
        tokenizer.pad_token_id or tokenizer.eos_token_id  # type: ignore
    )
    special_token_enc = special_token.encode()

    def is_byte_fallback(token: str) -> bool:
        return token.startswith("<0x") and token.endswith(">") and len(token) == 6

    def continuation(token: str) -> list[int]:
        if initial:
            cont = decoder.decode([token]).encode()
        else:
            cont = decoder.decode([special_token, token, special_token]).encode()
            cont = cont[
                len(special_token_enc) : -len(special_token_enc)
            ]  # remove padding
        return list(cont)

    continuations = []
    for token in ordered:
        if byte_fallback_enabled and is_byte_fallback(token):  # type: ignore
            hex = token[3:5]
            # convert hex to bytes
            cont = list(bytes.fromhex(hex))
        else:
            cont = continuation(token)
        continuations.append(cont)

    return continuations


def patch_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
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
        {{- message['content'] + eos_token }}
        {% endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- 'Answer:\n' }}
{%- endif %}"""
    tokenizer.chat_template = chat_template  # type: ignore
    return tokenizer
