from importlib import resources

from grammar_utils.parse import LR1Parser
from transformers import PreTrainedTokenizerBase


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = resources.read_text("grasp.baselines.grisp.grammar", "sparql.y")
    sparql_lexer = resources.read_text("grasp.baselines.grisp.grammar", "sparql.l")
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
