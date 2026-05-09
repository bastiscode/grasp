from typing import Any

from grammar_utils.parse import LR1Parser  # type: ignore
from pydantic import BaseModel

from grasp.model import Message
from grasp.sparql.utils import find_all, parse_string
from grasp.tasks.base import GraspTask
from grasp.utils import FunctionCallException


class ShapesSetupState(BaseModel):
    pattern: str | None = None
    description: str | None = None


def validate_pattern_format(pattern: str, parser: LR1Parser) -> None:
    if "{CLASS}" not in pattern:
        raise FunctionCallException(
            "Pattern must contain '{CLASS}' as the class placeholder."
        )
    class_pattern = pattern.replace("{CLASS}", "?class")
    test_query = f"SELECT DISTINCT ?class WHERE {{\n  {class_pattern}\n}} LIMIT 1"
    parse, _ = parse_string(test_query, parser)
    variables = {
        child["value"]
        for var in find_all(parse, "Var")
        for child in var.get("children", [])
    }
    if "?instance" not in variables:
        raise FunctionCallException(
            "Pattern must contain '?instance' as the entity variable."
        )


class ShapesSetupTask(GraspTask):
    name = "shapes-setup"

    def system_information(self) -> str:
        manager = self.managers[0]
        return f"""\
You are a knowledge graph shape index setup assistant. Your task is to \
explore the '{manager.kg}' knowledge graph and determine the SPARQL pattern \
that connects instances to their class-like nodes.

Different knowledge graphs use different idioms for grouping entities into concepts:
- Standard RDF/OWL: '?instance a {{CLASS}} .'
- Wikidata-style:   '?instance wdt:P31 {{CLASS}} .'
- SKOS:             '?instance skos:inScheme {{CLASS}} .'
- Custom:           '?instance ex:category {{CLASS}} .'
- UNION (mixed):    '{{ ?instance a {{CLASS}} . }} UNION {{ ?instance wdt:P31 {{CLASS}} . }}'
- Multi-hop:        '?instance ex:type ?t . ?t skos:broader {{CLASS}} .'

The pattern uses two fixed placeholders:
- '?instance' — the entity/instance node
- '{{CLASS}}' — replaced with '?class' for class discovery, or a concrete IRI for per-class profiling

Your goal:
1. Explore the KG structure with SPARQL queries to understand how entities are typed.
2. Identify the primary grouping idiom (or a combination via UNION/property paths).
3. Validate the candidate pattern by running:
   SELECT DISTINCT ?class WHERE {{ PATTERN[{{CLASS}}→?class] }} LIMIT 1
   Confirm at least one result is returned.
4. Call set_query with the validated pattern.
5. Optionally call set_description with a summary of what classes are covered.
6. Call stop when done."""

    def rules(self) -> list[str]:
        return []

    def function_definitions(self) -> list[dict]:
        return [
            {
                "name": "show_setup",
                "description": "Show the current pattern and description for the shape index.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "set_pattern",
                "description": "Set or clear the shape index' SPARQL pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": ["string", "null"],
                            "description": "The SPARQL pattern, or null to clear / unset the current pattern.",
                        },
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "set_description",
                "description": "Set a concise description of what the shape index contains. "
                "Typically a single sentence about covered classes and any caveats is sufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "The description",
                        },
                    },
                    "required": ["description"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "stop",
                "description": "Stop the setup process.",
            },
        ]

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.state, ShapesSetupState)
        manager = self.managers[0]

        if fn_name == "show_setup":
            return format_setup(self.state)

        elif fn_name == "set_query":
            pattern = fn_args["pattern"]
            validate_pattern_format(pattern, manager.sparql_parser)
            self.state.pattern = pattern
            return "Pattern updated. Use get_shape to test it live."

        elif fn_name == "set_description":
            self.state.description = fn_args["description"]
            return "Description updated."

        elif fn_name == "stop":
            return "Stopping"

        raise FunctionCallException(f"Unknown function {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, dict), "Input for shapes-setup must be a dict"
        self.input = input
        self.state = ShapesSetupState()
        manager = self.managers[0]
        if input.get("notes"):
            return input["notes"]
        return f'Set up the shape index pattern for the "{manager.kg}" knowledge graph.'

    def output(self, messages: list[Message]) -> dict:
        assert isinstance(self.state, ShapesSetupState)
        if self.state.pattern:
            formatted = (
                f"Shape setup complete.\n"
                f"Pattern:\n{self.state.pattern}\n"
                f"Description:\n{self.state.description or 'None'}"
            )
        else:
            formatted = "Shape setup stopped without a pattern."
        return {
            "type": "output",
            "pattern": self.state.pattern,
            "description": self.state.description,
            "formatted": formatted,
        }


def format_setup(state: ShapesSetupState) -> str:
    return f"""\
Pattern:
{state.pattern or "None"}

Description:
{state.description or "None"}"""
