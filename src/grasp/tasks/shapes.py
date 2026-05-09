from typing import Any

from search_rdf.model import SentenceTransformerModel

from grasp.configs import GraspConfig
from grasp.functions import find_manager, validate_page
from grasp.manager import KgManager
from grasp.tasks.examples import ExampleIndex
from grasp.tasks.utils import Sample
from grasp.utils import FunctionCallException, format_list


class ShapeSample(Sample):
    iri: str
    short_iri: str
    shex: str

    def input(self) -> str:
        return self.short_iri

    def queries(self) -> list[str]:
        return [self.shex, self.short_iri]


class ShapeIndex(ExampleIndex):
    sample_cls = ShapeSample
    pattern: str | None = None

    def __init__(
        self,
        data: Any,
        index: Any,
        model: SentenceTransformerModel,
        samples: list[ShapeSample],
        pattern: str | None = None,
    ) -> None:
        super().__init__(data, index, model, samples)
        self.pattern = pattern
        self._iri_map: dict[str, ShapeSample] = {s.iri: s for s in samples}

    @classmethod
    def load(cls, dir: str, model: SentenceTransformerModel) -> "ShapeIndex":
        import os
        instance = super().load(dir, model)
        shape_instance = cls.__new__(cls)
        shape_instance.data = instance.data
        shape_instance.index = instance.index
        shape_instance.model = instance.model
        shape_instance.samples = instance.samples
        shape_instance._iri_map = {s.iri: s for s in instance.samples}  # type: ignore

        pattern_file = os.path.join(dir, "pattern.sparql")
        if os.path.exists(pattern_file):
            with open(pattern_file) as f:
                shape_instance.pattern = f.read().strip()
        else:
            shape_instance.pattern = None

        return shape_instance  # type: ignore

    def get_by_iri(self, iri: str) -> "ShapeSample | None":
        return self._iri_map.get(iri)


def _expand_iri(iri: str, manager: KgManager) -> str:
    if iri.startswith("<") and iri.endswith(">"):
        return iri[1:-1]
    if ":" in iri and not iri.startswith("http"):
        prefix, local = iri.split(":", 1)
        full_prefix = manager.prefixes.get(prefix)
        if full_prefix is not None:
            return full_prefix + local
    return iri


def load_shape_indices(
    config: GraspConfig,
    model: SentenceTransformerModel | str | None = None,
) -> dict[str, ShapeIndex]:
    if isinstance(model, str):
        model = SentenceTransformerModel(model)

    indices: dict[str, ShapeIndex] = {}
    for kg in config.knowledge_graphs:
        if kg.shape_index is None:
            continue
        assert model is not None, "Model must be provided to load shape indices"
        indices[kg.kg] = ShapeIndex.load(kg.shape_index, model)

    return indices


def search_shape_functions(config: GraspConfig) -> list[dict]:
    shape_kgs = [kg.kg for kg in config.knowledge_graphs if kg.shape_index is not None]
    if not shape_kgs:
        return []

    kg_list = format_list(f'"{kg}"' for kg in shape_kgs)
    num_results = config.search_k

    return [
        {
            "name": "search_shape",
            "description": f"""\
Search for pseudo-ShEx schema patterns for concepts in the specified knowledge \
graph that match a semantic query. Returns the full shape description per result, \
including the pseudo-ShEx block and membership pattern.

Use this to discover which concepts exist in the KG and what their structure looks \
like before writing SPARQL queries.

Currently shapes are available for the following knowledge graphs:
{kg_list}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": shape_kgs,
                        "description": "The knowledge graph to search shapes in",
                    },
                    "query": {
                        "type": "string",
                        "description": "A semantic query describing the concept you are looking for, e.g. 'human', 'protein', 'scientific article'",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": config.search_max_pages,
                        "description": "Page number for pagination",
                    },
                },
                "required": ["kg", "query", "page"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "get_shape",
            "description": f"""\
Retrieve the pseudo-ShEx schema pattern for a specific concept IRI in the \
specified knowledge graph. Returns the complete shape description including \
membership pattern and pseudo-ShEx block.

If the shape is not in the pre-built index, it will be computed on the fly \
by running profiling queries against the endpoint. If on-the-fly computation \
fails, use SPARQL queries to explore the concept directly.

Currently shapes are available for the following knowledge graphs:
{kg_list}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": shape_kgs,
                        "description": "The knowledge graph to look up the shape in",
                    },
                    "iri": {
                        "type": "string",
                        "description": "The full or prefixed IRI of the concept, e.g. 'wd:Q5' or 'http://www.wikidata.org/entity/Q5'",
                    },
                },
                "required": ["kg", "iri"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def _format_shape(sample: ShapeSample, index: int | None = None) -> str:
    prefix = f"{index}. " if index is not None else ""
    return f"{prefix}{sample.shex}"


def call_shape_function(
    fn_name: str,
    fn_args: dict,
    shape_indices: dict[str, ShapeIndex],
    managers: list[KgManager],
    config: GraspConfig,
) -> str:
    kg = fn_args["kg"]
    if kg not in shape_indices:
        return f"No shape index available for knowledge graph '{kg}'"

    idx = shape_indices[kg]

    if fn_name == "search_shape":
        query = fn_args["query"]
        page = fn_args.get("page") or 1
        validate_page(page, config.search_max_pages)

        k = config.search_k
        start = (page - 1) * k
        end = page * k
        results = idx.search(query, end)[start:end]

        if not results:
            return f"No shapes found for query '{query}' (page {page})"

        parts = [f"Shapes matching '{query}' (page {page}):\n"]
        for i, sample in enumerate(results, start + 1):
            parts.append(_format_shape(sample, i))
        return "\n\n".join(parts)

    elif fn_name == "get_shape":
        iri_arg = fn_args["iri"]
        manager, _ = find_manager(managers, kg)
        expanded_iri = _expand_iri(iri_arg, manager)

        sample = idx.get_by_iri(expanded_iri)
        if sample is None:
            sample = idx.get_by_iri(iri_arg)

        if sample is not None:
            return _format_shape(sample)

        # Not in index — try on-the-fly computation
        if idx.pattern is not None:
            from grasp.build.shapes import SamplingConfig, compute_shape
            sampling = SamplingConfig(
                sparql_result_max_rows=config.sparql_result_max_rows or 5_000_000
            )
            result = compute_shape(expanded_iri, idx.pattern, manager, sampling)
            if result is not None:
                return result
            return (
                f"Shape for '{iri_arg}' is not in the index and could not be computed "
                "on the fly (query failed or timed out). "
                "Use SPARQL queries to explore this concept directly."
            )

        return (
            f"No shape found for IRI '{iri_arg}'. "
            "The IRI may not be a concept in the shape index."
        )

    raise FunctionCallException(f"Unknown shape function '{fn_name}'")
