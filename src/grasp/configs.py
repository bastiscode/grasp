from pydantic import BaseModel


class KgConfig(BaseModel):
    kg: str
    endpoint: str | None = None
    entities_dir: str | None = None
    properties_dir: str | None = None
    entities_type: str | None = None
    properties_type: str | None = None
    prefix_file: str | None = None
    notes_file: str | None = None
    example_index: str | None = None


class Config(BaseModel):
    model: str = "openai/gpt-4.1"
    model_endpoint: str | None = None

    seed: int | None = None
    fn_set: str = "search_extended"
    notes_file: str | None = None

    knowledge_graphs: list[KgConfig] = [KgConfig(kg="wikidata")]

    search_top_k: int = 10
    # 10 total rows, 5 top and 5 bottom
    result_max_rows: int = 10
    # same for columns
    result_max_columns: int = 10
    # 10 total results, 10 top
    list_k: int = 10
    # force that all IRIs used in a SPARQL query
    # were previously seen
    know_before_use: bool = False

    # model decoding parameters
    temperature: float | None = 0.2
    top_p: float | None = 0.9
    reasoning_effort: str | None = None

    # completion parameters
    max_completion_tokens: int = 16384  # 16k, leaves enough space for reasoning models
    completion_timeout: float = 120.0

    num_examples: int = 3
    force_examples: bool = False
    random_examples: bool = False

    # enable feedback loop
    feedback: bool = False


class AdaptInput(BaseModel):
    kg: str
    file: str


class Adapt(Config):
    # additional parameters specific to adaptation of GRASP
    # to knowledge graphs

    # method and input_type determine the way we adapt GRASP
    # iterative_note_taking + None => explore knowledge graph
    # iterative_note_taking + questions/pairs => run question
    # answering with some examples
    method: str = "iterative_note_taking"

    # if method = iterative_note_taking
    num_rounds: int = 5

    # optional input files with question-sparql pairs
    input: list[AdaptInput] = []

    # if input_files is non-empty
    samples_per_round: int = 3
    samples_per_file: int | None = None

    # adapt model can be different from the main model
    adapt_model: str | None = None
    adapt_model_endpoint: str | None = None
    # and have different decoding parameters
    adapt_temperature: float | None = None
    adapt_top_p: float | None = None
    adapt_reasoning_effort: str | None = None
