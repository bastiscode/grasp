from pydantic import BaseModel


class KgConfig(BaseModel):
    name: str
    endpoint: str | None = None
    entities_dir: str | None = None
    properties_dir: str | None = None
    entities_type: str | None = None
    properties_type: str | None = None
    prefix_file: str | None = None
    example_index: str | None = None


class Config(BaseModel):
    model: str
    model_endpoint: str | None = None
    api_key: str | None = None
    fn_set: str

    knowledge_graphs: list[KgConfig]

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
