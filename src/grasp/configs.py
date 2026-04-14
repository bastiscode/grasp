from typing import Any

from pydantic import BaseModel, conlist


class KgConfig(BaseModel):
    kg: str
    endpoint: str | None = None
    entities_type: str = "fuzzy"
    properties_type: str = "embedding"
    notes_file: str | None = None
    example_index: str | None = None

    # additional indices to load
    # built via search-rdf and exposed
    # via $GRASP_INDEX_DIR/{kg}/indices.yaml
    indices: list[str] = []


class ModelConfig(BaseModel):
    seed: int | None = None

    # model parameters
    model: str = "openai/gpt-5-mini"
    model_endpoint: str | None = None

    model_kwargs: dict[str, Any] = {}

    # decoding parameters
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    api: str | None = None
    parallel_tool_calls: bool = False
    tool_choice: str = "auto"

    # completion parameters
    max_completion_tokens: int = 8192  # 8k, leaves enough space for reasoning models
    completion_timeout: float = 120.0
    num_retries: int = 2


class GraspConfig(ModelConfig):
    # function set, notes, and knowledge graphs
    fn_set: str = "search_extended"
    notes_file: str | None = None

    knowledge_graphs: list[KgConfig] = [KgConfig(kg="wikidata")]

    # for embedding indices and example indices
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # optional task specific parameters
    # map[task_name, map[param_name, param_value]]
    task_kwargs: dict[str, dict[str, Any]] = {}

    # sparql query timeouts
    sparql_connection_timeout: float = 6.0
    sparql_query_timeout: float = 30.0
    sparql_read_timeout: float = 10.0

    # kg function parameters
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

    # interaction parameters
    max_steps: int = 100

    # example parameters
    num_examples: int = 3
    force_examples: str | None = None
    random_examples: bool = False

    # enable feedback loop
    feedback: bool = False
    max_feedbacks: int = 2
    notes_only_for_feedback: bool = False

    @property
    def sparql_request_timeout(self) -> tuple[float, float]:
        return self.sparql_connection_timeout, self.sparql_query_timeout


class ServerConfig(GraspConfig):
    port: int = 6789
    max_connections: int = 10
    max_generation_time: int = 300
    max_idle_time: int = 300
    log_outputs: str | None = None
    log_file: str | None = None
    share: str | None = None
    rate_limit: int | None = None
    rate_limit_window: int = 60


class NotesConfig(GraspConfig):
    # additional parameters specific to taking notes with GRASP
    max_notes: int = 16
    max_note_length: int = 512
    num_rounds: int = 5


class NoteTakingConfig(NotesConfig):
    # optional model config for the note taking step;
    # if set, fully replaces the parent GraspConfig's model config
    # for note taking (parent fields are not inherited)
    note_taking_model: ModelConfig | None = None


class NotesFromSamplesInput(BaseModel):
    kg: str
    file: str


class NotesFromSamplesConfig(NoteTakingConfig):
    # files with task examples
    samples: conlist(NotesFromSamplesInput, min_length=1)  # type: ignore
    samples_per_round: int = 3
    samples_per_file: int | None = None
    ignore_ground_truth: bool = False


class NotesFromOutputsConfig(NoteTakingConfig):
    # files with outputs only
    outputs: conlist(str, min_length=1)  # type: ignore
    outputs_per_round: int = 3
    outputs_per_file: int | None = None


class NotesFromExplorationConfig(NotesConfig):
    questions_per_round: int = 3
