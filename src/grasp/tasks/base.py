from abc import ABC, abstractmethod
from typing import Any

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.utils import Sample


class FeedbackTask(ABC):
    @abstractmethod
    def feedback_system_message(
        self,
        kg_notes: dict[str, list[str]],
        notes: list[str],
    ) -> str: ...

    @abstractmethod
    def feedback_instructions(self, inputs: list[str], output: dict) -> str: ...


class GraspTask(ABC):
    name: str

    def __init__(self, managers: list[KgManager], config: GraspConfig) -> None:
        self.managers = managers
        self.config = config

    @abstractmethod
    def system_information(self) -> str: ...

    @abstractmethod
    def rules(self) -> list[str]: ...

    @abstractmethod
    def function_definitions(self) -> list[dict]: ...

    @abstractmethod
    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        state: Any,
        example_indices: dict | None,
    ) -> str: ...

    @abstractmethod
    def done(self, fn_name: str) -> bool: ...

    @abstractmethod
    def output(self, messages: list[Message], state: Any) -> dict | None: ...

    def setup(self, input: Any) -> tuple[str, Any]:
        # default is no state, and string input
        assert isinstance(input, str), f"Input for {self.name} must be a string"
        return input, None

    @property
    def default_input_field(self) -> str | None:
        return None

    @classmethod
    def sample_cls(cls) -> type[Sample] | None:
        return None
