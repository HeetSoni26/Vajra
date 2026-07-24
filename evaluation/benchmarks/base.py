from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmark(ABC):
    """
    Abstractions for external benchmark datasets (HellaSwag, ARC, PIQA, etc.).
    Implementations should act as adapters loading data and formatting prompts
    for the Vajra evaluation pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load_dataset(self) -> Any:
        pass

    @abstractmethod
    def format_prompt(self, sample: Any) -> str:
        """Formats the dataset sample into a prompt suitable for the model."""

    @abstractmethod
    def evaluate_model(self, model, engine) -> dict[str, float]:
        """
        Executes the benchmark against the model.
        Returns a dictionary of metrics.
        """
