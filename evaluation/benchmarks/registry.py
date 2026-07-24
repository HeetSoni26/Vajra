from typing import Dict, Any, Type
import logging

logger = logging.getLogger(__name__)


class BenchmarkAdapter:
    """Base class for benchmark adapters."""

    def __init__(self, name: str):
        self.name = name

    def format_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError

    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        raise NotImplementedError


class BenchmarkRegistry:
    """Registry for managing available benchmarks."""

    _adapters: Dict[str, Type[BenchmarkAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(adapter_class: Type[BenchmarkAdapter]):
            cls._adapters[name] = adapter_class
            return adapter_class

        return decorator

    @classmethod
    def get_adapter(cls, name: str) -> BenchmarkAdapter:
        if name not in cls._adapters:
            raise ValueError(f"Benchmark '{name}' not found in registry.")
        return cls._adapters[name](name)

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        return list(cls._adapters.keys())
