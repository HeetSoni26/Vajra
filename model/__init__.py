from .config import ModelConfig

__all__ = ["ModelConfig", "FoundationLM"]


def __getattr__(name: str):
    if name == "FoundationLM":
        from .model import FoundationLM

        return FoundationLM
    raise AttributeError(name)
