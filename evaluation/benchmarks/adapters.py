from typing import Dict, Any
from evaluation.benchmarks.registry import BenchmarkAdapter, BenchmarkRegistry

@BenchmarkRegistry.register("hellaswag")
class HellaSwagAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"{example.get('ctx', '')}\n"

    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        # Dummy mock metric
        acc = sum(p == r for p, r in zip(predictions, references)) / max(len(references), 1)
        return {"accuracy": acc, "normalized_accuracy": acc}

@BenchmarkRegistry.register("arc_easy")
class ARCEasyAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"Question: {example.get('question', '')}\nAnswer:"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"accuracy": 0.0}

@BenchmarkRegistry.register("arc_challenge")
class ARCChallengeAdapter(ARCEasyAdapter):
    pass

@BenchmarkRegistry.register("piqa")
class PIQAAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"Goal: {example.get('goal', '')}\n"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"accuracy": 0.0}

@BenchmarkRegistry.register("winogrande")
class WinoGrandeAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"{example.get('sentence', '')}"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"accuracy": 0.0}

@BenchmarkRegistry.register("boolq")
class BoolQAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"{example.get('passage', '')}\nQuestion: {example.get('question', '')}?\n"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"accuracy": 0.0}

@BenchmarkRegistry.register("openbookqa")
class OpenBookQAAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"Question: {example.get('question_stem', '')}\n"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"accuracy": 0.0}

@BenchmarkRegistry.register("mmlu")
class MMLUAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"Question: {example.get('question', '')}\n"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"accuracy": 0.0}

@BenchmarkRegistry.register("gsm8k")
class GSM8KAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"Question: {example.get('question', '')}\nAnswer:"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"exact_match": 0.0}

@BenchmarkRegistry.register("humaneval")
class HumanEvalAdapter(BenchmarkAdapter):
    def format_prompt(self, example: Dict[str, Any]) -> str:
        return f"{example.get('prompt', '')}"
    
    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        return {"pass@1": 0.0}
