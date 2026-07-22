from abc import ABC, abstractmethod
from typing import Dict
import torch

class BaseMetric(ABC):
    """
    Base abstraction for a metric.
    """
    @abstractmethod
    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        pass
        
    @abstractmethod
    def compute(self) -> Dict[str, float]:
        pass
        
    @abstractmethod
    def reset(self):
        pass
