from typing import List, Any
from dataset.mixture.models import DatasetMixtureEntry


class SamplingStrategyBase:
    """
    Base interface for sampling strategies.
    Do not implement sampling algorithms, only the interfaces.
    """

    def __init__(self, entries: List[DatasetMixtureEntry]):
        self.entries = entries

    def sample(self) -> Any:
        """
        Samples datasets based on the defined strategy.
        """
        raise NotImplementedError


class FixedProportionStrategy(SamplingStrategyBase):
    pass


class TemperatureSamplingStrategy(SamplingStrategyBase):
    pass


class WeightedRandomStrategy(SamplingStrategyBase):
    pass


class CurriculumLearningStrategy(SamplingStrategyBase):
    pass


class DynamicSamplingStrategy(SamplingStrategyBase):
    pass
