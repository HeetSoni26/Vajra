from abc import ABC, abstractmethod
from typing import Optional
from dataset.preparation.models import Document, PreparationConfig, PreparationStatistics


class PipelineStage(ABC):
    """
    Abstract base class for all preparation pipeline stages.
    A stage accepts a Document, processes it, and returns the modified Document.
    If the Document should be discarded (filtered), it returns None.
    """

    def __init__(self, config: PreparationConfig):
        self.config = config

    @abstractmethod
    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        """
        Process the document. Return None if it fails filtering.
        Update `stats` appropriately.
        """
        pass
