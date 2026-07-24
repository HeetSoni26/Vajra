import time
from typing import List, Iterable
from dataset.preparation.models import Document, PreparationConfig, PreparationStatistics
from dataset.preparation.stages import (
    PipelineStage,
    UnicodeNormalizationStage,
    WhitespaceNormalizationStage,
    EmptyRemovalStage,
    LengthFilteringStage,
    CharacterRatioFilteringStage,
    WhitespaceRatioFilteringStage,
    ExactDeduplicationStage,
    LanguageDetectionStage,
)
from dataset.utils.logging import logger


class PreparationPipeline:
    """
    Orchestrates the sequence of processing stages for dataset preparation.
    """

    def __init__(self, config: PreparationConfig):
        self.config = config
        self.stats = PreparationStatistics()
        self.stages: List[PipelineStage] = self._build_pipeline()

    def _build_pipeline(self) -> List[PipelineStage]:
        """
        Instantiates the processing stages in the correct order.
        """
        stages = []

        # 1. Cleaning & Normalization
        stages.append(UnicodeNormalizationStage(self.config))
        stages.append(WhitespaceNormalizationStage(self.config))

        # 2. Quality Filtering
        stages.append(CharacterRatioFilteringStage(self.config))
        stages.append(WhitespaceRatioFilteringStage(self.config))

        # 3. Empty Removal & Size Filtering
        stages.append(EmptyRemovalStage(self.config))
        stages.append(LengthFilteringStage(self.config))

        # 4. Language & Deduplication
        stages.append(LanguageDetectionStage(self.config))
        stages.append(ExactDeduplicationStage(self.config))

        return stages

    def process_document(self, doc: Document) -> Document | None:
        """
        Process a single document through all stages.
        """
        self.stats.total_documents_read += 1
        self.stats.total_characters_read += len(doc.text)

        current_doc = doc
        for stage in self.stages:
            current_doc = stage.process(current_doc, self.stats)
            if current_doc is None:
                break

        if current_doc is not None:
            self.stats.total_documents_written += 1
            self.stats.total_characters_written += len(current_doc.text)

        return current_doc

    def process_stream(self, document_stream: Iterable[Document]) -> Iterable[Document]:
        """
        Process a stream of documents, yielding the surviving documents.
        """
        start_time = time.time()

        try:
            for doc in document_stream:
                processed = self.process_document(doc)
                if processed is not None:
                    yield processed
        finally:
            self.stats.processing_time_seconds += time.time() - start_time
            logger.info("Pipeline processing completed.")
            logger.info(
                f"Total read: {self.stats.total_documents_read}, Total written: {self.stats.total_documents_written}"
            )
