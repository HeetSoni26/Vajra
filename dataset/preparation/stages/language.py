from dataset.preparation.models import Document, PreparationStatistics
from dataset.preparation.stages.base import PipelineStage


class LanguageDetectionStage(PipelineStage):
    """
    Placeholder for language detection integration (e.g., fasttext, gcld3).
    Currently just passes the document through.
    """

    def process(self, doc: Document, stats: PreparationStatistics) -> Document | None:
        if not self.config.enable_language_detection:
            return doc

        # Placeholder: Add language to metadata
        # In the future: if doc.metadata['language'] != target_lang: return None
        doc.metadata["language"] = "unknown"
        return doc
