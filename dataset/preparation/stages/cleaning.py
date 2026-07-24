import re
import unicodedata
from typing import Optional
from dataset.preparation.models import Document, PreparationStatistics
from dataset.preparation.stages.base import PipelineStage


class UnicodeNormalizationStage(PipelineStage):
    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not self.config.enable_unicode_normalization:
            return doc
        doc.text = unicodedata.normalize("NFC", doc.text)
        return doc


class WhitespaceNormalizationStage(PipelineStage):
    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not self.config.enable_whitespace_normalization:
            return doc
        # Convert multiple spaces/tabs to a single space, but preserve newlines
        # This regex replaces horizontal whitespace runs with a single space
        doc.text = re.sub(r"[ \t]+", " ", doc.text)
        # Remove trailing/leading whitespace per line
        doc.text = "\n".join(line.strip() for line in doc.text.split("\n"))
        doc.text = doc.text.strip()
        return doc


class EmptyRemovalStage(PipelineStage):
    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not self.config.enable_empty_removal:
            return doc
        if not doc.text or len(doc.text.strip()) == 0:
            stats.filtered_empty += 1
            return None
        return doc
