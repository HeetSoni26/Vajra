from typing import Optional
from dataset.preparation.models import Document, PreparationStatistics
from dataset.preparation.stages.base import PipelineStage


class LengthFilteringStage(PipelineStage):
    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not self.config.enable_length_filtering:
            return doc
        length = len(doc.text)
        if length < self.config.min_length or length > self.config.max_length:
            stats.filtered_length += 1
            return None
        return doc


class CharacterRatioFilteringStage(PipelineStage):
    """
    Filters out documents that have a low ratio of alphabetic characters,
    often indicating log files, hex dumps, or excessive formatting.
    """

    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not doc.text:
            return doc

        # Simple heuristic: ratio of alphabetic chars to total chars
        alpha_chars = sum(1 for c in doc.text if c.isalpha())
        ratio = alpha_chars / len(doc.text)

        if ratio < self.config.min_char_ratio:
            stats.filtered_quality += 1
            return None
        return doc


class WhitespaceRatioFilteringStage(PipelineStage):
    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not doc.text:
            return doc

        space_chars = sum(1 for c in doc.text if c.isspace())
        ratio = space_chars / len(doc.text)

        if ratio > self.config.max_whitespace_ratio:
            stats.filtered_quality += 1
            return None
        return doc
