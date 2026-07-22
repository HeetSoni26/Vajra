import hashlib
from typing import Optional, Set
from dataset.preparation.models import Document, PreparationStatistics
from dataset.preparation.stages.base import PipelineStage

class ExactDeduplicationStage(PipelineStage):
    """
    Maintains an in-memory set of document hashes to detect exact duplicates.
    Suitable for single-node processing of moderate sized datasets.
    """
    def __init__(self, config):
        super().__init__(config)
        self.seen_hashes: Set[str] = set()

    def process(self, doc: Document, stats: PreparationStatistics) -> Optional[Document]:
        if not self.config.enable_exact_deduplication:
            return doc
            
        # Create MD5 hash of the normalized text
        doc_hash = hashlib.md5(doc.text.encode('utf-8')).hexdigest()
        
        if doc_hash in self.seen_hashes:
            stats.filtered_duplicates += 1
            return None
            
        self.seen_hashes.add(doc_hash)
        return doc
