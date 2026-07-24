from dataset.preparation.models import Document, PreparationConfig, PreparationStatistics
from dataset.preparation.pipeline import PreparationPipeline
from dataset.preparation.stages import (
    CharacterRatioFilteringStage,
    EmptyRemovalStage,
    ExactDeduplicationStage,
    LengthFilteringStage,
    UnicodeNormalizationStage,
    WhitespaceNormalizationStage,
)


def test_unicode_normalization():
    config = PreparationConfig()
    stage = UnicodeNormalizationStage(config)
    stats = PreparationStatistics()

    # \u0065\u0301 is e + acute accent. NFC normalizes it to \u00e9 (é)
    doc = Document(id="1", text="r\u0065\u0301sum\u0065\u0301")
    processed = stage.process(doc, stats)
    assert processed.text == "résumé"


def test_whitespace_normalization():
    config = PreparationConfig()
    stage = WhitespaceNormalizationStage(config)
    stats = PreparationStatistics()

    doc = Document(id="1", text="  This   is \t a test. \n  New   line  ")
    processed = stage.process(doc, stats)
    assert processed.text == "This is a test.\nNew line"


def test_empty_removal():
    config = PreparationConfig()
    stage = EmptyRemovalStage(config)
    stats = PreparationStatistics()

    doc = Document(id="1", text="   \n   ")
    processed = stage.process(doc, stats)
    assert processed is None
    assert stats.filtered_empty == 1


def test_length_filtering():
    config = PreparationConfig(min_length=10, max_length=50)
    stage = LengthFilteringStage(config)
    stats = PreparationStatistics()

    doc_short = Document(id="1", text="short")
    doc_valid = Document(id="2", text="this is a valid length string")
    doc_long = Document(id="3", text="x" * 60)

    assert stage.process(doc_short, stats) is None
    assert stage.process(doc_valid, stats) is not None
    assert stage.process(doc_long, stats) is None
    assert stats.filtered_length == 2


def test_character_ratio_filtering():
    config = PreparationConfig(min_char_ratio=0.5)
    stage = CharacterRatioFilteringStage(config)
    stats = PreparationStatistics()

    doc_hex = Document(id="1", text="0x00 0x01 0x02 0x03")
    doc_text = Document(id="2", text="This is mostly alphabetic characters with a few spaces.")

    assert stage.process(doc_hex, stats) is None
    assert stage.process(doc_text, stats) is not None


def test_exact_deduplication():
    config = PreparationConfig()
    stage = ExactDeduplicationStage(config)
    stats = PreparationStatistics()

    doc1 = Document(id="1", text="Same text")
    doc2 = Document(id="2", text="Same text")
    doc3 = Document(id="3", text="Different text")

    assert stage.process(doc1, stats) is not None
    assert stage.process(doc2, stats) is None
    assert stage.process(doc3, stats) is not None
    assert stats.filtered_duplicates == 1


def test_pipeline_integration():
    config = PreparationConfig(min_length=5)
    pipeline = PreparationPipeline(config)

    docs = [
        Document(id="1", text="Normal document."),
        Document(id="2", text="  \n "),  # empty
        Document(id="3", text="Normal document."),  # duplicate
        Document(id="4", text="ok"),  # too short
    ]

    processed = list(pipeline.process_stream(docs))
    assert len(processed) == 1
    assert processed[0].id == "1"
    assert pipeline.stats.total_documents_read == 4
    assert pipeline.stats.total_documents_written == 1
    assert pipeline.stats.filtered_empty == 1
    assert pipeline.stats.filtered_length == 1
    assert pipeline.stats.filtered_duplicates == 1
