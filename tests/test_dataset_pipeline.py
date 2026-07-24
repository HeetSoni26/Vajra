import json
from unittest import mock

from scripts.prepare_dataset import clean_document, prepare_huggingface


def test_clean_document():
    # HTML removal
    assert (
        clean_document(
            "<p>Hello world this is a test document with enough length to pass the filter.</p>"
        )
        == "Hello world this is a test document with enough length to pass the filter."
    )

    # Length constraints
    assert clean_document("Too short") == ""

    long_doc = "A" * 100001
    assert clean_document(long_doc) == ""

    # Normalization
    raw = "Hello   world\n\n\nThis is a \t test document that is long enough to pass."
    cleaned = clean_document(raw)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


@mock.patch("scripts.prepare_dataset.DatasetTokenizer")
@mock.patch("scripts.prepare_dataset.load_dataset")
def test_prepare_huggingface(mock_load_dataset, mock_tokenizer_class, tmp_path):
    mock_ds = [
        {
            "text": "This is a clean English document that is definitely long enough to pass the min length check."
        },
        {"text": "Too short"},
        {
            "text": "This is a duplicate English document that is definitely long enough to pass the min length check.",
            "language": "en",
        },
        {
            "text": "This is a duplicate English document that is definitely long enough to pass the min length check.",
            "language": "en",
        },
        {
            "text": "Ceci est un document en français qui est assez long pour passer.",
            "language": "fr",
        },
    ]
    mock_load_dataset.return_value = mock_ds

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.tokenize_documents.return_value = ([1] * 100, {})
    mock_tokenizer_class.return_value = mock_tokenizer

    output_dir = tmp_path / "data"

    report = prepare_huggingface(
        dataset_name="dummy",
        output_dir=output_dir,
        stream=False,
        max_docs=None,
    )

    assert report["validation_passed"] is True
    assert report["cleaning_stats"]["unique_docs"] == 2
    assert report["cleaning_stats"]["docs_processed"] == 2

    assert (output_dir / "train.bin").exists()
    assert (output_dir / "val.bin").exists()
    assert (output_dir / "test.bin").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "dataset_report.json").exists()

    meta = json.loads((output_dir / "metadata.json").read_text())
    assert meta["extra_info"]["docs_processed"] == 2
