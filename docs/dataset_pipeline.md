# Vajra Dataset Pipeline

The Vajra production dataset pipeline is designed to easily ingest, clean, tokenize, and format massive corpora from the Hugging Face hub (specifically optimized for `HuggingFaceFW/fineweb-edu`).

## Pipeline Workflow

1. **Download / Streaming**
   - The pipeline uses the `datasets` library to connect to Hugging Face.
   - It supports `streaming=True`, allowing for on-the-fly processing without needing terabytes of local storage.
   - It includes configurable limits such as `max-docs`, `max-tokens`, and `max-gb`.

2. **Preprocessing & Cleaning**
   - **Language Filtering**: Skips non-English documents (if metadata is available).
   - **Length Constraints**: Removes documents that are too short (<50 chars) or too long (>100,000 chars).
   - **HTML Removal**: Strips any residual HTML tags via regex.
   - **Normalization**: Applies strict Unicode NFC normalization and collapses multiple whitespaces and newlines.
   - **Empty & Duplicate Removal**: Hashes each cleaned document to remove exact duplicates and filters out empty lines.
   - **Statistics Collection**: Logs the number of processed documents, tokens, and data volume bytes.

3. **Tokenization**
   - Uses the Vajra tokenizer (`DatasetTokenizer`).
   - Appends an EOS token to each document.

4. **Splitting & Binary Formatting**
   - Splits the processed token stream into `train`, `val`, and `test` partitions based on user-defined ratios.
   - Saves them as `.bin` files (`uint32` memmaps) for extremely fast loading during training.
   - Generates `metadata.json` and `dataset_report.json` to keep track of splits, token counts, and cleaning statistics.

5. **Validation**
   - Ensures that output `.bin` files are not empty.
   - Verifies that the total tokens match the metadata statistics.
   - Checks the maximum token ID to ensure it is within the expected vocabulary range (e.g., `<128000`).

## Expected Directory Structure

After running the pipeline, the output directory (e.g., `data/fineweb`) will look like this:

```
data/fineweb/
├── train.bin             # Training tokens (uint32 memmap)
├── val.bin               # Validation tokens (uint32 memmap)
├── test.bin              # Test tokens (uint32 memmap)
├── metadata.json         # High-level dataset metadata (split info)
└── dataset_report.json   # Detailed stats on cleaning and processing
```

## Usage

Run the pipeline using the provided CLI:

```bash
python -m scripts.prepare_dataset \
    --dataset HuggingFaceFW/fineweb-edu \
    --stream \
    --max-docs 10000 \
    --output data/fineweb \
    --seed 42
```
