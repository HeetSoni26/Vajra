# Vajra Dataset Preparation Pipeline

The Dataset Preparation Pipeline is the third major component of the Vajra Dataset Framework. It sits between the raw downloaded datasets and the tokenization phase. Its primary responsibility is to accept raw text formats (JSONL, TXT) and output highly normalized, filtered, and deduplicated text streams ready for tokenizer training and sequence packing.

## Architecture
The pipeline is designed around a decoupled `PipelineStage` architecture orchestrated by the `PreparationPipeline`.

- **Models (`dataset/preparation/models.py`)**: `Document` (ID, text, metadata), `PreparationConfig` (thresholds and toggles), `PreparationStatistics` (tracking filtering ratios).
- **Pipeline (`dataset/preparation/pipeline.py`)**: Orchestrates the execution of a stream of `Document`s through all registered stages.
- **Stages (`dataset/preparation/stages/`)**:
  - `cleaning.py`: `UnicodeNormalizationStage`, `WhitespaceNormalizationStage`, `EmptyRemovalStage`
  - `filtering.py`: `LengthFilteringStage`, `CharacterRatioFilteringStage`, `WhitespaceRatioFilteringStage`
  - `deduplication.py`: `ExactDeduplicationStage` (In-memory exact MD5 hashing)
  - `language.py`: `LanguageDetectionStage` (Placeholder for fasttext integration)
- **IO (`dataset/preparation/readers.py`, `writers.py`)**: `DocumentReader` reads standard JSONL/TXT. `DocumentWriter` writes normalized JSONL containing the ID, cleaned text, and preserved metadata.

## Configurable Thresholds
Using `PreparationConfig`, you can disable any individual stage and adjust filtering heuristics:
- `min_length` / `max_length`
- `min_char_ratio`
- `max_whitespace_ratio`

## Usage
The CLI exposes a `prepare` command that accepts an input stream and an output path.

```bash
# Dry run to see statistics without writing
python dataset/scripts/manage_dataset.py prepare --input raw_data.jsonl --output cleaned_data.jsonl --dry-run

# Run full pipeline
python dataset/scripts/manage_dataset.py prepare --input raw_data.jsonl --output cleaned_data.jsonl
```
