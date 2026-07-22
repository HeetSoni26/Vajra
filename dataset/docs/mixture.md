# Vajra Dataset Selection & Mixture Builder

The Mixture framework is responsible for architecting the final distribution of training data before sequences are actually assembled and tokenized. It is isolated from data collection or shard generation, explicitly acting as the structural authority on what data the model is trained on, and in what proportions.

## Architecture

- **Models (`models.py`)**: 
  - `DatasetMixtureEntry`: Configures a specific versioned dataset, its sampling strategy interface, its intended weight in the mix, and caches metadata.
  - `DatasetMixture`: An overarching collection wrapper tracking the mixture name, description, and list of all entries.
- **Validation (`validators.py`)**: 
  - `MixtureValidator`: Ensures mathematical validity (e.g., weights summing to 100%), structural validity (no duplicates), and compliance checks against restrictive dataset licenses.
- **Analysis Engine (`analysis.py`)**: 
  - `MixtureAnalyzer`: Projects analytical graphs onto the given mixture configuration, assessing token distribution, language biases, and domain distributions.
- **Sampling Interfaces (`sampling.py`)**: 
  - Exposes abstract interfaces such as `TemperatureSamplingStrategy`, `CurriculumLearningStrategy`, etc., which will be implemented by the sequence packer later.
- **Registry & Management (`manager.py`)**:
  - `MixtureManager`: Provides a file-based registry mapping names to `.json` files.

## CLI Usage

Use `manage_dataset.py mixture <command>` to interact with the module:

```bash
# Create a new named mixture
python dataset/scripts/manage_dataset.py mixture create "vajra_370m_mix_v1"

# Import a manually constructed mixture config
python dataset/scripts/manage_dataset.py mixture import path/to/mixture.json

# Run validation checks (sum of weights, duplicates, licenses)
python dataset/scripts/manage_dataset.py mixture validate "vajra_370m_mix_v1"

# Generate statistical distribution reports
python dataset/scripts/manage_dataset.py mixture analyze "vajra_370m_mix_v1"
```
