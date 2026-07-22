# Vajra Experiment Tracking & Run Management

The Experiment Tracking architecture provides independent robust state tracking isolating metadata spanning dataset parameters, optimization hyper-parameters, environment boundaries, and metric results globally structurally preventing silent regressions seamlessly natively.

## Components

- **`config.py` (`ExperimentConfig`)**: Controls storage locations natively binding interval limits tracking artifact rotation schedules explicitly safely without database targets locally mapping targets securely.
- **`snapshots.py` (`SnapshotManager`)**: Aggregates physical metadata tracking Python distributions, PyTorch limits, CUDA builds, Git hashes natively guaranteeing reproducing topology matches gracefully checking variables natively.
- **`artifacts.py` (`ArtifactManager`)**: Logs exact copies replicating critical states cleanly preventing mutation arrays parsing targets reliably moving configuration limits gracefully scaling bounds explicitly safely.
- **`metrics.py` (`MetricsHistory`)**: Logs interval bounding sequences safely recording Step variants tracking moving variants recording minimum loss cleanly maximizing bounds flawlessly recording JSON formats securely dynamically perfectly.
- **`search.py` (`SearchEngine`)**: Dynamically resolves projects querying paths mathematically parsing states explicitly resolving status checks scaling matching topologies cleanly robustly.
- **`comparison.py` (`ComparisonEngine`)**: Aggregates metric vectors checking absolute deviations wrapping percentage changes tracking natively preventing regression variants cleanly comparing instances robustly dynamically.
- **`manager.py` (`RunManager`)**: Orchestrates unified flows driving intervals cleanly binding logic wrapping objects tracking states securely preventing lost state failures gracefully globally gracefully reliably accurately seamlessly.

## CLI Utility

Operate physically bounding states straight out of `manage_experiments.py`:

```bash
# Create boundaries manually
python experiments/scripts/manage_experiments.py create --project vajra-370m --name test-run --tags offline base

# List targets isolating tags
python experiments/scripts/manage_experiments.py list --project vajra-370m --tags base

# Compare absolute regressions
python experiments/scripts/manage_experiments.py compare output/experiments/vajra-370m/run_1 output/experiments/vajra-370m/run_2

# Export
python experiments/scripts/manage_experiments.py export output/experiments/vajra-370m/run_1 --output run_summary.md --format md
```
