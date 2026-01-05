# Hybrid Training User Guide

This guide explains how to train the QSight hybrid classical + quantum pipeline using the production code located in `model/production`. It covers environment setup, dataset preparation, command-line usage, staged training behaviour, and export artifacts.

## 1. Prerequisites

- Python 3.10 or later (matching the `qml_qiskit_env.yml` environment).
- GPU with CUDA support is recommended for the classical encoder. CPU-only training is supported but slower.
- Optional packages:
  - `wandb` for experiment tracking.
  - `qiskit` and `qiskit_machine_learning` for the quantum head. If they are not installed, the pipeline automatically falls back to classical-only training.
- Install dependencies (conda example):
  ```bash
  conda env create -f model/qml_qiskit_env.yml
  conda activate qml_qiskit
  pip install wandb qiskit qiskit_machine_learning  # optional
  ```

## 2. Dataset Preparation

The trainer supports two Kaggle-derived datasets. Place the unzipped data under a common root directory:

- **tanlikesmath**: expects `resized_train/resized_train/*.jpeg` and `resized_train/trainLabels.csv`.
- **sovitrath**: expects `gaussian_filtered_images/gaussian_filtered_images/<class>/*.png`.

Example layout (tanlikesmath):
```
/data/retinopathy/
  resized_train/
    resized_train/
      10_left.jpeg
      ...
    trainLabels.csv
```

## 3. Quick Start (CLI)

Use the packaged entry point to launch staged training:
```bash
python -m model.production.run_training \
  --dataset-root /data/retinopathy \
  --dataset-type tanlikesmath \
  --encoder-type vit \
  --batch-size 32 \
  --output-dir outputs/tanlikesmath_vit \
  --use-wandb --wandb-project QSight-Care
```

Key flags:
- `--dataset-root`: Directory containing the dataset.
- `--dataset-type`: `tanlikesmath` (default) or `sovitrath`.
- `--encoder-type`: `vit` (default) or `resnet`.
- `--no-pretrained`: Disable ImageNet weights for ViT/ResNet.
- `--stages-config`: Path to a JSON file defining custom training stages (see §4).
- `--grad-clip`: Gradient clipping value (set to `0` to disable).
- `--mixed-precision` / `--no-mixed-precision`: Force-enable or disable autocast (defaults to `cuda` availability).
- `--no-quantum`: Skip the quantum head entirely.
- `--quantum-qubits`, `--quantum-shots`: Configure quantum circuit size when Qiskit is installed.
- `--latency-tracking` / `--disable-latency`: Control latency instrumentation in the training loop.
- `--multi-gpu`: Enable PyTorch `DataParallel` across multiple GPUs; pair with `--device-ids` for explicit selection.
- `--use-wandb` and related flags: Enable Weights & Biases logging (see WandB guide for details).

The script seeds Python, NumPy, and PyTorch RNGs using `--seed` (default `42`).

### Optional local config file

To avoid typing long paths or leaking personal filesystem details, populate `config/training.local.template.json` and save a copy named `config/training.local.json` (ignored by Git):
```bash
cp config/training.local.template.json config/training.local.json
# edit config/training.local.json to point at your local dataset/output paths
```
Run the trainer with the config defaults plus any overrides:
```bash
python -m model.production.run_training --config-file config/training.local.json --use-wandb
```
Any CLI flag still wins over the JSON file, so you can keep secrets local while sharing reproducible commands in the docs.

Embed stage schedules directly in the same file using the `stages` array. Each entry mirrors the JSON accepted by `--stages-config`:
```json
{
  "stages": [
    {
      "name": "classical",
      "num_epochs": 8,
      "learning_rate": 0.0001,
      "patience": 4,
      "active_mask": [1, 1, 0]
    },
    {
      "name": "ensemble",
      "num_epochs": 4,
      "learning_rate": 5e-05,
      "loss_weights": {"ensemble": 1.0}
    }
  ]
}
```
When `stages` is present in the config file it takes precedence over `--stages-config` and the default schedule.

### Multi-GPU runs

Enable both RTX 3060 cards via DataParallel:
```bash
python -m model.production.run_training \
  --config-file config/training.local.json \
  --multi-gpu --device-ids 0,1 \
  --use-wandb --wandb-mode online
```
Set `multi_gpu` to `true` in the local config to make this the default and override `device_ids` when fewer or different GPUs are present.

## 4. Staged Training Behaviour

By default the trainer executes three stages:
1. **classical** (12 epochs, lr `1e-4`): trains encoder, compression module, classical heads, and ensemble while the quantum head is masked out.
2. **quantum** (8 epochs, lr `5e-5`): freezes encoder/compression/classical heads, trains only the quantum head + ensemble.
3. **ensemble** (5 epochs, lr `1e-4`): fine-tunes ensemble weights (all heads fixed).

Loss weights and active head masks change per stage, allowing stable optimisation of the hybrid model.

### Custom Stage Schedules

Pass `--stages-config path/to/stages.json` to override the defaults. The JSON file must contain a list of objects with the following shape:
```json
[
  {
    "name": "classical",
    "num_epochs": 10,
    "learning_rate": 0.0001,
    "weight_decay": 1e-5,
    "patience": 4,
    "active_mask": [1, 1, 0],
    "loss_weights": {
      "classical_a": 1.0,
      "classical_b": 1.0,
      "quantum": 0.0,
      "ensemble": 0.5
    }
  }
]
```
Fields:
- `active_mask`: `[use_head_a, use_head_b, use_quantum]`.
- `loss_weights`: Optional per-head loss multipliers. Missing keys inherit defaults.

## 5. Monitoring with Weights & Biases

Enable W&B logging with `--use-wandb` and configure project/entity/run names via the corresponding flags. Logged data includes:
- Loss and accuracy per head and ensemble.
- Average ensemble weights and uncertainty estimates.
- Validation confusion matrices and latency measurements.
- Stage summaries added to the run and the final report (`training_summary.json`).

Refer to `docs/WandB_Usage_Guide.md` for setup, filters, and report automation steps.

## 6. Using the Python API

For programmatic control import the training utilities directly:
```python
from model.production import (
    DatasetConfig,
    TrainingConfig,
    default_stage_schedule,
    train_hybrid_model,
)

config = DatasetConfig(
    root="/data/retinopathy",
    dataset_type="tanlikesmath",
    batch_size=16,
)
train_loader, val_loader, classes = create_dataloaders(config)
training_config = TrainingConfig(
    stages=default_stage_schedule(),
    checkpoint_dir=Path("outputs/checkpoints"),
)
model, summary = train_hybrid_model(
    train_loader,
    val_loader,
    classes,
    config=training_config,
    encoder_type="vit",
    quantum_enabled=True,
)
```

The returned `summary` contains per-stage metrics and the best validation snapshot.

## 7. Output Artifacts

The trainer populates the `--output-dir` (default `trained_model`) with:
- `phase1_classical_model.pth`: Core model weights.
- `complete_checkpoint.pth`: Model weights + training summary + metadata.
- `model_info.json`: Encoder/quantum configuration and class labels.
- `training_summary.json`: History and stage-level metrics.
- `quantum_training_data.pkl`: Cached compressed features for quantum fine-tuning.

If W&B logging is enabled, the run logs additionally store confusion matrices, latency histograms, and ensemble weights.

## 8. Troubleshooting & Tips

- **Quantum dependencies missing**: The trainer logs a warning and disables the quantum head automatically (`quantum_enabled=False`). Install `qiskit` and `qiskit_machine_learning` to re-enable it.
- **Mixed precision**: Enabled automatically when CUDA is available. Use `--no-mixed-precision` if you encounter numerical instability.
- **Checkpointing**: Provide `--checkpoint-dir` to save stage-level checkpoints (`stage_<name>_epoch_<n>.pth`).
- **Gradient clipping**: Defaults to `1.0`. Set `--grad-clip 0` to disable.
- **Dataset verification**: Ensure the labels CSV (tanlikesmath) or class folders (sovitrath) exist under `--dataset-root`; the loader raises descriptive errors otherwise.
- **Feature export**: Use the saved `quantum_training_data.pkl` to run quantum-only experiments without re-running the full classical stages.

With this guide you should be able to reproduce the hybrid training pipeline, customise stages for experiments, and capture complete logs for publication or benchmarking.
