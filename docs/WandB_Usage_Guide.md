# Weights & Biases Online Usage Guide

This guide summarizes how to run Phase 1 training, benchmarking, and automated reporting with Weights & Biases (W&B) in the QSight project.

## 1. Prerequisites
- Create a W&B account (https://wandb.ai) and generate an API key (Settings → API Keys).
- Ensure Python environment replicates `qml_qiskit_env.yml`.
- Install dependencies: `pip install wandb torchcam kagglehub qiskit qiskit_machine_learning`.

## 2. Authenticate W&B Online
```bash
wandb login <YOUR_API_KEY>
```
- Remove `WANDB_MODE=offline` from shell if previously set.
- Optional: set default entity/project:
```bash
export WANDB_ENTITY=<your_entity>
export WANDB_PROJECT=QSight-Care
```

## 3. Configure Training Run
The Phase 1 trainer inside `model/QML_Qiskit.ipynb` is W&B-aware.

1. Open the notebook and verify `CONFIG['use_wandb'] = True` and `CONFIG['wandb_project'] = 'QSight-Care'` (adjust as needed).
2. Set a distinctive run name: `CONFIG['wandb_run_name'] = 'phase1_hybrid_training_<timestamp>'`.
3. For multi-run experiments, vary `CONFIG['stages']`, `CONFIG['learning_rate']`, `CONFIG['compressed_dim']`, or `CONFIG['dataset']` before execution.

## 4. Launch Training
1. Execute cells sequentially (imports → dataset → model → trainer).
2. The trainer performs staged training (`classical`, `quantum`, `ensemble`) and logs:
   - Loss/accuracy per head and ensemble.
   - Per-head latencies.
   - Ensemble weights per epoch.
   - Confusion matrices and bar charts for selected validation samples.
   - Gradient-based saliency overlays.
3. Monitor progress in the browser under your W&B project.

### Recommendations
- Pin GPU/CPU usage values by adding `wandb.log({'hardware/gpu_mem': ...})` if needed.
- Tag runs (`wandb.config.update({'tag': 'baseline'})`) to filter dashboards easily.

## 5. Benchmark Evaluation (Testing)
1. Ensure the saved checkpoint (`complete_checkpoint.pth`) is present.
2. Open `model/Hybrid_Benchmark.ipynb`.
3. Adjust `CONFIG['class_mask_scenarios']` if you want to test additional head combinations.
4. Run cells to produce:
   - Accuracy/latency table comparing scenarios.
   - Confusion matrices for each head.
   - Confidence vs. class bar charts and gradient saliency previews.

To log benchmark outputs into the same W&B run:
```python
import wandb
wandb.init(project='QSight-Care', name='benchmark_phase1', resume='allow')
# After generating figures:
wandb.log({'benchmark/table': wandb.Table(dataframe=results_df)})
wandb.log({'benchmark/confusion_hybrid': wandb.Image(fig)})
```
Close the run with `wandb.finish()` when done.

## 6. Automated Reporting
W&B Reports consolidate runs, tables, and charts:
1. In the W&B web UI, navigate to Reports → New Report.
2. Add sections:
   - **Overview**: Aggregate metrics using Line Plot & Table blocks filtered by run tags.
   - **Per-Head Performance**: Embed confusion matrix images and latency metrics.
   - **Explainability**: Insert logged bar charts and saliency images.
3. Schedule auto-updates by saving the report with a filter (e.g., `tag = baseline`). Future runs matching the tag will refresh the report automatically.

## 7. Useful Run Filters
- `stage:classical` to isolate warm-up stage metrics.
- `stage:quantum` to diagnose quantum head fine-tuning.
- `stage:ensemble` for final calibration.
- Custom tags like `qubits=4` or `dataset=sovitrath` to compare experiments.

## 8. Housekeeping
- Archive experimental runs in W&B when no longer needed.
- Enable Team permissions for collaborators (Settings → Members) if working with the consortium.
- Periodically export metric tables using `wandb.Api()` for external reports or publications.

By following these steps, you can capture the full training lifecycle, compare classical vs. hybrid results, and produce automated reports for Hack the Horizon or future publications.
