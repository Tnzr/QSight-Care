# Recent Enhancements Summary

This page highlights the major improvements incorporated after stabilizing the training pipeline. Each section lists the motivation, the change itself, and the expected impact on model quality or operability.

---

## 1. Data Balance & Sampling
- Enabled **class-weighted cross entropy** when the dataset exposes imbalance statistics.
- Introduced an optional **`WeightedRandomSampler`** (controlled via `use_weighted_sampler`) seeded for reproducibility.
- Impact: minority classes receive more frequent gradient updates without overshooting batch sizes.

## 2. Stronger Data Augmentation
- Replaced simple resize-and-flip transforms with **random resized crops, affine jitter, blur, sharpness, and color adjustments** tailored for fundus imagery.
- Impact: improves generalization for subtle lesions and reduces overfitting to acquisition conditions.

## 3. Quantum Annealing Schedule
- Added `annealing_schedule` to `StageConfig` plus `QuantumClassificationHead.set_annealing_params`.
- Temperature and noise now follow cosine decay (configurable), logged each epoch as `quantum_temperature` / `quantum_noise`.
- Impact: smooths the quantum head’s optimization landscape, mimicking annealing as the circuit weights converge.

## 4. Visualization & Interpretability Backups
- Confusion matrices rendered with **row-normalized heatmaps** and saved locally under `trained_model/reports/visualizations/<stage>/epoch_##/` before WandB upload.
- Grad-CAM collages, overlays, and originals now persist to disk alongside logged artifacts.
- Impact: reproducible interpretability reviews even when WandB layout changes or network access is restricted.

## 5. Logging & Analytics Upgrades
- Epoch logs now include **`train_acc`** in addition to validation accuracy, giving tighter feedback on optimization phases.
- WandB metrics include annealing telemetry and balanced per-head accuracy summaries.
- Impact: easier triage of underperforming heads and hyperparameter scheduling decisions.

## 6. Configurability Improvements
- `run_training.py` and `StageConfig` accept additional knobs: `grad_accum_steps`, adaptive LR settings, forced `DataParallel`, and annealing payloads.
- Local config (`config/training.local.json`) showcases updated defaults (smaller batch for more updates, weighted sampler, tuned learning rates).
- Impact: simplifies experimentation and keeps experiments reproducible across team members.

---

**Next Candidates**
- Evaluate automated augmentation policies (e.g., RandAugment) once baseline stabilizes.
- Track annealing efficacy across runs; adjust schedule length when quantum stage early-stops frequently.
- Consider packaging visualization export into a reusable CLI for inference-time audits.
