# Debugging Sessions Retrospective

This document captures the major debugging efforts completed while stabilizing the QSight hybrid (classical + quantum) training pipeline. Each entry summarizes the symptoms, investigation path, implemented fix, and outstanding follow-up tasks.

---

## 1. Qiskit Sampler Crash Under `DataParallel`
- **Symptoms:** Training halted with a `QiskitError` whenever the quantum head executed inside `torch.nn.DataParallel`.
- **Investigation:** Reproduced the crash with the quantum stage active; traced stack to simultaneous sampler invocations. Confirmed Qiskit sampler back-end is not thread safe.
- **Resolution:** Wrapped sampler execution inside a global threading lock and disabled `DataParallel` for stages where only the quantum head runs.
- **Follow-up:** Monitor performance impact; revisit once Qiskit exposes a re-entrant GPU sampler API.

## 2. Aer GPU Backend Failing to Initialize
- **Symptoms:** Qiskit defaulted to CPU simulation despite GPUs being available; sampler logs lacked CUDA enablement.
- **Investigation:** Checked environment variables and `AerSimulator` diagnostics; `QISKIT_AER_CUDA` was unset.
- **Resolution:** Added auto-detection to set `QISKIT_AER_CUDA=1` when GPUs are present, and hardened backend option construction.
- **Follow-up:** Keep backend logging enabled in production runs to catch regression or driver mismatches.

## 3. Stage Scheduling Mask Regression
- **Symptoms:** Heads continued to train outside their scheduled stages; gradients leaked into frozen modules.
- **Investigation:** Reviewed mask propagation and `StageConfig` parsing; discovered stale masks were left on the model between stages.
- **Resolution:** Forced the trainer to set the active mask on every epoch and ensured stage transitions invoke `ClassicalDRModel.set_stage`.
- **Follow-up:** Consider regression tests around mask handling when adding new heads.

## 4. WandB Confidence Table Crash (`mask_tensor` undefined)
- **Symptoms:** Validation logging threw a `NameError` when confidence tables were generated without an active mask.
- **Investigation:** Replayed the run with verbose logging; pinpointed missing local assignment before using `mask_tensor`.
- **Resolution:** Stored the stage mask early in the visualization path so the table builder always sees a defined tensor.
- **Follow-up:** Add static typing (e.g., `mypy`) for the logging helpers to prevent similar mistakes.

## 5. Grad-CAM Duplicated Classes and Missing Visuals
- **Symptoms:** WandB Grad-CAM panels repeated the same class and occasionally failed to render.
- **Investigation:** Audited sampling logic; observed the helper reused the first few validation samples regardless of class.
- **Resolution:** Ensured `_ensure_interpretability_samples` gathers class-diverse examples and hardened the Grad-CAM helper against missing hooks.
- **Follow-up:** Automate a smoke test that exercises the interpretability pipeline in CI.

## 6. Visualization Artifacts Lost When WandB Layout Changed
- **Symptoms:** Confusion matrices and Grad-CAM collages disappeared from WandB reports after layout tweaks.
- **Investigation:** Confirmed Matplotlib figures were only streamed to WandB without local copies.
- **Resolution:** Added stage/epoch-specific directories under `trained_model/reports/visualizations` and saved normalized confusion matrices plus Grad-CAM originals/overlays locally before logging.
- **Follow-up:** Periodically prune old visualization folders or archive them with run metadata.

---

**Maintainer Notes**
- Keep the debugging log up to date whenever new production issues surface.
- Flag any recurring problem areas (quantum sampling, visualization exports, stage masking) for refactor prioritization.
