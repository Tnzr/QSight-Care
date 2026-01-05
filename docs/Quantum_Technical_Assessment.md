# Quantum-Classical Architecture Technical Assessment

## Context and Objectives
- **Program**: Hack the Horizon (African Quantum Consortium)
- **Goal**: Deliver a clinically useful diabetic retinopathy system that proves a modular quantum head can augment classical inference while remaining resilient to limited quantum access.
- **Drivers**:
  - Maintain >=80% classical workload to ensure reproducibility and real-world deployability.
  - Preserve the ability to benchmark and publish an interchangeable quantum module.
  - Keep the pipeline compatible with both emulated (Qiskit) and future hardware-backed runs.

## Current Baseline
- Vision encoder (ViT-B/16) → 2048-dim latent.
- Compression module reduces to 30-dim latent for quantum interfacing.
- Head portfolio:
  1. **Classical Head A** – operates on full 2048 latent.
  2. **Classical Head B** – operates on compressed 30 latent.
  3. **Quantum Classification Head** – TorchConnector around a SamplerQNN (4 qubits today).
- Dynamic ensemble learns head weights during training and supports per-head interpretability in the Streamlit UI.

## Architectural Questions Under Review
1. **Where to locate the dimensionality reduction?**
   - External compression module feeding both Classical Head B and Quantum Head.
   - Internal projection layer inside the Quantum Head only.
2. **How to train the hybrid pipeline?**
   - Joint training of encoder + classical heads + quantum head (slow in simulation, risk of gradient noise).
   - Two-stage training: classical stack first (freeze encoder), then quantum head fine-tuning.
3. **How many qubits are required?**
   - Trade-off between representational capacity and circuit fidelity/noise when moving to hardware.

## Option Analysis
| Option | Description | Pros | Cons | Recommendation |
| --- | --- | --- | --- | --- |
| **A. External Compression Module (status quo)** | Keep a shared compression module (2048→30) feeding classical head B and quantum head. | - One compression step supports both classical and quantum benchmarking.<br>- Encoder remains modular; compression weights trained with rich classical gradients.<br>- Simplifies logging of classical vs quantum performance on identical features. | - Quantum head still requires additional 30→qubit projection.<br>- Two-stage training must ensure compression weights remain useful for quantum head. | **Preferred for benchmarking parity**; retain while allowing quantum head to learn an additional projection internally. |
| **B. Internal Projection Only** | Remove shared compression; the quantum head performs its own linear projection 2048→qubits. | - Simpler conceptual integration; fewer moving parts.<br>- Quantum head fully encapsulates quantum-specific preprocessing. | - Breaks comparability with Classical Head B (no shared 30-dim latent).<br>- Encoder gradients exclusively classical; quantum head may underfit without an intermediate bottleneck.<br>- Harder to reuse 30-dim features for classical ablation studies. | Defer; useful for pure quantum-ablation experiments but weaker for research benchmarking. |
| **C. Dual Compression (shared 30-dim + quantum-internal 30→qubits)** | Keep current compression module and allow the quantum head to apply its own projection into qubits. | - Preserves benchmark fairness.<br>- Enables tuning of qubit count independent of compressed latent width.<br>- Provides an interpretable "quantum adapter" layer for hardware-specific calibration. | - Slightly deeper quantum head and more parameters.<br>- Requires careful freezing strategy to avoid training instability. | **Adopted path**: maintain shared compression and fine-tune the quantum adapter per deployment target. |

## Training Strategy Assessment
- **Joint Training**:
  - *Benefits*: Single optimization pass; ensemble learns true joint behavior.
  - *Risks*: SamplerQNN backprop in simulation is slow; gradients noisy; may block experimentation cadence.
- **Staged Training (recommended path)**:
  1. Train encoder + classical heads + ensemble without quantum head (faster convergence, stable features).
  2. Freeze encoder + compression; train quantum head (and optionally light-touch fine-tune classical head B for fairness).
  3. Finetune ensemble weights with small learning rate across all heads.
  - *Benefits*: Quantum training isolated; experiments reproducible; quantum hardware pipeline can swap in by replaying stage 2.
  - *Mitigations*: Capture compressed features after stage 1 (already supported) to accelerate stage 2.

## Qubit Count Considerations
- **4–6 qubits**: tractable on simulators and near-term hardware; limited expressivity but best for prototyping and hardware demos.
- **8–12 qubits**: sweet spot for richer decision boundaries; may require circuit transpilation and error mitigation plans.
- **>12 qubits**: currently impractical for reliable hardware access; simulation becomes exponential.

**Strategy**: Start with 4-qubit adapter for development, expand to 8 or 10 qubits during benchmarking once compression stability is verified. Maintain parameterized ansatz depth adjustable via configuration (reps).

## Benchmarking & Evaluation Plan
- **Per-Head Metrics**:
  - Accuracy, F1, confusion matrix for Head A/B/Quantum and ensemble.
  - Calibration plots + Brier score to relate confidence vs accuracy.
- **System Metrics**:
  - Inference latency per head (CPU vs GPU vs quantum simulator).
  - Energy usage proxies (GPU power draw; for quantum, document backend energy/shot cost when available).
  - Parameter count per head and overall memory footprint.
- **Quantum-Specific**:
  - Shot noise sensitivity (variance across repeated runs).
  - Circuit depth, 2-qubit gate count post-transpilation.
  - Hardware execution success probability and queue latency (once real backend is used).

Data logging hooks already present in the Streamlit inference UI should be extended to capture per-head runtime and prediction variance. Training loop must store head-specific confusion matrices and ensemble weights per epoch.

## Quantum Hardware Enablement
1. **Simulation Baseline**: Continue with `Sampler` (local statevector) for development.
2. **Qiskit Runtime Migration**:
   - Prepare transpilation workflow using `FakeBackend` for circuit validation.
   - Parameterize shots, optimization level, and error mitigation strategies (Zero-Noise Extrapolation, measurement error mitigation).
3. **Hardware Execution**:
   - Integrate Qiskit Runtime session management (token, backend selection).
   - Utilize saved compressed features to minimize active quantum runtime.
   - Record queue times and execution costs for reporting.

## Implementation Roadmap
1. **Documentation & Config** (this file): baseline decisions.
2. **Code Updates**:
   - Ensure quantum head exposes config for qubit count, ansatz depth, projection dims.
   - Add hooks in trainer for staged quantum fine-tuning.
   - Extend logging utilities for per-head confusion matrices + latency capture.
3. **Benchmark Campaign**:
   - Baseline classical-only vs hybrid on validation split.
   - Iterate qubit counts and record metrics table.
   - Produce report summarizing trade-offs for publication.
4. **Hardware Trial**:
   - Run reduced dataset batch on available hardware backend, collect fidelity/latency data.

## Publication & Consortium Alignment
- Highlight modular adapter concept enabling quantum plug-and-play.
- Emphasize 80% classical workload to satisfy reliability demands while showcasing quantum value-add.
- Prepare reproducible notebooks (Qiskit + PyTorch) demonstrating staged training for community adoption.

## Next Steps Checklist
- [ ] Update training code to expose staged training switches (freeze encoder/compression during quantum head fit).
- [ ] Implement metric logging for confusion matrices and latency per head.
- [ ] Draft benchmarking notebook capturing classical-only vs hybrid comparisons.
- [ ] Prototype 8-qubit configuration to stress test compression adequacy.
- [ ] Schedule initial Qiskit Runtime dry-run using fake backend to validate transpilation and measurement mitigation.
