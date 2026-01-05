# Hybrid Inference User Guide

This guide explains how to load trained checkpoints, run predictions, and integrate the hybrid model into downstream applications (Streamlit app, benchmarking notebooks, or custom Python scripts).

## 1. What the Inference Package Provides

The module `model/production/inference.py` exposes:
- `load_model(checkpoint_dir)`: Loads weights, metadata, and the saved training summary from an output directory.
- `predict_image(model, image_path, device, transform, class_names)`: Returns per-class probabilities, head-specific confidences, ensemble weights, and latency estimates for a single image.
- `DEFAULT_TRANSFORM` and `preprocess_image`: Consistent preprocessing utilities for fundus images.

All functions are re-exported via `model.production`, so you can import from either location.

## 2. Loading a Trained Model

```python
from model.production import load_model

checkpoint_dir = "outputs/tanlikesmath_vit"
model, metadata, summary = load_model(checkpoint_dir)
print(metadata["classes"])         # class labels
print(summary.get("history", [])[:1])  # optional: training history
```

The loader expects the directory produced by the trainer (see Training Guide §7) and reconstructs the model architecture using the stored metadata. If the checkpoint was trained without the quantum head, the loader keeps that configuration consistent.

## 3. Single-Image Prediction

```python
from pathlib import Path
from model.production import predict_image, DEFAULT_TRANSFORM

image_path = Path("/data/samples/000c1434d8d7.png")
result = predict_image(
    model,
    image_path,
    device="cuda",          # or "cpu"
    transform=DEFAULT_TRANSFORM,
)
print("Predicted class:", result["prediction"])
print("Ensemble probabilities:", result["probabilities"])
print("Head weights:", result["ensemble_weights"])
print("Latencies (s):", result["latencies"])
```

Returned fields:
- `prediction`: Label with the highest ensemble probability.
- `probabilities`: Ensemble softmax output per class.
- `per_head_probabilities`: Softmax outputs for Classical A, Classical B, and Quantum heads.
- `ensemble_weights`: Learned weights the ensemble used for the final decision.
- `uncertainties`: Average uncertainty estimated for each head.
- `latencies`: Measured latency for encoder, compression, each head, and ensemble fusion.

## 4. Batch Inference

For evaluating multiple images, wrap your own dataloader and call `model` directly:
```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

inference_transform = DEFAULT_TRANSFORM
folder_dataset = datasets.ImageFolder(
    root="/data/retinopathy/validation",
    transform=inference_transform,
)
loader = DataLoader(folder_dataset, batch_size=16, shuffle=False)

model = model.to("cuda").eval()
predictions = []
with torch.no_grad():
    for images, _ in loader:
        outputs = model(images.to("cuda"), return_all=True)
        probs = torch.softmax(outputs.final_output, dim=1)
        predictions.extend(probs.argmax(dim=1).cpu().tolist())
```

Use the structured output (`ModelForwardOutput`) to log per-head scores, ensemble weights, or latency statistics across the batch.

## 5. Integrating with the Streamlit App

The existing `streamlit_app.py` imports helpers from `model.production` (via `hybrid_components`). To point the UI at a trained checkpoint:
1. Place the exported directory (containing `complete_checkpoint.pth`) next to the app or update the path in the Streamlit configuration.
2. Inside the app, use:
   ```python
   from model.production import load_model, predict_image
   model, metadata, _ = load_model(checkpoint_dir)
   result = predict_image(model, uploaded_file, device="cuda")
   ```
3. Display fields such as `probabilities` and `per_head_probabilities` to show head-specific behaviour in the UI.

## 6. Notebook Convenience Wrapper

The file `model/hybrid_components.py` re-exports the production API for backward compatibility with notebooks (`Hybrid_Benchmark.ipynb`, `QML_Qiskit.ipynb`). Importing from `hybrid_components` still yields the production implementations, so no refactor is needed when migrating older notebooks.

## 7. Metadata & Feature Reuse

After training, the output directory also contains:
- `training_summary.json`: Stage-level metrics and confusion matrices (useful for dashboards or reports).
- `quantum_training_data.pkl`: Cached compressed features (`train`/`val`) that can be fed into quantum-only experiments without re-running the encoder.
- `model_info.json`: Records encoder type, whether the quantum head was enabled, qubit/shots configuration, and class labels.

Example: exporting features for a custom quantum study
```python
import pickle

with open(checkpoint_dir / "quantum_training_data.pkl", "rb") as handle:
    data = pickle.load(handle)

train_features = data["train_features"]
train_labels = data["train_labels"]
# Feed into quantum fine-tuning or classical baselines.
```

## 8. Troubleshooting

- **Mismatched device**: Ensure the model and input tensors share the same device (`model.to(device)` before inference).
- **Quantum head disabled warning**: If `qiskit` is missing, the model stores `quantum_enabled = False` and returns zero logits for the quantum head. Install `qiskit` and re-train to enable it.
- **Different image sizes**: Use `DEFAULT_TRANSFORM` or a custom transform that outputs tensors of shape `(3, 224, 224)` and matches the normalisation used during training.
- **Missing metadata**: Confirm that `model_info.json` and `complete_checkpoint.pth` exist. They must live inside the same directory passed to `load_model`.

Armed with these helpers you can embed the hybrid model in services, perform offline benchmarking, or extend the UI while retaining visibility into each head’s contribution.
