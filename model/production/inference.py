"""Inference helpers for the hybrid classical + quantum model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image
from torchvision import transforms

from .models import ClassicalDRModel, ModelForwardOutput, LATENCY_KEYS

LOGGER = logging.getLogger(__name__)


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse %s: %s", path, exc)
        return {}


def load_model(checkpoint_dir: Path | str) -> tuple[ClassicalDRModel, Dict[str, object], Dict[str, object]]:
    """Load a trained model along with metadata and optional training summary."""

    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "phase1_classical_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    metadata = _load_json(checkpoint_dir / "model_info.json")
    training_summary = _load_json(checkpoint_dir / "training_summary.json")

    num_classes = int(metadata.get("num_classes", 5))
    encoder_type = metadata.get("encoder_type", "vit")
    pretrained = bool(metadata.get("pretrained", True))
    quantum_enabled = bool(metadata.get("quantum_enabled", True))
    raw_qubits = metadata.get("quantum_qubits", 4)
    raw_shots = metadata.get("quantum_shots", 1024)
    quantum_qubits = int(raw_qubits) if raw_qubits is not None else 4
    quantum_shots = int(raw_shots) if raw_shots is not None else 1024

    model = ClassicalDRModel(
        num_classes=num_classes,
        encoder_type=encoder_type,
        pretrained=pretrained,
        quantum_enabled=quantum_enabled,
        quantum_qubits=quantum_qubits,
        quantum_shots=quantum_shots,
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    class_list = metadata.get("classes")
    if isinstance(class_list, Iterable):
        model.classes = list(class_list)

    return model, metadata, training_summary


def _class_mapping(class_names: Optional[Dict[int, str] | List[str]], num_classes: int) -> Dict[int, str]:
    if class_names is None:
        return {idx: str(idx) for idx in range(num_classes)}
    if isinstance(class_names, dict):
        return class_names
    return {idx: name for idx, name in enumerate(class_names)}


def _normalize_forward_output(raw_output, device: torch.device) -> ModelForwardOutput:
    if isinstance(raw_output, ModelForwardOutput):
        return raw_output

    if isinstance(raw_output, (list, tuple)) and len(raw_output) == 10:
        latencies_payload = raw_output[8]
        if isinstance(latencies_payload, dict):
            latencies = {key: float(value) for key, value in latencies_payload.items()}
        elif isinstance(latencies_payload, torch.Tensor):
            latencies_payload = latencies_payload.detach().cpu()
            latencies = {
                key: float(latencies_payload[idx].item())
                for idx, key in enumerate(LATENCY_KEYS)
                if idx < latencies_payload.numel()
            }
        else:
            latencies = {
                key: float(latencies_payload[idx])
                for idx, key in enumerate(LATENCY_KEYS)
                if idx < len(latencies_payload)
            }

        active_payload = raw_output[9]
        if isinstance(active_payload, torch.Tensor):
            active_mask = active_payload.to(device)
        elif active_payload is None:
            active_mask = None
        else:
            active_mask = torch.as_tensor(active_payload, device=device, dtype=torch.float32)

        return ModelForwardOutput(
            latent_features=raw_output[0],
            compressed_features=raw_output[1],
            output_a=raw_output[2],
            output_b=raw_output[3],
            output_c=raw_output[4],
            final_output=raw_output[5],
            ensemble_weights=raw_output[6],
            uncertainties=raw_output[7],
            latencies=latencies,
            active_mask=active_mask,
        )

    raise TypeError(f"Unexpected forward output type: {type(raw_output)!r}")


def _predict_logits(model: ClassicalDRModel, tensor: torch.Tensor) -> ModelForwardOutput:
    with torch.no_grad():
        raw_output = model(
            tensor,
            return_all=True,
            track_latency=True,
        )
    return _normalize_forward_output(raw_output, tensor.device)


def predict_image(
    model: ClassicalDRModel,
    image_path: Path | str,
    device: str = "cpu",
    transform=DEFAULT_TRANSFORM,
    class_names: Dict[int, str] | List[str] | None = None,
) -> Dict[str, object]:
    """Predict a single image and return per-head probabilities and metadata."""

    model = model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = _predict_logits(model, tensor)

    final_probs = torch.softmax(outputs.final_output, dim=1).cpu().numpy()[0]
    head_probs = {
        "classical_a": torch.softmax(outputs.output_a, dim=1).cpu().numpy()[0],
        "classical_b": torch.softmax(outputs.output_b, dim=1).cpu().numpy()[0],
        "quantum": torch.softmax(outputs.output_c, dim=1).cpu().numpy()[0],
    }

    mapping = _class_mapping(
        class_names or getattr(model, "classes", None),
        num_classes=final_probs.shape[0],
    )

    probabilities = {mapping[idx]: float(prob) for idx, prob in enumerate(final_probs)}
    per_head = {
        head: {mapping[idx]: float(prob) for idx, prob in enumerate(probs)}
        for head, probs in head_probs.items()
    }

    predicted_idx = int(final_probs.argmax())
    prediction = mapping[predicted_idx]

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "per_head_probabilities": per_head,
        "ensemble_weights": outputs.ensemble_weights.detach().cpu().numpy().tolist(),
        "uncertainties": outputs.uncertainties.detach().cpu().numpy().tolist(),
        "latencies": outputs.latencies,
    }


def preprocess_image(image):
    """Preprocess image for model input"""
    return DEFAULT_TRANSFORM(image).unsqueeze(0)


__all__ = ["load_model", "predict_image", "preprocess_image", "DEFAULT_TRANSFORM"]