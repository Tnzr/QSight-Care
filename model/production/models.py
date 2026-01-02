"""Hybrid model components for classical + quantum diabetic retinopathy pipeline."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _inject_cuda_library_paths() -> None:
    """Augment LD_LIBRARY_PATH with common CUDA library locations."""

    candidates: List[Path] = []
    preferred_cublas: Optional[Path] = None
    preferred_cublas_lt: Optional[Path] = None

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix_path = Path(conda_prefix) / "lib"
        if prefix_path.exists():
            candidates.append(prefix_path)

    cuda_packages = (
        "custatevec",
        "custabilizer",
        "cutensornet",
        "cuquantum",
        "cudensitymat",
        "nvidia/cublas",
        "nvidia/cusolver",
    )

    for entry in sys.path:
        try:
            site_path = Path(entry)
        except TypeError:
            continue
        if not site_path.exists() or "site-packages" not in site_path.as_posix():
            continue
        for package_name in cuda_packages:
            package_path = site_path / package_name
            if not package_path.exists():
                continue
            lib_path = package_path / "lib"
            if lib_path.exists():
                candidates.append(lib_path)
                if "nvidia/cublas" in package_name:
                    libcublas_candidate = lib_path / "libcublas.so.12"
                    if libcublas_candidate.exists():
                        preferred_cublas = libcublas_candidate
                    libcublas_lt_candidate = lib_path / "libcublasLt.so.12"
                    if libcublas_lt_candidate.exists():
                        preferred_cublas_lt = libcublas_lt_candidate
            else:
                candidates.append(package_path)
                if "nvidia/cublas" in package_name:
                    libcublas_candidate = package_path / "libcublas.so.12"
                    if libcublas_candidate.exists():
                        preferred_cublas = libcublas_candidate
                    libcublas_lt_candidate = package_path / "libcublasLt.so.12"
                    if libcublas_lt_candidate.exists():
                        preferred_cublas_lt = libcublas_lt_candidate

    if not candidates:
        return

    existing_raw = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing_raw.split(":") if part]
    normalized_existing = {Path(part).resolve() for part in existing_parts if part}
    updated = False

    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists() or resolved in normalized_existing:
            continue
        existing_parts.insert(0, str(resolved))
        normalized_existing.add(resolved)
        updated = True

    if updated:
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_parts)
        if sys.platform.startswith("linux"):
            try:
                import importlib

                importlib.invalidate_caches()
            except Exception:  # pragma: no cover - best effort
                pass

    if sys.platform.startswith("linux"):
        preload_existing = os.environ.get("LD_PRELOAD", "").split()
        for candidate in (preferred_cublas_lt, preferred_cublas):
            if candidate is None:
                continue
            candidate_str = str(candidate)
            if candidate_str not in preload_existing:
                preload_existing.insert(0, candidate_str)
        if preload_existing:
            os.environ["LD_PRELOAD"] = " ".join(preload_existing)

    if preferred_cublas_lt is not None and sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL(str(preferred_cublas_lt), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).debug("Failed to preload libcublasLt: %s", exc)

    if preferred_cublas is not None and sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL(str(preferred_cublas), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).debug("Failed to preload libcublas: %s", exc)


_inject_cuda_library_paths()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


try:
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.primitives import Sampler as BaseSampler
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import SamplerQNN

    QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BaseSampler = None  # type: ignore[assignment]
    QISKIT_AVAILABLE = False

if QISKIT_AVAILABLE:
    _inject_cuda_library_paths()
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import Sampler as AerSampler

        QISKIT_AER_AVAILABLE = True
        AER_IMPORT_ERROR: Optional[str] = None
    except ImportError as exc:  # pragma: no cover - optional dependency
        AerSimulator = None  # type: ignore[assignment]
        AerSampler = None  # type: ignore[assignment]
        QISKIT_AER_AVAILABLE = False
        AER_IMPORT_ERROR = f"ImportError: {exc}"
    except OSError as exc:  # pragma: no cover - diagnostic
        AerSimulator = None  # type: ignore[assignment]
        AerSampler = None  # type: ignore[assignment]
        QISKIT_AER_AVAILABLE = False
        AER_IMPORT_ERROR = f"OSError: {exc}"
else:
    QISKIT_AER_AVAILABLE = False
    AerSimulator = None  # type: ignore[assignment]
    AerSampler = None  # type: ignore[assignment]
    AER_IMPORT_ERROR = "Qiskit unavailable"


LOGGER = logging.getLogger(__name__)
SAMPLER_EXECUTION_LOCK = threading.Lock()

LATENCY_KEYS = [
    "encoder",
    "compression",
    "classical_a",
    "classical_b",
    "quantum",
    "ensemble",
]

if QISKIT_AVAILABLE:
    class DataParallelTorchConnector(TorchConnector):
        """TorchConnector variant that plays nicely with torch.nn.DataParallel."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if "_weights" in self._parameters:
                self._parameters.pop("_weights")
            object.__setattr__(self, "_weights", self._parameters.get("weight"))

        @property
        def weight(self):  # type: ignore[override]
            if "weight" not in self._parameters:
                raise AttributeError("weight")
            return self._parameters["weight"]

        @weight.setter  # type: ignore[override]
        def weight(self, value):
            if not isinstance(value, torch.nn.Parameter):
                value = torch.nn.Parameter(value)
            self._parameters["weight"] = value
            object.__setattr__(self, "_weights", value)


class VisionEncoder(nn.Module):
    """Backbone encoder supporting ViT or ResNet features."""

    def __init__(self, encoder_type: str = "vit", pretrained: bool = True) -> None:
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.feature_dim: int = 0
        self.cam_target_layer: Optional[nn.Module] = None

        if self.encoder_type in {"vit", "vit_b_16", "vitb16"}:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            vit = models.vit_b_16(weights=weights)
            vit.heads = nn.Identity()
            self.encoder = vit
            self.projection = nn.Linear(768, 2048)
            self.feature_dim = 2048
        elif self.encoder_type in {"resnet", "resnet50", "resnet_50"}:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool,
                nn.Flatten(),
            )
            self.feature_dim = int(resnet.fc.in_features)
            self.cam_target_layer = resnet.layer4
            self.projection = nn.Identity()
        elif self.encoder_type in {"resnet101", "resnet_101"}:
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet101(weights=weights)
            self.encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool,
                nn.Flatten(),
            )
            self.feature_dim = int(resnet.fc.in_features)
            self.cam_target_layer = resnet.layer4
            self.projection = nn.Identity()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.encoder_type in {"vit", "vit_b_16", "vitb16"}:
            features = self.projection(features)
        return features


class CompressionModule(nn.Module):
    def __init__(self, input_dim: int = 2048, compressed_dim: int = 30) -> None:
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, compressed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compressor(x)


class ClassicalHeadA(nn.Module):
    def __init__(self, input_dim: int = 2048, num_classes: int = 5) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ClassicalHeadB(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 5) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class QuantumClassificationHead(nn.Module):
    """Quantum classification head backed by a Qiskit SamplerQNN."""

    def __init__(
        self,
        input_dim: int = 32,
        num_classes: int = 5,
        num_qubits: int = 4,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "Qiskit is required for the quantum head. Install qiskit and qiskit-machine-learning."
            )
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.q_device = torch.device("cpu")
        self.input_projection = nn.Linear(input_dim, num_qubits)
        feature_map = ZZFeatureMap(num_qubits, reps=2)
        ansatz = RealAmplitudes(num_qubits, reps=2)
        circuit = feature_map.compose(ansatz)
        self._sampler_backend = None
        sampler_backend = None
        sampler = None
        if QISKIT_AER_AVAILABLE and AerSimulator is not None and AerSampler is not None:
            try:
                base_backend = AerSimulator()
                device_probe = getattr(base_backend, "available_devices", None)
                available_devices = tuple(device_probe() if callable(device_probe) else ())
                available_devices = tuple(device.upper() for device in available_devices)
                LOGGER.info(
                    "AerSimulator reported devices: %s",
                    ", ".join(available_devices) or "unknown",
                )
                use_gpu = "GPU" in available_devices or torch.cuda.is_available()
                if use_gpu and not os.environ.get("QISKIT_AER_CUDA"):
                    os.environ["QISKIT_AER_CUDA"] = "1"
                    LOGGER.debug("Auto-enabled QISKIT_AER_CUDA for GPU acceleration")
                backend_options: dict[str, Any] = {"device": "GPU" if use_gpu else "CPU"}
                if use_gpu:
                    backend_options["method"] = "statevector"
                    backend_options["cuStateVec_enable"] = True
                sampler = AerSampler(backend_options=backend_options, run_options={"shots": shots})
                sampler_backend = backend_options
                device_label = backend_options.get("device", "GPU" if use_gpu else "CPU")
                method_label = backend_options.get("method", "automatic")
                LOGGER.info(
                    "Quantum head using AerSimulator backend on %s with method %s",
                    device_label,
                    method_label,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                LOGGER.warning("Failed to initialize Aer GPU sampler; falling back to default Sampler: %s", exc)
                sampler = None

        if sampler is None:
            if BaseSampler is None:
                raise RuntimeError("Qiskit Sampler backend is unavailable")
            sampler = BaseSampler(options={"shots": shots})
            LOGGER.info("Quantum head using default Qiskit Sampler backend (CPU)")
        else:
            self._sampler_backend = sampler_backend
        self.qnn = SamplerQNN(
            sampler=sampler,
            circuit=circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            sparse=False,
            input_gradients=True,
        )
        initial_weights = torch.zeros(self.qnn.num_weights, dtype=torch.double)
        self.q_layer = DataParallelTorchConnector(self.qnn, initial_weights=initial_weights)
        output_dim = 2**num_qubits
        self.post_process = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("noise_scale", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        temperature = self.temperature.to(projected.device)
        scaled = projected / temperature.clamp_min(1e-3)
        if self.training and float(self.noise_scale.item()) > 0.0:
            scaled = scaled + torch.randn_like(scaled) * self.noise_scale.to(projected.device)
        projected_cpu = torch.tanh(scaled).to(self.q_device, dtype=torch.double)
        # Serialize access to the shared Qiskit sampler to avoid re-entrancy issues under DataParallel.
        with SAMPLER_EXECUTION_LOCK:
            quantum_raw = self.q_layer(projected_cpu)
        quantum_raw = quantum_raw.to(dtype=torch.float32)
        quantum_raw = quantum_raw.to(projected.device)
        logits = self.post_process(quantum_raw)
        return logits

    def set_annealing_params(self, temperature: float, noise_scale: float = 0.0) -> None:
        temp_value = max(float(temperature), 1e-3)
        noise_value = max(float(noise_scale), 0.0)
        self.temperature.copy_(torch.tensor(temp_value, dtype=torch.float32, device=self.temperature.device))
        self.noise_scale.copy_(torch.tensor(noise_value, dtype=torch.float32, device=self.noise_scale.device))

    def get_annealing_params(self) -> Tuple[float, float]:
        return float(self.temperature.item()), float(self.noise_scale.item())


class DynamicEnsemble(nn.Module):
    def __init__(self, num_heads: int = 3, init_temp: float = 1.0) -> None:
        super().__init__()
        self.base_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.uncertainty_scales = nn.Parameter(torch.ones(num_heads))

    def forward(
        self,
        head_outputs: List[torch.Tensor],
        uncertainties: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active_mask is None:
            mask = torch.ones(len(head_outputs), device=head_outputs[0].device)
        else:
            mask = active_mask.to(head_outputs[0].device).float()
        weights = F.softmax(self.base_weights / self.temperature, dim=0)
        weights = weights * mask
        if weights.sum() <= 0:
            weights = torch.ones_like(weights) * mask
        weights = weights / (weights.sum() + 1e-8)

        if uncertainties is not None and self.training:
            if uncertainties.dim() == 1:
                uncertainties = uncertainties.unsqueeze(0)
            scaled_uncertainties = uncertainties * self.uncertainty_scales.unsqueeze(0)
            confidence = 1.0 / (scaled_uncertainties + 1e-8)
            batch_confidence = confidence.mean(dim=0)
            confidence_weights = F.softmax(batch_confidence, dim=0)
            with torch.no_grad():
                predictions = torch.stack([torch.argmax(out, dim=1) for out in head_outputs], dim=1)
                predictions_float = predictions.float()
                max_vals, _ = predictions_float.max(dim=1)
                min_vals, _ = predictions_float.min(dim=1)
                agreement_mask = (max_vals == min_vals).float()
                agreement = agreement_mask.mean()
            uncertainty_weight = 0.7 * (1 - agreement) + 0.3
            weights = (1 - uncertainty_weight) * weights + uncertainty_weight * confidence_weights

        weights = weights * mask
        if weights.sum() <= 0:
            weights = mask / (mask.sum() + 1e-8)
        weights = weights / (weights.sum() + 1e-8)

        if not self.training:
            weights = weights.detach()

        final_output = None
        for idx, (weight, out) in enumerate(zip(weights, head_outputs)):
            if mask[idx] > 0:
                final_output = out * weight if final_output is None else final_output + weight * out
        if final_output is None:
            final_output = head_outputs[0]
        return final_output, weights


@dataclass
class ModelForwardOutput:
    latent_features: torch.Tensor
    compressed_features: torch.Tensor
    output_a: torch.Tensor
    output_b: torch.Tensor
    output_c: torch.Tensor
    final_output: torch.Tensor
    ensemble_weights: torch.Tensor
    uncertainties: torch.Tensor
    latencies: Dict[str, float]
    active_mask: Optional[torch.Tensor]


class ClassicalDRModel(nn.Module):
    """Classical + quantum hybrid model with dynamic ensemble."""

    def __init__(
        self,
        encoder_type: str = "vit",
        num_classes: int = 5,
        compressed_dim: int = 30,
        pretrained: bool = True,
        quantum_enabled: bool = True,
        quantum_qubits: int = 4,
        quantum_shots: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.compressed_dim = compressed_dim
        self.quantum_enabled = quantum_enabled and QISKIT_AVAILABLE
        self.vision_encoder = VisionEncoder(encoder_type=encoder_type, pretrained=pretrained)
        encoder_feature_dim = getattr(self.vision_encoder, "feature_dim", 2048) or 2048
        self.encoder_feature_dim = encoder_feature_dim
        self.compression = CompressionModule(input_dim=encoder_feature_dim, compressed_dim=compressed_dim)
        self.classical_head_a = ClassicalHeadA(input_dim=encoder_feature_dim, num_classes=num_classes)
        self.classical_head_b = ClassicalHeadB(input_dim=compressed_dim, num_classes=num_classes)
        self.quantum_qubits = quantum_qubits
        self.quantum_shots = quantum_shots
        if quantum_enabled and not QISKIT_AVAILABLE:
            LOGGER.warning("Qiskit not available; quantum head will be disabled.")
        if self.quantum_enabled:
            self.quantum_head = QuantumClassificationHead(
                input_dim=compressed_dim,
                num_classes=num_classes,
                num_qubits=quantum_qubits,
                shots=quantum_shots,
            )
        else:
            self.quantum_head = None
        self.ensemble = DynamicEnsemble(num_heads=3)
        self.head_order = ["classical_a", "classical_b", "quantum"]
        self._latency_tracking = False
        self._active_mask: Optional[List[float]] = None

    def enable_latency_tracking(self, enabled: bool = True) -> None:
        self._latency_tracking = enabled

    def set_active_mask(self, mask: Optional[Iterable[float]]) -> None:
        if mask is None:
            self._active_mask = None
            return
        mask_list = list(mask)
        if len(mask_list) != 3:
            raise ValueError("Active mask must contain exactly three entries")
        self._active_mask = [float(value) for value in mask_list]

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = True,
        active_mask: Optional[torch.Tensor] = None,
        track_latency: Optional[bool] = None,
    ) -> ModelForwardOutput | torch.Tensor:
        if track_latency is not None:
            self._latency_tracking = track_latency

        mask_source = active_mask
        if mask_source is None and self._active_mask is not None:
            mask_source = self._active_mask

        mask_tensor = None
        if mask_source is not None:
            mask_tensor = torch.as_tensor(mask_source, device=x.device, dtype=torch.float32)

        latencies: Dict[str, float] = {}

        start = time.perf_counter()
        latent_features = self.vision_encoder(x)
        latencies["encoder"] = time.perf_counter() - start

        start = time.perf_counter()
        compressed_features = self.compression(latent_features)
        latencies["compression"] = time.perf_counter() - start

        head_outputs: List[torch.Tensor] = []

        start = time.perf_counter()
        output_a = self.classical_head_a(latent_features)
        latencies["classical_a"] = time.perf_counter() - start
        head_outputs.append(output_a)

        start = time.perf_counter()
        output_b = self.classical_head_b(compressed_features)
        latencies["classical_b"] = time.perf_counter() - start
        head_outputs.append(output_b)

        quantum_active = self.quantum_enabled and self.quantum_head is not None
        if quantum_active and mask_tensor is not None and mask_tensor.numel() >= 3:
            quantum_active = bool(mask_tensor[2].item() > 0)

        if quantum_active:
            start = time.perf_counter()
            output_c = self.quantum_head(compressed_features)
            latencies["quantum"] = time.perf_counter() - start
        else:
            output_c = torch.zeros_like(output_b)
            latencies["quantum"] = 0.0
        head_outputs.append(output_c)

        with torch.no_grad():
            prob_a = F.softmax(output_a, dim=1)
            prob_b = F.softmax(output_b, dim=1)
            prob_c = F.softmax(output_c, dim=1)
            uncertainties = torch.tensor(
                [
                    1.0 - prob_a.max(dim=1)[0].mean(),
                    1.0 - prob_b.max(dim=1)[0].mean(),
                    1.0 - prob_c.max(dim=1)[0].mean(),
                ],
                device=x.device,
            )

        final_output, ensemble_weights = self.ensemble(head_outputs, uncertainties, active_mask=mask_tensor)
        latencies.setdefault("ensemble", 0.0)

        latency_values = [float(latencies.get(key, 0.0)) for key in LATENCY_KEYS]
        latency_tensor = torch.tensor(latency_values, device=x.device, dtype=torch.float32)

        payload = (
            latent_features,
            compressed_features,
            output_a,
            output_b,
            output_c,
            final_output,
            ensemble_weights,
            uncertainties,
            latency_tensor,
            mask_tensor if mask_tensor is not None else None,
        )
        if return_all:
            return payload
        return final_output

    def compute_losses(
        self,
        outputs: ModelForwardOutput,
        targets: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if loss_weights is None:
            loss_weights = {
                "classical_a": 1.0,
                "classical_b": 1.0,
                "quantum": 1.0,
                "ensemble": 0.5,
            }

        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=targets.device)
        weight_tensor: Optional[torch.Tensor] = None
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, device=targets.device, dtype=torch.float32)

        if loss_weights.get("classical_a", 0.0) > 0:
            loss_a = F.cross_entropy(outputs.output_a, targets, weight=weight_tensor)
            total_loss = total_loss + loss_weights["classical_a"] * loss_a
            losses["loss_a"] = loss_a
        else:
            losses["loss_a"] = torch.tensor(0.0, device=targets.device)

        if loss_weights.get("classical_b", 0.0) > 0:
            loss_b = F.cross_entropy(outputs.output_b, targets, weight=weight_tensor)
            total_loss = total_loss + loss_weights["classical_b"] * loss_b
            losses["loss_b"] = loss_b
        else:
            losses["loss_b"] = torch.tensor(0.0, device=targets.device)

        if loss_weights.get("quantum", 0.0) > 0:
            loss_c = F.cross_entropy(outputs.output_c, targets, weight=weight_tensor)
            total_loss = total_loss + loss_weights["quantum"] * loss_c
            losses["loss_c"] = loss_c
        else:
            losses["loss_c"] = torch.tensor(0.0, device=targets.device)

        if loss_weights.get("ensemble", 0.0) > 0:
            ensemble_loss = F.cross_entropy(outputs.final_output, targets, weight=weight_tensor)
            total_loss = total_loss + loss_weights["ensemble"] * ensemble_loss
            losses["ensemble_loss"] = ensemble_loss
        else:
            losses["ensemble_loss"] = torch.tensor(0.0, device=targets.device)

        losses["total_loss"] = total_loss
        return losses

    def extract_compressed_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent_features = self.vision_encoder(x)
            compressed_features = self.compression(latent_features)
        return compressed_features

    def set_stage(self, stage: str) -> None:
        stage = stage.lower()
        if stage == "classical":
            self._set_requires_grad(self.vision_encoder, True)
            self._set_requires_grad(self.compression, True)
            self._set_requires_grad(self.classical_head_a, True)
            self._set_requires_grad(self.classical_head_b, True)
            self._set_requires_grad(self.quantum_head, False)
            self._set_requires_grad(self.ensemble, True)
        elif stage == "quantum":
            self._set_requires_grad(self.vision_encoder, False)
            self._set_requires_grad(self.compression, False)
            self._set_requires_grad(self.classical_head_a, False)
            self._set_requires_grad(self.classical_head_b, False)
            self._set_requires_grad(self.quantum_head, True)
            self._set_requires_grad(self.ensemble, True)
        elif stage == "ensemble":
            self._set_requires_grad(self.vision_encoder, False)
            self._set_requires_grad(self.compression, False)
            self._set_requires_grad(self.classical_head_a, False)
            self._set_requires_grad(self.classical_head_b, False)
            self._set_requires_grad(self.quantum_head, False)
            self._set_requires_grad(self.ensemble, True)
        else:  # full
            self._set_requires_grad(self.vision_encoder, True)
            self._set_requires_grad(self.compression, True)
            self._set_requires_grad(self.classical_head_a, True)
            self._set_requires_grad(self.classical_head_b, True)
            self._set_requires_grad(self.quantum_head, True)
            self._set_requires_grad(self.ensemble, True)

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad


__all__ = [
    "VisionEncoder",
    "CompressionModule",
    "ClassicalHeadA",
    "ClassicalHeadB",
    "QuantumClassificationHead",
    "DynamicEnsemble",
    "ModelForwardOutput",
    "ClassicalDRModel",
    "QISKIT_AVAILABLE",
    "QISKIT_AER_AVAILABLE",
    "LATENCY_KEYS",
]
