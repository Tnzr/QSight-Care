"""Hybrid training utilities for the production pipeline."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, RandomSampler
from tqdm import tqdm

try:
    from graphviz import Digraph
except ImportError:  # pragma: no cover - optional dependency
    Digraph = None

from .models import ClassicalDRModel, ModelForwardOutput, LATENCY_KEYS
from .quantization import QuantizationConfig, QuantizationManager

if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

LOGGER = logging.getLogger(__name__)

HEAD_DISPLAY_NAMES = {
    "classical_a": "Full-Resolution",
    "classical_b": "Compressed",
    "quantum": "Quantum",
    "ensemble": "Ensemble",
}


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


@dataclass
class StageConfig:
    name: str
    num_epochs: int
    learning_rate: float
    weight_decay: float = 1e-5
    patience: int = 5
    active_mask: Optional[Iterable[int]] = None
    loss_weights: Optional[Dict[str, float]] = None
    grad_accum_steps: int = 1
    force_data_parallel: Optional[bool] = None
    adaptive_lr: bool = True
    adaptive_patience: int = 4
    adaptive_threshold: float = 1e-4
    lr_decay_factor: float = 0.5
    min_lr: float = 1e-5
    annealing_schedule: Optional[Dict[str, Any]] = None
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    log_alias: Optional[str] = None
    skip_ensemble: bool = False


@dataclass
class WandBConfig:
    use_wandb: bool = False
    project: str = "qsight-care"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: str = "online"
    notes: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@dataclass
class LossConfig:
    type: str = "class_balanced"
    focal_gamma: float = 2.0
    class_balanced_beta: float = 0.999


@dataclass
class TrainingConfig:
    stages: List[StageConfig] = field(default_factory=list)
    device: str = field(default_factory=_default_device)
    grad_clip: Optional[float] = 1.0
    mixed_precision: bool = True
    enable_latency_tracking: bool = True
    log_interval: int = 25
    checkpoint_dir: Optional[Path] = None
    wandb: Optional[WandBConfig] = None
    data_parallel: bool = False
    device_ids: Optional[List[int]] = None
    quantization: Optional[QuantizationConfig] = None
    class_weights: Optional[List[float]] = None
    class_counts: Optional[List[int]] = None
    collapse_detection_enabled: bool = True
    collapse_threshold: float = 0.9
    collapse_patience: int = 2
    collapse_cooldown: int = 1
    collapse_reweight_factor: float = 0.5
    collapse_min_class_weight: float = 0.05
    collapse_log_distributions: bool = True
    collapse_entropy_weight: float = 0.0
    collapse_entropy_target: float = 1.2
    collapse_distill_weight: float = 0.0
    collapse_distill_temperature: float = 1.5
    auto_tune_batch_size: bool = False
    batch_tuner_min_batch_size: Optional[int] = None
    batch_tuner_max_batch_size: Optional[int] = None
    batch_tuner_growth_factor: float = 1.5
    batch_tuner_target_utilization: float = 0.9
    batch_tuner_max_latency_increase: float = 1.3
    batch_tuner_include_val: bool = True
    batch_tuner_warmup_steps: int = 1
    ema: Optional["EMAConfig"] = None
    loss: LossConfig = field(default_factory=LossConfig)

    def ensure_stages(self) -> None:
        if self.stages:
            return
        self.stages = default_stage_schedule()


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    update_after_steps: int = 100
    update_every: int = 1


class GradCAMAttributor:
    """Produces Grad-CAM overlays for interpretability when ResNet backbones are active."""

    def __init__(
        self,
        model: ClassicalDRModel,
        device: torch.device,
        max_samples: int = 3,
        alpha: float = 0.45,
    ) -> None:
        self.model = model
        self.device = device
        self.max_samples = max(1, int(max_samples))
        self.alpha = float(alpha)
        self.enabled = False
        self.target_module: Optional[torch.nn.Module] = None

        vision_encoder = getattr(model, "vision_encoder", None)
        if vision_encoder is not None:
            target_layer = getattr(vision_encoder, "cam_target_layer", None)
            if isinstance(target_layer, torch.nn.Module):
                self.target_module = target_layer
                self.enabled = True

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.registered_mean = mean.view(1, 3, 1, 1).to(device)
        self.registered_std = std.view(1, 3, 1, 1).to(device)

        if not self.enabled:
            LOGGER.debug("Grad-CAM disabled: no compatible target layer detected.")

    def generate_overlays(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor],
        paths: Optional[List[str]],
        class_names: Iterable[str],
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        if images is None or not isinstance(images, torch.Tensor):
            return []
        total = images.shape[0]
        if total == 0:
            return []

        label_list: Optional[List[int]] = None
        if labels is not None and isinstance(labels, torch.Tensor):
            label_list = [int(x) for x in labels.detach().cpu().tolist()]

        selected_indices = self._select_indices(label_list, total)
        if not selected_indices:
            return []

        image_batch = images[selected_indices].to(self.device, non_blocking=True).clone().requires_grad_(True)
        label_batch = (
            torch.tensor([label_list[idx] for idx in selected_indices], device=self.device)
            if label_list is not None
            else None
        )
        sample_paths = [str((paths or [""] * total)[idx]) for idx in selected_indices]

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def _forward_hook(_module, _inputs, output):
            activations.clear()
            activations.append(output.detach())

        def _backward_hook(_module, _grad_inputs, grad_outputs):
            gradients.clear()
            gradients.append(grad_outputs[0].detach())

        handles: List[Any] = []
        try:
            handles.append(self.target_module.register_forward_hook(_forward_hook))  # type: ignore[arg-type]
            handles.append(self.target_module.register_full_backward_hook(_backward_hook))  # type: ignore[arg-type]
        except Exception:
            for handle in handles:
                handle.remove()
            return []

        prev_mode = self.model.training
        overlays: List[Dict[str, Any]] = []
        try:
            self.model.eval()
            with torch.enable_grad():
                outputs = self.model(image_batch, return_all=True, track_latency=False)
                normalized = outputs if isinstance(outputs, ModelForwardOutput) else self._normalize_tuple_output(outputs)
                logits = normalized.final_output
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                gradients_target = label_batch if label_batch is not None else preds
                gathered = logits.gather(1, gradients_target.view(-1, 1)).sum()
                self.model.zero_grad(set_to_none=True)
                gathered.backward()

            if not activations or not gradients:
                return []

            activation = activations[-1]
            gradient = gradients[-1]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image_batch.shape[-2:], mode="bilinear", align_corners=False)
            cam_flat = cam.flatten(1)
            cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            with torch.no_grad():
                denorm = self._denormalize(image_batch.detach()).clamp(0.0, 1.0)

            denorm_np = denorm.cpu().permute(0, 2, 3, 1).numpy()
            cam_np = cam.squeeze(1).detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            target_np = gradients_target.detach().cpu().numpy()
            label_names = list(class_names)

            try:
                from matplotlib import cm

                colormap = cm.get_cmap("inferno")
            except Exception:
                colormap = None

            for idx in range(len(selected_indices)):
                base_img = denorm_np[idx]
                heatmap = cam_np[idx]
                if colormap is not None:
                    heat_rgb = colormap(heatmap)[..., :3]
                else:
                    heat_rgb = np.stack([heatmap] * 3, axis=-1)
                overlay_img = np.clip((1.0 - self.alpha) * base_img + self.alpha * heat_rgb, 0.0, 1.0)
                pred_idx = int(preds_np[idx])
                tgt_idx = int(target_np[idx])
                overlays.append(
                    {
                        "overlay": overlay_img,
                        "heatmap": heatmap,
                        "base": base_img,
                        "pred_idx": pred_idx,
                        "pred_label": label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx),
                        "target_idx": tgt_idx,
                        "target_label": label_names[tgt_idx] if tgt_idx < len(label_names) else str(tgt_idx),
                        "confidence": float(probs_np[idx, pred_idx]),
                        "path": sample_paths[idx],
                    }
                )
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(prev_mode)
            self.model.zero_grad(set_to_none=True)

        return overlays

    def _select_indices(self, labels: Optional[List[int]], total: int) -> List[int]:
        indices = list(range(total))
        if not labels:
            return indices[: self.max_samples]
        selected: List[int] = []
        seen: set[int] = set()
        for idx in indices:
            label_val = int(labels[idx])
            if label_val not in seen:
                selected.append(idx)
                seen.add(label_val)
            if len(selected) >= self.max_samples:
                break
        if len(selected) < self.max_samples:
            for idx in indices:
                if idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= self.max_samples:
                    break
        return selected

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.registered_std + self.registered_mean

    @staticmethod
    def _normalize_tuple_output(raw_output: Any) -> ModelForwardOutput:
        if isinstance(raw_output, ModelForwardOutput):
            return raw_output
        if isinstance(raw_output, (list, tuple)) and len(raw_output) == 10:
            return ModelForwardOutput(
                latent_features=raw_output[0],
                compressed_features=raw_output[1],
                output_a=raw_output[2],
                output_b=raw_output[3],
                output_c=raw_output[4],
                final_output=raw_output[5],
                ensemble_weights=raw_output[6],
                uncertainties=raw_output[7],
                latencies=raw_output[8],
                active_mask=raw_output[9],
            )
        raise TypeError("Unsupported forward output for Grad-CAM normalization")


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, metric: float) -> None:
        score = -metric
        if self.best_score is None:
            self.best_score = score
            return
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def default_stage_schedule() -> List[StageConfig]:
    return [
        StageConfig(
            name="classical",
            num_epochs=12,
            learning_rate=1e-4,
            weight_decay=1e-5,
            patience=4,
            active_mask=[1, 1, 0],
            loss_weights={
                "classical_a": 1.0,
                "classical_b": 1.0,
                "quantum": 0.0,
                "ensemble": 0.5,
            },
        ),
        StageConfig(
            name="quantum",
            num_epochs=8,
            learning_rate=5e-5,
            weight_decay=1e-5,
            patience=3,
            active_mask=[0, 0, 1],
            loss_weights={
                "classical_a": 0.0,
                "classical_b": 0.0,
                "quantum": 1.0,
                "ensemble": 0.25,
            },
        ),
        StageConfig(
            name="ensemble",
            num_epochs=5,
            learning_rate=1e-4,
            weight_decay=5e-6,
            patience=3,
            active_mask=[1, 1, 1],
            loss_weights={
                "classical_a": 0.5,
                "classical_b": 0.5,
                "quantum": 0.5,
                "ensemble": 1.0,
            },
        ),
    ]


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    _, predicted = torch.max(logits, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return float(correct / total) if total else 0.0


def _convert_mask(mask: Optional[Iterable[int]]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask_tensor = torch.as_tensor(list(mask), dtype=torch.float32)
    if mask_tensor.ndim != 1 or mask_tensor.shape[0] != 3:
        raise ValueError("active_mask must contain three elements (heads A, B, quantum)")
    return mask_tensor


def _prepare_loss_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    base = {"classical_a": 1.0, "classical_b": 1.0, "quantum": 1.0, "ensemble": 0.5}
    if not weights:
        return base
    base.update(weights)
    return base


class Phase1Trainer:
    """Stage-aware trainer with optional W&B logging."""

    def __init__(
        self,
        model: ClassicalDRModel,
        config: TrainingConfig,
        classes: Iterable[str],
    ) -> None:
        self.config = config
        self.config.ensure_stages()
        requested_device = torch.device(self.config.device)
        self.model = model
        self.data_parallel = False
        self.device_ids = None

        if self.config.data_parallel and torch.cuda.device_count() > 1:
            available = torch.cuda.device_count()
            candidate_ids = self.config.device_ids or list(range(available))
            valid_ids = [idx for idx in candidate_ids if 0 <= idx < available]
            if len(valid_ids) >= 2:
                primary_index = requested_device.index if requested_device.type == "cuda" and requested_device.index is not None else valid_ids[0]
                if primary_index not in valid_ids:
                    valid_ids = [primary_index] + [idx for idx in valid_ids if idx != primary_index]
                else:
                    valid_ids = [primary_index] + [idx for idx in valid_ids if idx != primary_index]
                self.device_ids = valid_ids
                self.device = torch.device(f"cuda:{valid_ids[0]}")
                self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=valid_ids)
                self.data_parallel = True
                LOGGER.info("Enabled DataParallel across GPUs %s", valid_ids)
            else:
                LOGGER.warning("Multi-GPU requested but fewer than two valid device IDs provided; falling back to single GPU")

        if not self.data_parallel:
            self.device = requested_device
            self.model = self.model.to(self.device)
        self.base_model = _unwrap_model(self.model)
        self.gradcam_helper = GradCAMAttributor(self.base_model, self.device)
        self._interpretability_samples: Optional[Dict[str, Any]] = None
        self._architecture_logged: set[str] = set()
        self.classes = list(classes)
        if self.gradcam_helper.enabled:
            desired_samples = max(len(self.classes) * 2, 6)
            self.gradcam_helper.max_samples = max(self.gradcam_helper.max_samples, desired_samples)
        self.class_weight_tensor: Optional[torch.Tensor] = None
        if self.config.class_weights and len(self.config.class_weights) == len(self.classes):
            self.class_weight_tensor = torch.tensor(self.config.class_weights, dtype=torch.float32)
        elif self.config.class_weights:
            LOGGER.warning(
                "Ignoring configured class_weights: expected %d values but received %d",
                len(self.classes),
                len(self.config.class_weights),
            )
        self._base_class_weight_tensor: Optional[torch.Tensor] = None
        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor.to(self.device)
            self._base_class_weight_tensor = self.class_weight_tensor.detach().cpu().clone()
        self.class_counts = list(self.config.class_counts) if self.config.class_counts else None
        if self.class_counts and len(self.class_counts) != len(self.classes):
            LOGGER.warning(
                "Ignoring configured class_counts: expected %d values but received %d",
                len(self.classes),
                len(self.class_counts),
            )
            self.class_counts = None
        configured_loss = getattr(self.config.loss, "type", "cross_entropy")
        self._loss_type = str(configured_loss).lower()
        if self._loss_type == "balanced":
            self._loss_type = "class_balanced"
        if self._loss_type not in {"cross_entropy", "class_balanced", "focal_class_balanced"}:
            LOGGER.warning("Unknown loss type '%s'; defaulting to cross_entropy", self._loss_type)
            self._loss_type = "cross_entropy"
        self._focal_gamma = float(getattr(self.config.loss, "focal_gamma", 2.0))
        self._cb_beta = float(getattr(self.config.loss, "class_balanced_beta", 0.999))
        self._warned_missing_class_counts = False
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.history: List[Dict[str, Any]] = []
        self.best_state_meta: Optional[Dict[str, Any]] = None
        self.best_state_weights: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_loss: Optional[float] = None
        self.global_step = 0
        self.wandb_run = self._init_wandb()
        base_vis_root = Path(self.config.checkpoint_dir).parent if self.config.checkpoint_dir else Path("trained_model")
        self.visualization_dir = (base_vis_root / "reports" / "visualizations").resolve()
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        if self.wandb_run and self.config.class_weights:
            try:
                summary_payload = {
                    "class_weights": list(self.config.class_weights),
                }
                if self.class_counts is not None:
                    summary_payload["class_counts"] = list(self.class_counts)
                self.wandb_run.summary.update(summary_payload)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to log class weighting metadata to W&B: %s", exc)
        self._confidence_history: Dict[str, Dict[str, Dict[str, List[tuple[int, float]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self._collapse_tracker: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "streak": 0,
            "dominant_class": None,
            "cooldown": 0,
        })
        self._collapse_events: List[Dict[str, Any]] = []
        self._batch_tuner_cache: Dict[tuple[str, str], int] = {}
        self.ema_helper: Optional[AveragedModel] = None
        self._ema_update_after = 0
        self._ema_update_every = 1
        if self.config.ema and self.config.ema.enabled:
            decay = float(self.config.ema.decay)

            def _ema_avg_fn(averaged: torch.Tensor, new: torch.Tensor, num_averaged: int) -> torch.Tensor:
                return decay * averaged + (1.0 - decay) * new

            self.ema_helper = AveragedModel(self.base_model, avg_fn=_ema_avg_fn)
            self.ema_helper.to(self.device)
            self._ema_update_after = max(0, int(self.config.ema.update_after_steps))
            self._ema_update_every = max(1, int(self.config.ema.update_every))
        self.quant_manager: Optional[QuantizationManager] = None
        if self.config.quantization and self.config.quantization.enabled:
            self.quant_manager = QuantizationManager(self.config.quantization)
        if self.config.enable_latency_tracking and hasattr(self.base_model, "enable_latency_tracking"):
            self.base_model.enable_latency_tracking(True)

    def load_initial_state(
        self,
        checkpoint_source: Path | str | Dict[str, Any],
        strict: bool = False,
        restore_history: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(checkpoint_source, (str, Path)):
            checkpoint_path = Path(checkpoint_source).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            source_name = str(checkpoint_path)
        elif isinstance(checkpoint_source, dict):
            checkpoint_data = checkpoint_source
            source_name = "<in-memory checkpoint>"
        else:  # pragma: no cover - defensive
            raise TypeError("checkpoint_source must be a path or a checkpoint dictionary")

        state_dict: Optional[Dict[str, torch.Tensor]] = None
        if isinstance(checkpoint_data, dict):
            candidate = checkpoint_data.get("model_state_dict")
            if isinstance(candidate, dict):
                state_dict = candidate
            elif all(isinstance(key, str) for key in checkpoint_data.keys()):
                state_dict = checkpoint_data  # assume raw state dict

        if state_dict is None:
            raise ValueError("Checkpoint does not contain a valid model_state_dict")

        base_model = _unwrap_model(self.model)
        load_result = base_model.load_state_dict(state_dict, strict=strict)
        if isinstance(load_result, tuple):  # older PyTorch versions
            missing_keys, unexpected_keys = load_result
        else:
            missing_keys = getattr(load_result, "missing_keys", [])
            unexpected_keys = getattr(load_result, "unexpected_keys", [])
        missing_keys = list(missing_keys)
        unexpected_keys = list(unexpected_keys)

        if self.ema_helper and isinstance(checkpoint_data, dict):
            ema_state = checkpoint_data.get("ema_state_dict")
            if isinstance(ema_state, dict):
                try:
                    self.ema_helper.load_state_dict(ema_state, strict=False)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to load EMA state from checkpoint: %s", exc)

        history_count = 0
        if restore_history and isinstance(checkpoint_data, dict):
            prior_history = checkpoint_data.get("history")
            if isinstance(prior_history, list):
                self.history.extend(prior_history)
                history_count = len(prior_history)

        metadata = {
            "source": source_name,
            "stage": checkpoint_data.get("stage") if isinstance(checkpoint_data, dict) else None,
            "epoch": checkpoint_data.get("epoch") if isinstance(checkpoint_data, dict) else None,
            "history_restored": bool(history_count),
            "history_entries": history_count,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

        if missing_keys or unexpected_keys:
            LOGGER.debug(
                "Checkpoint load mismatch -> missing: %s unexpected: %s",
                missing_keys,
                unexpected_keys,
            )

        return metadata

    def _ensure_interpretability_samples(self, val_loader: DataLoader) -> None:
        if not self.gradcam_helper.enabled or self._interpretability_samples is not None:
            return
        collected_images: List[torch.Tensor] = []
        collected_labels: List[int] = []
        collected_paths: List[str] = []
        seen_classes: set[int] = set()
        per_class_counts: Dict[int, int] = defaultdict(int)
        max_samples = self.gradcam_helper.max_samples
        max_total = max(max_samples * 2, len(self.classes) * 3, 8)
        try:
            for images, labels, paths in val_loader:
                batch_size = images.shape[0]
                for idx in range(batch_size):
                    label_val = int(labels[idx].item()) if labels is not None else -1
                    if len(collected_images) >= max_total:
                        break

                    should_add = False
                    class_count = per_class_counts[label_val]
                    if label_val >= 0:
                        if class_count == 0:
                            should_add = True
                        elif class_count < 2 and len(collected_images) < max_total:
                            should_add = True
                    else:
                        should_add = len(collected_images) < max_samples

                    if not should_add and len(collected_images) < max_samples:
                        should_add = True

                    if should_add:
                        collected_images.append(images[idx].cpu())
                        collected_labels.append(label_val)
                        collected_paths.append(str(paths[idx]) if paths else "")
                        per_class_counts[label_val] += 1
                        if label_val >= 0:
                            seen_classes.add(label_val)

                    if len(seen_classes) >= len(self.classes) and len(collected_images) >= max_samples:
                        break
                if len(seen_classes) >= len(self.classes) and len(collected_images) >= max_samples:
                    break
                if len(collected_images) >= max_total:
                    break
        except StopIteration:
            pass

        if not collected_images:
            return

        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label_val in enumerate(collected_labels):
            label_to_indices[label_val].append(idx)

        selected_indices: List[int] = []
        seen_indices: set[int] = set()

        for class_idx in range(len(self.classes)):
            if class_idx in label_to_indices:
                choice = label_to_indices[class_idx][0]
                selected_indices.append(choice)
                seen_indices.add(choice)

        for idx, label_val in enumerate(collected_labels):
            if len(selected_indices) >= max_samples:
                break
            if idx in seen_indices:
                continue
            selected_indices.append(idx)
            seen_indices.add(idx)

        trimmed_indices = selected_indices[:max_samples] if selected_indices else list(range(min(max_samples, len(collected_images))))
        stacked_images = torch.stack([collected_images[i] for i in trimmed_indices])
        stacked_labels = torch.tensor([collected_labels[i] for i in trimmed_indices]) if collected_labels else None
        stacked_paths = [collected_paths[i] for i in trimmed_indices]
        self._interpretability_samples = {
            "images": stacked_images,
            "labels": stacked_labels,
            "paths": stacked_paths,
        }

    def _head_display_name(self, key: str) -> str:
        return HEAD_DISPLAY_NAMES.get(key, key.replace("_", " ").title())

    def _init_wandb(self):
        wandb_cfg = self.config.wandb
        if not wandb_cfg or not wandb_cfg.use_wandb:
            return None
        try:
            import wandb

            run = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=wandb_cfg.run_name,
                notes=wandb_cfg.notes,
                tags=wandb_cfg.tags,
                mode=wandb_cfg.mode,
                config=wandb_cfg.config,
            )
            return run
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning("W&B requested but wandb is not installed; skipping logging")
            return None

    def _log_to_wandb(self, metrics: Dict[str, Any], stage_key: str, epoch: int) -> None:
        if not self.wandb_run:
            return
        metrics = {f"{stage_key}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        metrics["global_step"] = self.global_step
        if LOGGER.isEnabledFor(logging.DEBUG):
            preview = {k: v for k, v in metrics.items() if k not in {"epoch", "global_step"}}
            LOGGER.debug("Logging to W&B at step %d: %s", self.global_step, preview)
        try:
            import wandb

            wandb.log(metrics, step=self.global_step)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to log to W&B: %s", exc)

    def _stage_log_key(self, stage: StageConfig) -> str:
        alias = getattr(stage, "log_alias", None)
        if alias:
            return str(alias)
        return stage.name

    def _stage_log_dir(self, stage: StageConfig) -> Path:
        key = self._stage_log_key(stage)
        safe_key = key.replace("/", "-").replace(" ", "_").replace("[", "").replace("]", "")
        return self.visualization_dir / safe_key

    def _compute_losses(
        self,
        outputs: ModelForwardOutput,
        targets: torch.Tensor,
        loss_weights: Optional[Dict[str, float]],
    ) -> Dict[str, torch.Tensor]:
        loss_mode = self._loss_type
        include_cb = loss_mode in {"class_balanced", "focal_class_balanced"}
        class_vector = self._resolve_class_weight_vector(targets.device, include_cb)

        if loss_mode == "focal_class_balanced":
            return self._compute_focal_class_balanced_losses(outputs, targets, loss_weights, class_vector)
        if loss_mode == "class_balanced":
            return self._compute_weighted_losses(outputs, targets, loss_weights, class_vector)
        weights = _prepare_loss_weights(loss_weights)
        base_losses = self.base_model.compute_losses(
            outputs,
            targets,
            loss_weights=loss_weights,
            class_weights=class_vector or self.class_weight_tensor,
        )
        base_losses["total_loss"] = base_losses.get("total_loss", torch.tensor(0.0, device=targets.device))
        return self._apply_quantum_regularizers(outputs, base_losses, weights, targets.device)

    def _compute_weighted_losses(
        self,
        outputs: ModelForwardOutput,
        targets: torch.Tensor,
        loss_weights: Optional[Dict[str, float]],
        class_vector: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        weights = _prepare_loss_weights(loss_weights)
        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=targets.device)
        head_map = {
            "classical_a": ("loss_a", outputs.output_a),
            "classical_b": ("loss_b", outputs.output_b),
            "quantum": ("loss_c", outputs.output_c),
            "ensemble": ("ensemble_loss", outputs.final_output),
        }
        for head_key, (loss_key, logits) in head_map.items():
            weight = float(weights.get(head_key, 0.0))
            if weight <= 0:
                losses[loss_key] = torch.tensor(0.0, device=targets.device)
                continue
            head_loss = F.cross_entropy(logits, targets, weight=class_vector)
            losses[loss_key] = head_loss
            total_loss = total_loss + weight * head_loss
        losses["total_loss"] = total_loss
        return self._apply_quantum_regularizers(outputs, losses, weights, targets.device)

    def _compute_focal_class_balanced_losses(
        self,
        outputs: ModelForwardOutput,
        targets: torch.Tensor,
        loss_weights: Optional[Dict[str, float]],
        class_vector: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        weights = _prepare_loss_weights(loss_weights)
        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=targets.device)
        head_map = {
            "classical_a": ("loss_a", outputs.output_a),
            "classical_b": ("loss_b", outputs.output_b),
            "quantum": ("loss_c", outputs.output_c),
            "ensemble": ("ensemble_loss", outputs.final_output),
        }
        for head_key, (loss_key, logits) in head_map.items():
            weight = float(weights.get(head_key, 0.0))
            if weight <= 0:
                losses[loss_key] = torch.tensor(0.0, device=targets.device)
                continue
            head_loss = self._focal_class_balanced_loss(logits, targets, class_vector)
            losses[loss_key] = head_loss
            total_loss = total_loss + weight * head_loss
        losses["total_loss"] = total_loss
        return self._apply_quantum_regularizers(outputs, losses, weights, targets.device)

    def _apply_quantum_regularizers(
        self,
        outputs: ModelForwardOutput,
        losses: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        weight_map = weights or {}
        quantum_weight = float(weight_map.get("quantum", 0.0))
        zero_tensor = torch.tensor(0.0, device=device)
        if quantum_weight <= 0:
            losses.setdefault("quantum_entropy_penalty", zero_tensor.clone())
            losses.setdefault("quantum_distill_penalty", zero_tensor.clone())
            return losses

        entropy_weight = float(getattr(self.config, "collapse_entropy_weight", 0.0))
        if entropy_weight > 0:
            probs = torch.softmax(outputs.output_c, dim=1)
            entropy = -(probs * torch.log(torch.clamp(probs, min=1e-6))).sum(dim=1).mean()
            entropy_target = torch.tensor(
                float(getattr(self.config, "collapse_entropy_target", 0.0)),
                device=entropy.device,
            )
            entropy_gap = torch.relu(entropy_target - entropy)
            entropy_penalty = entropy_gap * entropy_weight
            losses["quantum_entropy_penalty"] = entropy_penalty
            losses["total_loss"] = losses["total_loss"] + entropy_penalty
        else:
            losses.setdefault("quantum_entropy_penalty", zero_tensor.clone())

        distill_weight = float(getattr(self.config, "collapse_distill_weight", 0.0))
        if distill_weight > 0:
            teacher_logits = outputs.output_b.detach()
            temperature = max(1e-3, float(getattr(self.config, "collapse_distill_temperature", 1.0)))
            student_log_probs = F.log_softmax(outputs.output_c / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
            distill_penalty = distill_weight * distill_loss
            losses["quantum_distill_penalty"] = distill_penalty
            losses["total_loss"] = losses["total_loss"] + distill_penalty
        else:
            losses.setdefault("quantum_distill_penalty", zero_tensor.clone())

        return losses

    def _resolve_class_weight_vector(
        self,
        device: torch.device,
        include_class_balanced: bool,
    ) -> Optional[torch.Tensor]:
        base_weights: Optional[torch.Tensor] = None
        if self.class_weight_tensor is not None:
            base_weights = self.class_weight_tensor.to(device, dtype=torch.float32)
        cb_weights = self._class_balanced_weights(device) if include_class_balanced else None
        if base_weights is None and cb_weights is None:
            return None
        if base_weights is None:
            combined = cb_weights
        elif cb_weights is None:
            combined = base_weights
        else:
            combined = base_weights * cb_weights
        if combined is None:
            return None
        combined = torch.clamp(combined, min=1e-6)
        total = combined.sum()
        if total > 0:
            combined = combined * (combined.shape[0] / total)
        return combined

    def _class_balanced_weights(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.class_counts:
            if self._loss_type in {"class_balanced", "focal_class_balanced"} and not self._warned_missing_class_counts:
                LOGGER.warning(
                    "%s loss requested but class_counts were not provided; falling back to base class weights",
                    self._loss_type.replace("_", " "),
                )
                self._warned_missing_class_counts = True
            return None
        counts = torch.tensor(self.class_counts, device=device, dtype=torch.float32)
        beta = max(0.0, min(float(self._cb_beta), 0.999999))
        if beta <= 0.0:
            weights = torch.ones_like(counts)
        else:
            beta_tensor = torch.tensor(beta, device=device, dtype=torch.float32)
            valid_mask = counts > 0
            effective = torch.where(
                valid_mask,
                1.0 - torch.pow(beta_tensor, counts),
                torch.ones_like(counts),
            )
            weights = torch.where(
                valid_mask,
                (1.0 - beta_tensor) / (effective + 1e-8),
                torch.ones_like(counts),
            )
        weights = torch.clamp(weights, min=1e-6)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights * (weights.shape[0] / weight_sum)
        return weights

    def _focal_class_balanced_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        gamma = max(0.0, float(self._focal_gamma))
        if gamma > 0:
            pt = torch.exp(-ce)
            modulating = torch.pow(1.0 - pt, gamma)
            loss_values = modulating * ce
        else:
            loss_values = ce
        if class_weights is not None:
            weights = class_weights.to(logits.device, dtype=torch.float32).detach().clamp(min=1e-6)
            sample_weights = weights[targets]
            loss_values = loss_values * sample_weights
        return loss_values.mean()

    def _log_model_architecture(
        self,
        stage: StageConfig,
        epoch: int,
        epoch_dir: Path,
        mask_tensor: Optional[torch.Tensor],
    ) -> None:
        stage_key = self._stage_log_key(stage)
        if stage_key in self._architecture_logged:
            return
        diagram = self._build_architecture_overview(stage, mask_tensor)
        if diagram is None:
            return
        stage_label = stage_key.title()
        if stage_label.lower() != stage.name.lower():
            title = f"{stage_label} Architecture (Stage '{stage.name}' Epoch {epoch})"
        else:
            title = f"{stage_label} Architecture (Epoch {epoch})"
        try:
            base_filename = str((epoch_dir / "architecture_overview").resolve())
            output_path = Path(diagram.render(base_filename, format="png", cleanup=True))
            if self.wandb_run:
                try:
                    import wandb

                    wandb.log(
                        {
                            f"{stage_key}/architecture": wandb.Image(
                                str(output_path),
                                caption=title,
                            )
                        },
                        step=self.global_step,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to log architecture diagram to W&B: %s", exc)
            self._architecture_logged.add(stage_key)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to generate architecture diagram for stage %s: %s", stage.name, exc)

    def _build_architecture_overview(
        self,
        stage: StageConfig,
        mask_tensor: Optional[torch.Tensor],
    ) -> Optional[Digraph]:
        if Digraph is None:
            LOGGER.debug("graphviz package unavailable; skipping architecture overview for stage %s", stage.name)
            return None

        diagram = Digraph(comment=f"{stage.name.title()} Architecture")
        diagram.attr(rankdir="LR", nodesep="0.75", ranksep="0.9")
        diagram.attr(label=f"{stage.name.title()} Stage Architecture", labelloc="t", fontsize="20")

        encoder_type = getattr(self.base_model, "encoder_type", "Vision Encoder")
        quantum_enabled = bool(getattr(self.base_model, "quantum_enabled", False))
        mask_list: Optional[List[float]] = None
        if mask_tensor is not None:
            mask_list = mask_tensor.detach().cpu().tolist()

        def _is_active(idx: int, default: bool = True) -> bool:
            if mask_list is None:
                return default
            if idx >= len(mask_list):
                return default
            return bool(mask_list[idx] > 0)

        def _head_style(active: bool) -> Dict[str, str]:
            if active:
                return {"style": "filled", "fillcolor": "#a5d8ff"}
            return {"style": "filled", "fillcolor": "#d3d3d3"}

        diagram.node("input", "Retinal Image\n224x224 RGB", shape="oval", style="filled", fillcolor="#e6f2ff")
        diagram.node("encoder", f"Vision Encoder\n({encoder_type})", shape="box", style="filled", fillcolor="#a5d8ff")
        diagram.node("latent", "Latent Features", shape="box", style="filled", fillcolor="#f8f9fa")

        head_a_active = _is_active(0)
        head_b_active = _is_active(1)
        head_c_active = quantum_enabled and _is_active(2)

        compression_needed = head_b_active or head_c_active
        if compression_needed:
            diagram.node("compression", "Compression Module\n(2048 -> 30)", shape="box", style="filled", fillcolor="#f8f9fa")

        active_head_nodes: List[str] = []
        if head_a_active:
            diagram.node("head_a", "Classification Head A\nFull-Resolution", shape="box", **_head_style(True))
            active_head_nodes.append("head_a")
        if head_b_active:
            diagram.node("head_b", "Classification Head B\nCompressed", shape="box", **_head_style(True))
            active_head_nodes.append("head_b")
        if head_c_active:
            diagram.node("head_c", "Quantum Head", shape="box", **_head_style(True))
            active_head_nodes.append("head_c")

        diagram.node("ensemble", "Dynamic Ensemble", shape="box", style="filled", fillcolor="#bee3f8")
        diagram.node("output", "Final Predictions\n(5 DR classes)", shape="oval", style="filled", fillcolor="#e6f2ff")

        diagram.edge("input", "encoder")
        diagram.edge("encoder", "latent")
        if head_a_active:
            diagram.edge("latent", "head_a")
        if compression_needed:
            diagram.edge("latent", "compression")
        if head_b_active:
            diagram.edge("compression", "head_b")
        if head_c_active:
            diagram.edge("compression", "head_c")

        for head_node in active_head_nodes:
            diagram.edge(head_node, "ensemble")
        diagram.edge("ensemble", "output")

        if not active_head_nodes:
            diagram.edge("latent", "ensemble")

        return diagram

    def _log_epoch_visualizations(self, stage: StageConfig, epoch: int, val_outcome: Dict[str, Any]) -> None:
        if not self.wandb_run:
            return
        preds = val_outcome.get("preds") or []
        targets = val_outcome.get("targets") or []
        if not preds or not targets:
            return
        mask_tensor = self._stage_mask(stage)
        stage_dir = self._stage_log_dir(stage)
        stage_key = self._stage_log_key(stage)
        stage_label = stage_key if stage_key == stage.name else f"{stage_key} [{stage.name}]"
        epoch_suffix = f"{stage.name}_epoch_{epoch:02d}" if stage_key != stage.name else f"epoch_{epoch:02d}"
        epoch_dir = stage_dir / epoch_suffix
        epoch_dir.mkdir(parents=True, exist_ok=True)
        tracked_heads = self._resolve_tracked_heads(mask_tensor)
        head_display_map = {head: self._head_display_name(head) for head in tracked_heads}
        reverse_head_label_map = {display: original for original, display in head_display_map.items()}

        self._log_model_architecture(stage, epoch, epoch_dir, mask_tensor)

        try:
            import wandb

            confusion_payload = val_outcome.get("confusion_matrix", None)
            per_head_accuracy = val_outcome.get("per_head_class_accuracy")
            per_head_confusion = val_outcome.get("head_confusion_matrices")
            metrics_log: Dict[str, Any] = {}
            if confusion_payload:
                cm_array = np.array(confusion_payload, dtype=np.int64)
                cm_table = wandb.Table(columns=["Actual", "Predicted", "Count"])
                for actual_idx, actual_label in enumerate(self.classes):
                    for pred_idx, pred_label in enumerate(self.classes):
                        cm_table.add_data(actual_label, pred_label, int(cm_array[actual_idx, pred_idx]))
                wandb.log({f"{stage_key}/confusion_matrix_table": cm_table}, step=self.global_step)

                conf_float = cm_array.astype(np.float64)
                row_totals = conf_float.sum(axis=1)
                per_class_recall = np.zeros(conf_float.shape[0], dtype=np.float64)
                nonzero_mask = row_totals > 0
                if np.any(nonzero_mask):
                    per_class_recall[nonzero_mask] = np.diag(conf_float)[nonzero_mask] / row_totals[nonzero_mask]
                balanced_acc = float(per_class_recall.mean()) if per_class_recall.size else 0.0
                metrics_log[f"{stage_key}/balanced_acc_ensemble"] = balanced_acc

                try:
                    import matplotlib.pyplot as plt

                    row_sums = cm_array.sum(axis=1, keepdims=True)
                    normalized = np.zeros_like(cm_array, dtype=np.float64)
                    nonzero_rows = row_sums.squeeze() > 0
                    normalized[nonzero_rows] = cm_array[nonzero_rows] / row_sums[nonzero_rows]
                    annotation_grid = np.empty_like(normalized, dtype=object)
                    for row_idx in range(cm_array.shape[0]):
                        for col_idx in range(cm_array.shape[1]):
                            percent = normalized[row_idx, col_idx] * 100.0
                            count = int(cm_array[row_idx, col_idx])
                            if count == 0 and not nonzero_rows[row_idx]:
                                annotation_grid[row_idx, col_idx] = "n/a"
                            else:
                                annotation_grid[row_idx, col_idx] = f"{percent:.1f}%\n({count})"

                    try:
                        import seaborn as sns  # type: ignore

                        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                        sns.heatmap(
                            normalized,
                            annot=annotation_grid,
                            fmt="",
                            cmap="Blues",
                            cbar_kws={"label": "Recall"},
                            xticklabels=self.classes,
                            yticklabels=self.classes,
                            ax=ax,
                        )
                        ax.set_xticklabels(self.classes, rotation=45, ha="right")
                        ax.set_yticklabels(self.classes, rotation=0)
                    except Exception:
                        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                        im = ax.imshow(normalized, cmap="Blues")
                        ax.figure.colorbar(im, ax=ax, label="Recall")
                        ax.set_xticks(np.arange(len(self.classes)))
                        ax.set_yticks(np.arange(len(self.classes)))
                        ax.set_xticklabels(self.classes, rotation=45, ha="right")
                        ax.set_yticklabels(self.classes)
                        for i in range(cm_array.shape[0]):
                            for j in range(cm_array.shape[1]):
                                ax.text(j, i, annotation_grid[i, j], ha="center", va="center", color="black")

                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{stage_label} Confusion Matrix (Epoch {epoch})")
                    fig_path = epoch_dir / "confusion_matrix.png"
                    fig.savefig(fig_path, dpi=200)
                    wandb.log({f"{stage_key}/confusion_matrix": wandb.Image(fig)}, step=self.global_step)
                    plt.close(fig)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to render confusion matrix heatmap: %s", exc)

            if per_head_confusion:
                for head_name, matrix in per_head_confusion.items():
                    display_name = head_display_map.get(head_name, head_name)
                    cm_array = np.asarray(matrix, dtype=np.int64)
                    head_table = wandb.Table(columns=["Actual", "Predicted", "Count"])
                    for actual_idx, actual_label in enumerate(self.classes):
                        for pred_idx, pred_label in enumerate(self.classes):
                            head_table.add_data(actual_label, pred_label, int(cm_array[actual_idx, pred_idx]))
                    table_key = display_name.lower().replace(" ", "_") + "_confusion_matrix_table"
                    wandb.log({f"{stage_key}/{table_key}": head_table}, step=self.global_step)

                    try:
                        import matplotlib.pyplot as plt
                        row_sums = cm_array.sum(axis=1, keepdims=True)
                        normalized = np.zeros_like(cm_array, dtype=np.float64)
                        nonzero_rows = row_sums.squeeze() > 0
                        normalized[nonzero_rows] = cm_array[nonzero_rows] / row_sums[nonzero_rows]
                        annotation_grid = np.empty_like(normalized, dtype=object)
                        for row_idx in range(cm_array.shape[0]):
                            for col_idx in range(cm_array.shape[1]):
                                percent = normalized[row_idx, col_idx] * 100.0
                                count = int(cm_array[row_idx, col_idx])
                                if count == 0 and not nonzero_rows[row_idx]:
                                    annotation_grid[row_idx, col_idx] = "n/a"
                                else:
                                    annotation_grid[row_idx, col_idx] = f"{percent:.1f}%\n({count})"

                        try:
                            import seaborn as sns  # type: ignore

                            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                            sns.heatmap(
                                normalized,
                                annot=annotation_grid,
                                fmt="",
                                cmap="Blues",
                                cbar_kws={"label": "Recall"},
                                xticklabels=self.classes,
                                yticklabels=self.classes,
                                ax=ax,
                            )
                            ax.set_xticklabels(self.classes, rotation=45, ha="right")
                            ax.set_yticklabels(self.classes, rotation=0)
                        except Exception:
                            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                            im = ax.imshow(normalized, cmap="Blues")
                            ax.figure.colorbar(im, ax=ax, label="Recall")
                            ax.set_xticks(np.arange(len(self.classes)))
                            ax.set_yticks(np.arange(len(self.classes)))
                            ax.set_xticklabels(self.classes, rotation=45, ha="right")
                            ax.set_yticklabels(self.classes)
                            for i in range(cm_array.shape[0]):
                                for j in range(cm_array.shape[1]):
                                    ax.text(j, i, annotation_grid[i, j], ha="center", va="center", color="black")

                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title(f"{display_name} Confusion Matrix (Epoch {epoch})")
                        fig_path = epoch_dir / f"{display_name.lower().replace(' ', '_')}_confusion_matrix.png"
                        fig.savefig(fig_path, dpi=200)
                        wandb.log({f"{stage_key}/{display_name.lower().replace(' ', '_')}_confusion_matrix": wandb.Image(fig)}, step=self.global_step)
                        plt.close(fig)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.warning("Failed to render %s confusion matrix heatmap: %s", display_name, exc)

            bar_payload_available = per_head_accuracy or per_head_confusion
            if bar_payload_available:
                try:
                    import matplotlib.pyplot as plt

                    accuracy_matrix: Dict[str, np.ndarray] = {}
                    if per_head_accuracy:
                        for head_name, values in per_head_accuracy.items():
                            display_name = head_display_map.get(head_name, head_name)
                            accuracy_matrix[display_name] = np.asarray(values, dtype=np.float64)
                    elif per_head_confusion:
                        for head_name, matrix in per_head_confusion.items():
                            display_name = head_display_map.get(head_name, head_name)
                            conf = np.asarray(matrix, dtype=np.float64)
                            row_totals = conf.sum(axis=1)
                            recall = np.zeros(conf.shape[0], dtype=np.float64)
                            nonzero = row_totals > 0
                            recall[nonzero] = np.diag(conf)[nonzero] / row_totals[nonzero]
                            accuracy_matrix[display_name] = recall
                    balanced_head_metrics: Dict[str, float] = {}
                    for display_name, recalls in accuracy_matrix.items():
                        if recalls.size:
                            balanced_head_metrics[display_name] = float(np.nanmean(recalls))
                    for display_name, value in balanced_head_metrics.items():
                        original_name = reverse_head_label_map.get(display_name, display_name)
                        metric_alias = {
                            "classical_a": "balanced_acc_full_res",
                            "classical_b": "balanced_acc_compressed",
                            "quantum": "balanced_acc_quantum",
                            "ensemble": "balanced_acc_ensemble",
                        }.get(original_name)
                        key = metric_alias or f"balanced_acc_{original_name.lower().replace(' ', '_')}"
                        metrics_log[f"{stage_key}/{key}"] = value

                    if accuracy_matrix:
                        head_names = list(accuracy_matrix.keys())
                        num_heads = len(head_names)
                        sample_values = next(iter(accuracy_matrix.values()))
                        num_classes = len(sample_values)
                        class_labels = [self.classes[idx] if idx < len(self.classes) else str(idx) for idx in range(num_classes)]
                        bar_width = 0.75 / max(1, num_heads)
                        indices = np.arange(num_classes, dtype=np.float64)
                        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                        for offset, head_name in enumerate(head_names):
                            values = accuracy_matrix[head_name]
                            positions = indices + (offset - (num_heads - 1) / 2) * bar_width
                            ax.bar(positions, values, bar_width, label=head_name)
                        ax.set_xticks(indices)
                        ax.set_xticklabels(class_labels, rotation=45, ha="right")
                        ax.set_ylim(0.0, 1.0)
                        ax.set_ylabel("Recall")
                        ax.set_title(f"{stage_label} Head Recall by Class (Epoch {epoch})")
                        ax.legend()

                        for offset, head_name in enumerate(head_names):
                            values = accuracy_matrix[head_name]
                            positions = indices + (offset - (num_heads - 1) / 2) * bar_width
                            for xpos, value in zip(positions, values):
                                ax.text(xpos, max(value - 0.02, 0.01), f"{value*100:.1f}%", ha="center", va="bottom", fontsize=8, rotation=90)

                        bar_path = epoch_dir / "head_confusion_bar.png"
                        fig.savefig(bar_path, dpi=200)
                        wandb.log({f"{stage_key}/head_confusion_bar": wandb.Image(fig)}, step=self.global_step)
                        plt.close(fig)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to render head confusion bar chart: %s", exc)

            per_head_accuracy = val_outcome.get("per_head_class_accuracy")
            normalized_accuracy: Dict[str, np.ndarray] = {}
            if per_head_accuracy:
                normalized_accuracy = {
                    head_display_map[head_name]: np.array(values, dtype=np.float64)
                    for head_name, values in per_head_accuracy.items()
                    if values is not None and head_name in head_display_map
                }
                if not normalized_accuracy:
                    per_head_accuracy = None
            if per_head_accuracy and normalized_accuracy:
                try:
                    import matplotlib.pyplot as plt

                    sample_head = next((vals for vals in normalized_accuracy.values() if vals is not None), None)
                    if sample_head is None:
                        raise ValueError("per_head_class_accuracy did not include any active entries")
                    num_classes = len(sample_head)
                    class_labels = [self.classes[idx] if idx < len(self.classes) else str(idx) for idx in range(num_classes)]
                    head_names = list(normalized_accuracy.keys())
                    index = np.arange(len(class_labels), dtype=np.float64)
                    bar_width = 0.8 / max(1, len(head_names))
                    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
                    for head_offset, head_name in enumerate(head_names):
                        values = normalized_accuracy[head_name]
                        positions = index + (head_offset - (len(head_names) - 1) / 2) * bar_width
                        ax.bar(positions, values, bar_width, label=head_name)
                    ax.set_xticks(index)
                    ax.set_xticklabels(class_labels, rotation=45, ha="right")
                    ax.set_ylim(0.0, 1.0)
                    ax.set_ylabel("Accuracy")
                    ax.set_title(f"{stage_label} Head Accuracy (Epoch {epoch})")
                    ax.legend()
                    acc_path = epoch_dir / "head_accuracy.png"
                    fig.savefig(acc_path, dpi=200)
                    wandb.log({f"{stage_key}/head_accuracy_bar": wandb.Image(fig)}, step=self.global_step)
                    plt.close(fig)

                    summary_table = wandb.Table(columns=["class"] + head_names)
                    for class_idx, class_label in enumerate(class_labels):
                        row = [class_label]
                        for head_name in head_names:
                            row.append(float(normalized_accuracy[head_name][class_idx]))
                        summary_table.add_data(*row)
                    wandb.log({f"{stage_key}/head_accuracy_table": summary_table}, step=self.global_step)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to create head accuracy bar chart: %s", exc)

                stage_history = self._confidence_history[stage_key]
                for head_name, accuracies in normalized_accuracy.items():
                    for class_idx, value in enumerate(accuracies):
                        class_label = self.classes[class_idx] if class_idx < len(self.classes) else str(class_idx)
                        stage_history[class_label][head_name].append((epoch, float(value)))

                for class_label, head_series in stage_history.items():
                    epochs = sorted({ep for series in head_series.values() for ep, _ in series})
                    if not epochs:
                        continue
                    xs = epochs
                    ys: List[List[float]] = []
                    keys: List[str] = []
                    for head_name, series in head_series.items():
                        value_map = {ep: val for ep, val in series}
                        ys.append([value_map.get(ep, float("nan")) for ep in xs])
                        keys.append(head_name)
                    if xs and ys:
                        line_plot = wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title=f"{stage_label} {class_label} Head Accuracy",
                            xname="Epoch",
                        )
                        wandb.log({f"{stage_key}/head_accuracy_trend/{class_label}": line_plot}, step=self.global_step)
                        trend_table = wandb.Table(columns=["epoch"] + keys)
                        for x_idx, epoch_value in enumerate(xs):
                            row = [epoch_value]
                            for series_idx, key in enumerate(keys):
                                value = ys[series_idx][x_idx]
                                row.append(None if math.isnan(value) else float(value))
                            trend_table.add_data(*row)
                        wandb.log({f"{stage_key}/head_accuracy_trend/{class_label}_table": trend_table}, step=self.global_step)

            if self.gradcam_helper.enabled and self._interpretability_samples:
                try:
                    gradcam_epoch_dir = epoch_dir / "gradcam"
                    gradcam_epoch_dir.mkdir(parents=True, exist_ok=True)
                    cam_payloads = self.gradcam_helper.generate_overlays(
                        images=self._interpretability_samples["images"],
                        labels=self._interpretability_samples["labels"],
                        paths=self._interpretability_samples.get("paths"),
                        class_names=self.classes,
                    )
                    if cam_payloads:
                        collage_payloads: List[Dict[str, Any]] = []
                        try:
                            import matplotlib.pyplot as plt

                            max_columns = min(len(self.classes), 5)
                            covered_classes: set[str] = set()
                            selected_indices: set[int] = set()
                            ordered_payloads: List[Dict[str, Any]] = []

                            # Prefer ground-truth coverage first.
                            for idx, payload in enumerate(cam_payloads):
                                target_label = payload.get("target_label")
                                if target_label and target_label in self.classes and target_label not in covered_classes:
                                    ordered_payloads.append(payload)
                                    covered_classes.add(target_label)
                                    selected_indices.add(idx)
                                if len(ordered_payloads) >= len(self.classes):
                                    break

                            # Back-fill with predictions for any remaining classes.
                            if len(ordered_payloads) < len(self.classes):
                                for idx, payload in enumerate(cam_payloads):
                                    if idx in selected_indices:
                                        continue
                                    pred_label = payload.get("pred_label")
                                    if pred_label and pred_label in self.classes and pred_label not in covered_classes:
                                        ordered_payloads.append(payload)
                                        covered_classes.add(pred_label)
                                        selected_indices.add(idx)
                                    if len(ordered_payloads) >= len(self.classes):
                                        break

                            # Fill remaining slots (up to max_columns) with any unused payloads.
                            if len(ordered_payloads) < max_columns:
                                for idx, payload in enumerate(cam_payloads):
                                    if idx in selected_indices:
                                        continue
                                    ordered_payloads.append(payload)
                                    selected_indices.add(idx)
                                    if len(ordered_payloads) >= max_columns:
                                        break

                            collage_payloads = ordered_payloads[:max_columns]
                            if collage_payloads:
                                num_cols = len(collage_payloads)
                                fig, axes = plt.subplots(2, num_cols, figsize=(3.5 * num_cols, 6), constrained_layout=True)
                                fig.suptitle(f"{stage_label} Grad-CAM (Epoch {epoch})", fontsize=14)
                                axes = np.array(axes).reshape(2, num_cols)
                                for col, payload in enumerate(collage_payloads):
                                    base_img = np.clip(payload["base"], 0.0, 1.0)
                                    heatmap = np.clip(payload["heatmap"], 0.0, 1.0)
                                    axes[0, col].imshow(base_img)
                                    axes[0, col].axis("off")
                                    axes[0, col].set_title(payload.get("target_label", ""), fontsize=11)
                                    axes[1, col].imshow(base_img)
                                    axes[1, col].imshow(heatmap, cmap="inferno", alpha=0.55)
                                    axes[1, col].axis("off")
                                    axes[1, col].set_title(
                                        f"Pred: {payload.get('pred_label', '')} ({payload.get('confidence', 0.0):.2f})",
                                        fontsize=10,
                                    )
                                collage_path = gradcam_epoch_dir / "collage.png"
                                fig.savefig(collage_path, dpi=200)
                                wandb.log({f"{stage_key}/gradcam_collage": wandb.Image(fig)}, step=self.global_step)
                                plt.close(fig)
                        except Exception as collage_exc:  # pragma: no cover - defensive
                            LOGGER.warning("Failed to construct Grad-CAM collage: %s", collage_exc)

                        overlay_images = []
                        original_images = []
                        for item_idx, payload in enumerate(collage_payloads):
                            overlay_uint8 = np.clip(payload["overlay"] * 255.0, 0.0, 255.0).astype(np.uint8)
                            base_uint8 = np.clip(payload["base"] * 255.0, 0.0, 255.0).astype(np.uint8)
                            caption_parts = [
                                payload.get("path") or "sample",
                                f"target: {payload.get('target_label', '')}",
                                f"pred: {payload.get('pred_label', '')} ({payload.get('confidence', 0.0):.2f})",
                            ]
                            caption = " | ".join(part for part in caption_parts if part)
                            overlay_images.append(wandb.Image(overlay_uint8, caption=f"Overlay | {caption}"))
                            original_images.append(wandb.Image(base_uint8, caption=f"Original | {caption}"))

                            try:
                                overlay_path = gradcam_epoch_dir / f"overlay_{item_idx:02d}.png"
                                base_path = gradcam_epoch_dir / f"original_{item_idx:02d}.png"
                                Image.fromarray(overlay_uint8).save(overlay_path)
                                Image.fromarray(base_uint8).save(base_path)
                            except Exception as img_exc:  # pragma: no cover - defensive
                                LOGGER.debug("Failed to persist Grad-CAM images: %s", img_exc)
                        log_payload: Dict[str, Any] = {f"{stage_key}/gradcam_overlays": overlay_images}
                        if original_images:
                            log_payload[f"{stage_key}/gradcam_originals"] = original_images
                        wandb.log(log_payload, step=self.global_step)
                except Exception as cam_exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to log Grad-CAM overlays: %s", cam_exc)

            if metrics_log:
                wandb.log(metrics_log, step=self.global_step)
        except ImportError:  # pragma: no cover - optional dependency
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to log W&B visualizations: %s", exc)

    def _create_optimizer(self, stage: StageConfig) -> torch.optim.Optimizer:
        params = [p for p in self.base_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=stage.learning_rate, weight_decay=stage.weight_decay)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    def _stage_mask(self, stage: StageConfig) -> Optional[torch.Tensor]:
        return _convert_mask(stage.active_mask)

    def _resolve_tracked_heads(self, mask_tensor: Optional[torch.Tensor]) -> List[str]:
        heads: List[str] = []
        mask_list: Optional[List[float]] = None
        if mask_tensor is not None:
            mask_list = mask_tensor.detach().cpu().tolist()
        if mask_list is None or (len(mask_list) > 0 and mask_list[0] > 0):
            heads.append("classical_a")
        if mask_list is None or (len(mask_list) > 1 and mask_list[1] > 0):
            heads.append("classical_b")
        quantum_active = getattr(self.base_model, "quantum_enabled", False)
        if quantum_active and (mask_list is None or (len(mask_list) > 2 and mask_list[2] > 0)):
            heads.append("quantum")
        heads.append("ensemble")
        return heads

    def _apply_quantum_annealing(self, stage: StageConfig, epoch_idx: int) -> Optional[Tuple[float, float]]:
        if stage.name.lower() != "quantum":
            return None
        quantum_head = getattr(self.base_model, "quantum_head", None)
        if quantum_head is None or not hasattr(quantum_head, "set_annealing_params"):
            return None
        schedule = stage.annealing_schedule or {}
        if not schedule:
            return None

        start_temp = float(schedule.get("start_temp", schedule.get("start_temperature", 2.0)))
        end_temp = float(schedule.get("end_temp", schedule.get("end_temperature", 0.6)))
        start_noise = float(schedule.get("start_noise", schedule.get("start_noise_scale", 0.15)))
        end_noise = float(schedule.get("end_noise", schedule.get("end_noise_scale", 0.0)))
        strategy = str(schedule.get("strategy", "cosine")).lower()
        gamma = float(schedule.get("gamma", 3.0))
        steepness = float(schedule.get("steepness", 6.0))

        total_epochs = max(1, stage.num_epochs - 1)
        progress = min(1.0, max(0.0, epoch_idx / total_epochs))

        if strategy == "linear":
            weight = progress
        elif strategy == "exponential":
            weight = 1.0 - math.exp(-gamma * progress)
        elif strategy == "sigmoid":
            weight = 1.0 / (1.0 + math.exp(-steepness * (progress - 0.5)))
        else:  # cosine
            weight = 0.5 * (1.0 - math.cos(math.pi * progress))

        temperature = start_temp + (end_temp - start_temp) * weight
        noise = start_noise + (end_noise - start_noise) * weight
        temperature = float(max(1e-3, temperature))
        noise = float(max(0.0, noise))
        quantum_head.set_annealing_params(temperature, noise)
        return temperature, noise

    def _should_disable_data_parallel(self, stage: StageConfig) -> bool:
        if not self.data_parallel:
            return False
        if stage.force_data_parallel is not None:
            return not bool(stage.force_data_parallel)
        mask = stage.active_mask
        if not mask:
            return False
        if len(mask) < 3:
            return False
        quantum_active = mask[2] > 0
        classical_active = (mask[0] > 0) or (mask[1] > 0)
        return quantum_active and not classical_active

    def _compute_metrics(
        self,
        outputs: ModelForwardOutput,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        mask_tensor = outputs.active_mask
        mask_list: Optional[List[float]] = None
        if mask_tensor is not None:
            mask_list = mask_tensor.detach().cpu().tolist()

        def _is_active(idx: int, default: bool = True) -> bool:
            if mask_list is None:
                return default
            if idx >= len(mask_list):
                return default
            return bool(mask_list[idx] > 0)

        quantum_enabled = getattr(self.base_model, "quantum_enabled", False)

        metrics: Dict[str, float] = {}
        if _is_active(0):
            acc_full_res = compute_accuracy(outputs.output_a, targets)
            metrics["acc_full_res"] = acc_full_res
            metrics["acc_a"] = acc_full_res
        if _is_active(1):
            acc_compressed = compute_accuracy(outputs.output_b, targets)
            metrics["acc_compressed"] = acc_compressed
            metrics["acc_b"] = acc_compressed
        if quantum_enabled and _is_active(2):
            acc_quantum = compute_accuracy(outputs.output_c, targets)
            metrics["acc_quantum"] = acc_quantum
            metrics["acc_c"] = acc_quantum
        metrics["acc_ensemble"] = compute_accuracy(outputs.final_output, targets)
        return metrics

    def _aggregate_latencies(self, latency_accumulator: Dict[str, List[float]]) -> Dict[str, float]:
        return {key: float(statistics.mean(values)) for key, values in latency_accumulator.items() if values}

    def _clone_loader(
        self,
        loader: DataLoader,
        batch_size: int,
        is_train: bool,
    ) -> DataLoader:
        if batch_size <= 0:
            raise ValueError("Batch size override must be positive")
        base_sampler = loader.sampler
        generator = getattr(loader, "generator", None)
        sampler = None
        if isinstance(base_sampler, WeightedRandomSampler):
            weight_tensor = base_sampler.weights.clone() if hasattr(base_sampler, "weights") else base_sampler.weights
            sampler = WeightedRandomSampler(
                weights=weight_tensor,
                num_samples=base_sampler.num_samples,
                replacement=base_sampler.replacement,
                generator=generator,
            )
        elif isinstance(base_sampler, SequentialSampler):
            sampler = SequentialSampler(loader.dataset)
        elif isinstance(base_sampler, RandomSampler):
            sampler = None
        elif base_sampler is not None:
            sampler = base_sampler

        shuffle = bool(is_train) and sampler is None
        drop_last = loader.drop_last if is_train else False
        loader_kwargs: Dict[str, Any] = {
            "dataset": loader.dataset,
            "batch_size": int(batch_size),
            "shuffle": shuffle,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "generator": generator,
            "persistent_workers": getattr(loader, "persistent_workers", False),
        }
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        if prefetch_factor is not None and loader.num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        pin_memory_device = getattr(loader, "pin_memory_device", None)
        if pin_memory_device:
            loader_kwargs["pin_memory_device"] = pin_memory_device
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        return DataLoader(**loader_kwargs)

    def _prepare_stage_loader(
        self,
        loader: DataLoader,
        stage: StageConfig,
        is_train: bool,
    ) -> DataLoader:
        desired_batch = stage.train_batch_size if is_train else stage.val_batch_size
        if desired_batch is None:
            return loader
        current_batch = loader.batch_size
        if current_batch is None or int(desired_batch) != int(current_batch):
            return self._clone_loader(loader, int(desired_batch), is_train)
        return loader

    def _maybe_auto_tune_batch_size(
        self,
        loader: DataLoader,
        stage: StageConfig,
        mask_tensor: Optional[torch.Tensor],
        loss_weights: Dict[str, float],
        stage_model: torch.nn.Module,
        is_train: bool,
    ) -> Optional[int]:
        if not self.config.auto_tune_batch_size or self.device.type != "cuda":
            return None
        attr_name = "train_batch_size" if is_train else "val_batch_size"
        if getattr(stage, attr_name) is not None:
            return None
        if not is_train and not self.config.batch_tuner_include_val:
            return None
        cache_key = (stage.name, attr_name)
        if cache_key in self._batch_tuner_cache:
            return self._batch_tuner_cache[cache_key]
        tuned = self._auto_tune_batch_size(loader, stage, mask_tensor, loss_weights, stage_model, is_train)
        if tuned:
            self._batch_tuner_cache[cache_key] = tuned
            phase = "train" if is_train else "val"
            LOGGER.info("Batch tuner selected batch_size=%d for stage %s (%s)", tuned, stage.name, phase)
            if self.wandb_run:
                try:
                    import wandb

                    stage_key = self._stage_log_key(stage)
                    wandb.log({f"tuner/{stage_key}_{phase}_batch_size": tuned}, step=self.global_step)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to log batch tuner metrics: %s", exc)
        return tuned

    def _auto_tune_batch_size(
        self,
        loader: DataLoader,
        stage: StageConfig,
        mask_tensor: Optional[torch.Tensor],
        loss_weights: Dict[str, float],
        stage_model: torch.nn.Module,
        include_backward: bool,
    ) -> Optional[int]:
        base_batch = max(1, int(loader.batch_size or 1))
        min_batch = int(self.config.batch_tuner_min_batch_size or base_batch)
        max_batch = int(self.config.batch_tuner_max_batch_size or base_batch * 4)
        if max_batch < min_batch:
            max_batch = min_batch
        growth = max(1.1, float(self.config.batch_tuner_growth_factor) if self.config.batch_tuner_growth_factor else 1.5)
        target_util = float(min(max(self.config.batch_tuner_target_utilization, 0.1), 0.99))
        max_latency_factor = float(max(self.config.batch_tuner_max_latency_increase, 1.0))
        total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
        best_batch = min_batch
        baseline_latency = None
        candidate = max(min_batch, base_batch)
        tried_any = False
        while candidate <= max_batch:
            try:
                metrics = self._evaluate_batch_candidate(
                    loader,
                    mask_tensor,
                    loss_weights,
                    stage_model,
                    int(candidate),
                    include_backward,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message or "cuda error" in message:
                    LOGGER.info("Batch tuning hit CUDA OOM at batch size %d; backing off", candidate)
                    torch.cuda.empty_cache()
                    break
                raise
            if metrics is None:
                break
            tried_any = True
            utilization = metrics["peak_memory"] / max(total_memory, 1.0)
            latency = metrics["elapsed_ms"]
            if baseline_latency is None:
                baseline_latency = latency
            if utilization > target_util + 1e-3:
                LOGGER.debug(
                    "Batch tuner stopping at size %d (utilization %.2f%% > target %.2f%%)",
                    candidate,
                    utilization * 100.0,
                    target_util * 100.0,
                )
                break
            if latency <= baseline_latency * max_latency_factor:
                best_batch = int(candidate)
            else:
                LOGGER.debug(
                    "Batch tuner rejected size %d due to latency %.2fms (baseline %.2fms)",
                    candidate,
                    latency,
                    baseline_latency,
                )
                break
            next_candidate = max(candidate + 1, int(candidate * growth))
            if next_candidate == candidate:
                next_candidate += 1
            candidate = next_candidate
            torch.cuda.empty_cache()
        if not tried_any:
            return None
        return best_batch

    def _evaluate_batch_candidate(
        self,
        loader: DataLoader,
        mask_tensor: Optional[torch.Tensor],
        loss_weights: Dict[str, float],
        stage_model: torch.nn.Module,
        batch_size: int,
        include_backward: bool,
    ) -> Optional[Dict[str, float]]:
        if batch_size <= 0:
            return None
        trial_loader = self._clone_loader(loader, batch_size, include_backward)
        iterator = iter(trial_loader)
        try:
            batch = next(iterator)
        except StopIteration:
            return None
        images, targets, _ = batch
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        prev_mode = stage_model.training
        stage_model.train(include_backward)
        mask_values = mask_tensor.tolist() if mask_tensor is not None else None
        self.base_model.set_active_mask(mask_values)
        if include_backward:
            self.base_model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        autocast_device = "cuda" if self.device.type == "cuda" else self.device.type
        with torch.amp.autocast(autocast_device, enabled=self.config.mixed_precision and self.device.type == "cuda"):
            raw_outputs = stage_model(
                images,
                return_all=True,
                track_latency=False,
            )
            outputs = self._normalize_output(raw_outputs)
            losses = self._compute_losses(outputs, targets, loss_weights)
            loss = losses["total_loss"]
        if include_backward:
            loss.backward()
        end_event.record()
        torch.cuda.synchronize(self.device)
        peak_memory = float(torch.cuda.max_memory_allocated(self.device))
        elapsed_ms = float(start_event.elapsed_time(end_event))
        if include_backward:
            self.base_model.zero_grad(set_to_none=True)
        stage_model.train(prev_mode)
        del images, targets
        torch.cuda.empty_cache()
        return {
            "peak_memory": peak_memory,
            "elapsed_ms": elapsed_ms,
        }

    @contextlib.contextmanager
    def _swap_ema_parameters(self) -> Iterable[bool]:
        if not self.ema_helper or self.ema_helper.n_averaged == 0:
            yield False
            return
        self.ema_helper.to(self.device)
        self.ema_helper.module.to(self.device)
        self.ema_helper.eval()
        self.ema_helper.swap_parameters(self.base_model)
        try:
            yield True
        finally:
            self.ema_helper.swap_parameters(self.base_model)

    def _log_distribution_metrics(
        self,
        stage_name: str,
        epoch: int,
        preds: List[int],
        targets: List[int],
    ) -> Optional[Dict[str, Any]]:
        if not preds:
            return None
        num_classes = len(self.classes)
        pred_counts = np.bincount(np.asarray(preds, dtype=np.int64), minlength=num_classes)
        target_counts = (
            np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes)
            if targets
            else np.zeros(num_classes, dtype=np.int64)
        )
        pred_total = max(int(pred_counts.sum()), 1)
        target_total = max(int(target_counts.sum()), 1)
        pred_dist = pred_counts.astype(np.float64) / float(pred_total)
        target_dist = target_counts.astype(np.float64) / float(target_total)

        if self.wandb_run and self.config.collapse_log_distributions:
            distribution_metrics: Dict[str, Any] = {}
            for idx, class_name in enumerate(self.classes):
                safe_label = class_name.replace(" ", "_").lower()
                distribution_metrics[f"val/pred_ratio/{safe_label}"] = float(pred_dist[idx])
                distribution_metrics[f"val/target_ratio/{safe_label}"] = float(target_dist[idx])
                distribution_metrics[f"val/pred_count/{safe_label}"] = int(pred_counts[idx])
                distribution_metrics[f"val/target_count/{safe_label}"] = int(target_counts[idx])
            self._log_to_wandb(distribution_metrics, stage_name, epoch)

        return {
            "pred_counts": pred_counts,
            "target_counts": target_counts,
            "pred_dist": pred_dist,
            "target_dist": target_dist,
        }

    def _monitor_prediction_collapse(
        self,
        stage_name: str,
        epoch: int,
        distribution_info: Optional[Dict[str, Any]],
    ) -> None:
        if not self.config.collapse_detection_enabled or not distribution_info:
            return
        pred_dist = distribution_info.get("pred_dist")
        if pred_dist is None or len(pred_dist) == 0:
            return
        dominant_idx = int(np.argmax(pred_dist))
        dominant_ratio = float(pred_dist[dominant_idx])
        tracker = self._collapse_tracker[stage_name]
        tracker["cooldown"] = max(0, tracker.get("cooldown", 0) - 1)
        threshold = float(self.config.collapse_threshold)
        patience = max(1, int(self.config.collapse_patience))

        if dominant_ratio >= threshold:
            if tracker.get("dominant_class") == dominant_idx:
                tracker["streak"] += 1
            else:
                tracker["streak"] = 1
                tracker["dominant_class"] = dominant_idx
            class_label = self.classes[dominant_idx] if dominant_idx < len(self.classes) else str(dominant_idx)
            LOGGER.warning(
                "%s | Epoch %02d | validation predictions dominated by '%s' (%.2f%%)",
                stage_name.upper(),
                epoch,
                class_label,
                dominant_ratio * 100.0,
            )
            if tracker["streak"] >= patience and tracker.get("cooldown", 0) == 0:
                self._apply_collapse_reweighting(
                    stage_name,
                    epoch,
                    dominant_idx,
                    dominant_ratio,
                )
                tracker["cooldown"] = max(0, int(self.config.collapse_cooldown))
                tracker["streak"] = 0
        else:
            tracker["streak"] = 0
            tracker["dominant_class"] = None

    def _apply_collapse_reweighting(
        self,
        stage_name: str,
        epoch: int,
        class_idx: int,
        dominant_ratio: float,
    ) -> None:
        if self.class_weight_tensor is None:
            LOGGER.warning(
                "Prediction collapse detected on class index %d but class weights are unavailable for adjustment",
                class_idx,
            )
            return

        factor = float(self.config.collapse_reweight_factor)
        min_weight = max(float(self.config.collapse_min_class_weight), 1e-5)
        base_weights = (
            self._base_class_weight_tensor.clone()
            if self._base_class_weight_tensor is not None
            else self.class_weight_tensor.detach().cpu().clone()
        )
        adjusted = self.class_weight_tensor.detach().cpu().clone()
        adjusted[class_idx] = max(float(adjusted[class_idx]) * factor, min_weight)
        adjusted = torch.clamp(adjusted, min=min_weight)
        adjusted = adjusted / adjusted.sum() * adjusted.shape[0]

        self.class_weight_tensor = adjusted.to(self.device)
        self.config.class_weights = adjusted.tolist()
        class_label = self.classes[class_idx] if class_idx < len(self.classes) else str(class_idx)
        LOGGER.warning(
            "Detected prediction collapse on '%s' (ratio %.3f); updated class weights -> %s",
            class_label,
            dominant_ratio,
            ", ".join(f"{float(weight):.3f}" for weight in adjusted.tolist()),
        )

        event_record = {
            "stage": stage_name,
            "epoch": epoch,
            "dominant_class": class_label,
            "dominant_ratio": dominant_ratio,
            "updated_weights": adjusted.tolist(),
            "baseline_weights": base_weights.tolist(),
        }
        self._collapse_events.append(event_record)

        if self.wandb_run:
            weight_metrics: Dict[str, Any] = {
                "collapse/dominant_ratio": dominant_ratio,
                "collapse/dominant_class_idx": class_idx,
            }
            for idx, weight in enumerate(adjusted.tolist()):
                label = self.classes[idx] if idx < len(self.classes) else str(idx)
                weight_metrics[f"collapse/weight/{label.replace(' ', '_').lower()}"] = float(weight)
            self._log_to_wandb(weight_metrics, stage_name, epoch)
    def _normalize_output(
        self,
        raw_output: Any,
    ) -> ModelForwardOutput:
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
                active_mask = active_payload.to(self.device)
            elif active_payload is None:
                stored = getattr(self.base_model, "_active_mask", None)
                active_mask = (
                    torch.as_tensor(stored, device=self.device, dtype=torch.float32)
                    if stored is not None
                    else None
                )
            else:
                active_mask = torch.as_tensor(active_payload, device=self.device, dtype=torch.float32)

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

        raise TypeError("Unexpected forward output type from model: %r" % (type(raw_output),))

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        stage: StageConfig,
        epoch_idx: int,
        train: bool,
        mask_tensor: Optional[torch.Tensor],
        loss_weights: Dict[str, float],
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        stage_key = self._stage_log_key(stage)
        is_train = train and optimizer is not None
        skip_ensemble = bool(getattr(stage, "skip_ensemble", False))
        ema_ctx = self._swap_ema_parameters() if (not is_train and self.ema_helper and self.ema_helper.n_averaged > 0) else contextlib.nullcontext()
        with ema_ctx:
            model.train(is_train)
            epoch_loss = 0.0
            metrics_accumulator = defaultdict(float)
            latency_accumulator: Dict[str, List[float]] = defaultdict(list)
            ensemble_weights: List[np.ndarray] = []
            total_batches = 0
            preds: List[int] = []
            targets_all: List[int] = []

            autocast_enabled = self.config.mixed_precision and self.device.type == "cuda" and torch.cuda.is_available()
            autocast_device = "cuda" if autocast_enabled else self.device.type

            accum_steps = max(1, getattr(stage, "grad_accum_steps", 1))
            accum_counter = 0
            if is_train:
                optimizer.zero_grad()

            mask_values = mask_tensor.tolist() if mask_tensor is not None else None
            self.base_model.set_active_mask(mask_values)

            per_class_confidence: Optional[Dict[str, np.ndarray]] = None
            per_class_counts: Optional[Dict[str, np.ndarray]] = None
            per_head_predictions: Optional[Dict[str, List[int]]] = None
            head_label_map: Dict[str, str] = {}
            reverse_head_label_map: Dict[str, str] = {}

            iterator = tqdm(
                loader,
                desc=f"{stage.name.capitalize()} | Epoch {epoch_idx + 1} {'Train' if is_train else 'Val'}",
                dynamic_ncols=True,
            )
            for batch_idx, (images, targets, _) in enumerate(iterator):
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.set_grad_enabled(is_train):
                    with torch.amp.autocast(autocast_device, enabled=autocast_enabled):
                        raw_outputs = model(
                            images,
                            return_all=True,
                            track_latency=self.config.enable_latency_tracking,
                            skip_ensemble=skip_ensemble,
                        )
                        outputs = self._normalize_output(raw_outputs)
                        losses = self._compute_losses(outputs, targets, loss_weights)
                        loss = losses["total_loss"]

                if is_train:
                    scale_loss = loss / accum_steps
                    if self.config.mixed_precision:
                        self.scaler.scale(scale_loss).backward()
                    else:
                        scale_loss.backward()

                    accum_counter += 1
                    if accum_counter % accum_steps == 0:
                        if self.config.mixed_precision:
                            if self.config.grad_clip:
                                self.scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            if self.config.grad_clip:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                            optimizer.step()
                        optimizer.zero_grad()
                        if self.ema_helper:
                            step_idx = self.global_step + 1
                            if step_idx >= self._ema_update_after and ((step_idx - self._ema_update_after) % self._ema_update_every == 0):
                                self.ema_helper.update_parameters(self.base_model)
                        accum_counter = 0

                epoch_loss += float(loss.item())
                batch_metrics = self._compute_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    metrics_accumulator[key] += value
                for aux_key in ("quantum_entropy_penalty", "quantum_distill_penalty"):
                    aux_val = losses.get(aux_key)
                    if isinstance(aux_val, torch.Tensor):
                        metrics_accumulator[aux_key] += float(aux_val.detach().item())

                if outputs.latencies:
                    for key, value in outputs.latencies.items():
                        latency_accumulator[key].append(float(value))

                ensemble_weights.append(outputs.ensemble_weights.detach().cpu().numpy())

                if not is_train:
                    predictions = torch.argmax(outputs.final_output, dim=1)
                    preds.extend(predictions.cpu().numpy().tolist())
                    targets_all.extend(targets.cpu().numpy().tolist())

                    if per_class_confidence is None:
                        num_classes = len(self.classes)
                        tracked_heads = self._resolve_tracked_heads(mask_tensor)
                        head_label_map = {head_name: self._head_display_name(head_name) for head_name in tracked_heads}
                        per_class_confidence = {
                            head_label_map[head_name]: np.zeros(num_classes, dtype=np.float64)
                            for head_name in tracked_heads
                        }
                        per_class_counts = {
                            head_label_map[head_name]: np.zeros(num_classes, dtype=np.int64)
                            for head_name in tracked_heads
                        }
                        per_head_predictions = {head_label_map[head_name]: [] for head_name in tracked_heads}
                        reverse_head_label_map = {display: original for original, display in head_label_map.items()}

                    head_probabilities = {
                        "classical_a": torch.softmax(outputs.output_a, dim=1),
                        "classical_b": torch.softmax(outputs.output_b, dim=1),
                        "quantum": torch.softmax(outputs.output_c, dim=1),
                        "ensemble": torch.softmax(outputs.final_output, dim=1),
                    }
                    targets_np = targets.cpu().numpy()
                    sample_indices = np.arange(targets_np.shape[0])
                    for head_name, probs in head_probabilities.items():
                        if per_class_confidence is None:
                            continue
                        display_key = head_label_map.get(head_name, head_name)
                        if display_key not in per_class_confidence:
                            continue
                        probs_np = probs.detach().cpu().numpy()
                        head_conf = per_class_confidence[display_key]
                        head_counts = per_class_counts[display_key]
                        np.add.at(head_conf, targets_np, probs_np[sample_indices, targets_np])
                        np.add.at(head_counts, targets_np, 1)

                    if per_head_predictions is not None:
                        head_prediction_tensors: Dict[str, torch.Tensor] = {
                            "classical_a": torch.argmax(outputs.output_a, dim=1),
                            "classical_b": torch.argmax(outputs.output_b, dim=1),
                            "ensemble": predictions,
                        }
                        if head_label_map.get("quantum", "quantum") in per_head_predictions:
                            head_prediction_tensors["quantum"] = torch.argmax(outputs.output_c, dim=1)
                        for head_name, tensor_preds in head_prediction_tensors.items():
                            display_key = head_label_map.get(head_name, head_name)
                            if display_key not in per_head_predictions:
                                continue
                            per_head_predictions[display_key].extend(tensor_preds.cpu().numpy().tolist())

                total_batches += 1
                self.global_step += 1

                if is_train and batch_idx % max(1, self.config.log_interval) == 0:
                    iterator.set_postfix({"loss": loss.item()})
                    if self.wandb_run:
                        batch_log = {"batch/loss": float(loss.item())}
                        for metric_name, metric_value in batch_metrics.items():
                            if metric_name.startswith("acc_"):
                                batch_log[f"batch/{metric_name}"] = float(metric_value)
                        for aux_key in ("quantum_entropy_penalty", "quantum_distill_penalty"):
                            aux_val = losses.get(aux_key)
                            if isinstance(aux_val, torch.Tensor):
                                batch_log[f"batch/{aux_key}"] = float(aux_val.detach().item())
                        self._log_to_wandb(batch_log, stage_key, epoch_idx + 1)

        if total_batches == 0:
            return {
                "loss": float("inf"),
                "metrics": {},
                "latencies": {},
                "ensemble_weights": None,
                "preds": preds,
                "targets": targets_all,
            }

        metrics = {key: value / total_batches for key, value in metrics_accumulator.items()}
        epoch_loss /= total_batches
        avg_weights = np.mean(ensemble_weights, axis=0) if ensemble_weights else None
        latencies = self._aggregate_latencies(latency_accumulator)
        result = {
            "loss": epoch_loss,
            "metrics": metrics,
            "latencies": latencies,
            "ensemble_weights": avg_weights,
            "preds": preds,
            "targets": targets_all,
        }
        if not is_train:
            LOGGER.debug(
                "Stage %s epoch %d validation sample counts: preds=%d targets=%d",
                stage.name,
                epoch_idx + 1,
                len(preds),
                len(targets_all),
            )

        if not is_train:
            class_labels = list(range(len(self.classes)))
            conf_matrix = confusion_matrix(targets_all, preds, labels=class_labels) if targets_all and preds else None
            if conf_matrix is not None:
                result["confusion_matrix"] = conf_matrix.tolist()
            if per_head_predictions and targets_all:
                head_conf_mats: Dict[str, List[List[int]]] = {}
                head_class_accuracy: Dict[str, List[float]] = {}
                for head_name, head_pred_list in per_head_predictions.items():
                    if not head_pred_list:
                        continue
                    conf = confusion_matrix(targets_all, head_pred_list, labels=class_labels)
                    head_conf_mats[head_name] = conf.tolist()
                    conf_np = conf.astype(np.float64)
                    row_totals = conf_np.sum(axis=1)
                    diag_vals = np.diag(conf_np)
                    per_class_acc = np.zeros_like(diag_vals)
                    nonzero_rows = row_totals > 0
                    per_class_acc[nonzero_rows] = diag_vals[nonzero_rows] / row_totals[nonzero_rows]
                    head_class_accuracy[head_name] = per_class_acc.tolist()
                if head_conf_mats:
                    result["head_confusion_matrices"] = head_conf_mats
                    result["per_head_class_accuracy"] = head_class_accuracy
                if head_class_accuracy:
                    result["per_class_confidence"] = head_class_accuracy
            targets_total = len(targets_all)
            if targets_total:
                result["val_distribution"] = {
                    "targets": dict(zip(class_labels, np.bincount(np.asarray(targets_all, dtype=np.int64), minlength=len(class_labels)).tolist())),
                    "preds": dict(zip(class_labels, np.bincount(np.asarray(preds, dtype=np.int64), minlength=len(class_labels)).tolist())),
                    "total_samples": targets_total,
                }
            LOGGER.debug(
                "Stage %s epoch %d validation summary: preds=%d targets=%d",
                stage.name,
                epoch_idx + 1,
                len(preds),
                len(targets_all),
            )

        return result

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        stage_summaries: List[Dict[str, Any]] = []
        self._ensure_interpretability_samples(val_loader)
        for stage_idx, stage in enumerate(self.config.stages):
            LOGGER.info("Starting stage %s (%d/%d) for %d epochs", stage.name, stage_idx + 1, len(self.config.stages), stage.num_epochs)
            stage_key = self._stage_log_key(stage)
            stage_train_loader = self._prepare_stage_loader(train_loader, stage, is_train=True)
            stage_val_loader = self._prepare_stage_loader(val_loader, stage, is_train=False)
            self._ensure_interpretability_samples(stage_val_loader)
            mask_tensor = self._stage_mask(stage)
            mask_values = mask_tensor.tolist() if mask_tensor is not None else None
            self.base_model.set_active_mask(mask_values)
            loss_weights = _prepare_loss_weights(stage.loss_weights)
            if hasattr(self.base_model, "set_stage"):
                self.base_model.set_stage(stage.name)
            use_data_parallel = not self._should_disable_data_parallel(stage)
            stage_model = self.model if use_data_parallel else self.base_model
            if not use_data_parallel and self.data_parallel:
                LOGGER.info("Running stage %s without DataParallel to stabilize quantum head execution", stage.name)
            tuned_train = self._maybe_auto_tune_batch_size(
                stage_train_loader,
                stage,
                mask_tensor,
                loss_weights,
                stage_model,
                is_train=True,
            )
            if tuned_train:
                stage_train_loader = self._clone_loader(stage_train_loader, tuned_train, True)
            tuned_val = self._maybe_auto_tune_batch_size(
                stage_val_loader,
                stage,
                mask_tensor,
                loss_weights,
                stage_model,
                is_train=False,
            )
            if tuned_val:
                stage_val_loader = self._clone_loader(stage_val_loader, tuned_val, False)
            optimizer = self._create_optimizer(stage)
            scheduler = self._create_scheduler(optimizer)
            early_stopper = EarlyStopping(patience=stage.patience)
            adaptive_state = {
                "best_loss": float("inf"),
                "best_acc": 0.0,
                "plateau_epochs": 0,
            }

            for epoch in range(stage.num_epochs):
                annealing_state = self._apply_quantum_annealing(stage, epoch)
                train_outcome = self._run_epoch(
                    stage_train_loader,
                    optimizer,
                    stage,
                    epoch,
                    train=True,
                    mask_tensor=mask_tensor,
                    loss_weights=loss_weights,
                    model=stage_model,
                )
                val_outcome = self._run_epoch(
                    stage_val_loader,
                    optimizer,
                    stage,
                    epoch,
                    train=False,
                    mask_tensor=mask_tensor,
                    loss_weights=loss_weights,
                    model=stage_model,
                )

                scheduler.step(val_outcome["loss"])
                early_stopper.step(val_outcome["loss"])

                history_entry = {
                    "stage": stage.name,
                    "epoch": epoch + 1,
                    "train_loss": train_outcome["loss"],
                    "val_loss": val_outcome["loss"],
                    "train_metrics": train_outcome["metrics"],
                    "val_metrics": val_outcome["metrics"],
                    "latencies": val_outcome["latencies"],
                    "ensemble_weights": (
                        val_outcome["ensemble_weights"].tolist()
                        if isinstance(val_outcome["ensemble_weights"], np.ndarray)
                        else None
                    ),
                }
                self.history.append(history_entry)

                wandb_metrics = {
                    "train/loss": train_outcome["loss"],
                    "val/loss": val_outcome["loss"],
                }
                wandb_metrics.update({f"train/{k}": v for k, v in train_outcome["metrics"].items()})
                wandb_metrics.update({f"val/{k}": v for k, v in val_outcome["metrics"].items()})
                if annealing_state is not None:
                    wandb_metrics[f"{stage.name}/quantum_temperature"] = float(annealing_state[0])
                    wandb_metrics[f"{stage.name}/quantum_noise"] = float(annealing_state[1])
                self._log_to_wandb(wandb_metrics, stage_key, epoch + 1)
                self._log_epoch_visualizations(stage, epoch + 1, val_outcome)
                distribution_info = self._log_distribution_metrics(
                    stage_key,
                    epoch + 1,
                    val_outcome.get("preds", []),
                    val_outcome.get("targets", []),
                )
                self._monitor_prediction_collapse(stage_key, epoch + 1, distribution_info)

                train_acc = train_outcome["metrics"].get("acc_ensemble", 0.0) * 100
                val_acc = val_outcome["metrics"].get("acc_ensemble", 0.0) * 100
                LOGGER.info(
                    "%s | Epoch %02d | train_loss %.4f | train_acc %.2f | val_loss %.4f | val_acc %.2f",
                    stage.name.upper(),
                    epoch + 1,
                    train_outcome["loss"],
                    train_acc,
                    val_outcome["loss"],
                    val_acc,
                )
                if annealing_state is not None:
                    LOGGER.info(
                        "%s | Epoch %02d | annealing temp %.3f | noise %.3f",
                        stage.name.upper(),
                        epoch + 1,
                        float(annealing_state[0]),
                        float(annealing_state[1]),
                    )

                current_val_loss = val_outcome["loss"]
                current_val_acc = val_outcome["metrics"].get("acc_ensemble", 0.0)
                loss_improvement = adaptive_state["best_loss"] - current_val_loss
                acc_improvement = current_val_acc - adaptive_state["best_acc"]

                if loss_improvement > stage.adaptive_threshold or acc_improvement > stage.adaptive_threshold:
                    adaptive_state["best_loss"] = min(adaptive_state["best_loss"], current_val_loss)
                    adaptive_state["best_acc"] = max(adaptive_state["best_acc"], current_val_acc)
                    adaptive_state["plateau_epochs"] = 0
                else:
                    adaptive_state["plateau_epochs"] += 1
                    if stage.adaptive_lr and adaptive_state["plateau_epochs"] >= stage.adaptive_patience:
                        lr_strings: List[str] = []
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            new_lr = max(old_lr * stage.lr_decay_factor, stage.min_lr)
                            if new_lr < old_lr:
                                param_group["lr"] = new_lr
                            lr_strings.append(f"{param_group['lr']:.2e}")
                        adaptive_state["plateau_epochs"] = 0
                        LOGGER.info(
                            "%s | Adaptive LR decay triggered -> %s",
                            stage.name.upper(),
                            ", ".join(lr_strings),
                        )

                if self.best_val_loss is None or current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.best_state_meta = {
                        "stage": stage.name,
                        "epoch": epoch + 1,
                        "val_loss": current_val_loss,
                    }
                    base_model = _unwrap_model(self.model)
                    state_copy = {
                        key: value.detach().cpu().clone()
                        for key, value in base_model.state_dict().items()
                    }
                    self.best_state_weights = state_copy
                    LOGGER.info("New best validation loss: %.4f", current_val_loss)
                    if self.config.checkpoint_dir:
                        self._save_checkpoint(stage, epoch + 1)

                if early_stopper.early_stop:
                    LOGGER.info("Early stopping triggered for stage %s at epoch %d", stage.name, epoch + 1)
                    break

            summary = self._build_stage_summary(stage, stage_val_loader, mask_tensor, loss_weights, stage_model)
            stage_summaries.append(summary)

        if self.best_state_weights:
            _unwrap_model(self.model).load_state_dict(self.best_state_weights)

        training_summary = {
            "history": self.history,
            "stage_summaries": stage_summaries,
            "best_state": self.best_state_meta,
            "classes": self.classes,
        }
        if self._collapse_events:
            training_summary["collapse_events"] = self._collapse_events
        if self.quant_manager:
            try:
                quantization_info = self.quant_manager.generate_artifacts(self.base_model, val_loader)
                if quantization_info:
                    training_summary["quantization"] = quantization_info
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Quantization pipeline failed: %s", exc)
        if self.wandb_run:
            try:
                import wandb

                wandb.summary.update(training_summary)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to push summary to W&B: %s", exc)
        return training_summary

    def _build_stage_summary(
        self,
        stage: StageConfig,
        val_loader: DataLoader,
        mask_tensor: Optional[torch.Tensor],
        loss_weights: Dict[str, float],
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        model.eval()
        preds: List[int] = []
        targets: List[int] = []
        mask_values = mask_tensor.tolist() if mask_tensor is not None else None
        self.base_model.set_active_mask(mask_values)
        skip_ensemble = bool(getattr(stage, "skip_ensemble", False))
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                raw_outputs = model(
                    images,
                    return_all=True,
                    track_latency=self.config.enable_latency_tracking,
                    skip_ensemble=skip_ensemble,
                )
                outputs = self._normalize_output(raw_outputs)
                _ = self._compute_losses(outputs, labels, loss_weights)
                preds.extend(torch.argmax(outputs.final_output, dim=1).cpu().numpy().tolist())
                targets.extend(labels.cpu().numpy().tolist())

        conf = confusion_matrix(targets, preds, labels=list(range(len(self.classes))))
        report = classification_report(
            targets,
            preds,
            target_names=self.classes,
            zero_division=0,
            output_dict=True,
        )
        summary = {
            "stage": stage.name,
            "confusion_matrix": conf.tolist(),
            "classification_report": report,
        }
        self._log_confusion_matrix_to_wandb(stage.name, targets, preds)
        return summary

    def _log_confusion_matrix_to_wandb(self, stage_name: str, targets: List[int], preds: List[int]) -> None:
        if not self.wandb_run:
            return
        try:
            import wandb

            cm_plot = wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets,
                preds=preds,
                class_names=self.classes,
            )
            wandb.log({f"{stage_name}/confusion_matrix": cm_plot}, step=self.global_step)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to log confusion matrix to W&B: %s", exc)

    def _save_checkpoint(self, stage: StageConfig, epoch: int) -> None:
        if not self.config.checkpoint_dir:
            return
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"stage_{stage.name}_epoch_{epoch}.pth"
        base_model = _unwrap_model(self.model)
        torch.save(
            {
                "model_state_dict": base_model.state_dict(),
                "stage": stage.name,
                "epoch": epoch,
                "history": self.history,
            },
            checkpoint_path,
        )
        LOGGER.debug("Saved checkpoint to %s", checkpoint_path)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def save_model_and_features(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: Iterable[str],
    device: str,
    training_summary: Dict[str, Any],
    output_dir: Path | str = "trained_model",
) -> Dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model(model)
    base_model.eval()

    train_features: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    val_features: List[np.ndarray] = []
    val_labels: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in tqdm(train_loader, desc="Extract train features"):
            images = images.to(device)
            features = base_model.extract_compressed_features(images).cpu().numpy()
            train_features.append(features)
            train_labels.append(labels.numpy())
        for images, labels, _ in tqdm(val_loader, desc="Extract val features"):
            images = images.to(device)
            features = base_model.extract_compressed_features(images).cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.numpy())

    train_features_np = np.concatenate(train_features, axis=0)
    train_labels_np = np.concatenate(train_labels, axis=0)
    val_features_np = np.concatenate(val_features, axis=0)
    val_labels_np = np.concatenate(val_labels, axis=0)

    model_info = {
        "compressed_dim": base_model.compressed_dim,
        "num_classes": base_model.num_classes,
        "classes": list(classes),
        "encoder_type": getattr(base_model, "encoder_type", "vit"),
        "pretrained": getattr(base_model, "pretrained", True),
        "quantum_enabled": getattr(base_model, "quantum_enabled", False),
        "quantum_qubits": getattr(base_model, "quantum_qubits", None),
        "quantum_shots": getattr(base_model, "quantum_shots", None),
    }

    torch.save(base_model.state_dict(), output_dir / "phase1_classical_model.pth")
    torch.save(
        {
            "model": base_model.state_dict(),
            "training_summary": training_summary,
            "model_info": model_info,
        },
        output_dir / "complete_checkpoint.pth",
    )

    (output_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))
    (output_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2))

    quantum_data = {
        "train_features": train_features_np,
        "train_labels": train_labels_np,
        "val_features": val_features_np,
        "val_labels": val_labels_np,
    }
    with (output_dir / "quantum_training_data.pkl").open("wb") as handle:
        import pickle

        pickle.dump(quantum_data, handle)

    return quantum_data


def train_hybrid_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: Iterable[str],
    config: Optional[TrainingConfig] = None,
    output_dir: Path | str = "trained_model",
    encoder_type: str = "vit",
    pretrained: bool = True,
    quantum_enabled: bool = True,
    quantum_qubits: int = 4,
    quantum_shots: int = 1024,
    init_checkpoint: Path | str | None = None,
    init_strict: bool = False,
    restore_history: bool = False,
) -> tuple[ClassicalDRModel, Dict[str, Any]]:
    config = config or TrainingConfig()
    config.ensure_stages()
    if config.loss is None:
        config.loss = LossConfig()
    loss_name = getattr(config.loss, "type", "cross_entropy")
    loss_name_lower = str(loss_name).lower()
    LOGGER.info("Configured loss objective: %s", loss_name)
    if loss_name_lower == "balanced":
        loss_name_lower = "class_balanced"
    if loss_name_lower in {"class_balanced", "focal_class_balanced"}:
        LOGGER.info(
            "Class-balanced weighting beta=%.6f",
            float(getattr(config.loss, "class_balanced_beta", 0.999)),
        )
        if not config.class_counts:
            LOGGER.warning(
                "%s loss enabled but class_counts are missing; class-balanced weighting will fall back to base weights",
                loss_name,
            )
    if loss_name_lower == "focal_class_balanced":
        LOGGER.info(
            "Focal modulation gamma=%.3f",
            float(getattr(config.loss, "focal_gamma", 2.0)),
        )
    classes = list(classes)

    sampler = getattr(train_loader, "sampler", None)
    if isinstance(sampler, WeightedRandomSampler):
        LOGGER.info(
            "Training sampler: WeightedRandomSampler (replacement=%s, num_samples=%s)",
            getattr(sampler, "replacement", "?"),
            getattr(sampler, "num_samples", "?"),
        )

    if config.class_weights is None or config.class_counts is None:
        train_dataset = getattr(train_loader, "dataset", None)
        dataset_weights = getattr(train_dataset, "class_weights", None) if train_dataset is not None else None
        dataset_counts = getattr(train_dataset, "class_counts", None) if train_dataset is not None else None
        if config.class_weights is None and dataset_weights is not None:
            weights_array = np.asarray(dataset_weights, dtype=np.float64)
            config.class_weights = weights_array.tolist()
        if config.class_counts is None and dataset_counts is not None:
            counts_array = np.asarray(dataset_counts, dtype=np.int64)
            config.class_counts = counts_array.tolist()

    if config.class_weights:
        LOGGER.info(
            "Using class weights: %s",
            ", ".join(f"{float(weight):.3f}" for weight in config.class_weights),
        )
    if config.class_counts:
        LOGGER.info(
            "Training class counts: %s",
            ", ".join(str(int(count)) for count in config.class_counts),
        )

    model = ClassicalDRModel(
        num_classes=len(classes),
        encoder_type=encoder_type,
        pretrained=pretrained,
        quantum_enabled=quantum_enabled,
        quantum_qubits=quantum_qubits,
        quantum_shots=quantum_shots,
    )

    trainer = Phase1Trainer(model, config, classes)
    training_summary = trainer.train(train_loader, val_loader)

    if init_checkpoint:
        checkpoint_report = trainer.load_initial_state(init_checkpoint, strict=init_strict, restore_history=restore_history)
        stage_name = checkpoint_report.get("stage") or "unknown"
        epoch_index = checkpoint_report.get("epoch")
        LOGGER.info(
            "Initialized weights from %s (stage=%s, epoch=%s)",
            checkpoint_report.get("source"),
            stage_name,
            epoch_index if epoch_index is not None else "unknown",
        )
        history_entries = checkpoint_report.get("history_entries") or 0
        if history_entries:
            LOGGER.info("Restored %d prior history entries", history_entries)
        missing_keys = checkpoint_report.get("missing_keys") or []
        unexpected_keys = checkpoint_report.get("unexpected_keys") or []
        if missing_keys:
            formatted = ", ".join(sorted(str(key) for key in missing_keys))
            LOGGER.warning("Missing keys when loading checkpoint: %s", formatted)
        if unexpected_keys:
            formatted = ", ".join(sorted(str(key) for key in unexpected_keys))
            LOGGER.warning("Unexpected keys when loading checkpoint: %s", formatted)

    final_model = _unwrap_model(trainer.model)

    save_model_and_features(
        final_model,
        train_loader,
        val_loader,
        classes,
        config.device,
        training_summary=training_summary,
        output_dir=output_dir,
    )

    return final_model, training_summary


__all__ = [
    "StageConfig",
    "WandBConfig",
    "LossConfig",
    "TrainingConfig",
    "default_stage_schedule",
    "Phase1Trainer",
    "train_hybrid_model",
    "save_model_and_features",
]
