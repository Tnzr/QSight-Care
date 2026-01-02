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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

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
    adaptive_patience: int = 2
    adaptive_threshold: float = 5e-4
    lr_decay_factor: float = 0.5
    min_lr: float = 1e-6
    annealing_schedule: Optional[Dict[str, Any]] = None


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

    def ensure_stages(self) -> None:
        if self.stages:
            return
        self.stages = default_stage_schedule()


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
        self.classes = list(classes)
        self.class_weight_tensor: Optional[torch.Tensor] = None
        if self.config.class_weights and len(self.config.class_weights) == len(self.classes):
            self.class_weight_tensor = torch.tensor(self.config.class_weights, dtype=torch.float32)
        elif self.config.class_weights:
            LOGGER.warning(
                "Ignoring configured class_weights: expected %d values but received %d",
                len(self.classes),
                len(self.config.class_weights),
            )
        if self.class_weight_tensor is not None:
            self.class_weight_tensor = self.class_weight_tensor.to(self.device)
        self.class_counts = list(self.config.class_counts) if self.config.class_counts else None
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
        self.quant_manager: Optional[QuantizationManager] = None
        if self.config.quantization and self.config.quantization.enabled:
            self.quant_manager = QuantizationManager(self.config.quantization)
        if self.config.enable_latency_tracking and hasattr(self.base_model, "enable_latency_tracking"):
            self.base_model.enable_latency_tracking(True)

    def _ensure_interpretability_samples(self, val_loader: DataLoader) -> None:
        if not self.gradcam_helper.enabled or self._interpretability_samples is not None:
            return
        collected_images: List[torch.Tensor] = []
        collected_labels: List[int] = []
        collected_paths: List[str] = []
        seen_classes: set[int] = set()
        max_samples = self.gradcam_helper.max_samples
        try:
            for images, labels, paths in val_loader:
                batch_size = images.shape[0]
                for idx in range(batch_size):
                    label_val = int(labels[idx].item()) if labels is not None else -1
                    if label_val not in seen_classes or len(collected_images) < max_samples:
                        collected_images.append(images[idx].cpu())
                        collected_labels.append(label_val)
                        collected_paths.append(str(paths[idx]) if paths else "")
                        if label_val >= 0:
                            seen_classes.add(label_val)
                    if len(collected_images) >= max_samples and len(seen_classes) >= len(self.classes):
                        break
                if len(collected_images) >= max_samples:
                    break
        except StopIteration:
            pass

        if not collected_images:
            return

        stacked_images = torch.stack(collected_images[:max_samples])
        stacked_labels = torch.tensor(collected_labels[:max_samples]) if collected_labels else None
        stacked_paths = collected_paths[:max_samples]
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

    def _log_to_wandb(self, metrics: Dict[str, Any], stage_name: str, epoch: int) -> None:
        if not self.wandb_run:
            return
        metrics = {f"{stage_name}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        metrics["global_step"] = self.global_step
        try:
            import wandb

            wandb.log(metrics, step=self.global_step)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to log to W&B: %s", exc)

    def _log_epoch_visualizations(self, stage: StageConfig, epoch: int, val_outcome: Dict[str, Any]) -> None:
        if not self.wandb_run:
            return
        preds = val_outcome.get("preds") or []
        targets = val_outcome.get("targets") or []
        if not preds or not targets:
            return
        mask_tensor = self._stage_mask(stage)

        try:
            import wandb

            confusion_payload = val_outcome.get("confusion_matrix")
            if confusion_payload:
                try:
                    import matplotlib.pyplot as plt

                    cm_array = np.array(confusion_payload, dtype=np.int64)
                    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=self.classes)
                    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
                    ax.tick_params(axis="x", rotation=45)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{stage.name.title()} Confusion Matrix (Epoch {epoch})")
                    ax.grid(False)
                    wandb.log({f"{stage.name}/confusion_matrix": wandb.Image(fig)}, step=self.global_step)
                    plt.close(fig)

                    cm_table = wandb.Table(columns=["Actual", "Predicted", "Count"])
                    for actual_idx, actual_label in enumerate(self.classes):
                        for pred_idx, pred_label in enumerate(self.classes):
                            cm_table.add_data(actual_label, pred_label, int(cm_array[actual_idx, pred_idx]))
                    wandb.log({f"{stage.name}/confusion_matrix_table": cm_table}, step=self.global_step)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to render confusion matrix heatmap: %s", exc)

            per_head_accuracy = val_outcome.get("per_head_class_accuracy")
            normalized_accuracy: Dict[str, np.ndarray] = {}
            if per_head_accuracy:
                tracked_heads = self._resolve_tracked_heads(mask_tensor)
                head_display_map = {head: self._head_display_name(head) for head in tracked_heads}
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
                    ax.set_title(f"{stage.name.title()} Head Accuracy (Epoch {epoch})")
                    ax.legend()
                    acc_path = epoch_dir / "head_accuracy.png"
                    fig.savefig(acc_path, dpi=200)
                    wandb.log({f"{stage.name}/head_accuracy_bar": wandb.Image(fig)}, step=self.global_step)
                    plt.close(fig)

                    summary_table = wandb.Table(columns=["class"] + head_names)
                    for class_idx, class_label in enumerate(class_labels):
                        row = [class_label]
                        for head_name in head_names:
                            row.append(float(normalized_accuracy[head_name][class_idx]))
                        summary_table.add_data(*row)
                    wandb.log({f"{stage.name}/head_accuracy_table": summary_table}, step=self.global_step)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to create head accuracy bar chart: %s", exc)

                stage_history = self._confidence_history[stage.name]
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
                            title=f"{stage.name.title()} {class_label} Head Accuracy",
                            xname="Epoch",
                        )
                        wandb.log({f"{stage.name}/head_accuracy_trend/{class_label}": line_plot}, step=self.global_step)
                        trend_table = wandb.Table(columns=["epoch"] + keys)
                        for x_idx, epoch_value in enumerate(xs):
                            row = [epoch_value]
                            for series_idx, key in enumerate(keys):
                                value = ys[series_idx][x_idx]
                                row.append(None if math.isnan(value) else float(value))
                            trend_table.add_data(*row)
                        wandb.log({f"{stage.name}/head_accuracy_trend/{class_label}_table": trend_table}, step=self.global_step)

            if self.gradcam_helper.enabled and self._interpretability_samples:
                try:
                    cam_payloads = self.gradcam_helper.generate_overlays(
                        images=self._interpretability_samples["images"],
                        labels=self._interpretability_samples["labels"],
                        paths=self._interpretability_samples.get("paths"),
                        class_names=self.classes,
                    )
                    if cam_payloads:
                        try:
                            import matplotlib.pyplot as plt

                            max_columns = min(len(self.classes), 5)
                            collage_payloads: List[Dict[str, Any]] = []
                            seen_targets: set[int] = set()
                            for payload in cam_payloads:
                                target_idx = int(payload.get("target_idx", -1))
                                if target_idx in seen_targets:
                                    continue
                                collage_payloads.append(payload)
                                seen_targets.add(target_idx)
                                if len(collage_payloads) >= max_columns:
                                    break
                            if collage_payloads:
                                num_cols = len(collage_payloads)
                                fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 6), constrained_layout=True)
                                axes = np.array(axes).reshape(2, num_cols)
                                for col, payload in enumerate(collage_payloads):
                                    base_img = np.clip(payload["base"], 0.0, 1.0)
                                    heatmap = np.clip(payload["heatmap"], 0.0, 1.0)
                                    axes[0, col].imshow(base_img)
                                    axes[0, col].axis("off")
                                    axes[0, col].set_title(payload.get("target_label", ""), fontsize=12)
                                    axes[1, col].imshow(base_img)
                                    axes[1, col].imshow(heatmap, cmap="inferno", alpha=0.6)
                                    axes[1, col].axis("off")
                                    axes[1, col].set_title(
                                        f"Pred: {payload.get('pred_label', '')} ({payload.get('confidence', 0.0):.2f})",
                                        fontsize=11,
                                    )
                                gradcam_dir = epoch_dir / "gradcam"
                                gradcam_dir.mkdir(parents=True, exist_ok=True)
                                collage_path = gradcam_dir / "collage.png"
                                fig.savefig(collage_path, dpi=200)
                                wandb.log({f"{stage.name}/gradcam_collage": wandb.Image(fig)}, step=self.global_step)
                                plt.close(fig)
                        except Exception as collage_exc:  # pragma: no cover - defensive
                            LOGGER.warning("Failed to construct Grad-CAM collage: %s", collage_exc)

                        overlay_images = []
                        original_images = []
                        gradcam_dir = epoch_dir / "gradcam"
                        gradcam_dir.mkdir(parents=True, exist_ok=True)
                        for item_idx, payload in enumerate(cam_payloads):
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
                                overlay_path = gradcam_dir / f"overlay_{item_idx:02d}.png"
                                base_path = gradcam_dir / f"original_{item_idx:02d}.png"
                                Image.fromarray(overlay_uint8).save(overlay_path)
                                Image.fromarray(base_uint8).save(base_path)
                            except Exception as img_exc:  # pragma: no cover - defensive
                                LOGGER.debug("Failed to persist Grad-CAM images: %s", img_exc)
                        log_payload: Dict[str, Any] = {f"{stage.name}/gradcam_overlays": overlay_images}
                        if original_images:
                            log_payload[f"{stage.name}/gradcam_originals"] = original_images
                        wandb.log(log_payload, step=self.global_step)
                except Exception as cam_exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to log Grad-CAM overlays: %s", cam_exc)
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
        metrics = {
            "acc_full_res": compute_accuracy(outputs.output_a, targets),
            "acc_compressed": compute_accuracy(outputs.output_b, targets),
            "acc_quantum": compute_accuracy(outputs.output_c, targets),
            "acc_ensemble": compute_accuracy(outputs.final_output, targets),
        }
        metrics["acc_a"] = metrics["acc_full_res"]
        metrics["acc_b"] = metrics["acc_compressed"]
        metrics["acc_c"] = metrics["acc_quantum"]
        return metrics

    def _aggregate_latencies(self, latency_accumulator: Dict[str, List[float]]) -> Dict[str, float]:
        return {key: float(statistics.mean(values)) for key, values in latency_accumulator.items() if values}

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
        is_train = train and optimizer is not None
        model.train(is_train)
        epoch_loss = 0.0
        metrics_accumulator = defaultdict(float)
        latency_accumulator: Dict[str, List[float]] = defaultdict(list)
        ensemble_weights: List[np.ndarray] = []
        total_batches = 0
        preds: List[int] = []
        targets_all: List[int] = []

        if self.device.type == "cuda" and torch.cuda.is_available() and is_train and self.config.mixed_precision:
            autocast_ctx = torch.amp.autocast("cuda", enabled=True)
        else:
            autocast_ctx = contextlib.nullcontext()

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

        iterator = tqdm(
            loader,
            desc=f"{stage.name.capitalize()} | Epoch {epoch_idx + 1} {'Train' if is_train else 'Val'}",
            dynamic_ncols=True,
        )
        for batch_idx, (images, targets, _) in enumerate(iterator):
            images = images.to(self.device)
            targets = targets.to(self.device)

            with autocast_ctx:
                raw_outputs = model(
                    images,
                    return_all=True,
                    track_latency=self.config.enable_latency_tracking,
                )
                outputs = self._normalize_output(raw_outputs)
                losses = self.base_model.compute_losses(
                    outputs,
                    targets,
                    loss_weights=loss_weights,
                    class_weights=self.class_weight_tensor,
                )
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
                    accum_counter = 0

            epoch_loss += float(loss.item())
            batch_metrics = self._compute_metrics(outputs, targets)
            for key, value in batch_metrics.items():
                metrics_accumulator[key] += value

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
                    batch_log.update({f"batch/{k}": float(v) for k, v in batch_metrics.items()})
                    self._log_to_wandb(batch_log, stage.name, epoch_idx + 1)

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

        return result

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        stage_summaries: List[Dict[str, Any]] = []
        self._ensure_interpretability_samples(val_loader)
        for stage_idx, stage in enumerate(self.config.stages):
            LOGGER.info("Starting stage %s (%d/%d) for %d epochs", stage.name, stage_idx + 1, len(self.config.stages), stage.num_epochs)
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
                    train_loader,
                    optimizer,
                    stage,
                    epoch,
                    train=True,
                    mask_tensor=mask_tensor,
                    loss_weights=loss_weights,
                    model=stage_model,
                )
                val_outcome = self._run_epoch(
                    val_loader,
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
                self._log_to_wandb(wandb_metrics, stage.name, epoch + 1)
                self._log_epoch_visualizations(stage, epoch + 1, val_outcome)

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

            summary = self._build_stage_summary(stage, val_loader, mask_tensor, loss_weights, stage_model)
            stage_summaries.append(summary)

        if self.best_state_weights:
            _unwrap_model(self.model).load_state_dict(self.best_state_weights)

        training_summary = {
            "history": self.history,
            "stage_summaries": stage_summaries,
            "best_state": self.best_state_meta,
            "classes": self.classes,
        }
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
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                raw_outputs = model(
                    images,
                    return_all=True,
                    track_latency=self.config.enable_latency_tracking,
                )
                outputs = self._normalize_output(raw_outputs)
                _ = self.base_model.compute_losses(
                    outputs,
                    labels,
                    loss_weights=loss_weights,
                    class_weights=self.class_weight_tensor,
                )
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
) -> tuple[ClassicalDRModel, Dict[str, Any]]:
    config = config or TrainingConfig()
    config.ensure_stages()
    classes = list(classes)

    sampler = getattr(train_loader, "sampler", None)
    if isinstance(sampler, WeightedRandomSampler):
        LOGGER.info("Training sampler: WeightedRandomSampler (replacement=True) enabled for class balancing")

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
    "TrainingConfig",
    "default_stage_schedule",
    "Phase1Trainer",
    "train_hybrid_model",
    "save_model_and_features",
]
