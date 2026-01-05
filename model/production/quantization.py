"""Utilities for aligning the hybrid model with post-training quantization and exports."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)

__all__ = ["QuantizationConfig", "QuantizationManager"]


@dataclass
class QuantizationConfig:
    """Configuration for post-training quantization pipelines."""

    enabled: bool = False
    approach: str = "ptq_dynamic"  # "ptq_dynamic", "ptq_static", or "qat"
    backend: str = "qnnpack"
    modules: Iterable[str] = ("compression", "classical_head_a", "classical_head_b", "ensemble")
    calibration_batches: int = 16
    prepare_on_stage_end: Optional[str] = None
    calibrate_on_stage_end: Optional[str] = None
    convert_on_stage_end: Optional[str] = None
    export_onnx_path: Optional[str] = None
    onnx_opset: int = 17
    onnx_dynamic_axes: bool = True
    onnx_input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)


class QuantizationManager:
    """Handles dynamic/static quantization and optional ONNX export."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        self.quantized_model: Optional[nn.Module] = None
        self._prepared = False
        self._calibrated = False
        self._converted = False

    def _set_backend(self) -> None:
        try:
            torch.backends.quantized.engine = self.config.backend
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to set quantized backend %s: %s", self.config.backend, exc)

    def _clone_model(self, model: nn.Module) -> nn.Module:
        return copy.deepcopy(model).cpu()

    def _modules_to_quantize(self, model: nn.Module) -> Dict[str, nn.Module]:
        result: Dict[str, nn.Module] = {}
        for name in self.config.modules:
            module = getattr(model, name, None)
            if module is not None:
                result[name] = module
            else:
                LOGGER.debug("Quantization module '%s' not found on model", name)
        return result

    def run_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        self._set_backend()
        model.eval()
        modules = {nn.Linear}
        quantized_model = torch.ao.quantization.quantize_dynamic(
            self._clone_model(model),
            {m for m in modules},
            dtype=torch.qint8,
        )
        self.quantized_model = quantized_model
        self._converted = True
        LOGGER.info("Dynamic PTQ complete: quantized %d linear modules", sum(1 for _ in quantized_model.modules() if isinstance(_, nn.Linear)))
        return quantized_model

    def _prepare_static(self, model: nn.Module) -> nn.Module:
        self._set_backend()
        model.eval()
        prepared = self._clone_model(model)
        prepared.eval()
        qconfig = torch.ao.quantization.get_default_qconfig(self.config.backend)
        prepared.qconfig = None
        for module in self._modules_to_quantize(prepared).values():
            module.qconfig = qconfig
        torch.ao.quantization.prepare(prepared, inplace=True)
        self._prepared = True
        LOGGER.info("Static PTQ prepare complete using backend %s", self.config.backend)
        return prepared

    def _calibrate_static(self, prepared: nn.Module, calibration_loader: Optional[Iterable]) -> None:
        if calibration_loader is None:
            LOGGER.warning("Static PTQ requested without calibration data; skipping calibration")
            return
        prepared.eval()
        batches_ran = 0
        with torch.inference_mode():
            for images, *_ in calibration_loader:
                if not isinstance(images, torch.Tensor):
                    continue
                batch = images.detach().to(dtype=torch.float32, device="cpu")
                prepared(batch)
                batches_ran += 1
                if batches_ran >= self.config.calibration_batches:
                    break
        self._calibrated = True
        LOGGER.info("Static PTQ calibration complete over %d batches", batches_ran)

    def _convert_static(self, prepared: nn.Module) -> nn.Module:
        quantized = torch.ao.quantization.convert(prepared, inplace=True)
        self.quantized_model = quantized
        self._converted = True
        LOGGER.info("Static PTQ conversion complete")
        return quantized

    def _export_onnx(self, model: nn.Module) -> Optional[Path]:
        export_path = self.config.export_onnx_path
        if not export_path:
            return None
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_cpu = model.cpu()
        model_cpu.eval()
        dummy_input = torch.randn(*self.config.onnx_input_shape)
        dynamic_axes = None
        if self.config.onnx_dynamic_axes:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(path),
            opset_version=self.config.onnx_opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        LOGGER.info("Exported ONNX model to %s (opset %d)", path, self.config.onnx_opset)
        return path

    def generate_artifacts(
        self,
        model: nn.Module,
        calibration_loader: Optional[Iterable] = None,
    ) -> Dict[str, Optional[str]]:
        if not self.config.enabled:
            return {}

        approach = self.config.approach.lower()
        if approach == "ptq_dynamic":
            quantized = self.run_dynamic_quantization(model)
        elif approach == "ptq_static":
            prepared = self._prepare_static(model)
            self._calibrate_static(prepared, calibration_loader)
            quantized = self._convert_static(prepared)
        else:
            raise NotImplementedError(
                "Only post-training dynamic or static quantization is implemented in QuantizationManager"
            )

        artifact_path = self._export_onnx(quantized)
        return {
            "approach": approach,
            "onnx_path": str(artifact_path) if artifact_path else None,
            "quantized_model_available": bool(self.quantized_model),
        }
