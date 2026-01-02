"""Command-line entry point for hybrid classical + quantum training."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

try:  # pragma: no cover - allow script execution outside package context
    from .datasets import DatasetConfig, create_dataloaders
    from .models import AER_IMPORT_ERROR, QISKIT_AER_AVAILABLE, QISKIT_AVAILABLE
    from .training import (
        StageConfig,
        TrainingConfig,
        WandBConfig,
        default_stage_schedule,
        train_hybrid_model,
    )
    from .quantization import QuantizationConfig
except ImportError:
    from datasets import DatasetConfig, create_dataloaders  # type: ignore
    from models import AER_IMPORT_ERROR, QISKIT_AER_AVAILABLE, QISKIT_AVAILABLE  # type: ignore
    from training import (  # type: ignore
        StageConfig,
        TrainingConfig,
        WandBConfig,
        default_stage_schedule,
        train_hybrid_model,
    )
    from quantization import QuantizationConfig  # type: ignore

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

LOGGER = logging.getLogger(__name__)


def _import_module(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None, None
    version = getattr(module, "__version__", None)
    return module, version


def log_environment_diagnostics(
    args: argparse.Namespace,
    device: str,
    use_mixed_precision: bool,
    quantization_config: Optional[QuantizationConfig],
) -> None:
    LOGGER.info("=== Environment Diagnostics ===")
    LOGGER.info("Python %s (%s)", sys.version.split()[0], platform.platform())
    LOGGER.info(
        "PyTorch %s | CUDA available: %s | CUDA runtime: %s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.version.cuda,
    )
    cudnn_available = torch.backends.cudnn.is_available()
    LOGGER.info(
        "cuDNN available: %s (version %s)",
        cudnn_available,
        torch.backends.cudnn.version() if cudnn_available else "n/a",
    )
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        LOGGER.info("Detected %d CUDA device(s)", device_count)
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            LOGGER.info(
                "GPU %d: %s | compute capability %d.%d | %.2f GB",
                idx,
                props.name,
                props.major,
                props.minor,
                props.total_memory / (1024 ** 3),
            )
        LOGGER.info("Selected device argument: %s", device)
        device_ids = args.device_ids if args.device_ids else None
        LOGGER.info(
            "Requested device IDs: %s",
            ", ".join(str(i) for i in device_ids) if device_ids else "default",
        )
    else:
        LOGGER.info("CUDA unavailable, training will run on %s", device)

    mixed_precision_source = "auto" if args.mixed_precision is None else "cli"
    LOGGER.info("Mixed precision (%s): %s", mixed_precision_source, use_mixed_precision)
    LOGGER.info(
        "AMP bf16 support: %s",
        torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )
    LOGGER.info("AMP fp16 support: %s", torch.cuda.is_available())
    LOGGER.info("DataParallel requested: %s", bool(args.multi_gpu))
    LOGGER.info("Gradient clipping: %s", args.grad_clip if args.grad_clip else "disabled")
    LOGGER.info("Log interval (batches): %s", args.log_interval)

    LOGGER.info(
        "W&B logging enabled: %s (mode=%s)",
        bool(args.use_wandb),
        getattr(args, "wandb_mode", "n/a"),
    )

    LOGGER.info("Quantization backend engine: %s", torch.backends.quantized.engine)
    if quantization_config and quantization_config.enabled:
        from dataclasses import asdict

        LOGGER.info("Quantization configuration: %s", asdict(quantization_config))
    else:
        LOGGER.info("Quantization configuration: disabled")

    LOGGER.info("Qiskit Machine Learning available: %s", QISKIT_AVAILABLE)
    qiskit_ml_mod, qiskit_ml_version = _import_module("qiskit_machine_learning")
    if qiskit_ml_mod:
        LOGGER.info(" - qiskit_machine_learning version: %s", qiskit_ml_version or "unknown")
    qiskit_mod, qiskit_version = _import_module("qiskit")
    if qiskit_mod:
        LOGGER.info(" - qiskit version: %s", qiskit_version or "unknown")

    LOGGER.info("Qiskit Aer GPU available: %s", QISKIT_AER_AVAILABLE)
    if not QISKIT_AER_AVAILABLE and AER_IMPORT_ERROR:
        LOGGER.info(" - Aer import diagnostic: %s", AER_IMPORT_ERROR)
    if QISKIT_AER_AVAILABLE:
        try:
            from qiskit_aer import AerSimulator

            backend = AerSimulator()
            available_devices = tuple(
                getattr(backend, "available_devices", lambda: ("CPU",))()
            )
            if available_devices:
                LOGGER.info(
                    " - AerSimulator reported devices: %s",
                    ", ".join(available_devices),
                )
            device_option = getattr(getattr(backend, "options", None), "device", None)
            method_option = getattr(getattr(backend, "options", None), "method", None)
            LOGGER.info(" - AerSimulator default device option: %s", device_option or "unspecified")
            if method_option:
                LOGGER.info(" - AerSimulator default method option: %s", method_option)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(" - Failed to instantiate AerSimulator: %s", exc)

    onnx_mod, onnx_version = _import_module("onnx")
    LOGGER.info("ONNX available: %s", bool(onnx_mod))
    if onnx_mod:
        LOGGER.info(" - onnx version: %s", onnx_version or "unknown")

    ort_mod, ort_version = _import_module("onnxruntime")
    LOGGER.info("ONNX Runtime available: %s", bool(ort_mod))
    if ort_mod:
        LOGGER.info(" - onnxruntime version: %s", ort_version or "unknown")

    trt_mod, trt_version = _import_module("tensorrt")
    LOGGER.info("TensorRT available: %s", bool(trt_mod))
    if trt_mod:
        LOGGER.info(" - tensorrt version: %s", trt_version or "unknown")

    LOGGER.info("================================")


def _load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    expanded_path = config_path.expanduser()
    if not expanded_path.exists():
        raise FileNotFoundError(f"Config file not found: {expanded_path}")
    try:
        return json.loads(expanded_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
        raise ValueError(f"Failed to parse config file {expanded_path}: {exc}") from exc


def _config_path(config: dict, key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    return Path(value).expanduser()


def _build_stage_configs(raw_stages: List[dict]) -> List[StageConfig]:
    stages: List[StageConfig] = []
    for idx, entry in enumerate(raw_stages):
        if not isinstance(entry, dict):
            raise ValueError(f"Stage definition at index {idx} must be a JSON object")
        name = entry.get("name") or f"stage_{idx + 1}"
        num_epochs = int(entry.get("num_epochs", 1))
        learning_rate = float(entry.get("learning_rate", 1e-4))
        weight_decay = float(entry.get("weight_decay", 1e-5))
        patience = int(entry.get("patience", 3))
        active_mask = entry.get("active_mask")
        loss_weights = entry.get("loss_weights")
        grad_accum_steps = int(entry.get("grad_accum_steps", 1))
        force_data_parallel = entry.get("force_data_parallel")
        adaptive_lr = entry.get("adaptive_lr", True)
        adaptive_patience = int(entry.get("adaptive_patience", 2))
        adaptive_threshold = float(entry.get("adaptive_threshold", 5e-4))
        lr_decay_factor = float(entry.get("lr_decay_factor", 0.5))
        min_lr = float(entry.get("min_lr", 1e-6))
        annealing_schedule = entry.get("annealing_schedule")
        stages.append(
            StageConfig(
                name=name,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                active_mask=active_mask,
                loss_weights=loss_weights,
                grad_accum_steps=grad_accum_steps,
                force_data_parallel=force_data_parallel,
                adaptive_lr=bool(adaptive_lr),
                adaptive_patience=adaptive_patience,
                adaptive_threshold=adaptive_threshold,
                lr_decay_factor=lr_decay_factor,
                min_lr=min_lr,
                annealing_schedule=annealing_schedule,
            )
        )
    return stages


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to JSON file supplying default arguments (not committed)",
    )
    config_args, remaining = base_parser.parse_known_args()
    config_data = _load_config(config_args.config_file)

    dataset_root_default = _config_path(config_data, "dataset_root")
    output_dir_default = _config_path(config_data, "output_dir") or Path("trained_model")
    stages_config_default = _config_path(config_data, "stages_config")
    checkpoint_dir_default = _config_path(config_data, "checkpoint_dir")
    pretrained_default = config_data.get("pretrained")
    if pretrained_default is None:
        pretrained_default = not bool(config_data.get("no_pretrained", False))
    if pretrained_default is None:
        pretrained_default = True

    quantum_default = config_data.get("quantum_enabled")
    if quantum_default is None:
        quantum_default = config_data.get("quantum")
    if quantum_default is None:
        quantum_default = not bool(config_data.get("no_quantum", False))
    if quantum_default is None:
        quantum_default = True

    mixed_precision_default = config_data.get("mixed_precision")
    if mixed_precision_default is None and "mixed_precision_enabled" in config_data:
        mixed_precision_default = bool(config_data["mixed_precision_enabled"])

    use_wandb_default = config_data.get("use_wandb")
    if use_wandb_default is None:
        use_wandb_default = False

    latency_tracking_default = config_data.get("latency_tracking")
    if latency_tracking_default is None and "enable_latency_tracking" in config_data:
        latency_tracking_default = bool(config_data["enable_latency_tracking"])
    if latency_tracking_default is None and "disable_latency" in config_data:
        latency_tracking_default = not bool(config_data["disable_latency"])
    if latency_tracking_default is None:
        latency_tracking_default = True

    multi_gpu_default = config_data.get("multi_gpu")
    if multi_gpu_default is None:
        multi_gpu_default = config_data.get("data_parallel", False)
    multi_gpu_default = bool(multi_gpu_default)

    quant_defaults = config_data.get("quantization")
    if quant_defaults is not None and not isinstance(quant_defaults, dict):
        raise ValueError("Config field 'quantization' must be a JSON object when provided")
    if quant_defaults is None:
        quant_defaults = {}

    config_stage_entries = config_data.get("stages")
    if config_stage_entries is not None and not isinstance(config_stage_entries, list):
        raise ValueError("Config field 'stages' must be a list of stage definitions")

    parser = argparse.ArgumentParser(
        description="Train hybrid DR model (classical + quantum)",
        parents=[base_parser],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root_default,
        required=dataset_root_default is None,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=config_data.get("dataset_type", "tanlikesmath"),
        choices=["tanlikesmath", "sovitrath"],
        help="Dataset variant",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir_default)
    parser.add_argument("--batch-size", type=int, default=config_data.get("batch_size", 32))
    parser.add_argument("--train-ratio", type=float, default=config_data.get("train_ratio", 0.8))
    parser.add_argument("--num-workers", type=int, default=config_data.get("num_workers", 2))
    parser.add_argument("--seed", type=int, default=config_data.get("seed", 42))
    parser.add_argument(
        "--weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("use_weighted_sampler", False),
        help="Enable weighted sampling to counter class imbalance",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=pretrained_default,
        help="Enable or disable pretrained weights",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default=config_data.get("encoder_type", "vit"),
        choices=["vit", "resnet"],
        help="Backbone encoder",
    )
    parser.add_argument("--device", type=str, default=config_data.get("device"), help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--grad-clip", type=float, default=config_data.get("grad_clip", 1.0), help="Gradient clipping value (0 to disable)")
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=mixed_precision_default,
        help="Enable or disable autocast mixed precision (default: auto)",
    )
    parser.add_argument(
        "--latency-tracking",
        action=argparse.BooleanOptionalAction,
        default=latency_tracking_default,
        dest="latency_tracking",
        help="Enable or disable latency tracking",
    )
    parser.add_argument(
        "--disable-latency",
        action="store_false",
        dest="latency_tracking",
        help="Disable latency tracking (alias)",
    )
    parser.add_argument("--log-interval", type=int, default=config_data.get("log_interval", 25), help="Training log interval (batches)")
    parser.add_argument(
        "--stages-config",
        type=Path,
        default=stages_config_default,
        help="Optional JSON file describing training stages",
    )
    parser.add_argument("--quantum-qubits", type=int, default=config_data.get("quantum_qubits", 4), help="Number of qubits for quantum head")
    parser.add_argument("--quantum-shots", type=int, default=config_data.get("quantum_shots", 1024), help="Shots for quantum sampler")
    parser.add_argument(
        "--quantum",
        action=argparse.BooleanOptionalAction,
        default=quantum_default,
        help="Enable or disable the quantum head",
    )
    quant_export_default = quant_defaults.get("export_onnx_path")
    if quant_export_default is not None:
        quant_export_default = Path(quant_export_default).expanduser()
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=quant_defaults.get("enabled", config_data.get("quantize", False)),
        help="Enable post-training quantization artifact generation",
    )
    parser.add_argument(
        "--quantization-approach",
        type=str,
        choices=["ptq_dynamic", "ptq_static"],
        default=quant_defaults.get("approach", config_data.get("quantization_approach", "ptq_dynamic")),
        help="Post-training quantization approach",
    )
    parser.add_argument(
        "--quantization-backend",
        type=str,
        default=quant_defaults.get("backend", config_data.get("quantization_backend", "qnnpack")),
        help="Quantized backend engine (e.g. qnnpack, fbgemm)",
    )
    parser.add_argument(
        "--quantization-calibration-batches",
        type=int,
        default=quant_defaults.get("calibration_batches", config_data.get("quantization_calibration_batches", 16)),
        help="Number of batches to use for static PTQ calibration",
    )
    parser.add_argument(
        "--quantization-export-onnx",
        type=Path,
        default=quant_export_default,
        help="Optional path to export the quantized model to ONNX",
    )
    parser.add_argument(
        "--quantization-onnx-opset",
        type=int,
        default=quant_defaults.get("onnx_opset", config_data.get("quantization_onnx_opset", 17)),
        help="ONNX opset version to use for quantized export",
    )
    parser.add_argument(
        "--quantization-onnx-dynamic",
        action=argparse.BooleanOptionalAction,
        default=quant_defaults.get("onnx_dynamic_axes", config_data.get("quantization_onnx_dynamic_axes", True)),
        help="Enable dynamic axes when exporting to ONNX",
    )
    parser.add_argument(
        "--quantization-onnx-shape",
        type=int,
        nargs=4,
        default=quant_defaults.get("onnx_input_shape"),
        metavar=("B", "C", "H", "W"),
        help="Input tensor shape for ONNX export (batch, channels, height, width)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=checkpoint_dir_default,
        help="Optional directory to store per-stage checkpoints",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=use_wandb_default,
        help="Enable or disable Weights & Biases logging",
    )
    parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=multi_gpu_default,
        help="Enable DataParallel training across multiple GPUs",
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated GPU indices to use with --multi-gpu (e.g. '0,1')",
    )
    parser.add_argument("--wandb-project", type=str, default=config_data.get("wandb_project", "qsight-care"), help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=config_data.get("wandb_entity"), help="W&B entity/team")
    parser.add_argument("--wandb-run-name", type=str, default=config_data.get("wandb_run_name"), help="Custom W&B run name")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=config_data.get("wandb_tags"),
        help="Optional list of W&B tags",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=config_data.get("wandb_mode", "online"),
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    args = parser.parse_args(remaining)
    for attr in ["dataset_root", "output_dir", "stages_config", "checkpoint_dir"]:
        value = getattr(args, attr, None)
        if isinstance(value, Path):
            setattr(args, attr, value.expanduser())

    if args.device_ids is None:
        config_device_ids = config_data.get("device_ids")
        if isinstance(config_device_ids, str):
            args.device_ids = config_device_ids
        elif isinstance(config_device_ids, (list, tuple)):
            args.device_ids = ",".join(str(idx) for idx in config_device_ids)
    if isinstance(args.device_ids, str) and args.device_ids:
        args.device_ids = [int(idx.strip()) for idx in args.device_ids.split(",") if idx.strip()]
    else:
        args.device_ids = None

    args.config_stages = config_stage_entries
    args.config_file = config_args.config_file
    args.quant_defaults = quant_defaults
    return args


def load_stages_from_file(path: Path) -> List[StageConfig]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Stage config file must contain a list, found {type(data)!r}")
    return _build_stage_configs(data)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_config = DatasetConfig(
        root=args.dataset_root,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        use_weighted_sampler=bool(args.weighted_sampler),
    )
    train_loader, val_loader, classes = create_dataloaders(dataset_config)
    if args.config_stages:
        stages = _build_stage_configs(args.config_stages)
    elif args.stages_config:
        stages = load_stages_from_file(args.stages_config)
    else:
        stages = default_stage_schedule()

    if args.mixed_precision is None:
        use_mixed_precision = torch.cuda.is_available()
    else:
        use_mixed_precision = args.mixed_precision

    quantum_enabled = True if args.quantum is None else bool(args.quantum)

    quantization_config = None
    quant_dict = dict(args.quant_defaults or {})
    quant_dict["enabled"] = bool(args.quantize)
    quant_dict["approach"] = args.quantization_approach
    quant_dict["backend"] = args.quantization_backend
    quant_dict["calibration_batches"] = max(1, int(args.quantization_calibration_batches))
    if args.quantization_export_onnx is not None:
        quant_dict["export_onnx_path"] = str(args.quantization_export_onnx.expanduser())
    quant_dict["onnx_opset"] = int(args.quantization_onnx_opset)
    quant_dict["onnx_dynamic_axes"] = bool(args.quantization_onnx_dynamic)
    if args.quantization_onnx_shape is not None:
        quant_dict["onnx_input_shape"] = tuple(int(value) for value in args.quantization_onnx_shape)
    if quant_dict.get("enabled"):
        quantization_config = QuantizationConfig(**quant_dict)

    wandb_config = None
    if args.use_wandb:
        wandb_config = WandBConfig(
            use_wandb=True,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config={
                "encoder_type": args.encoder_type,
                "batch_size": args.batch_size,
                "train_ratio": args.train_ratio,
                "quantum_enabled": quantum_enabled,
                "quantum_qubits": args.quantum_qubits,
                "quantum_shots": args.quantum_shots,
            },
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    training_config = TrainingConfig(
        stages=stages,
        device=device,
        grad_clip=max(0.0, args.grad_clip) or None,
        mixed_precision=use_mixed_precision,
        enable_latency_tracking=args.latency_tracking,
        log_interval=max(1, args.log_interval),
        checkpoint_dir=args.checkpoint_dir,
        wandb=wandb_config,
        data_parallel=args.multi_gpu,
        device_ids=args.device_ids,
        quantization=quantization_config,
    )

    log_environment_diagnostics(args, device, use_mixed_precision, quantization_config)

    train_hybrid_model(
        train_loader,
        val_loader,
        classes,
        config=training_config,
        output_dir=args.output_dir,
        encoder_type=args.encoder_type,
        pretrained=bool(args.pretrained),
        quantum_enabled=quantum_enabled,
        quantum_qubits=args.quantum_qubits,
        quantum_shots=args.quantum_shots,
    )


if __name__ == "__main__":
    main()
