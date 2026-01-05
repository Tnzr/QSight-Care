"""Backwards-compatible wrapper exposing production hybrid components."""

from model.production import (
    ClassicalDRModel,
    DatasetConfig,
    ModelForwardOutput,
    Phase1Trainer,
    StageConfig,
    TrainingConfig,
    WandBConfig,
    create_dataloaders,
    default_stage_schedule,
    load_model,
    predict_image,
    preprocess_image,
    save_model_and_features,
    train_hybrid_model,
)

__all__ = [
    "ClassicalDRModel",
    "DatasetConfig",
    "ModelForwardOutput",
    "Phase1Trainer",
    "StageConfig",
    "TrainingConfig",
    "WandBConfig",
    "create_dataloaders",
    "default_stage_schedule",
    "load_model",
    "predict_image",
    "preprocess_image",
    "save_model_and_features",
    "train_hybrid_model",
]
