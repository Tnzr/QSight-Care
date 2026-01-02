"""Dataset utilities for diabetic retinopathy classification."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np

LOGGER = logging.getLogger(__name__)

TANLIKESMATH_CLASS_NAMES: Tuple[str, ...] = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)


def _default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)],
                p=0.6,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.2),
            transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _default_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class DatasetConfig:
    root: Path
    dataset_type: str = "tanlikesmath"
    image_size: int = 224
    train_ratio: float = 0.8
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    transform: Optional[Callable] = None
    val_transform: Optional[Callable] = None
    use_weighted_sampler: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if self.transform is None:
            self.transform = _default_transform(self.image_size)
        if self.val_transform is None:
            self.val_transform = _default_val_transform(self.image_size)


class DiabeticRetinopathyDataset(Dataset):
    """Dataset supporting Sovitrath (preprocessed) and Tanlikesmath (resized) variants."""

    def __init__(
        self,
        config: DatasetConfig,
        mode: str = "train",
        transform: Optional[Callable] = None,
        image_paths: Optional[Sequence[Path]] = None,
        labels: Optional[Sequence[int]] = None,
        classes: Optional[Sequence[str]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.transform = transform or config.transform
        if image_paths is not None and labels is not None and classes is not None:
            self.image_paths = list(map(Path, image_paths))
            self.labels = list(labels)
            self.classes = list(classes)
            self.class_to_idx = class_to_idx or {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.image_paths: list[Path] = []
            self.labels: list[int] = []
            self.classes: Sequence[str] = []
            self.class_to_idx: dict[str, int] = {}
            loader = self._loaders().get(config.dataset_type.lower())
            if loader is None:
                raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
            loader()

        self._compute_class_stats()
        LOGGER.info("Loaded %d samples for %s (%s)", len(self.image_paths), config.dataset_type, mode)

    # Loader selection ---------------------------------------------------
    def _loaders(self) -> dict[str, Callable[[], None]]:
        return {
            "sovitrath": self._load_sovitrath,
            "tanlikesmath": self._load_tanlikesmath,
        }

    # Dataset loaders ----------------------------------------------------
    def _load_sovitrath(self) -> None:
        base_path = self.config.root / "gaussian_filtered_images" / "gaussian_filtered_images"
        classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
        self._load_from_directory(base_path, classes)

    def _load_tanlikesmath(self) -> None:
        base_candidates = [
            self.config.root / "resized_train" / "resized_train",
            self.config.root / "resized_train",
        ]
        base_path = next((path for path in base_candidates if path.exists()), None)
        if base_path is None:
            raise FileNotFoundError(
                "Could not locate image directory. Expected either 'resized_train/resized_train' or 'resized_train' under the dataset root."
            )

        labels_candidates = [
            self.config.root / "resized_train" / "trainLabels.csv",
            self.config.root / "trainLabels.csv",
        ]
        labels_path = next((path for path in labels_candidates if path.exists()), None)
        if labels_path is None:
            raise FileNotFoundError(
                "Missing labels CSV. Expected 'trainLabels.csv' inside dataset root or the 'resized_train' directory."
            )

        labels_df = pd.read_csv(labels_path)
        self.classes = list(TANLIKESMATH_CLASS_NAMES)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for _, row in labels_df.iterrows():
            img_name = f"{row['image']}.jpeg"
            img_path = base_path / img_name
            if img_path.exists():
                self.image_paths.append(img_path)
                self.labels.append(int(row["level"]))
            else:
                LOGGER.debug("Skipping missing image: %s", img_path)

    def _load_from_directory(self, base_path: Path, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        if not base_path.exists():
            raise FileNotFoundError(f"Image directory not found: {base_path}")
        for class_name in self.classes:
            class_path = base_path / class_name
            if not class_path.exists():
                LOGGER.warning("Skipping missing class directory: %s", class_path)
                continue
            for image_name in class_path.glob("*.jp*g"):
                self.image_paths.append(image_name)
                self.labels.append(self.class_to_idx[class_name])

    # Dataset interface --------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)

    def _compute_class_stats(self) -> None:
        num_classes = len(self.classes)
        if num_classes == 0:
            self.class_counts = np.zeros(0, dtype=np.int64)
            self.class_weights = np.zeros(0, dtype=np.float64)
            return
        if not self.labels:
            self.class_counts = np.zeros(num_classes, dtype=np.int64)
            self.class_weights = np.ones(num_classes, dtype=np.float64) / max(num_classes, 1)
            return
        counts = np.bincount(self.labels, minlength=num_classes).astype(np.int64)
        self.class_counts = counts
        total = counts.sum()
        if total == 0:
            self.class_weights = np.ones(num_classes, dtype=np.float64) / max(num_classes, 1)
            return
        safe_counts = counts.astype(np.float64)
        zero_mask = safe_counts == 0.0
        safe_counts[zero_mask] = 1.0
        weights = (total / num_classes) / safe_counts
        weights[zero_mask] = 0.0
        self.class_weights = weights


def create_dataloaders(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    base_dataset = DiabeticRetinopathyDataset(config, mode="full", transform=None)
    indices = list(range(len(base_dataset)))
    labels = [base_dataset.labels[i] for i in indices]
    train_indices, val_indices = train_test_split(
        indices,
        train_size=config.train_ratio,
        stratify=labels,
        random_state=config.seed,
    )

    def subset(indices_subset: Sequence[int], transform: Callable) -> DiabeticRetinopathyDataset:
        image_paths = [base_dataset.image_paths[i] for i in indices_subset]
        subset_labels = [base_dataset.labels[i] for i in indices_subset]
        return DiabeticRetinopathyDataset(
            config,
            mode="subset",
            transform=transform,
            image_paths=image_paths,
            labels=subset_labels,
            classes=base_dataset.classes,
            class_to_idx=base_dataset.class_to_idx,
        )

    train_dataset = subset(train_indices, config.transform)
    val_dataset = subset(val_indices, config.val_transform)

    sampler = None
    sampler_generator = None
    if config.use_weighted_sampler and getattr(train_dataset, "class_weights", None) is not None:
        class_weights = np.asarray(train_dataset.class_weights, dtype=np.float64)
        if class_weights.size == len(train_dataset.classes) and class_weights.sum() > 0:
            class_weights = class_weights / (class_weights.sum() + 1e-8)
            sample_labels = np.asarray(train_dataset.labels, dtype=np.int64)
            sample_weights = class_weights[np.clip(sample_labels, 0, len(class_weights) - 1)]
            sample_weights = np.clip(sample_weights, 1e-3, None)
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(config.seed)
            sampler = WeightedRandomSampler(
                weights=sample_weights.tolist(),
                num_samples=len(sample_weights),
                replacement=True,
                generator=sampler_generator,
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        sampler=sampler,
        generator=sampler_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, base_dataset.classes


if __name__ == "__main__":
    config = DatasetConfig(root="/data/diabetic_retinopathy", dataset_type="tanlikesmath")
    train_loader, val_loader, classes = create_dataloaders(config)
    LOGGER.info("Number of classes: %d", len(classes))
    LOGGER.info("Classes: %s", classes)
    LOGGER.info("Training samples: %d", len(train_loader.dataset))
    LOGGER.info("Validation samples: %d", len(val_loader.dataset))