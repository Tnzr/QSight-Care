# ============================================================================
# STREAMLIT DEPLOYMENT APP FOR DIABETIC RETINOPATHY CLASSIFICATION
# UPDATED WITH DATABASE SUPPORT AND HEAD CONFIDENCE VISUALISATIONS
# ============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from database import (
    create_patient,
    delete_patient,
    get_patient,
    init_db,
    list_assessments,
    list_patients,
    record_assessment,
    update_patient,
)

# ============================================================================
# 1. PAGE CONFIGURATION AND CONSTANTS
# ============================================================================
st.set_page_config(
    page_title="Diabetic Retinopathy Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path("./trained_model")
CLASS_COLORS = {
    "No_DR": "#4CAF50",
    "Mild": "#8BC34A",
    "Moderate": "#FFC107",
    "Severe": "#FF9800",
    "Proliferate_DR": "#F44336",
}
HEAD_DISPLAY_NAMES = {
    "fullres": "FullResHead",
    "comp": "CompHead",
    "quantum": "QuantumHead",
    "ensemble": "Final Ensemble",
}
SEX_OPTIONS = ["Male", "Female", "Other"]

# ============================================================================
# 2. CUSTOM STYLING
# ============================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2E86AB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E86AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 8px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .head-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# 3. DATABASE BOOTSTRAP
# ============================================================================


@st.cache_resource
def bootstrap_database() -> bool:
    init_db()
    return True


bootstrap_database()

# ============================================================================
# 4. MODEL COMPONENTS (MUST MATCH TRAINING PIPELINE)
# ============================================================================


class VisionEncoder(nn.Module):
    def __init__(self, encoder_type: str = "vit", pretrained: bool = False) -> None:
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "vit":
            self.encoder = models.vit_b_16(pretrained=pretrained)
            self.encoder.heads = nn.Identity()
            self.projection = nn.Linear(768, 2048)
        else:
            resnet = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.encoder_type == "vit":
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


class FullResHead(nn.Module):
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


class CompHead(nn.Module):
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


class QuantumHead(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 5, quantum_layers: int = 3) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for layer_idx in range(quantum_layers):
            next_dim = 64 if layer_idx < quantum_layers - 1 else 32
            activation: nn.Module = nn.Tanh() if layer_idx < quantum_layers - 1 else nn.ReLU()
            layers.extend(
                [
                    nn.Linear(current_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    activation,
                    nn.Dropout(0.2),
                ]
            )
            current_dim = next_dim
        layers.append(nn.Linear(32, num_classes))
        self.quantum_sim = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum_sim(x)


class DynamicEnsemble(nn.Module):
    def __init__(self, num_heads: int = 3, init_temp: float = 1.0) -> None:
        super().__init__()
        self.base_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.uncertainty_scales = nn.Parameter(torch.ones(num_heads))

    def forward(
        self,
        head_outputs: Dict[str, torch.Tensor],
        uncertainties: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.base_weights / self.temperature, dim=0)

        if uncertainties is not None and self.training:
            if uncertainties.dim() == 1:
                uncertainties = uncertainties.unsqueeze(0)
            scaled_uncertainties = uncertainties * self.uncertainty_scales.unsqueeze(0)
            confidence = 1.0 / (scaled_uncertainties + 1e-8)
            batch_confidence = confidence.mean(dim=0)
            confidence_weights = F.softmax(batch_confidence, dim=0)
            with torch.no_grad():
                predictions = torch.stack(
                    [torch.argmax(output, dim=1) for output in head_outputs.values()], dim=1
                ).float()
                max_vals, _ = predictions.max(dim=1)
                min_vals, _ = predictions.min(dim=1)
                agreement = (max_vals == min_vals).float().mean()
            uncertainty_weight = 0.7 * (1 - agreement) + 0.3
            weights = (1 - uncertainty_weight) * weights + uncertainty_weight * confidence_weights

        weights = weights / weights.sum()
        if not self.training:
            weights = weights.detach()

        final_output = sum(weight * output for weight, output in zip(weights, head_outputs.values()))
        return final_output, weights


class HybridDRModel(nn.Module):
    def __init__(self, model_info: Dict[str, Any]) -> None:
        super().__init__()
        self.classes = model_info.get(
            "classes",
            ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
        )
        self.num_classes = len(self.classes)
        self.compressed_dim = model_info.get("compressed_dim", 30)
        self.encoder_type = model_info.get("encoder_type", "vit")

        self.vision_encoder = VisionEncoder(encoder_type=self.encoder_type, pretrained=False)
        self.compression = CompressionModule(input_dim=2048, compressed_dim=self.compressed_dim)
        self.fullres_head = FullResHead(input_dim=2048, num_classes=self.num_classes)
        self.comp_head = CompHead(input_dim=self.compressed_dim, num_classes=self.num_classes)
        self.quantum_head = QuantumHead(input_dim=self.compressed_dim, num_classes=self.num_classes)
        self.ensemble = DynamicEnsemble(num_heads=3)

        if "ensemble_weights" in model_info:
            with torch.no_grad():
                weights_tensor = torch.tensor(model_info["ensemble_weights"], dtype=torch.float32)
                if weights_tensor.numel() == 3:
                    self.ensemble.base_weights.data = weights_tensor

        self.uncertainty_fullres = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.uncertainty_comp = nn.Sequential(
            nn.Linear(self.compressed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.uncertainty_quantum = nn.Sequential(
            nn.Linear(self.compressed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_all: bool = True) -> Dict[str, torch.Tensor]:
        latent_features = self.vision_encoder(x)
        compressed_features = self.compression(latent_features)

        outputs = {
            "fullres": self.fullres_head(latent_features),
            "comp": self.comp_head(compressed_features),
            "quantum": self.quantum_head(compressed_features),
        }

        uncertainties = torch.cat(
            [
                self.uncertainty_fullres(latent_features),
                self.uncertainty_comp(compressed_features),
                self.uncertainty_quantum(compressed_features),
            ],
            dim=1,
        ).squeeze()

        final_output, ensemble_weights = self.ensemble(outputs, uncertainties)

        if not return_all:
            return {"final_output": final_output}

        probabilities = F.softmax(final_output, dim=1)

        return {
            "fullres": outputs["fullres"],
            "comp": outputs["comp"],
            "quantum": outputs["quantum"],
            "final_output": final_output,
            "ensemble_weights": ensemble_weights,
            "uncertainties": uncertainties,
            "probabilities": probabilities,
            "latent_features": latent_features,
            "compressed_features": compressed_features,
        }


# ============================================================================
# 5. MODEL LOADING AND INFERENCE UTILITIES
# ============================================================================


def compute_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if not weight_kg or not height_cm or height_cm == 0:
        return None
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m**2), 1)


@st.cache_resource
def load_model():
    try:
        model_path = MODEL_DIR / "phase1_classical_model.pth"
        info_path = MODEL_DIR / "model_info.pkl"

        if not model_path.exists():
            st.error(f"❌ Model file not found at: {model_path}")
            return None, None, None
        if not info_path.exists():
            st.error(f"❌ Model info file not found at: {info_path}")
            return None, None, None

        import pickle

        with open(info_path, "rb") as handle:
            model_info = pickle.load(handle)

        model = HybridDRModel(model_info)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        class_names = model_info.get(
            "classes",
            ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
        )
        ensemble_weights = model_info.get("ensemble_weights", [0.333, 0.333, 0.333])
        return model, class_names, ensemble_weights

    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Error loading model: {exc}")
        return None, None, None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def predict_images(
    model: HybridDRModel,
    image_tensor: torch.Tensor,
    eye_labels: List[str],
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    with torch.no_grad():
        outputs = model(image_tensor, return_all=True)

    prob_tensor = outputs["probabilities"]
    if prob_tensor.dim() == 1:
        prob_tensor = prob_tensor.unsqueeze(0)

    uncertainties_tensor = outputs["uncertainties"]
    if uncertainties_tensor.dim() == 1:
        uncertainties_tensor = uncertainties_tensor.unsqueeze(0)

    head_tensors = {}
    for key in ["fullres", "comp", "quantum"]:
        head_tensor = outputs[key]
        if head_tensor.dim() == 1:
            head_tensor = head_tensor.unsqueeze(0)
        head_tensors[key] = head_tensor

    ensemble_weights = outputs["ensemble_weights"].detach().cpu().numpy()
    if ensemble_weights.ndim > 1:
        ensemble_weights = ensemble_weights.squeeze()

    batch_size = prob_tensor.shape[0]
    results: List[Dict[str, Any]] = []
    for idx in range(batch_size):
        probs_np = prob_tensor[idx].detach().cpu().numpy()
        final_idx = int(probs_np.argmax())
        final_confidence = float(probs_np[final_idx])

        head_probabilities: Dict[str, np.ndarray] = {}
        head_predictions: Dict[str, int] = {}
        head_confidences: Dict[str, float] = {}
        for key in ["fullres", "comp", "quantum"]:
            logits = head_tensors[key][idx]
            head_probs = F.softmax(logits, dim=0).detach().cpu().numpy()
            head_probabilities[key] = head_probs
            head_predictions[key] = int(head_probs.argmax())
            head_confidences[key] = float(head_probs[final_idx])

        uncertainty_row = uncertainties_tensor[idx].detach().cpu().numpy()
        eye_label = eye_labels[idx] if idx < len(eye_labels) else f"Image {idx + 1}"
        class_label = (
            class_names[final_idx]
            if class_names and final_idx < len(class_names)
            else str(final_idx)
        )

        results.append(
            {
                "eye": eye_label,
                "final_index": final_idx,
                "final_label": class_label,
                "final_confidence": final_confidence,
                "probabilities": probs_np,
                "ensemble_weights": ensemble_weights.copy(),
                "uncertainties": uncertainty_row,
                "head_predictions": head_predictions,
                "head_probabilities": head_probabilities,
                "head_confidences": head_confidences,
            }
        )

    return results


def run_inference_for_eyes(
    model: HybridDRModel,
    images: List[Image.Image],
    eye_labels: List[str],
    requested_mode: str,
    class_names: Optional[List[str]],
) -> Dict[str, Any]:
    if not images:
        raise ValueError("At least one eye image is required for inference.")

    batched_requested = requested_mode.startswith("Batch") and len(images) > 1
    results: List[Dict[str, Any]] = []

    if batched_requested:
        batch_tensor = torch.cat([preprocess_image(image) for image in images], dim=0)
        batch_results = predict_images(model, batch_tensor, eye_labels, class_names)
        for entry in batch_results:
            entry["execution_mode"] = "batch"
        results.extend(batch_results)
        execution_mode = "batch"
    else:
        execution_mode = "sequential"
        for image, label in zip(images, eye_labels):
            tensor = preprocess_image(image)
            single_results = predict_images(model, tensor, [label], class_names)
            for entry in single_results:
                entry["execution_mode"] = "sequential"
            results.extend(single_results)

    ensemble_weights = results[0]["ensemble_weights"] if results else None

    return {
        "mode_requested": requested_mode,
        "execution_mode": execution_mode,
        "results": results,
        "ensemble_weights": ensemble_weights,
        "saved_to_patient": False,
    }


# ============================================================================
# 6. SESSION STATE HELPERS
# ============================================================================


def ensure_session_state() -> None:
    defaults = {
        "model": None,
        "class_names": None,
        "ensemble_reference": None,
        "prediction_result": None,
        "uploaded_image": None,
        "left_eye_image": None,
        "right_eye_image": None,
        "inference_mode": "Batch (single pass)",
        "active_patient_id": None,
        "profile_form_loaded_for": None,
        "profile_name": "",
        "profile_age": 0,
        "profile_sex": SEX_OPTIONS[0],
        "profile_weight": 0.0,
        "profile_height": 0.0,
        "profile_insulin": False,
        "profile_smoking": False,
        "profile_alcohol": False,
        "profile_vascular": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


ensure_session_state()


def load_profile_into_state(patient: Dict[str, Any], patient_id: Optional[int]) -> None:
    st.session_state.profile_form_loaded_for = patient_id
    st.session_state.profile_name = patient.get("name", "")
    st.session_state.profile_age = patient.get("age", 0) or 0
    st.session_state.profile_sex = patient.get("sex", SEX_OPTIONS[0]) or SEX_OPTIONS[0]
    st.session_state.profile_weight = float(patient.get("weight_kg")) if patient.get("weight_kg") else 0.0
    st.session_state.profile_height = float(patient.get("height_cm")) if patient.get("height_cm") else 0.0
    st.session_state.profile_insulin = bool(patient.get("insulin"))
    st.session_state.profile_smoking = bool(patient.get("smoking"))
    st.session_state.profile_alcohol = bool(patient.get("alcohol"))
    st.session_state.profile_vascular = bool(patient.get("vascular_disease"))


def reset_profile_state() -> None:
    st.session_state.profile_form_loaded_for = None
    st.session_state.profile_name = ""
    st.session_state.profile_age = 0
    st.session_state.profile_sex = SEX_OPTIONS[0]
    st.session_state.profile_weight = 0.0
    st.session_state.profile_height = 0.0
    st.session_state.profile_insulin = False
    st.session_state.profile_smoking = False
    st.session_state.profile_alcohol = False
    st.session_state.profile_vascular = False


# ============================================================================
# 7. SIDEBAR: PATIENT PROFILE BUILDER
# ============================================================================


def render_profile_builder() -> None:
    st.markdown("## 🧑‍⚕️ Patient Profile Builder")
    patients = list_patients()
    options = {"➕ New Profile": None}
    for patient in patients:
        label = f"{patient['name']} (ID {patient['id']})"
        options[label] = patient["id"]

    option_labels = list(options.keys())
    current_selection = None
    if st.session_state.active_patient_id is not None:
        current_label = next(
            (label for label, pid in options.items() if pid == st.session_state.active_patient_id),
            option_labels[0],
        )
        current_selection = option_labels.index(current_label)
    else:
        current_selection = 0

    selection = st.selectbox("Select or create a patient", option_labels, index=current_selection)
    selected_id = options[selection]

    if st.session_state.profile_form_loaded_for != selected_id:
        if selected_id is None:
            reset_profile_state()
        else:
            patient_record = get_patient(selected_id)
            if patient_record:
                load_profile_into_state(patient_record, selected_id)
                st.session_state.active_patient_id = selected_id
            else:
                reset_profile_state()
                st.session_state.active_patient_id = None

    with st.form("patient_profile_form", clear_on_submit=False):
        st.session_state.profile_name = st.text_input("Name", value=st.session_state.profile_name)
        st.session_state.profile_age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            step=1,
            value=int(st.session_state.profile_age),
        )
        st.session_state.profile_sex = st.selectbox(
            "Sex", SEX_OPTIONS, index=SEX_OPTIONS.index(st.session_state.profile_sex)
        )
        st.session_state.profile_weight = st.number_input(
            "Weight (kg)",
            min_value=0.0,
            max_value=300.0,
            step=0.5,
            value=float(st.session_state.profile_weight),
        )
        st.session_state.profile_height = st.number_input(
            "Height (cm)",
            min_value=0.0,
            max_value=250.0,
            step=0.5,
            value=float(st.session_state.profile_height),
        )

        bmi = compute_bmi(st.session_state.profile_weight, st.session_state.profile_height)
        if bmi:
            st.caption(f"Calculated BMI: {bmi}")
        else:
            st.caption("Provide weight and height to compute BMI.")

        st.session_state.profile_insulin = st.checkbox(
            "Insulin Therapy",
            value=bool(st.session_state.profile_insulin),
        )
        st.session_state.profile_smoking = st.checkbox(
            "Smoking",
            value=bool(st.session_state.profile_smoking),
        )
        st.session_state.profile_alcohol = st.checkbox(
            "Alcohol Consumption",
            value=bool(st.session_state.profile_alcohol),
        )
        st.session_state.profile_vascular = st.checkbox(
            "Vascular Disease",
            value=bool(st.session_state.profile_vascular),
        )

        submitted = st.form_submit_button("Save Profile", use_container_width=True)

    if submitted:
        if not st.session_state.profile_name.strip():
            st.warning("Name is required to save the profile.")
        else:
            bmi_value = compute_bmi(st.session_state.profile_weight, st.session_state.profile_height)
            profile_payload = {
                "name": st.session_state.profile_name.strip(),
                "age": int(st.session_state.profile_age),
                "sex": st.session_state.profile_sex,
                "weight_kg": float(st.session_state.profile_weight) if st.session_state.profile_weight else None,
                "height_cm": float(st.session_state.profile_height) if st.session_state.profile_height else None,
                "bmi": bmi_value,
                "obesity_flag": 1 if bmi_value and bmi_value >= 30 else 0,
                "insulin": int(st.session_state.profile_insulin),
                "smoking": int(st.session_state.profile_smoking),
                "alcohol": int(st.session_state.profile_alcohol),
                "vascular_disease": int(st.session_state.profile_vascular),
            }
            if selected_id is None:
                new_id = create_patient(profile_payload)
                st.session_state.active_patient_id = new_id
                st.session_state.profile_form_loaded_for = new_id
                st.success(f"✅ Created profile for {profile_payload['name']} (ID {new_id}).")
            else:
                update_patient(selected_id, profile_payload)
                st.session_state.active_patient_id = selected_id
                st.session_state.profile_form_loaded_for = selected_id
                st.success("✅ Profile updated successfully.")
            st.rerun()

    if selected_id is not None:
        if st.button("Delete Profile", use_container_width=True):
            delete_patient(selected_id)
            st.success("🗑️ Profile deleted.")
            st.session_state.active_patient_id = None
            reset_profile_state()
            st.rerun()

        assessments = list_assessments(selected_id, limit=5)
        if assessments:
            st.markdown("### 📈 Recent Assessments")
            for record in assessments:
                timestamp = record.get("created_at", "")
                label = record.get("final_label", "Unknown")
                confidence = record.get("final_confidence")
                st.write(
                    f"- {timestamp}: **{label}** (confidence {confidence:.2%})"
                )
    else:
        st.info("Create or select a patient profile to enable history tracking.")


# ============================================================================
# 8. STREAMLIT APPLICATION BODY
# ============================================================================


def main() -> None:
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Hybrid Classical-Quantum Deep Learning Model with Dynamic Ensemble")
    st.markdown("Upload a retinal fundus image for automated classification of diabetic retinopathy severity.")

    with st.sidebar:
        render_profile_builder()
        st.markdown("---")
        st.markdown("## 🚀 Quick Start")

        if st.button("🔄 Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                model, class_names, ensemble_reference = load_model()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.class_names = class_names
                    st.session_state.ensemble_reference = ensemble_reference
                    st.success("✅ Model loaded!")
                    st.info(f"Classes: {class_names}")
                    st.info(f"Reference Ensemble Weights: {ensemble_reference}")
                else:
                    st.error("❌ Failed to load model")

        st.markdown("---")
        st.markdown("## 📁 Model Files")
        required_files = [
            (MODEL_DIR / "phase1_classical_model.pth", "Model Weights"),
            (MODEL_DIR / "model_info.pkl", "Model Info"),
        ]
        missing = False
        for path, desc in required_files:
            if path.exists():
                st.success(f"✅ {desc}")
            else:
                st.error(f"❌ {desc} missing: {path}")
                missing = True
        if missing:
            st.warning(
                "Place the exported `phase1_classical_model.pth` and `model_info.pkl` files inside the `trained_model/` folder."
            )

        st.markdown("---")
        st.markdown("## 📸 Upload Eye Images")
        left_file = st.file_uploader(
            "Left Eye Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="left_eye_upload",
            help="Upload the left eye retinal fundus image",
        )
        if left_file is not None:
            try:
                st.session_state.left_eye_image = Image.open(left_file).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unable to read left eye image: {exc}")

        right_file = st.file_uploader(
            "Right Eye Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="right_eye_upload",
            help="Upload the right eye retinal fundus image",
        )
        if right_file is not None:
            try:
                st.session_state.right_eye_image = Image.open(right_file).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unable to read right eye image: {exc}")

        if st.button("🧹 Clear Uploaded Eye Images", use_container_width=True):
            st.session_state.left_eye_image = None
            st.session_state.right_eye_image = None
            st.session_state.prediction_result = None

        st.markdown("---")
        st.markdown("## 🧠 Inference Settings")
        st.radio(
            "Select inference mode",
            ("Batch (single pass)", "Sequential (memory saver)"),
            key="inference_mode",
            help="Batch mode runs both eyes in a single forward pass. Sequential mode processes each eye separately to reduce memory footprint.",
        )

        if st.session_state.model is not None:
            if st.button("🔍 Analyze Eyes", type="primary", use_container_width=True):
                eye_images: List[Image.Image] = []
                eye_labels: List[str] = []
                if st.session_state.left_eye_image is not None:
                    eye_images.append(st.session_state.left_eye_image)
                    eye_labels.append("Left Eye")
                if st.session_state.right_eye_image is not None:
                    eye_images.append(st.session_state.right_eye_image)
                    eye_labels.append("Right Eye")

                if not eye_images:
                    st.warning("Upload at least one eye image before running inference.")
                else:
                    with st.spinner("Analyzing eye images..."):
                        try:
                            summary = run_inference_for_eyes(
                                st.session_state.model,
                                eye_images,
                                eye_labels,
                                st.session_state.inference_mode,
                                st.session_state.class_names,
                            )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"❌ Inference failed: {exc}")
                        else:
                            patient_id = st.session_state.active_patient_id
                            if patient_id is not None:
                                per_eye_metadata = []
                                confidences_fullres: List[float] = []
                                confidences_comp: List[float] = []
                                confidences_quantum: List[float] = []
                                final_confidences: List[float] = []
                                for entry in summary["results"]:
                                    per_eye_metadata.append(
                                        {
                                            "eye": entry["eye"],
                                            "final_label": entry["final_label"],
                                            "final_confidence": entry["final_confidence"],
                                            "probabilities": entry["probabilities"].tolist(),
                                            "head_probabilities": {
                                                key: value.tolist()
                                                for key, value in entry["head_probabilities"].items()
                                            },
                                            "head_confidences": entry["head_confidences"],
                                            "ensemble_weights": entry["ensemble_weights"].tolist()
                                            if isinstance(entry["ensemble_weights"], np.ndarray)
                                            else entry["ensemble_weights"],
                                            "uncertainties": entry["uncertainties"].tolist()
                                            if isinstance(entry["uncertainties"], np.ndarray)
                                            else entry["uncertainties"],
                                            "execution_mode": entry.get("execution_mode"),
                                        }
                                    )
                                    final_confidences.append(entry["final_confidence"])
                                    confidences_fullres.append(entry["head_confidences"]["fullres"])
                                    confidences_comp.append(entry["head_confidences"]["comp"])
                                    confidences_quantum.append(entry["head_confidences"]["quantum"])

                                aggregate_label = "; ".join(
                                    f"{entry['eye']}: {entry['final_label']}" for entry in summary["results"]
                                )
                                record_assessment(
                                    patient_id,
                                    {
                                        "inference_mode": summary["execution_mode"],
                                        "final_label": aggregate_label,
                                        "final_confidence": float(np.mean(final_confidences)) if final_confidences else None,
                                        "head_fullres_conf": float(np.mean(confidences_fullres)) if confidences_fullres else None,
                                        "head_comp_conf": float(np.mean(confidences_comp)) if confidences_comp else None,
                                        "head_quantum_conf": float(np.mean(confidences_quantum)) if confidences_quantum else None,
                                        "ensemble_weights": summary["ensemble_weights"].tolist() if summary["ensemble_weights"] is not None else [],
                                        "metadata": {
                                            "mode_requested": summary["mode_requested"],
                                            "execution_mode": summary["execution_mode"],
                                            "per_eye": per_eye_metadata,
                                        },
                                    },
                                )
                                summary["saved_to_patient"] = True
                            else:
                                summary["saved_to_patient"] = False

                            st.session_state.prediction_result = summary
                            st.rerun()
        else:
            st.warning("⚠️ Load the model first to enable analysis.")

    st.markdown('<h3 class="sub-header">Input Eye Images</h3>', unsafe_allow_html=True)
    if st.session_state.left_eye_image is None and st.session_state.right_eye_image is None:
        st.info("👈 Upload left and/or right eye images using the sidebar uploader.")
    else:
        eye_cols = st.columns(2)
        with eye_cols[0]:
            st.markdown("**Left Eye**")
            if st.session_state.left_eye_image is not None:
                st.image(st.session_state.left_eye_image, use_column_width=True)
                width, height = st.session_state.left_eye_image.size
                st.caption(f"📐 {width}×{height} pixels")
            else:
                st.caption("No left eye image provided.")
        with eye_cols[1]:
            st.markdown("**Right Eye**")
            if st.session_state.right_eye_image is not None:
                st.image(st.session_state.right_eye_image, use_column_width=True)
                width, height = st.session_state.right_eye_image.size
                st.caption(f"📐 {width}×{height} pixels")
            else:
                st.caption("No right eye image provided.")

    st.markdown('<h3 class="sub-header">Analysis Results</h3>', unsafe_allow_html=True)
    if st.session_state.model is None:
        st.warning("⚠️ Model not loaded. Click 'Load Model' in the sidebar.")
    elif st.session_state.prediction_result is not None:
        display_prediction_results(st.session_state.prediction_result)
    elif st.session_state.left_eye_image is not None or st.session_state.right_eye_image is not None:
        st.info("📸 Eye images are ready. Click 'Analyze Eyes' in the sidebar to run inference.")
    else:
        st.success("✅ Model loaded and standing by. Upload eye images to begin.")

    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        <p><strong>Hybrid Classical-Quantum Deep Learning Model for Diabetic Retinopathy Classification</strong></p>
        <p>⚠️ <em>This tool is for research and educational purposes only. Not for clinical diagnosis.</em></p>
        <p>Model uses dynamic ensemble learning with adaptive weight allocation.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 9. RESULT RENDERING
# ============================================================================


def render_eye_prediction(entry: Dict[str, Any], class_names: List[str]) -> None:
    predicted_class = entry.get("final_label", "Unknown")
    confidence = entry.get("final_confidence", 0.0)
    display_color = CLASS_COLORS.get(predicted_class, "#2E86AB")
    execution_mode = entry.get("execution_mode", "").capitalize()

    st.markdown(f"### {entry.get('eye', 'Eye')} Prediction")
    if execution_mode:
        st.caption(f"Processed via {execution_mode} inference path")

    st.markdown(
        f"""
        <div class="prediction-box" style="border-left-color: {display_color};">
            <h4 style="color: #2E86AB; margin-top: 0;">Final Prediction</h4>
            <h1 style="color: {display_color}; margin: 10px 0;">{predicted_class}</h1>
            <p style="font-size: 1.1rem;">Confidence: <strong>{confidence:.2%}</strong></p>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; margin: 10px 0;">
                <div style="width: {confidence * 100:.2f}%; background-color: {display_color}; height: 100%; border-radius: 10px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_weights = entry.get("ensemble_weights")
    head_weights = (
        np.array(raw_weights, dtype=float)
        if raw_weights is not None
        else np.array([], dtype=float)
    )
    raw_uncertainties = entry.get("uncertainties")
    uncertainties = (
        np.array(raw_uncertainties, dtype=float)
        if raw_uncertainties is not None
        else np.array([], dtype=float)
    )
    head_predictions = entry.get("head_predictions", {})
    head_conf = entry.get("head_confidences", {})

    cols = st.columns(3)
    for idx, key in enumerate(["fullres", "comp", "quantum"]):
        with cols[idx]:
            display_name = HEAD_DISPLAY_NAMES[key]
            st.markdown('<div class="head-card">', unsafe_allow_html=True)
            st.markdown(f"**{display_name}**")
            pred_idx = head_predictions.get(key)
            head_label = class_names[pred_idx] if pred_idx is not None and pred_idx < len(class_names) else "N/A"
            st.metric("Prediction", head_label)
            if idx < len(head_weights):
                st.caption(f"⚖️ Weight: {head_weights[idx]:.3f}")
            if key in head_conf:
                st.caption(f"🎯 Confidence: {head_conf[key]:.3f}")
            if idx < len(uncertainties):
                st.caption(f"❓ Uncertainty: {uncertainties[idx]:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 📊 Probability Distribution")
    fig_prob, ax_prob = plt.subplots(figsize=(8, 4))
    raw_probs = entry.get("probabilities")
    if raw_probs is None:
        probs = np.zeros(len(class_names))
    else:
        probs = np.array(raw_probs, dtype=float)
    if probs.shape[0] != len(class_names):
        adjusted = np.zeros(len(class_names))
        limit = min(len(class_names), probs.shape[0])
        adjusted[:limit] = probs[:limit]
        probs = adjusted
    colors = [CLASS_COLORS.get(name, "#2E86AB") for name in class_names]
    bars = ax_prob.bar(class_names, probs, color=colors)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim([0, 1])
    ax_prob.grid(True, alpha=0.3)
    for bar, prob in zip(bars, probs):
        ax_prob.text(bar.get_x() + bar.get_width() / 2.0, prob + 0.02, f"{prob:.2%}", ha="center", va="bottom")
    st.pyplot(fig_prob)
    plt.close(fig_prob)

    st.markdown("#### 🎚 Head Confidence Comparison")
    head_labels = [
        HEAD_DISPLAY_NAMES["fullres"],
        HEAD_DISPLAY_NAMES["comp"],
        HEAD_DISPLAY_NAMES["quantum"],
        HEAD_DISPLAY_NAMES["ensemble"],
    ]
    head_values = [
        head_conf.get("fullres", 0.0),
        head_conf.get("comp", 0.0),
        head_conf.get("quantum", 0.0),
        confidence,
    ]
    fig_head, ax_head = plt.subplots(figsize=(6, 3))
    bar_colors = ["#2E86AB", "#1ABC9C", "#9B59B6", "#E67E22"]
    bars = ax_head.bar(head_labels, head_values, color=bar_colors)
    ax_head.set_ylim(0, 1)
    ax_head.set_ylabel("Confidence (Probability)")
    ax_head.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, head_values):
        ax_head.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.2%}", ha="center", va="bottom")
    st.pyplot(fig_head)
    plt.close(fig_head)

    st.markdown("#### 🧮 Per-Class Confidence by Head")
    comparison_heads = ["fullres", "comp", "quantum", "ensemble"]
    comparison_labels = [
        HEAD_DISPLAY_NAMES["fullres"],
        HEAD_DISPLAY_NAMES["comp"],
        HEAD_DISPLAY_NAMES["quantum"],
        HEAD_DISPLAY_NAMES["ensemble"],
    ]
    comparison_data: List[np.ndarray] = []
    head_prob_dict = entry.get("head_probabilities") or {}
    for head_key in comparison_heads:
        if head_key == "ensemble":
            comparison_data.append(probs.copy())
        else:
            raw = head_prob_dict.get(head_key)
            if raw is None:
                comparison_data.append(np.zeros(len(class_names)))
            else:
                arr = np.array(raw, dtype=float)
                if arr.shape[0] != len(class_names):
                    tmp = np.zeros(len(class_names))
                    limit = min(len(class_names), arr.shape[0])
                    tmp[:limit] = arr[:limit]
                    arr = tmp
                comparison_data.append(arr)

    x_positions = np.arange(len(class_names))
    num_heads = len(comparison_heads)
    width = 0.18 if num_heads > 0 else 0.2
    fig_compare, ax_compare = plt.subplots(figsize=(max(8, len(class_names) * 1.2), 4))
    color_palette = ["#2E86AB", "#1ABC9C", "#9B59B6", "#E67E22"]
    for idx, data in enumerate(comparison_data):
        offset = (idx - (num_heads - 1) / 2) * width
        ax_compare.bar(
            x_positions + offset,
            data,
            width,
            label=comparison_labels[idx],
            color=color_palette[idx % len(color_palette)],
        )
    ax_compare.set_xticks(x_positions)
    ax_compare.set_xticklabels(class_names, rotation=15)
    ax_compare.set_ylim(0, 1)
    ax_compare.set_ylabel("Probability")
    ax_compare.set_title("Head-Level Probabilities per Severity Class")
    ax_compare.grid(True, axis="y", alpha=0.3)
    ax_compare.legend(loc="upper right")
    st.pyplot(fig_compare)
    plt.close(fig_compare)


def display_prediction_results(summary: Dict[str, Any]) -> None:
    class_names = st.session_state.class_names or ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
    results = summary.get("results", [])
    if not results:
        st.info("No predictions available yet.")
        return

    execution_mode_raw = summary.get("execution_mode", "unknown") or "unknown"
    execution_mode = execution_mode_raw.capitalize()
    requested_mode = summary.get("mode_requested")
    mode_caption = f"Inference executed in **{execution_mode}** mode."
    if requested_mode:
        requested_tag = "batch" if requested_mode.startswith("Batch") else "sequential"
        executed_tag = "batch" if execution_mode_raw.startswith("batch") else "sequential"
        if requested_tag != executed_tag:
            mode_caption += f" (Requested: {requested_mode})"
    st.caption(mode_caption)

    if summary.get("saved_to_patient"):
        st.success("📒 Assessment saved to the selected patient profile.")
    elif st.session_state.active_patient_id is None:
        st.info("Select or create a patient profile to save future assessments.")

    for entry in results:
        render_eye_prediction(entry, class_names)

    if (
        st.session_state.ensemble_reference is not None
        and summary.get("ensemble_weights") is not None
        and len(summary["ensemble_weights"]) == 3
    ):
        st.markdown("### ⚖️ Ensemble Weight Comparison")
        ref = np.array(st.session_state.ensemble_reference)
        current = np.array(summary["ensemble_weights"], dtype=float)
        labels = [HEAD_DISPLAY_NAMES["fullres"], HEAD_DISPLAY_NAMES["comp"], HEAD_DISPLAY_NAMES["quantum"]]
        indices = np.arange(len(labels))
        width = 0.35
        fig_weights, ax_weights = plt.subplots(figsize=(8, 3))
        ax_weights.bar(indices - width / 2, ref, width, label="Trained Reference", color="#2E86AB")
        ax_weights.bar(indices + width / 2, current, width, label="Current Batch", color="#4CAF50")
        ax_weights.set_ylabel("Weight")
        ax_weights.set_xticks(indices)
        ax_weights.set_xticklabels(labels)
        ax_weights.legend()
        ax_weights.grid(True, alpha=0.3)
        st.pyplot(fig_weights)
        plt.close(fig_weights)


# ============================================================================
# 10. ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        st.warning(f"📁 Created '{MODEL_DIR}' directory. Place your model files here.")
    main()
