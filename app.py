"""
app.py — DermaDx Lite Streamlit Application
============================================
Hybrid AI skin disease classifier for erythemato-squamous diseases.

Usage:
    streamlit run app.py

Requires model.pkl and feature_schema.json to be present.
Run `python train.py` first to generate these files.
"""

import json
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris",
}

ERYTHEMA_THRESHOLDS = [0.38, 0.44, 0.50]
SCALING_THRESHOLDS = [0.05, 0.15, 0.30]

CLINICAL_FEATURES = {
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "age",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_color(conf: float) -> str:
    """Return a color string based on confidence level."""
    if conf >= 0.75:
        return "high"
    elif conf >= 0.50:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Resource loading (cached -- runs only once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """
    Load model.pkl and feature_schema.json exactly once at startup.
    Stops the app with a clear error message if either file is missing.

    Returns:
        Tuple (model, schema_dict) where schema_dict has keys:
          - "features"   : list of feature definition dicts
          - "class_order": list of class labels in XGBoost's internal order
    """
    if not os.path.exists("model.pkl"):
        st.error(
            "model.pkl not found.\n\n"
            "Please run the training script first:\n```\npython train.py\n```"
        )
        st.stop()

    if not os.path.exists("feature_schema.json"):
        st.error(
            "feature_schema.json not found.\n\n"
            "Please run the training script first:\n```\npython train.py\n```"
        )
        st.stop()

    model = joblib.load("model.pkl")

    with open("feature_schema.json", "r") as f:
        schema = json.load(f)

    return model, schema


# ---------------------------------------------------------------------------
# ImageAnalyzer -- OpenCV-based feature extraction
# ---------------------------------------------------------------------------

class ImageAnalyzer:
    """
    Extracts erythema and scaling scores from a skin image using OpenCV.
    Image is ONLY used to autofill 2 of the 34 features -- never for prediction.
    """

    @staticmethod
    def _compute_erythema(bgr_image: np.ndarray) -> int:
        """
        Compute erythema score (0-3) from mean red-channel ratio.

        Method:
          1. Convert BGR -> RGB
          2. Compute per-pixel ratio: R / (R + G + B + epsilon)
          3. Take mean ratio across all pixels
          4. Map to 0-3 using ERYTHEMA_THRESHOLDS
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        ratio = r / (r + g + b + 1e-6)
        mean_ratio = float(np.mean(ratio))

        if mean_ratio < ERYTHEMA_THRESHOLDS[0]:
            return 0
        elif mean_ratio < ERYTHEMA_THRESHOLDS[1]:
            return 1
        elif mean_ratio < ERYTHEMA_THRESHOLDS[2]:
            return 2
        else:
            return 3

    @staticmethod
    def _compute_scaling(bgr_image: np.ndarray) -> int:
        """
        Compute scaling score (0-3) from Canny edge density.

        Method:
          1. Convert BGR -> Grayscale
          2. Apply Gaussian blur (5x5) to reduce noise
          3. Run Canny edge detection (low=50, high=150)
          4. Compute edge_density = edge_pixels / total_pixels
          5. Map to 0-3 using SCALING_THRESHOLDS
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        if edge_density < SCALING_THRESHOLDS[0]:
            return 0
        elif edge_density < SCALING_THRESHOLDS[1]:
            return 1
        elif edge_density < SCALING_THRESHOLDS[2]:
            return 2
        else:
            return 3

    @staticmethod
    def analyze(image_bytes: bytes) -> tuple:
        """
        Decode image bytes and extract erythema + scaling scores.

        Args:
            image_bytes: Raw bytes from an uploaded image file.

        Returns:
            Tuple (erythema_score, scaling_score) -- both ints in {0, 1, 2, 3}.

        Raises:
            ValueError: If the bytes cannot be decoded as a valid image.
        """
        img_array = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if bgr is None:
            raise ValueError(
                "The uploaded file could not be decoded as a valid image. "
                "Please upload a JPEG or PNG file."
            )

        erythema = ImageAnalyzer._compute_erythema(bgr)
        scaling = ImageAnalyzer._compute_scaling(bgr)
        return erythema, scaling


# ---------------------------------------------------------------------------
# Validator -- input checking before prediction
# ---------------------------------------------------------------------------

def validate_inputs(feature_values: dict, schema_features: list) -> list:
    """
    Validate that all 34 features are present and within valid ranges.

    Args:
        feature_values  : Dict mapping feature_name -> value.
        schema_features : List of feature definition dicts from feature_schema.json.

    Returns:
        List of error strings. Empty list means all inputs are valid.
    """
    errors = []

    for entry in schema_features:
        name = entry["name"]
        min_val = entry["min"]
        max_val = entry["max"]

        if name not in feature_values or feature_values[name] is None:
            errors.append(f"'{FEATURE_UI_CONFIG.get(name, {}).get('label', name)}' is missing.")
            continue

        val = feature_values[name]
        if not (min_val <= val <= max_val):
            errors.append(
                f"'{FEATURE_UI_CONFIG.get(name, {}).get('label', name)}' value {val} "
                f"is out of range [{min_val}, {max_val}]."
            )

    return errors


# ---------------------------------------------------------------------------
# PredictionEngine -- assembles feature vector and runs inference
# ---------------------------------------------------------------------------

class PredictionEngine:
    """
    Assembles the ordered 34-feature vector and runs XGBoost inference.
    Feature order is determined exclusively by feature_schema.json.
    """

    def __init__(self, model, schema: dict) -> None:
        """
        Args:
            model : Loaded XGBoost classifier.
            schema: Parsed feature_schema.json dict with "features" and "class_order".
        """
        self.model = model
        self.schema_features = sorted(schema["features"], key=lambda x: x["index"])
        self.class_order = schema["class_order"]

    def _assemble_vector(self, feature_values: dict) -> np.ndarray:
        """
        Build a (1, 34) numpy array ordered strictly by schema index.
        This ensures train/serve consistency -- the model always sees features
        in the same order it was trained on.
        """
        values = [feature_values[entry["name"]] for entry in self.schema_features]
        return np.array([values], dtype=np.float32)

    def predict(self, feature_values: dict) -> dict:
        """
        Run inference and return structured prediction results.

        Args:
            feature_values: Dict mapping feature_name -> numeric value.

        Returns:
            Dict with keys:
              - prediction  : str  -- top-1 disease name
              - confidence  : float -- top-1 probability (0.0-1.0)
              - top_3       : list of {"disease": str, "confidence": float}
              - all_probs   : dict mapping disease_name -> probability
        """
        vector = self._assemble_vector(feature_values)
        proba = self.model.predict_proba(vector)[0]

        prob_dict = {}
        for col_idx, class_label in enumerate(self.class_order):
            disease_name = CLASS_NAMES.get(class_label, f"Class {class_label}")
            prob_dict[disease_name] = float(proba[col_idx])

        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        top_prediction = sorted_probs[0][0]
        top_confidence = sorted_probs[0][1]

        top_3 = [
            {"disease": name, "confidence": conf}
            for name, conf in sorted_probs[:3]
        ]

        return {
            "prediction":  top_prediction,
            "confidence":  top_confidence,
            "top_3":       top_3,
            "all_probs":   prob_dict,
        }


# ---------------------------------------------------------------------------
# Severity scale constants
# ---------------------------------------------------------------------------

# Severity scale: maps display label -> numeric value (0-3)
SEVERITY_OPTIONS = ["None", "Mild", "Moderate", "Severe"]
SEVERITY_TO_INT  = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
INT_TO_SEVERITY  = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}


# ---------------------------------------------------------------------------
# Dual-layer naming system
# ---------------------------------------------------------------------------

FEATURE_UI_CONFIG = {
    # Symptoms
    "erythema": {
        "label":    "Skin redness",
        "clinical": "Erythema",
        "desc":     "Red or inflamed areas on skin",
    },
    "scaling": {
        "label":    "Flaky or peeling skin",
        "clinical": "Scaling",
        "desc":     "Dry, flaking, or peeling skin patches",
    },
    "itching": {
        "label":    "Itching sensation",
        "clinical": "Itching",
        "desc":     "Presence and severity of skin itching",
    },
    "definite_borders": {
        "label":    "Clearly defined rash edges",
        "clinical": "Definite Borders",
        "desc":     "Rash has sharp, well-defined borders",
    },
    # Appearance
    "koebner_phenomenon": {
        "label":    "New lesions after skin injury",
        "clinical": "Koebner Phenomenon",
        "desc":     "Lesions appear at sites of skin trauma",
    },
    "polygonal_papules": {
        "label":    "Flat multi-sided bumps",
        "clinical": "Polygonal Papules",
        "desc":     "Flat-topped bumps with angular shape",
    },
    "follicular_papules": {
        "label":    "Bumps around hair follicles",
        "clinical": "Follicular Papules",
        "desc":     "Small raised bumps at hair follicle openings",
    },
    "oral_mucosal_involvement": {
        "label":    "Lesions inside mouth",
        "clinical": "Oral Mucosal Involvement",
        "desc":     "Involvement of oral mucosa or gums",
    },
    # Location
    "knee_and_elbow_involvement": {
        "label":    "Rash on knees or elbows",
        "clinical": "Knee and Elbow Involvement",
        "desc":     "Lesions present on knees or elbows",
    },
    "scalp_involvement": {
        "label":    "Scalp involvement",
        "clinical": "Scalp Involvement",
        "desc":     "Rash or scaling present on scalp",
    },
    # Patient Info
    "family_history": {
        "label":    "Family history of skin disease",
        "clinical": "Family History",
        "desc":     "First-degree relative with similar condition",
    },
    "age": {
        "label":    "Age",
        "clinical": "Age",
        "desc":     "Patient age in years",
    },
    # Histopathology
    "melanin_incontinence": {
        "label":    "Melanin leakage",
        "clinical": "Melanin Incontinence",
        "desc":     "Melanin pigment in dermis on biopsy",
    },
    "eosinophils_in_the_infiltrate": {
        "label":    "Eosinophils in tissue",
        "clinical": "Eosinophils in Infiltrate",
        "desc":     "Eosinophil cells present in skin infiltrate",
    },
    "PNL_infiltrate": {
        "label":    "Neutrophil infiltration",
        "clinical": "PNL Infiltrate",
        "desc":     "Polymorphonuclear leukocytes in dermis",
    },
    "fibrosis_of_the_papillary_dermis": {
        "label":    "Dermal fibrosis",
        "clinical": "Fibrosis of Papillary Dermis",
        "desc":     "Fibrosis in the papillary dermis layer",
    },
    "exocytosis": {
        "label":    "Immune cells in epidermis",
        "clinical": "Exocytosis",
        "desc":     "Lymphocytes migrating into epidermis",
    },
    "acanthosis": {
        "label":    "Skin thickening",
        "clinical": "Acanthosis",
        "desc":     "Epidermal thickening on biopsy",
    },
    "hyperkeratosis": {
        "label":    "Excess keratin layer",
        "clinical": "Hyperkeratosis",
        "desc":     "Thickened outer skin layer",
    },
    "parakeratosis": {
        "label":    "Abnormal keratin cells",
        "clinical": "Parakeratosis",
        "desc":     "Nuclei retained in stratum corneum",
    },
    "clubbing_of_the_rete_ridges": {
        "label":    "Clubbed rete ridges",
        "clinical": "Clubbing of Rete Ridges",
        "desc":     "Bulbous widening of epidermal rete ridges",
    },
    "elongation_of_the_rete_ridges": {
        "label":    "Elongated rete ridges",
        "clinical": "Elongation of Rete Ridges",
        "desc":     "Downward extension of epidermal ridges",
    },
    "thinning_of_the_suprapapillary_epidermis": {
        "label":    "Thinned epidermis over papillae",
        "clinical": "Thinning of Suprapapillary Epidermis",
        "desc":     "Reduced epidermal thickness above dermal papillae",
    },
    "spongiform_pustule": {
        "label":    "Spongiform pustule",
        "clinical": "Spongiform Pustule",
        "desc":     "Neutrophil-filled spaces in epidermis",
    },
    "munro_microabcess": {
        "label":    "Munro microabscess",
        "clinical": "Munro Microabscess",
        "desc":     "Neutrophil clusters in stratum corneum",
    },
    "focal_hypergranulosis": {
        "label":    "Focal granular layer thickening",
        "clinical": "Focal Hypergranulosis",
        "desc":     "Localized increase in granular cell layer",
    },
    "disappearance_of_the_granular_layer": {
        "label":    "Loss of granular layer",
        "clinical": "Disappearance of Granular Layer",
        "desc":     "Absent or reduced stratum granulosum",
    },
    "vacuolisation_and_damage_of_basal_layer": {
        "label":    "Basal layer damage",
        "clinical": "Vacuolisation and Damage of Basal Layer",
        "desc":     "Vacuolar degeneration of basal keratinocytes",
    },
    "spongiosis": {
        "label":    "Intercellular fluid accumulation",
        "clinical": "Spongiosis",
        "desc":     "Edema between epidermal cells",
    },
    "saw_tooth_appearance_of_retes": {
        "label":    "Saw-tooth rete pattern",
        "clinical": "Saw-Tooth Appearance of Retes",
        "desc":     "Irregular pointed rete ridge pattern",
    },
    "follicular_horn_plug": {
        "label":    "Follicular horn plug",
        "clinical": "Follicular Horn Plug",
        "desc":     "Keratin plug within hair follicle opening",
    },
    "perifollicular_parakeratosis": {
        "label":    "Abnormal cells around follicles",
        "clinical": "Perifollicular Parakeratosis",
        "desc":     "Parakeratosis surrounding hair follicles",
    },
    "inflammatory_monoluclear_infiltrate": {
        "label":    "Mononuclear cell infiltrate",
        "clinical": "Inflammatory Mononuclear Infiltrate",
        "desc":     "Lymphocytes and monocytes in dermis",
    },
    "band_like_infiltrate": {
        "label":    "Band-like dermal infiltrate",
        "clinical": "Band-Like Infiltrate",
        "desc":     "Dense inflammatory band at dermal-epidermal junction",
    },
}


# ---------------------------------------------------------------------------
# Feature input renderer
# ---------------------------------------------------------------------------

def render_feature_input(
    feature_key: str,
    schema_entry: dict,
    default_value: int = 0,
) -> int:
    """
    Render a single feature input widget using user-friendly labels.

    For 0-3 range features: renders st.select_slider with severity labels.
    Returns the numeric integer value (0-3) for model input.

    Args:
        feature_key  : Original dataset feature name (used as widget key).
        schema_entry : Feature schema dict with min/max.
        default_value: Pre-filled value (e.g. from image analysis).

    Returns:
        Integer value in [min, max] for model input.
    """
    ui = FEATURE_UI_CONFIG.get(feature_key, {"label": feature_key, "clinical": feature_key, "desc": ""})
    label    = ui["label"]
    clinical = ui.get("clinical", "")
    desc     = ui["desc"]
    min_v = schema_entry["min"]
    max_v = schema_entry["max"]

    # Dual-label display: friendly name (bold) + clinical name (small gray) + description
    st.markdown(
        f'<div class="feat-label">'
        f'{label}'
        f'<span class="feat-clinical">{clinical}</span>'
        f'</div>'
        f'<div class="feat-desc">{desc}</div>',
        unsafe_allow_html=True,
    )

    if min_v == 0 and max_v == 3:
        # Severity scale: None | Mild | Moderate | Severe
        default_label = INT_TO_SEVERITY.get(default_value, "None")
        selected = st.select_slider(
            label,
            options=SEVERITY_OPTIONS,
            value=default_label,
            key=f"feat_{feature_key}",
            label_visibility="collapsed",
        )
        return SEVERITY_TO_INT[selected]
    else:
        # Fallback for non-0-3 features (e.g. age handled separately)
        val = st.slider(
            label,
            min_value=min_v,
            max_value=max_v,
            value=default_value,
            key=f"feat_{feature_key}",
            label_visibility="collapsed",
        )
        return val


# ---------------------------------------------------------------------------
# UI -> model input mapper
# ---------------------------------------------------------------------------

def map_ui_to_model_input(
    ui_values: dict,
    schema_features: list,
) -> dict:
    """
    Transform UI-collected values into model-ready feature dict.

    Ensures:
    - All 34 original feature keys are present
    - Values are integers
    - Order will be enforced by PredictionEngine via schema index

    Args:
        ui_values      : Dict of {feature_key: value} collected from UI widgets.
        schema_features: Ordered list of feature dicts from feature_schema.json.

    Returns:
        Dict mapping original feature_key -> int value, ready for PredictionEngine.
    """
    model_input = {}
    for entry in schema_features:
        key = entry["name"]
        val = ui_values.get(key, 0)
        model_input[key] = int(val)
    return model_input


# ---------------------------------------------------------------------------
# Clinical CSS
# ---------------------------------------------------------------------------

CLINICAL_CSS = """
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 1280px;
}

html, body, [class*="css"] {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #FFFFFF;
    color: #374151;
}

/* -- Top bar -- */
.topbar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 20px 0 16px;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 20px;
}
.topbar-left {}
.topbar-title {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.01em;
    margin: 0;
}
.topbar-subtitle {
    font-size: 13px;
    color: #6B7280;
    margin-top: 2px;
}
.topbar-right {
    display: flex;
    gap: 32px;
    align-items: flex-end;
}
.stat-item {
    text-align: right;
}
.stat-value {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
    display: block;
}
.stat-label {
    font-size: 11px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* -- Disclaimer -- */
.disclaimer-bar {
    border: 1px solid #E5E7EB;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 12px;
    color: #6B7280;
    margin-bottom: 20px;
    background: #F9FAFB;
}

/* -- Panel card -- */
.panel {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 20px;
    height: 100%;
}
.panel-title {
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #E5E7EB;
}

/* -- Upload box -- */
.upload-box {
    border: 1px dashed #D1D5DB;
    border-radius: 4px;
    padding: 24px 16px;
    text-align: center;
    background: #FFFFFF;
    margin-bottom: 12px;
}
.upload-box-text {
    font-size: 13px;
    color: #6B7280;
}

/* -- Image status -- */
.img-status {
    font-size: 12px;
    color: #166534;
    margin-top: 8px;
}
.img-status-error {
    font-size: 12px;
    color: #991B1B;
    margin-top: 8px;
}

/* -- Score row -- */
.score-row {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.score-tag {
    font-size: 12px;
    font-weight: 500;
    color: #374151;
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 3px;
    padding: 3px 10px;
}
.score-tag span {
    color: #1D4ED8;
    font-weight: 600;
}

/* -- Feature group header -- */
.feat-group {
    font-size: 11px;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 16px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #F3F4F6;
}
.feat-group:first-child { margin-top: 0; }

/* -- Slider label row -- */
.slider-label-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 2px;
}
.slider-label {
    font-size: 12px;
    font-weight: 500;
    color: #374151;
}
.slider-value {
    font-size: 12px;
    font-weight: 600;
    color: #1D4ED8;
}

/* -- Button -- */
.stButton > button {
    background: #1D4ED8;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    font-family: system-ui, sans-serif;
    letter-spacing: 0;
    transition: background 0.15s ease;
}
.stButton > button:hover { background: #1E40AF; }
.stButton > button:active { background: #1E3A8A; }

/* -- Slider track -- */
.stSlider > div > div > div { background: #1D4ED8; }

/* -- Selectbox / number input -- */
.stSelectbox > div > div { border-radius: 4px; }
.stNumberInput > div > div > input { border-radius: 4px; }

/* -- Results card -- */
.results-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 20px 24px;
    margin-top: 20px;
}
.results-title {
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #E5E7EB;
}

/* -- Primary diagnosis -- */
.dx-primary {
    margin-bottom: 16px;
}
.dx-label {
    font-size: 11px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.dx-name {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.dx-conf-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 12px 0 6px;
}
.dx-conf-label { font-size: 12px; color: #6B7280; }
.dx-conf-value { font-size: 13px; font-weight: 600; color: #111827; }
.conf-track {
    background: #E5E7EB;
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 8px;
}
.conf-fill {
    height: 100%;
    border-radius: 2px;
    background: #1D4ED8;
}
.conf-status {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 3px;
    display: inline-block;
}
.conf-high   { background: #DCFCE7; color: #166534; }
.conf-medium { background: #FEF3C7; color: #92400E; }
.conf-low    { background: #FEE2E2; color: #991B1B; }

/* -- Top-3 table -- */
.dx-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4px;
}
.dx-table th {
    font-size: 10px;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    text-align: left;
    padding: 0 0 8px;
    border-bottom: 1px solid #E5E7EB;
}
.dx-table th:last-child { text-align: right; }
.dx-table td {
    font-size: 13px;
    color: #374151;
    padding: 9px 0;
    border-bottom: 1px solid #F3F4F6;
    vertical-align: middle;
}
.dx-table td:last-child { text-align: right; font-weight: 600; color: #1D4ED8; }
.dx-table tr:last-child td { border-bottom: none; }
.dx-rank {
    font-size: 11px;
    color: #9CA3AF;
    font-weight: 600;
    width: 24px;
    display: inline-block;
}

/* -- Footer -- */
.clinical-footer {
    margin-top: 32px;
    padding: 14px 0;
    border-top: 1px solid #E5E7EB;
    font-size: 11px;
    color: #9CA3AF;
    text-align: center;
}

/* -- Feature label/desc -- */
.feat-label {
    font-size: 13px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 1px;
    margin-top: 12px;
}
.feat-clinical {
    font-size: 11px;
    font-weight: 400;
    color: #9CA3AF;
    margin-left: 6px;
    font-style: normal;
}
.feat-desc {
    font-size: 11px;
    color: #9CA3AF;
    margin-bottom: 2px;
    line-height: 1.4;
}
/* Tighten select_slider spacing */
.stSlider, [data-testid="stSlider"] {
    margin-bottom: 4px;
}
"""


# ---------------------------------------------------------------------------
# Results renderer -- clinical version
# ---------------------------------------------------------------------------

def render_results(result: dict) -> None:
    """
    Display prediction results with clinical dashboard UI:
    primary diagnosis card with confidence bar, differential diagnosis table,
    and a Plotly horizontal bar chart for probability distribution.
    """
    conf_pct = result["confidence"] * 100
    conf_level = _confidence_color(result["confidence"])
    conf_class = {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}[conf_level]
    conf_text  = {"high": "High", "medium": "Moderate", "low": "Low"}[conf_level]

    # Build top-3 table rows
    rows_html = ""
    for i, item in enumerate(result["top_3"]):
        pct = item["confidence"] * 100
        rows_html += f"""
        <tr>
            <td><span class="dx-rank">{i+1}</span>{item['disease']}</td>
            <td>{pct:.1f}%</td>
        </tr>"""

    # Build Plotly chart
    sorted_probs = sorted(result["all_probs"].items(), key=lambda x: x[1])
    diseases = [d for d, _ in sorted_probs]
    probs = [p * 100 for _, p in sorted_probs]
    colors = ["#DBEAFE"] * len(diseases)
    if diseases:
        colors[-1] = "#1D4ED8"  # highlight top disease

    fig = go.Figure(go.Bar(
        x=probs, y=diseases, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#374151"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=56, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   range=[0, max(probs) * 1.3] if probs else [0, 100]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#374151")),
        height=220, showlegend=False,
    )

    st.markdown(f"""
<div class="results-card">
    <div class="results-title">Diagnostic Output</div>
    <div style="display:flex; gap:32px; align-items:flex-start;">
        <div style="flex:1;">
            <div class="dx-primary">
                <div class="dx-label">Primary Classification</div>
                <div class="dx-name">{result['prediction']}</div>
            </div>
            <div class="dx-conf-row">
                <span class="dx-conf-label">Confidence</span>
                <span class="dx-conf-value">{conf_pct:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct:.1f}%;"></div>
            </div>
            <span class="conf-status {conf_class}">{conf_text}</span>
        </div>
        <div style="flex:1;">
            <table class="dx-table">
                <thead>
                    <tr>
                        <th>Differential Diagnosis</th>
                        <th>Probability</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Probability chart below
    st.markdown(
        '<div style="margin-top:16px;font-size:11px;font-weight:600;color:#9CA3AF;'
        'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">'
        'Probability Distribution</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main Streamlit app -- clinical dashboard
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="DermaDx AI -- Clinical Decision Support",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{CLINICAL_CSS}</style>", unsafe_allow_html=True)

    # Top bar
    st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-title">AI Skin Disease Diagnosis</div>
        <div class="topbar-subtitle">Clinical decision support tool for dermatological classification</div>
    </div>
    <div class="topbar-right">
        <div class="stat-item"><span class="stat-value">98.6%</span><span class="stat-label">Accuracy</span></div>
        <div class="stat-item"><span class="stat-value">34</span><span class="stat-label">Features</span></div>
        <div class="stat-item"><span class="stat-value">6</span><span class="stat-label">Classes</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="disclaimer-bar">
    For educational and research use only. Not a substitute for professional medical advice or clinical diagnosis.
</div>
""", unsafe_allow_html=True)

    model, schema = load_artifacts()
    schema_features = sorted(schema["features"], key=lambda x: x["index"])
    engine = PredictionEngine(model, schema)

    # Build a lookup dict: feature_name -> schema_entry (for render_feature_input)
    schema_lookup = {f["name"]: f for f in schema_features}

    if "erythema_from_image" not in st.session_state:
        st.session_state["erythema_from_image"] = None
    if "scaling_from_image" not in st.session_state:
        st.session_state["scaling_from_image"] = None

    col_left, col_right = st.columns([1, 1.6])
    feature_values = {}

    # LEFT PANEL
    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Image Input</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload skin image",
            type=["jpg", "jpeg", "png"],
            help="JPEG or PNG. Used only to estimate skin redness and flaking scores.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            st.image(image_bytes, use_column_width=True)
            try:
                with st.spinner("Extracting features..."):
                    erythema_score, scaling_score = ImageAnalyzer.analyze(image_bytes)
                st.session_state["erythema_from_image"] = erythema_score
                st.session_state["scaling_from_image"] = scaling_score
                st.markdown(
                    f'<div class="img-status">Image analyzed. Skin redness and flaking scores pre-filled.</div>'
                    f'<div class="score-row">'
                    f'<span class="score-tag">Skin Redness <span>{erythema_score}/3</span></span>'
                    f'<span class="score-tag">Skin Flaking <span>{scaling_score}/3</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except ValueError as e:
                st.markdown(f'<div class="img-status-error">{e}</div>', unsafe_allow_html=True)
                st.session_state["erythema_from_image"] = None
                st.session_state["scaling_from_image"] = None
        else:
            st.markdown("""
<div class="upload-box">
    <div class="upload-box-text">Select a JPEG or PNG image<br>
    <span style="font-size:11px;color:#9CA3AF;">Used only to estimate skin redness and flaking scores</span></div>
</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT PANEL
    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Clinical Features</div>', unsafe_allow_html=True)

        def get_default(name):
            if name == "erythema" and st.session_state["erythema_from_image"] is not None:
                return st.session_state["erythema_from_image"]
            if name == "scaling" and st.session_state["scaling_from_image"] is not None:
                return st.session_state["scaling_from_image"]
            return 0

        # Group: Symptoms
        st.markdown('<div class="feat-group">Symptoms</div>', unsafe_allow_html=True)
        symptoms = ["erythema", "scaling", "itching", "definite_borders"]
        c1, c2 = st.columns(2)
        for i, name in enumerate(symptoms):
            col = c1 if i % 2 == 0 else c2
            with col:
                feature_values[name] = render_feature_input(
                    name, schema_lookup[name], get_default(name)
                )

        # Group: Appearance
        st.markdown('<div class="feat-group">Appearance</div>', unsafe_allow_html=True)
        appearance = ["koebner_phenomenon", "polygonal_papules", "follicular_papules", "oral_mucosal_involvement"]
        c3, c4 = st.columns(2)
        for i, name in enumerate(appearance):
            col = c3 if i % 2 == 0 else c4
            with col:
                feature_values[name] = render_feature_input(name, schema_lookup[name])

        # Group: Location
        st.markdown('<div class="feat-group">Location</div>', unsafe_allow_html=True)
        location = ["knee_and_elbow_involvement", "scalp_involvement"]
        c5, c6 = st.columns(2)
        for i, name in enumerate(location):
            col = c5 if i % 2 == 0 else c6
            with col:
                feature_values[name] = render_feature_input(name, schema_lookup[name])

        # Group: Patient Info
        st.markdown('<div class="feat-group">Patient Info</div>', unsafe_allow_html=True)
        pi_c1, pi_c2 = st.columns(2)
        with pi_c1:
            st.markdown(
                '<div class="feat-label">Family history of skin disease'
                '<span class="feat-clinical">Family History</span></div>'
                '<div class="feat-desc">First-degree relative with similar condition</div>',
                unsafe_allow_html=True,
            )
            fh = st.selectbox(
                "Family History of Skin Disease",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                key="feat_family_history",
                label_visibility="collapsed",
            )
            feature_values["family_history"] = fh
        with pi_c2:
            st.markdown(
                '<div class="feat-label">Age'
                '<span class="feat-clinical">Age</span></div>'
                '<div class="feat-desc">Patient age in years</div>',
                unsafe_allow_html=True,
            )
            age_val = st.number_input(
                "Age",
                min_value=0, max_value=120, value=30, step=1,
                key="feat_age",
                label_visibility="collapsed",
            )
            feature_values["age"] = age_val

        # Group: Histopathological Features (expander)
        histopath_names = [
            "melanin_incontinence", "eosinophils_in_the_infiltrate", "PNL_infiltrate",
            "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
            "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
            "elongation_of_the_rete_ridges", "thinning_of_the_suprapapillary_epidermis",
            "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
            "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer",
            "spongiosis", "saw_tooth_appearance_of_retes", "follicular_horn_plug",
            "perifollicular_parakeratosis", "inflammatory_monoluclear_infiltrate",
            "band_like_infiltrate",
        ]
        with st.expander("Histopathological Features", expanded=False):
            st.caption("Biopsy-derived features. Leave at None if unavailable.")
            h1, h2 = st.columns(2)
            for i, name in enumerate(histopath_names):
                col = h1 if i % 2 == 0 else h2
                with col:
                    feature_values[name] = render_feature_input(name, schema_lookup[name])

        st.markdown('</div>', unsafe_allow_html=True)

    # Run Diagnosis button
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)
    if st.button("Run Diagnosis", use_container_width=True):
        # Map UI values to model input (original feature keys, integer values)
        model_input = map_ui_to_model_input(feature_values, schema_features)
        errors = validate_inputs(model_input, schema_features)
        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Running classification..."):
                try:
                    result = engine.predict(model_input)
                    st.session_state["result"] = result
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    st.session_state.pop("result", None)
    st.markdown('</div>', unsafe_allow_html=True)

    if "result" in st.session_state:
        render_results(st.session_state["result"])

    st.markdown("""
<div class="clinical-footer">
    For educational use only &nbsp;&middot;&nbsp; DermaDx AI &nbsp;&middot;&nbsp; XGBoost classifier trained on UCI Dermatology dataset
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
