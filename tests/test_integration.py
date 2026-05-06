"""
tests/test_integration.py — Integration tests for DermaDx Lite
===============================================================
Tests the full pipeline end-to-end using synthetic data:
  train.py pipeline → artifacts on disk → app.py loads and predicts
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (
    apply_smote,
    evaluate_model,
    load_and_clean,
    save_artifacts,
    split_data,
    train_models,
)
from app import (
    CLASS_NAMES,
    ImageAnalyzer,
    PredictionEngine,
    validate_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_csv(tmp_path, n_rows: int = 180) -> str:
    """
    Create a synthetic CSV that mimics the erythemato-squamous dataset.
    Uses 34 features + 1 class column with 6 balanced classes.
    """
    rng = np.random.default_rng(42)

    # 33 features with range 0–3, plus age (0–80)
    feature_cols = [
        "erythema", "scaling", "definite_borders", "itching",
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_and_elbow_involvement",
        "scalp_involvement", "family_history", "melanin_incontinence",
        "eosinophils_in_the_infiltrate", "PNL_infiltrate",
        "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
        "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
        "elongation_of_the_rete_ridges",
        "thinning_of_the_suprapapillary_epidermis",
        "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
        "disappearance_of_the_granular_layer",
        "vacuolisation_and_damage_of_basal_layer", "spongiosis",
        "saw_tooth_appearance_of_retes", "follicular_horn_plug",
        "perifollicular_parakeratosis", "inflammatory_monoluclear_infiltrate",
        "band_like_infiltrate",
    ]

    data = {}
    for col in feature_cols:
        if col == "family_history":
            data[col] = rng.integers(0, 2, size=n_rows)
        else:
            data[col] = rng.integers(0, 4, size=n_rows)

    data["age"] = rng.integers(10, 80, size=n_rows)
    # 6 balanced classes (labels 1–6)
    data["class"] = np.tile([1, 2, 3, 4, 5, 6], n_rows // 6 + 1)[:n_rows]

    df = pd.DataFrame(data)
    csv_path = str(tmp_path / "synthetic_dermatology.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def run_full_pipeline(csv_path: str, tmp_path) -> tuple:
    """Run the complete training pipeline and return (model_path, schema_path)."""
    df = load_and_clean(csv_path)
    X_train, X_test, y_train, y_test = split_data(df)

    feature_names = list(X_train.columns)
    feature_stats = {
        name: {
            "min": int(X_train[name].min()),
            "max": int(X_train[name].max()),
        }
        for name in feature_names
    }
    if "age" in feature_stats:
        feature_stats["age"]["max"] = max(feature_stats["age"]["max"], 120)

    X_res, y_res = apply_smote(X_train, y_train)
    _, xgb_model = train_models(X_res, y_res)

    model_path = str(tmp_path / "model.pkl")
    schema_path = str(tmp_path / "feature_schema.json")
    save_artifacts(xgb_model, feature_names, feature_stats, model_path, schema_path)

    return model_path, schema_path


# ---------------------------------------------------------------------------
# Test 18.1: Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestFullPipelineSmoke:
    def test_pipeline_creates_model_pkl(self, tmp_path):
        """train.py pipeline must create model.pkl."""
        csv_path = make_synthetic_csv(tmp_path)
        model_path, _ = run_full_pipeline(csv_path, tmp_path)
        assert os.path.exists(model_path), "model.pkl was not created"

    def test_pipeline_creates_schema_json(self, tmp_path):
        """train.py pipeline must create feature_schema.json."""
        csv_path = make_synthetic_csv(tmp_path)
        _, schema_path = run_full_pipeline(csv_path, tmp_path)
        assert os.path.exists(schema_path), "feature_schema.json was not created"

    def test_schema_has_34_features(self, tmp_path):
        """feature_schema.json must contain exactly 34 feature entries."""
        csv_path = make_synthetic_csv(tmp_path)
        _, schema_path = run_full_pipeline(csv_path, tmp_path)

        with open(schema_path) as f:
            schema = json.load(f)

        assert len(schema["features"]) == 34, (
            f"Expected 34 features, got {len(schema['features'])}"
        )

    def test_schema_has_class_order(self, tmp_path):
        """feature_schema.json must contain 'class_order' key."""
        csv_path = make_synthetic_csv(tmp_path)
        _, schema_path = run_full_pipeline(csv_path, tmp_path)

        with open(schema_path) as f:
            schema = json.load(f)

        assert "class_order" in schema, "Schema missing 'class_order' key"
        assert len(schema["class_order"]) == 6, (
            f"Expected 6 classes in class_order, got {len(schema['class_order'])}"
        )

    def test_schema_indices_are_unique_and_complete(self, tmp_path):
        """Schema feature indices must form a complete set {0, …, 33}."""
        csv_path = make_synthetic_csv(tmp_path)
        _, schema_path = run_full_pipeline(csv_path, tmp_path)

        with open(schema_path) as f:
            schema = json.load(f)

        indices = {entry["index"] for entry in schema["features"]}
        expected = set(range(34))
        assert indices == expected, (
            f"Schema indices {indices} != expected {expected}"
        )


# ---------------------------------------------------------------------------
# Test 18.2: App startup with artifacts
# ---------------------------------------------------------------------------

class TestAppStartupWithArtifacts:
    def test_load_artifacts_returns_model_and_schema(self, tmp_path):
        """load_artifacts() equivalent: loading model.pkl and schema must succeed."""
        import joblib
        from xgboost import XGBClassifier

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        # Simulate what load_artifacts() does
        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        assert isinstance(model, XGBClassifier), "Loaded model is not XGBClassifier"
        assert isinstance(schema["features"], list), "Schema features is not a list"
        assert len(schema["features"]) == 34

    def test_prediction_engine_initializes(self, tmp_path):
        """PredictionEngine must initialize without error from real artifacts."""
        import joblib

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        engine = PredictionEngine(model, schema)
        assert engine is not None
        assert len(engine.schema_features) == 34


# ---------------------------------------------------------------------------
# Test 18.3: End-to-end prediction
# ---------------------------------------------------------------------------

class TestEndToPrediction:
    def test_full_flow_returns_valid_result(self, tmp_path):
        """Full flow: train → load → predict must return valid result structure."""
        import joblib

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        engine = PredictionEngine(model, schema)

        # Build a valid feature dict using schema minimums
        feature_values = {
            entry["name"]: entry["min"]
            for entry in schema["features"]
        }

        result = engine.predict(feature_values)

        # Validate result structure
        assert "prediction" in result
        assert "confidence" in result
        assert "top_3" in result
        assert "all_probs" in result

    def test_prediction_is_known_disease(self, tmp_path):
        """Predicted disease must be one of the 6 known disease names."""
        import joblib

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        engine = PredictionEngine(model, schema)
        feature_values = {entry["name"]: entry["min"] for entry in schema["features"]}
        result = engine.predict(feature_values)

        valid_names = set(CLASS_NAMES.values())
        assert result["prediction"] in valid_names, (
            f"'{result['prediction']}' is not a valid disease name"
        )

    def test_confidence_in_valid_range(self, tmp_path):
        """Confidence must be a float in [0.0, 1.0]."""
        import joblib

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        engine = PredictionEngine(model, schema)
        feature_values = {entry["name"]: entry["min"] for entry in schema["features"]}
        result = engine.predict(feature_values)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_3_exactly_3_entries(self, tmp_path):
        """top_3 must always contain exactly 3 entries."""
        import joblib

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        engine = PredictionEngine(model, schema)
        feature_values = {entry["name"]: entry["min"] for entry in schema["features"]}
        result = engine.predict(feature_values)

        assert len(result["top_3"]) == 3

    def test_validation_passes_for_valid_inputs(self, tmp_path):
        """validate_inputs must return no errors for a valid feature dict."""
        csv_path = make_synthetic_csv(tmp_path)
        _, schema_path = run_full_pipeline(csv_path, tmp_path)

        with open(schema_path) as f:
            schema = json.load(f)

        feature_values = {
            entry["name"]: entry["min"]
            for entry in schema["features"]
        }

        errors = validate_inputs(feature_values, schema["features"])
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_deterministic_predictions(self, tmp_path):
        """Same inputs must produce identical predictions on repeated calls."""
        import joblib

        csv_path = make_synthetic_csv(tmp_path)
        model_path, schema_path = run_full_pipeline(csv_path, tmp_path)

        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)

        engine = PredictionEngine(model, schema)
        feature_values = {entry["name"]: entry["min"] for entry in schema["features"]}

        result1 = engine.predict(feature_values)
        result2 = engine.predict(feature_values)

        assert result1["prediction"] == result2["prediction"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["top_3"] == result2["top_3"]
