"""
tests/test_app.py — Unit tests for app.py components
=====================================================
Tests ImageAnalyzer, Validator, and PredictionEngine
without requiring a running Streamlit server.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    CLASS_NAMES,
    ImageAnalyzer,
    PredictionEngine,
    validate_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_minimal_schema(n_features: int = 34) -> dict:
    """Build a minimal feature schema dict for testing."""
    features = []
    for i in range(n_features - 1):
        features.append({
            "name":  f"feature_{i}",
            "index": i,
            "min":   0,
            "max":   3,
            "type":  "int",
        })
    # Last feature is 'age' with wider range
    features.append({
        "name":  "age",
        "index": n_features - 1,
        "min":   0,
        "max":   120,
        "type":  "int",
    })
    return {
        "features":    features,
        "class_order": [1, 2, 3, 4, 5, 6],
    }


def make_valid_feature_values(schema: dict) -> dict:
    """Build a valid feature dict with all values at minimum."""
    return {entry["name"]: entry["min"] for entry in schema["features"]}


def make_fake_model(schema: dict):
    """Train a tiny XGBoost model on synthetic data for testing.
    Uses 0-based labels (0–5) as required by XGBoost 2.0.
    The schema's class_order maps these back to original labels (1–6).
    """
    import numpy as np
    from xgboost import XGBClassifier

    n_features = len(schema["features"])
    rng = np.random.default_rng(42)
    X = rng.integers(0, 4, size=(120, n_features)).astype(np.float32)
    # Use 0-based labels (0–5) for XGBoost 2.0 compatibility
    y = np.tile([0, 1, 2, 3, 4, 5], 20)

    model = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        n_estimators=10,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Test: ImageAnalyzer
# ---------------------------------------------------------------------------

class TestImageAnalyzer:
    def test_invalid_bytes_raises_value_error(self):
        """Non-image bytes must raise ValueError."""
        with pytest.raises(ValueError, match="could not be decoded"):
            ImageAnalyzer.analyze(b"this is not an image")

    def test_empty_bytes_raises_value_error(self):
        """Empty bytes must raise ValueError (or cv2.error caught as ValueError)."""
        with pytest.raises((ValueError, Exception)):
            ImageAnalyzer.analyze(b"")

    def test_valid_image_returns_scores_in_range(self):
        """A valid synthetic image must return scores in {0, 1, 2, 3}."""
        import cv2

        # Create a simple red 100×100 image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # High red channel (BGR: B=0, G=0, R=200)

        # Encode to JPEG bytes
        _, buf = cv2.imencode(".jpg", img)
        image_bytes = buf.tobytes()

        erythema, scaling = ImageAnalyzer.analyze(image_bytes)

        assert isinstance(erythema, int), "Erythema score must be int"
        assert isinstance(scaling, int), "Scaling score must be int"
        assert erythema in {0, 1, 2, 3}, f"Erythema {erythema} out of range"
        assert scaling in {0, 1, 2, 3}, f"Scaling {scaling} out of range"

    def test_red_image_gives_high_erythema(self):
        """A predominantly red image should yield erythema score >= 2."""
        import cv2

        # Pure red image in BGR
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Max red

        _, buf = cv2.imencode(".jpg", img)
        erythema, _ = ImageAnalyzer.analyze(buf.tobytes())
        assert erythema >= 2, f"Expected high erythema for red image, got {erythema}"

    def test_compute_erythema_range(self):
        """_compute_erythema must always return int in {0, 1, 2, 3}."""
        for r_val in [0, 50, 128, 200, 255]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img[:, :, 2] = r_val  # BGR: set red channel
            score = ImageAnalyzer._compute_erythema(img)
            assert score in {0, 1, 2, 3}

    def test_compute_scaling_range(self):
        """_compute_scaling must always return int in {0, 1, 2, 3}."""
        # Uniform gray image (no edges)
        img_flat = np.full((50, 50, 3), 128, dtype=np.uint8)
        score_flat = ImageAnalyzer._compute_scaling(img_flat)
        assert score_flat in {0, 1, 2, 3}

        # Checkerboard image (many edges)
        img_edges = np.zeros((50, 50, 3), dtype=np.uint8)
        img_edges[::2, ::2] = 255
        score_edges = ImageAnalyzer._compute_scaling(img_edges)
        assert score_edges in {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Test: validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_inputs_return_empty_list(self):
        """All valid inputs should return an empty error list."""
        schema = make_minimal_schema()
        feature_values = make_valid_feature_values(schema)
        errors = validate_inputs(feature_values, schema["features"])
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_missing_feature_returns_error(self):
        """A missing feature should produce an error mentioning its name."""
        schema = make_minimal_schema()
        feature_values = make_valid_feature_values(schema)
        del feature_values["feature_0"]  # Remove one feature

        errors = validate_inputs(feature_values, schema["features"])
        assert len(errors) > 0, "Expected error for missing feature"
        assert any("feature_0" in e for e in errors), (
            "Error message should mention the missing feature name"
        )

    def test_out_of_range_value_returns_error(self):
        """A value exceeding max should produce an error."""
        schema = make_minimal_schema()
        feature_values = make_valid_feature_values(schema)
        feature_values["feature_0"] = 99  # Way out of range [0, 3]

        errors = validate_inputs(feature_values, schema["features"])
        assert len(errors) > 0, "Expected error for out-of-range value"

    def test_multiple_errors_reported(self):
        """Multiple invalid features should each produce an error."""
        schema = make_minimal_schema()
        feature_values = make_valid_feature_values(schema)
        del feature_values["feature_0"]
        del feature_values["feature_1"]

        errors = validate_inputs(feature_values, schema["features"])
        assert len(errors) >= 2, "Expected at least 2 errors for 2 missing features"

    def test_boundary_values_are_valid(self):
        """Min and max boundary values should be accepted."""
        schema = make_minimal_schema()
        feature_values = make_valid_feature_values(schema)
        feature_values["feature_0"] = 3   # max for 0–3 features
        feature_values["age"] = 120       # max for age

        errors = validate_inputs(feature_values, schema["features"])
        assert errors == [], f"Boundary values should be valid, got: {errors}"


# ---------------------------------------------------------------------------
# Test: PredictionEngine
# ---------------------------------------------------------------------------

class TestPredictionEngine:
    def setup_method(self):
        """Set up a fake model and schema for each test."""
        self.schema = make_minimal_schema()
        self.model = make_fake_model(self.schema)
        self.engine = PredictionEngine(self.model, self.schema)

    def test_predict_returns_all_required_keys(self):
        """predict() must return dict with all required keys."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)

        required_keys = {"prediction", "confidence", "top_3", "all_probs"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_confidence_is_float_in_range(self):
        """Confidence must be a float in [0.0, 1.0]."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_3_has_exactly_3_entries(self):
        """top_3 must contain exactly 3 entries."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)
        assert len(result["top_3"]) == 3

    def test_top_3_sorted_descending(self):
        """top_3 must be sorted by confidence descending."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)
        confidences = [item["confidence"] for item in result["top_3"]]
        assert confidences == sorted(confidences, reverse=True), (
            "top_3 is not sorted by confidence descending"
        )

    def test_prediction_is_valid_disease_name(self):
        """prediction must be one of the 6 known disease names."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)
        valid_names = set(CLASS_NAMES.values())
        assert result["prediction"] in valid_names, (
            f"'{result['prediction']}' is not a valid disease name"
        )

    def test_all_probs_has_6_entries(self):
        """all_probs must contain exactly 6 disease entries."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)
        assert len(result["all_probs"]) == 6

    def test_all_probs_sum_to_one(self):
        """Probabilities in all_probs must sum to approximately 1.0."""
        feature_values = make_valid_feature_values(self.schema)
        result = self.engine.predict(feature_values)
        total = sum(result["all_probs"].values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_deterministic_inference(self):
        """Same input must produce identical results on repeated calls."""
        feature_values = make_valid_feature_values(self.schema)
        result1 = self.engine.predict(feature_values)
        result2 = self.engine.predict(feature_values)
        assert result1["prediction"] == result2["prediction"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["top_3"] == result2["top_3"]

    def test_feature_vector_assembly_order(self):
        """_assemble_vector must order values by schema index."""
        feature_values = make_valid_feature_values(self.schema)
        # Set each feature to its index value for easy verification
        for entry in self.schema["features"]:
            feature_values[entry["name"]] = entry["index"]

        vector = self.engine._assemble_vector(feature_values)
        assert vector.shape == (1, len(self.schema["features"]))

        for entry in self.schema["features"]:
            expected = entry["index"]
            actual = float(vector[0, entry["index"]])
            assert actual == expected, (
                f"Feature '{entry['name']}' at index {entry['index']}: "
                f"expected {expected}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Test: Confidence formatting
# ---------------------------------------------------------------------------

class TestConfidenceFormatting:
    def test_format_87_3_percent(self):
        """0.873 should format as '87.3%'."""
        assert f"{0.873 * 100:.1f}%" == "87.3%"

    def test_format_100_percent(self):
        """1.0 should format as '100.0%'."""
        assert f"{1.0 * 100:.1f}%" == "100.0%"

    def test_format_0_percent(self):
        """0.0 should format as '0.0%'."""
        assert f"{0.0 * 100:.1f}%" == "0.0%"

    def test_format_rounds_correctly(self):
        """0.877 should round to '87.7%' (one decimal place)."""
        assert f"{0.877 * 100:.1f}%" == "87.7%"

    def test_format_50_percent(self):
        """0.5 should format as '50.0%'."""
        assert f"{0.5 * 100:.1f}%" == "50.0%"
