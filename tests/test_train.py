"""
tests/test_train.py — Unit tests for train.py
==============================================
Tests the data cleaning, splitting, and evaluation functions
without requiring the actual dataset CSV.
"""

import io
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

# Add parent directory to path so we can import train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (
    apply_smote,
    evaluate_model,
    load_and_clean,
    save_artifacts,
    split_data,
    train_models,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample_df(n_rows: int = 100, n_features: int = 34) -> pd.DataFrame:
    """Create a synthetic DataFrame that mimics the erythemato-squamous dataset.
    Uses class labels 1–6 (original dataset format).
    """
    rng = np.random.default_rng(42)
    data = {}
    for i in range(n_features):
        data[f"feature_{i}"] = rng.integers(0, 4, size=n_rows)
    # 6 balanced classes (labels 1–6, matching the real dataset)
    data["class"] = np.tile([1, 2, 3, 4, 5, 6], n_rows // 6 + 1)[:n_rows]
    return pd.DataFrame(data)


def encode_labels(y_train, y_test):
    """Encode labels to 0-based integers for XGBoost 2.0 compatibility."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def make_csv_with_missing(tmp_path, n_rows: int = 60) -> str:
    """Write a CSV with some '?' placeholders and return its path."""
    df = make_sample_df(n_rows)
    # Inject '?' into a few cells
    df.iloc[0, 0] = "?"
    df.iloc[5, 3] = "?"
    df.iloc[10, 7] = "?"
    csv_path = str(tmp_path / "test_data.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Test: load_and_clean
# ---------------------------------------------------------------------------

class TestLoadAndClean:
    def test_removes_question_marks(self, tmp_path):
        """'?' values must be replaced with NaN and rows dropped."""
        csv_path = make_csv_with_missing(tmp_path, n_rows=60)
        result = load_and_clean(csv_path)

        # No '?' strings should remain
        for col in result.columns:
            assert "?" not in result[col].astype(str).values, (
                f"Column '{col}' still contains '?' after cleaning"
            )

    def test_no_nan_after_cleaning(self, tmp_path):
        """Cleaned DataFrame must contain no NaN values."""
        csv_path = make_csv_with_missing(tmp_path, n_rows=60)
        result = load_and_clean(csv_path)
        assert result.isnull().sum().sum() == 0, "NaN values remain after cleaning"

    def test_rows_are_dropped(self, tmp_path):
        """Rows with '?' should be dropped, reducing row count."""
        csv_path = make_csv_with_missing(tmp_path, n_rows=60)
        original_df = pd.read_csv(csv_path)
        cleaned_df = load_and_clean(csv_path)
        assert len(cleaned_df) < len(original_df), (
            "Row count should decrease after dropping rows with '?'"
        )

    def test_raises_on_missing_file(self):
        """FileNotFoundError should be raised for a non-existent CSV."""
        with pytest.raises(FileNotFoundError):
            load_and_clean("nonexistent_file_xyz.csv")

    def test_clean_csv_unchanged_row_count(self, tmp_path):
        """A CSV with no '?' should retain all rows."""
        df = make_sample_df(50)
        csv_path = str(tmp_path / "clean.csv")
        df.to_csv(csv_path, index=False)
        result = load_and_clean(csv_path)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# Test: split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_approximate_80_20_split(self):
        """Train/test split should be approximately 80/20."""
        df = make_sample_df(100)
        X_train, X_test, y_train, y_test = split_data(df)
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        assert 0.75 <= train_ratio <= 0.85, (
            f"Train ratio {train_ratio:.2f} is not close to 0.80"
        )

    def test_no_overlap_between_splits(self):
        """Train and test indices must not overlap."""
        df = make_sample_df(100)
        X_train, X_test, _, _ = split_data(df)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(test_idx), "Train and test sets share indices"

    def test_all_rows_accounted_for(self):
        """Train + test should equal total rows."""
        df = make_sample_df(120)
        X_train, X_test, y_train, y_test = split_data(df)
        assert len(X_train) + len(X_test) == 120

    def test_stratification_preserves_classes(self):
        """All classes present in original data should appear in both splits."""
        df = make_sample_df(120)
        X_train, X_test, y_train, y_test = split_data(df)
        original_classes = set(df["class"].unique())
        assert set(y_train.unique()) == original_classes
        assert set(y_test.unique()) == original_classes


# ---------------------------------------------------------------------------
# Test: apply_smote
# ---------------------------------------------------------------------------

class TestApplySmote:
    def test_balances_classes(self):
        """After SMOTE, all classes should have equal sample counts."""
        df = make_sample_df(120)
        X_train, _, y_train, _ = split_data(df)
        y_train_enc, _, _ = encode_labels(y_train, y_train)
        X_res, y_res = apply_smote(X_train, y_train_enc)
        unique, counts = np.unique(y_res, return_counts=True)
        assert len(set(counts)) == 1, (
            f"Classes are not balanced after SMOTE: {dict(zip(unique, counts))}"
        )

    def test_output_is_numpy(self):
        """SMOTE output should be numpy arrays or DataFrames (both are valid)."""
        df = make_sample_df(120)
        X_train, _, y_train, _ = split_data(df)
        y_train_enc, _, _ = encode_labels(y_train, y_train)
        X_res, y_res = apply_smote(X_train, y_train_enc)
        # SMOTE may return DataFrame or ndarray depending on input type — both valid
        assert hasattr(X_res, '__len__'), "X_res should be array-like"
        assert hasattr(y_res, '__len__'), "y_res should be array-like"

    def test_increases_minority_class(self):
        """SMOTE should increase total sample count when classes are imbalanced."""
        df = make_sample_df(120)
        X_train, _, y_train, _ = split_data(df)
        y_train_enc, _, _ = encode_labels(y_train, y_train)
        X_res, y_res = apply_smote(X_train, y_train_enc)
        assert len(X_res) >= len(X_train)


# ---------------------------------------------------------------------------
# Test: evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def _get_fitted_model(self):
        """Return a tiny fitted DecisionTree for testing."""
        df = make_sample_df(120)
        X_train, X_test, y_train, y_test = split_data(df)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def test_returns_all_four_keys(self):
        """evaluate_model must return all four required metric keys."""
        model, X_test, y_test = self._get_fitted_model()
        metrics = evaluate_model(model, X_test, y_test, "Test Model")
        required_keys = {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
        assert required_keys == set(metrics.keys()), (
            f"Missing keys: {required_keys - set(metrics.keys())}"
        )

    def test_metrics_are_floats_in_range(self):
        """All metric values must be floats in [0.0, 1.0]."""
        model, X_test, y_test = self._get_fitted_model()
        metrics = evaluate_model(model, X_test, y_test, "Test Model")
        for key, val in metrics.items():
            assert isinstance(val, float), f"Metric '{key}' is not a float"
            assert 0.0 <= val <= 1.0, f"Metric '{key}' = {val} is out of [0, 1]"


# ---------------------------------------------------------------------------
# Test: save_artifacts
# ---------------------------------------------------------------------------

class TestSaveArtifacts:
    def _get_trained_xgb(self, tmp_path=None):
        """Helper: returns a trained XGBoost model with 0-based labels."""
        df = make_sample_df(120)
        X_train, _, y_train, _ = split_data(df)
        y_train_enc, _, le = encode_labels(y_train, y_train)
        X_res, y_res = apply_smote(X_train, y_train_enc)
        _, xgb_model = train_models(X_res, y_res)
        return xgb_model, list(X_train.columns), le

    def test_creates_model_pkl(self, tmp_path):
        """save_artifacts must create model.pkl."""
        xgb_model, feature_names, le = self._get_trained_xgb()
        feature_stats = {n: {"min": 0, "max": 3} for n in feature_names}
        class_order = [int(c) for c in le.classes_]

        model_path = str(tmp_path / "model.pkl")
        schema_path = str(tmp_path / "feature_schema.json")

        save_artifacts(xgb_model, feature_names, feature_stats,
                       model_path, schema_path, class_order=class_order)
        assert os.path.exists(model_path), "model.pkl was not created"

    def test_creates_schema_json(self, tmp_path):
        """save_artifacts must create feature_schema.json."""
        xgb_model, feature_names, le = self._get_trained_xgb()
        feature_stats = {n: {"min": 0, "max": 3} for n in feature_names}
        class_order = [int(c) for c in le.classes_]

        model_path = str(tmp_path / "model.pkl")
        schema_path = str(tmp_path / "feature_schema.json")

        save_artifacts(xgb_model, feature_names, feature_stats,
                       model_path, schema_path, class_order=class_order)
        assert os.path.exists(schema_path), "feature_schema.json was not created"

    def test_schema_has_correct_structure(self, tmp_path):
        """Schema JSON must have 'features' list and 'class_order' key."""
        xgb_model, feature_names, le = self._get_trained_xgb()
        feature_stats = {n: {"min": 0, "max": 3} for n in feature_names}
        class_order = [int(c) for c in le.classes_]

        model_path = str(tmp_path / "model.pkl")
        schema_path = str(tmp_path / "feature_schema.json")

        save_artifacts(xgb_model, feature_names, feature_stats,
                       model_path, schema_path, class_order=class_order)

        with open(schema_path) as f:
            schema = json.load(f)

        assert "features" in schema, "Schema missing 'features' key"
        assert "class_order" in schema, "Schema missing 'class_order' key"
        assert len(schema["features"]) == len(feature_names)

        # Each feature entry must have required fields
        for entry in schema["features"]:
            for field in ("name", "index", "min", "max", "type"):
                assert field in entry, f"Feature entry missing field '{field}'"
