"""
train.py — DermaDx Lite Training Pipeline
==========================================
Trains a Decision Tree (baseline) and XGBoost (primary) classifier on the
erythemato-squamous disease dataset, then saves:
  - model.pkl          : serialized XGBoost model
  - feature_schema.json: feature definitions (name, index, min, max, type)
                         plus class_order for correct label mapping at inference

Usage:
    python train.py                          # uses default 'dermatology.csv'
    python train.py --csv path/to/data.csv   # custom CSV path
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Fixed random seed for full reproducibility
RANDOM_STATE = 42

# Target column name in the CSV
TARGET_COL = "class"


# ---------------------------------------------------------------------------
# Step 1: Load and clean the dataset
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV, replace '?' placeholders with NaN, and drop incomplete rows.

    Args:
        csv_path: Path to the erythemato-squamous CSV file.

    Returns:
        Cleaned DataFrame with no NaN values.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Please provide the correct path with --csv."
        )

    df = pd.read_csv(csv_path)

    # Replace '?' placeholders (common in UCI datasets) with NaN
    df = df.replace("?", np.nan)

    # Convert all columns to numeric where possible (some may be read as str)
    for col in df.columns:
        if col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)

    print(f"[Data] Loaded {rows_before} rows from '{csv_path}'")
    print(f"[Data] Dropped {rows_dropped} rows with missing values")
    print(f"[Data] Clean dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"[Data] Class distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")

    return df


# ---------------------------------------------------------------------------
# Step 2: Stratified train/test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """
    Perform a stratified 80/20 train/test split.

    Args:
        df           : Cleaned DataFrame.
        target_col   : Name of the target column.
        test_size    : Fraction of data for the test set (default 0.2).
        random_state : Random seed for reproducibility.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) as DataFrames/Series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"[Split] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 3: SMOTE oversampling (training split only — no test leakage)
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
):
    """
    Apply SMOTE to balance class distribution in the training split.
    SMOTE is NEVER applied to the test split to prevent data leakage.

    Args:
        X_train      : Training features.
        y_train      : Training labels.
        random_state : Random seed for reproducibility.

    Returns:
        Tuple (X_resampled, y_resampled) as numpy arrays.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"[SMOTE] Before: {len(X_train)} samples | After: {len(X_res)} samples")
    unique, counts = np.unique(y_res, return_counts=True)
    print(f"[SMOTE] Balanced class counts: {dict(zip(unique, counts))}\n")

    return X_res, y_res


# ---------------------------------------------------------------------------
# Step 4: Train both models
# ---------------------------------------------------------------------------

def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
):
    """
    Train a Decision Tree (baseline) and XGBoost (primary) classifier.

    Args:
        X_train      : SMOTE-balanced training features.
        y_train      : SMOTE-balanced training labels.
        random_state : Random seed for reproducibility.

    Returns:
        Tuple (dt_model, xgb_model).
    """
    print("[Train] Training Decision Tree (baseline)...")
    dt_model = DecisionTreeClassifier(random_state=random_state)
    dt_model.fit(X_train, y_train)

    print("[Train] Training XGBoost (primary)...")
    xgb_model = XGBClassifier(
        random_state=random_state,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    xgb_model.fit(X_train, y_train)

    print("[Train] Both models trained successfully.\n")
    return dt_model, xgb_model


# ---------------------------------------------------------------------------
# Step 5: Evaluate a model on the test split
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> dict:
    """
    Evaluate a trained classifier and print metrics.

    Args:
        model  : Fitted classifier with a predict() method.
        X_test : Test features.
        y_test : True test labels.
        name   : Human-readable model name for display.

    Returns:
        Dict with keys: accuracy, precision_macro, recall_macro, f1_macro.
        All values are floats in [0.0, 1.0].
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print(f"[Eval] {name}")
    print(f"       Accuracy  : {metrics['accuracy']:.4f}")
    print(f"       Precision : {metrics['precision_macro']:.4f}")
    print(f"       Recall    : {metrics['recall_macro']:.4f}")
    print(f"       F1-Score  : {metrics['f1_macro']:.4f}\n")

    return metrics


# ---------------------------------------------------------------------------
# Step 6: Save model and feature schema
# ---------------------------------------------------------------------------

def save_artifacts(
    xgb_model: XGBClassifier,
    feature_names: list,
    feature_stats: dict,
    model_path: str = "model.pkl",
    schema_path: str = "feature_schema.json",
    class_order: list = None,
) -> None:
    """
    Serialize the XGBoost model and write the feature schema JSON.

    The schema JSON has two top-level keys:
      - "features"   : list of {name, index, min, max, type} dicts
      - "class_order": list of ORIGINAL class labels (e.g. [1,2,3,4,5,6])
                       in the order they were encoded (0→class_order[0], etc.)
                       Required to map predict_proba() columns → CLASS_NAMES.

    Args:
        xgb_model     : Trained XGBoost classifier.
        feature_names : Ordered list of feature column names (34 items).
        feature_stats : Dict mapping feature_name → {"min": ..., "max": ...}.
        model_path    : Output path for model.pkl.
        schema_path   : Output path for feature_schema.json.
        class_order   : Original class labels in encoded order (e.g. [1,2,3,4,5,6]).
    """
    # Save the XGBoost model
    joblib.dump(xgb_model, model_path)
    print(f"[Save] Model saved to '{model_path}'")

    # Build the feature schema list
    features_schema = []
    for i, name in enumerate(feature_names):
        features_schema.append({
            "name":  name,
            "index": i,
            "min":   int(feature_stats[name]["min"]),
            "max":   int(feature_stats[name]["max"]),
            "type":  "int",
        })

    # Use provided class_order, or fall back to model.classes_ (0-based)
    if class_order is None:
        class_order = [int(c) for c in xgb_model.classes_]

    schema = {
        "features":    features_schema,
        "class_order": class_order,
    }

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"[Save] Feature schema saved to '{schema_path}'")
    print(f"[Save] Features: {len(features_schema)} | Class order: {class_order}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(csv_path: str = "dermatology.csv") -> None:
    """
    Full training pipeline:
      load → clean → split → encode labels → SMOTE → train → evaluate → save
    """
    print("=" * 60)
    print("  DermaDx Lite — Training Pipeline")
    print("=" * 60 + "\n")

    # 1. Load and clean
    df = load_and_clean(csv_path)

    # 2. Split (before SMOTE to prevent leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Encode class labels to 0-based integers (required by XGBoost 2.0)
    #    Save the original class labels so app.py can map back to disease names.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    original_class_order = [int(c) for c in le.classes_]
    print(f"[Labels] Original classes: {original_class_order}")
    print(f"[Labels] Encoded as 0-based: {list(range(len(original_class_order)))}\n")

    # 4. Compute feature stats from training split (before SMOTE)
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

    # 5. Apply SMOTE to training split only (on encoded labels)
    X_res, y_res = apply_smote(X_train, y_train_enc)

    # 6. Train both models (on 0-based encoded labels)
    dt_model, xgb_model = train_models(X_res, y_res)

    # 7. Evaluate on the untouched test split (using encoded labels)
    print("[Eval] === Model Evaluation on Test Set ===\n")
    evaluate_model(dt_model, X_test, y_test_enc, "Decision Tree (baseline)")
    evaluate_model(xgb_model, X_test, y_test_enc, "XGBoost (primary)")

    # 8. Save artifacts — pass original_class_order so app.py can decode predictions
    save_artifacts(xgb_model, feature_names, feature_stats,
                   class_order=original_class_order)

    print("=" * 60)
    print("  Training complete! Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DermaDx Lite models on the erythemato-squamous dataset."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="dermatology.csv",
        help="Path to the dataset CSV file (default: dermatology.csv)",
    )
    args = parser.parse_args()
    main(csv_path=args.csv)
