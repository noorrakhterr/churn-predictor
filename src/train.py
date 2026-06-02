"""
Trains an XGBoost churn classifier and saves the model artifact.

Product context: the goal isn't the highest possible AUC — it's a model that
CSMs can trust.  That means:
  - Calibrated probabilities (so "80% risk" actually means something)
  - A threshold tuned for recall (missing a churning account is worse than a
    false alarm)
  - Feature importance surfaced so the team can sanity-check the model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# Make intra-package imports work whether run as `python src/train.py` or
# `python -m src.train` (mirrors the bootstrap in app/app.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NUMERIC_COLS / CATEGORICAL_COLS are imported for display only — the data is
# already encoded by the saved preprocessor, so training never transforms.
from preprocess import NUMERIC_COLS, CATEGORICAL_COLS, PROCESSED_DIR, MODELS_DIR

MODEL_PATH = MODELS_DIR / "churn_model.pkl"
RANDOM_SEED = 42


def build_model() -> CalibratedClassifierCV:
    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    # Isotonic calibration so probability outputs are trustworthy. The model
    # consumes already-encoded arrays (data/processed/*.npy), so there is no
    # preprocessor in the estimator — encoding is owned solely by preprocess.py.
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def choose_threshold(model, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Pick the threshold that maximises F-beta (beta=2) to favour recall.
    Product rationale: missing a churning account costs more than a false
    positive that prompts an unnecessary check-in call.
    """
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    beta = 2
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
    best_idx = np.argmax(f_beta[:-1])
    return float(thresholds[best_idx])


def train() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load the already-encoded arrays + feature names produced by
    # `python -m src.preprocess`. No transformation happens here.
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    print(f"Training on {len(X_train)} accounts ({y_train.mean():.1%} churn rate)")
    print(f"Features: {len(NUMERIC_COLS)} numeric + {len(CATEGORICAL_COLS)} "
          f"categorical → {len(feature_names)} after encoding")

    model = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_train, y_train)

    threshold = choose_threshold(model, X_train, y_train)
    print(f"Selected decision threshold: {threshold:.3f}")

    # Honest holdout evaluation: report on the untouched test split.
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=["retained", "churned"]))

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_cols": feature_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
