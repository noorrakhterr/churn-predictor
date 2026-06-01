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

from preprocess import PROCESSED_PATH, get_features_and_target, engineer_features, load_raw, RAW_PATH

MODEL_PATH = Path("models/churn_model.pkl")
RANDOM_SEED = 42


def build_model() -> CalibratedClassifierCV:
    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    # Isotonic calibration so probability outputs are trustworthy
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


def train(processed_path: Path = PROCESSED_PATH) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists():
        df = pd.read_csv(processed_path)
    else:
        print("Processed data not found — running preprocessing first.")
        df = engineer_features(load_raw(RAW_PATH))

    X, y = get_features_and_target(df)

    print(f"Training on {len(X)} accounts ({y.mean():.1%} churn rate)")

    model = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X, y)

    threshold = choose_threshold(model, X, y)
    print(f"Selected decision threshold: {threshold:.3f}")

    y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    print(classification_report(y, y_pred, target_names=["retained", "churned"]))

    artifact = {"model": model, "threshold": threshold, "feature_cols": list(X.columns)}
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
