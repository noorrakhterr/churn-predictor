"""
Training and evaluation pipeline for the B2B SaaS churn predictor.

Flow (see main):
    load_processed_data  →  cross_validate_models (3-model baseline comparison)
    →  tune_xgboost (RandomizedSearchCV)  →  evaluate_on_test (metrics + plots)
    →  find_optimal_threshold  →  persist all artifacts

Evaluation philosophy
---------------------
PR-AUC (average_precision) is the PRIMARY scoring metric because:
  - Under class imbalance (~20% churn), ROC-AUC can be misleadingly high.
    A model can score 0.90 ROC-AUC while still missing most churners, because
    ROC-AUC averages over all decision thresholds including those where the
    classifier trivially separates the abundant majority class.
  - PR-AUC focuses exclusively on the positive (churn) class: it measures how
    well the model ranks true churners near the top of its score distribution
    while maintaining acceptable precision.  It is directly proportional to
    the business value of the risk ranking.

Recall and F1 are secondary because the cost asymmetry favours recall: a
missed churn (false negative) loses the account with no chance of
intervention, while a false alarm triggers an unnecessary CSM call — costly
but recoverable.  Threshold tuning (find_optimal_threshold) shifts the
default 0.5 cut-off toward higher recall.

Calibration: the output probabilities feed downstream risk-tier dashboards.
A well-calibrated model gives probabilities that are trustworthy at face
value — a score of 0.70 means roughly 70% of those accounts actually churn.

Runnable as:  python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
PLOTS_DIR     = ROOT / "docs" / "plots"

RANDOM_STATE  = 42
N_SPLITS      = 5   # folds for both baseline CV and hyperparameter search
N_ITER_SEARCH = 20  # RandomizedSearchCV draws

# Scoring keys for cross_validate and RandomizedSearchCV.
# 'average_precision' is sklearn's scorer name for PR-AUC.
CV_SCORING = {
    "roc_auc":   "roc_auc",
    "pr_auc":    "average_precision",
    "f1":        "f1",
    "precision": "precision",
    "recall":    "recall",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the NumPy arrays written by src.preprocess."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def load_feature_names() -> list[str]:
    """Return feature names saved by src.preprocess, or [] if absent."""
    path = MODELS_DIR / "feature_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_models(scale_pos_weight: float) -> dict:
    """
    Construct the three candidate estimators with imbalance handling enabled.

    Imbalance strategy per model
    ----------------------------
    LogisticRegression / RandomForest — class_weight='balanced':
        sklearn reweights each training sample by n_samples / (n_classes * n_i),
        giving the minority class (churn) proportionally more influence on the
        decision boundary and split criterion without resampling the data.

    XGBClassifier — scale_pos_weight = n_negatives / n_positives (≈ 3.97):
        XGBoost's native equivalent of class_weight.  The positive-class
        gradient is multiplied by this factor at each boosting step, so the
        trees attend to churners even when they are a small fraction of rows.
        Unlike sklearn's class_weight, scale_pos_weight also influences leaf
        value updates, which tends to preserve probability calibration better
        under imbalance.
    """
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,       # extra iterations to converge on 31 scaled features
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation baseline
# ---------------------------------------------------------------------------

def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Run 5-fold StratifiedKFold CV for every candidate model.

    Why StratifiedKFold?
    With ~20% churn, plain KFold can produce folds whose positive rate drifts
    from ~15% to ~25%, inflating fold-to-fold variance and obscuring which
    model is genuinely better.  Stratification pins each fold's churn rate to
    approximately the training-set rate so cross-model comparisons are fair.

    Returns
    -------
    {model_name: {metric_mean: float, metric_std: float, ...}}
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        print(f"  {N_SPLITS}-fold CV: {name}...", flush=True)
        raw = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
            return_train_score=False,
        )
        summary: dict[str, float] = {}
        for metric in CV_SCORING:
            scores = raw[f"test_{metric}"]
            summary[f"{metric}_mean"] = float(scores.mean())
            summary[f"{metric}_std"]  = float(scores.std())
        results[name] = summary

    return results


def print_cv_table(cv_results: dict[str, dict[str, float]]) -> None:
    """Print CV results as a fixed-width aligned table."""
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    col_w   = 22
    width   = 22 + col_w * len(metrics)

    print("\n" + "=" * width)
    print(f"{'5-Fold Stratified CV Results':^{width}}")
    print("=" * width)
    print(f"{'Model':<22}" + "".join(f"{m:>{col_w}}" for m in metrics))
    print("-" * width)
    for model_name, summary in cv_results.items():
        row = f"{model_name:<22}"
        for m in metrics:
            cell = f"{summary[f'{m}_mean']:.4f} ±{summary[f'{m}_std']:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)
    print("=" * width)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> XGBClassifier:
    """
    RandomizedSearchCV over the XGBoost grid, optimising PR-AUC.

    Why PR-AUC as the tuning objective?
        The search must optimise the same criterion we report.  Maximising
        average_precision rewards models that rank churners near the top of
        the score distribution — exactly what a CSM escalation workflow needs
        so high-risk accounts receive attention before submitting a
        cancellation notice.

    Why RandomizedSearch over GridSearch?
        The grid has 3×3×3×2 = 54 combinations × 5 folds = 270 fits.
        20 random draws × 5 folds = 100 fits, typically recovering ≥95% of
        the best configuration at ~37% of the compute cost.
        (Bergstra & Bengio 2012: random search outperforms grid search in
        the same budget when a few hyperparameters dominate and the rest
        have little effect.)

    scale_pos_weight is fixed across all search configurations — it is a
        function of the training-set class ratio, not a tunable algorithm
        parameter, and must be preserved in every trial.

    refit=True (default): after the search, the best config is re-fitted
        on the full training set so the returned estimator is ready for
        test evaluation without a separate fit() call.
    """
    param_dist = {
        "n_estimators":  [100, 200, 400],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="average_precision",  # PR-AUC
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Best PR-AUC : {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate the tuned model on the held-out test set and save diagnostic plots.

    Four plots are saved to docs/plots/ at 150 dpi:

    confusion_matrix.png
        Raw TP/FP/TN/FN counts at the default 0.5 threshold.  Immediately
        shows the magnitude of missed churners (FN) vs false alarms (FP).

    roc_curve.png
        TPR vs FPR across all thresholds.  The dashed diagonal anchors the
        random-classifier baseline (AUC = 0.50).  Included for stakeholder
        familiarity, but interpret with care under imbalance.

    pr_curve.png
        Precision vs Recall for the positive class.  The horizontal dashed
        line is the no-skill baseline at y = churn_rate — a model below it
        is no better than predicting the majority class for every sample.
        This is the primary diagnostic for imbalanced classification.

    calibration_curve.png
        Reliability diagram with 10 equal-width probability bins.  The
        diagonal represents perfect calibration.  Points above → model
        under-confident; below → over-confident.  Miscalibration matters
        because scores feed downstream risk-tier systems where absolute
        probability values drive business decisions.

    Returns
    -------
    metrics dict with roc_auc, pr_auc, f1, precision, recall at threshold 0.5.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        "pr_auc":    float(average_precision_score(y_test, y_proba)),
        "f1":        float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall":    float(recall_score(y_test, y_pred)),
    }

    print("\n" + "=" * 60)
    print("Test-Set Evaluation  (threshold = 0.50)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Churn", "Churn"],
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix  (threshold = 0.50)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- ROC curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="XGBoost (tuned)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")
    ax.set_title(f"ROC Curve  (AUC = {metrics['roc_auc']:.4f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # --- PR curve with no-skill baseline ---
    # The no-skill line sits at precision = churn_rate: a model that predicts
    # every account as churn achieves exactly this average precision value.
    no_skill = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, ax=ax, name="XGBoost (tuned)"
    )
    ax.axhline(
        no_skill, color="grey", linestyle="--", lw=0.8,
        label=f"No-skill baseline (precision = {no_skill:.2f})",
    )
    ax.set_title(f"Precision-Recall Curve  (PR-AUC = {metrics['pr_auc']:.4f})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    # --- Calibration (reliability) curve ---
    # n_bins=10 divides [0,1] into 10 equal-width bins; each point shows the
    # fraction of true positives among samples in that predicted-probability bin.
    # Points on the diagonal = perfect calibration.
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(
        y_test, y_proba, n_bins=10, ax=ax, name="XGBoost (tuned)"
    )
    ax.set_title("Calibration Curve (Reliability Diagram)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved → {PLOTS_DIR}/")
    return metrics


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 on the evaluation set.

    Why tune the threshold?
        The default 0.5 cut-off is calibrated for 50/50 class balance.  With
        ~20% positive rate and asymmetric costs (missed churn >> false alarm),
        the optimal PR-curve operating point typically sits at a *lower*
        threshold: we trade some precision for meaningfully higher recall,
        catching more at-risk accounts while keeping false-alarm volume
        manageable.

    Why F1 as the optimisation criterion?
        F1 = 2·P·R / (P+R) is the harmonic mean of precision and recall,
        penalising extreme operating points.  Predicting every account as
        churn gives recall=1.0 but precision≈0.20, so F1≈0.33 — a natural
        guard against trivial solutions.  For stricter recall prioritisation,
        replace with F_beta (beta > 1): F_β = (1+β²)·P·R / (β²·P + R).

    Caveat: the threshold is chosen on the test set for reporting purposes
        only.  In production, use a time-based hold-out to avoid threshold
        overfitting on the same partition used for evaluation.

    Parameters
    ----------
    y_true  : 1-D integer array of ground-truth labels.
    y_proba : 1-D float array of positive-class probabilities.

    Returns
    -------
    best_threshold : float in [0, 1]
    """
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns n+1 precision/recall values and n
    # thresholds.  Trim the trailing pair so arrays align element-wise.
    f1_arr      = (2 * prec_arr[:-1] * rec_arr[:-1]
                   / (prec_arr[:-1] + rec_arr[:-1] + 1e-9))
    best_idx    = int(np.argmax(f1_arr))
    best_thresh = float(thresholds[best_idx])

    y_pred_default = (y_proba >= 0.50).astype(int)
    y_pred_optimal = (y_proba >= best_thresh).astype(int)

    print("\n" + "=" * 55)
    print(f"{'Threshold Optimisation  (maximise F1)':^55}")
    print("=" * 55)
    print(f"{'Metric':<20}{'Default (0.50)':>17}{'Optimal':>18}")
    print("-" * 55)
    for label, dval, oval in [
        ("Threshold",
         f"{0.50:.4f}",
         f"{best_thresh:.4f}"),
        ("F1",
         f"{f1_score(y_true, y_pred_default):.4f}",
         f"{f1_score(y_true, y_pred_optimal):.4f}"),
        ("Precision",
         f"{precision_score(y_true, y_pred_default):.4f}",
         f"{precision_score(y_true, y_pred_optimal):.4f}"),
        ("Recall",
         f"{recall_score(y_true, y_pred_default):.4f}",
         f"{recall_score(y_true, y_pred_optimal):.4f}"),
    ]:
        print(f"  {label:<18}{dval:>17}{oval:>18}")
    print("=" * 55)

    return best_thresh


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict[str, float]) -> None:
    """Print final test-set metrics in a clean aligned table."""
    print("\n" + "=" * 40)
    print(f"{'Final Test-Set Metrics':^40}")
    print("=" * 40)
    for key, val in metrics.items():
        print(f"  {key:<18} {val:.4f}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full training, tuning, and evaluation pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B2B SaaS Churn Predictor — Training Pipeline")
    print("=" * 60)

    # ---- 1. Load data -------------------------------------------------------
    print("\n[1/4] Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    print(f"  Train : {X_train.shape[0]:>5} samples  "
          f"({n_pos} churn / {n_neg} no-churn,  "
          f"scale_pos_weight = {scale_pos_weight:.2f})")
    print(f"  Test  : {X_test.shape[0]:>5} samples")
    print(f"  Feats : {X_train.shape[1]}")

    # ---- 2. Baseline CV on all three models ---------------------------------
    print("\n[2/4] Baseline cross-validation (5-fold StratifiedKFold)...")
    models   = make_models(scale_pos_weight)
    cv_stats = cross_validate_models(models, X_train, y_train)
    print_cv_table(cv_stats)

    # ---- 3. Hyperparameter tuning on XGBoost --------------------------------
    print("\n[3/4] Tuning XGBoost (RandomizedSearchCV, 20 iter × 5 folds)...")
    best_model = tune_xgboost(X_train, y_train, scale_pos_weight)

    # ---- 4. Test-set evaluation + plots -------------------------------------
    print("\n[4/4] Test-set evaluation...")
    test_metrics = evaluate_on_test(best_model, X_test, y_test)

    # ---- 5. Threshold optimisation (reported here, after test eval) ---------
    y_proba           = best_model.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, y_proba)

    print_metrics_table(test_metrics)

    # ---- 6. Persist artifacts -----------------------------------------------
    print("\nPersisting artifacts...")

    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")
    print(f"  model         → {MODELS_DIR}/churn_model.pkl")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  metrics       → {MODELS_DIR}/metrics.json")

    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump(
            {"optimal_threshold": optimal_threshold, "default_threshold": 0.5},
            f,
            indent=2,
        )
    print(f"  threshold     → {MODELS_DIR}/threshold.json")

    # Feature names are written by src.preprocess; save as fallback only.
    fn_path = MODELS_DIR / "feature_names.json"
    if not fn_path.exists():
        feature_names = load_feature_names()
        with open(fn_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"  feature_names → {fn_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
