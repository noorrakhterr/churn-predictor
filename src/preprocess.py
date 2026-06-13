"""
Production-quality preprocessing pipeline for the B2B SaaS churn dataset.

Flow (see `main`):
    load_data → identify_column_types → split_data (stratified)
    → build_preprocessor → FIT ON TRAIN ONLY → transform train/test
    → persist fitted preprocessor, processed arrays, feature names, metadata.

Design principle: every transform that *learns* from data (imputation medians,
scaler mean/std, one-hot vocabularies) is fit on the training split only, then
applied to the test split. That keeps the test set a faithful stand-in for
unseen production data — no information from it leaks into the model.

Runnable as:  python -m src.preprocess
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Resolve paths relative to the repo root (src/ -> repo root) so the script
# works regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "saas_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET_COL = "churned"
# Identifiers carry no generalizable signal — a model that keys off account_id
# or company_name would memorize the training rows (a subtle leakage / overfit
# trap), so we drop them up front. This dataset is a point-in-time snapshot with
# no post-outcome columns, so there are no other leakage features to remove.
ID_LEAKAGE_COLS = ["account_id", "company_name"]

# Canonical column lists, mirroring what identify_column_types() yields on the
# standard schema (booleans are cast to int in load_data(), so flags like
# exec_sponsor_changed_last_180d live in NUMERIC_COLS). Exported for display and
# metadata use by downstream files — NOT for transformation: that is the saved
# preprocessor's job (models/preprocessor.pkl).
NUMERIC_COLS = [
    "company_size",
    "acv_usd",
    "tenure_months",
    "seats_purchased",
    "seats_active_last_30d",
    "seat_utilization_rate",
    "logins_last_30d",
    "admin_logins_last_30d",
    "features_adopted",
    "mfa_enabled_pct",
    "api_calls_last_30d",
    "support_tickets_last_90d",
    "critical_tickets_last_90d",
    "nps_score",
    "qbr_attendance_rate",
    "exec_sponsor_changed_last_180d",
    "days_since_last_login",
    "discount_pct",
    "payment_delays_last_year",
    "expansion_revenue_last_year_usd",
]
CATEGORICAL_COLS = ["industry", "contract_type"]


def load_data(path: str | Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw CSV and split it into a feature matrix X and target vector y.

    Drops the target column and any ID/leakage columns from X.

    Args:
        path: Path to the raw churn CSV.

    Returns:
        (X, y) where X is the feature DataFrame and y is the 'churned' Series.
    """
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found in {path}")

    # --- Data cleaning ----------------------------------------------------
    # Cast boolean columns to int (0/1). Left as bool, identify_column_types
    # would treat them as categorical and OneHotEncoder would expand each into
    # two perfectly collinear columns (e.g. *_True and *_False). As int they
    # become a single, clean 0/1 numeric feature that the scaler handles fine.
    #
    # (Note: there are no numeric-stored-as-string columns in this dataset — all
    # object columns are genuinely categorical/ID — so no pd.to_numeric coercion
    # step is needed here. Add one only if a future source ships e.g. "1,234" or
    # blank-padded numerics:  df[col] = pd.to_numeric(df[col], errors="coerce").)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    y = df[TARGET_COL]
    # Drop the target plus any ID/leakage columns that happen to be present.
    drop_cols = [TARGET_COL] + [c for c in ID_LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)

    return X, y


def identify_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Programmatically classify columns as numeric or categorical by dtype.

    Numeric  = integer/float columns.
    Categorical = object, boolean, and pandas 'category' columns. Booleans are
    treated as categorical so they get one-hot encoded (a True/False flag is a
    category, not a magnitude).

    Args:
        X: Feature DataFrame.

    Returns:
        (numeric_cols, categorical_cols) as lists of column names.
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select the canonical feature columns (NUMERIC_COLS + CATEGORICAL_COLS) and
    the target from a raw DataFrame. A lightweight convenience wrapper for
    callers/tests that already hold a DataFrame; the full load-from-disk path is
    load_data(). Encoding still happens via the preprocessor, not here.

    Args:
        df: Raw churn DataFrame that includes all canonical feature columns and
            the TARGET_COL ('churned') column.

    Returns:
        (X, y) where X contains only the canonical feature columns and y is the
        'churned' Series.
    """
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    return X, y


def build_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that encodes all model inputs.

    Column lists default to the module-level NUMERIC_COLS / CATEGORICAL_COLS when
    omitted, so callers can do `build_preprocessor()`; main() passes the
    programmatically-detected lists explicitly.

    Numeric branch:  SimpleImputer(median) -> StandardScaler
        - Median imputation is robust to outliers and skew: unlike the mean, the
          median is barely moved by a few extreme values (e.g. a handful of
          accounts with huge api_calls), so imputed values stay representative.
        - StandardScaler centers/scales features to comparable ranges, which
          helps distance- and gradient-based models and keeps coefficients
          interpretable.

    Categorical branch: SimpleImputer(most_frequent) -> OneHotEncoder
        - handle_unknown='ignore' means categories unseen during fit (e.g. a new
          industry that appears only at inference time) are encoded as all-zeros
          instead of raising — the pipeline degrades gracefully in production.
        - sparse_output=False returns a dense array, simpler to save as .npy and
          to feed to downstream tools like SHAP.

    remainder='drop' guarantees only the explicitly listed columns flow through,
    so no stray/unexpected column silently enters the model.

    Args:
        numeric_cols: Names of numeric feature columns.
        categorical_cols: Names of categorical feature columns.

    Returns:
        An unfitted ColumnTransformer.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 train/test split.

    stratify=y preserves the churn class balance in both splits. With imbalanced
    data (~20% churn here) a plain random split can drift the minority rate
    between train and test, which biases both training and evaluation.
    Stratification keeps both splits faithful to the population rate.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction held out for testing.
        random_state: Seed for a reproducible split.

    Returns:
        (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """
    Return the full list of output feature names after one-hot expansion.

    This is critical for downstream explainability (e.g. SHAP), where each model
    input column needs a human-readable name. Prefers the fitted transformer's
    native get_feature_names_out(); falls back to building names manually from
    the fitted OneHotEncoder if that isn't available.

    Args:
        preprocessor: A *fitted* ColumnTransformer.
        numeric_cols: Numeric column names (passed through, in order).
        categorical_cols: Categorical column names fed to the encoder.

    Returns:
        Ordered list of output feature names.
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except (AttributeError, NotImplementedError):
        # Manual fallback: numeric columns are passed through 1:1; categorical
        # columns are expanded by the fitted OneHotEncoder.
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        return list(numeric_cols) + cat_names


def main() -> None:
    """Orchestrate the full preprocessing flow and persist all artifacts."""
    # Ensure output directories exist (idempotent).
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and separate features/target.
    X, y = load_data(RAW_DATA_PATH)

    # 2. Identify column types programmatically.
    numeric_cols, categorical_cols = identify_column_types(X)

    # 3. Stratified split BEFORE any fitting — see step 4 for why order matters.
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Build and fit the preprocessor on TRAIN ONLY.
    #    Anti-leakage: imputation medians, scaler statistics, and one-hot
    #    vocabularies are *learned* parameters. If we fit them on the full
    #    dataset (or the test set), information from the held-out data bleeds
    #    into the transform, inflating evaluation scores and producing a model
    #    that looks better offline than it performs on truly unseen data.
    #    Fitting on train only keeps the test set an honest proxy for production.
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)  # transform only — never fit

    # 5. Resolve output feature names (for SHAP / interpretability later).
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    # 6. Persist artifacts.
    # Fitted preprocessor — reused at inference so train/serve transforms match.
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    # Processed arrays as .npy (dense float matrices + integer labels).
    np.save(PROCESSED_DIR / "X_train.npy", X_train_processed)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_processed)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    # Feature names — ordered to match the columns of the processed arrays.
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Column metadata — which raw columns were numeric vs categorical.
    with open(MODELS_DIR / "column_metadata.json", "w") as f:
        json.dump(
            {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
            f,
            indent=2,
        )

    # 7. Report. Train/test churn rates should be nearly identical — that
    #    near-equality is the visible proof that stratification worked.
    print("Preprocessing complete.")
    print(
        f"  Raw features:          {X.shape[1]} cols "
        f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)"
    )
    print(f"  Processed feature dim: {X_train_processed.shape[1]} " f"(after one-hot encoding)")
    print(f"  X_train shape:         {X_train_processed.shape}")
    print(f"  X_test shape:          {X_test_processed.shape}")
    print(f"  Churn rate (train):    {y_train.mean():.4f}")
    print(
        f"  Churn rate (test):     {y_test.mean():.4f}  "
        f"(Δ={abs(y_train.mean() - y_test.mean()):.4f} — proof of stratification)"
    )
    print(f"  Artifacts saved to:    {MODELS_DIR}/ and {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
