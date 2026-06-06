"""Train and evaluate CS2 match-winner models.

Loads the processed match dataset, splits it *chronologically* (oldest
matches train, newest test) to avoid leakage, trains a logistic-regression
baseline and an XGBoost model, prints accuracy + AUC-ROC for both, and saves
the higher-AUC model to disk with joblib.

Expected dataset schema (one row per match):
    - a date column (default "date") for chronological ordering
    - the feature columns produced by src.features (e.g. kd_diff, adr_diff, ...)
    - a binary target column (default "label"): 1 if team A won, else 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import MODEL_FEATURE_COLS

DEFAULT_DATA = Path("data/processed/matches.csv")
DEFAULT_ELO_FEATURES = Path("data/processed/elo_features.csv")
DEFAULT_MODEL_OUT = Path("data/processed/best_model.joblib")
# The production model uses the recent-form stat diffs AND the FACEIT-Elo gap.
FEATURE_COLS = MODEL_FEATURE_COLS


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the processed dataset from parquet or CSV based on file suffix."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Build it with the feature pipeline first."
        )
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def chronological_split(
    df: pd.DataFrame, date_col: str, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by date and hold out the most recent ``test_size`` fraction as test."""
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_size))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
    """Return (accuracy, AUC-ROC) for a fitted classifier on the test set."""
    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return accuracy_score(y_test, preds), roc_auc_score(y_test, proba)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--elo-features", type=Path, default=DEFAULT_ELO_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_dataset(args.data)
    # Merge the FACEIT-Elo feature (built by src.elo) onto each match.
    if args.elo_features.exists():
        elo = load_dataset(args.elo_features)
        df = df.merge(elo, on="match_id", how="inner")
    missing = [c for c in (*FEATURE_COLS, args.target, args.date_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset is missing required columns: {missing}. "
            f"Build the Elo features first: python -m src.elo --data {args.data}"
        )

    train_df, test_df = chronological_split(df, args.date_col, args.test_size)
    x_train, y_train = train_df[FEATURE_COLS], train_df[args.target]
    x_test, y_test = test_df[FEATURE_COLS], test_df[args.target]
    print(f"Train: {len(train_df)} matches | Test: {len(test_df)} matches\n")

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        acc, auc = evaluate(model, x_test, y_test)
        results[name] = (model, acc, auc)
        print(f"{name:8s}  accuracy={acc:.3f}  auc={auc:.3f}")

    best_name = max(results, key=lambda n: results[n][2])  # highest AUC
    best_model = results[best_name][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.out)
    print(f"\nBest model: {best_name} (auc={results[best_name][2]:.3f}) -> saved to {args.out}")


if __name__ == "__main__":
    main()
