import json
import shutil
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_BANK_DIR,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_CUTOFF_DATE,
)
from utils.data_processing import load_training_dataset


class NumpyCreditRiskModel:
    def __init__(self, l2: float = 0.01, learning_rate: float = 0.08, epochs: int = 900):
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, frame: pd.DataFrame, labels: pd.Series):
        self.numeric_medians_ = frame[NUMERIC_FEATURES].median(numeric_only=True)
        numeric = frame[NUMERIC_FEATURES].fillna(self.numeric_medians_).astype(float)
        self.numeric_means_ = numeric.mean()
        self.numeric_stds_ = numeric.std().replace(0, 1).fillna(1)

        self.category_levels_ = {}
        for col in CATEGORICAL_FEATURES:
            values = frame[col].fillna("missing").astype(str)
            levels = values.value_counts(normalize=True)
            self.category_levels_[col] = levels.loc[levels.ge(0.01)].index.tolist()

        x = self._transform(frame)
        y = labels.astype(float).to_numpy()
        positive_rate = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.intercept_ = float(np.log(positive_rate / (1 - positive_rate)))
        self.weights_ = np.zeros(x.shape[1], dtype=float)

        for _ in range(self.epochs):
            logits = np.clip(x @ self.weights_ + self.intercept_, -30, 30)
            preds = 1 / (1 + np.exp(-logits))
            error = preds - y
            grad_w = (x.T @ error) / len(y) + self.l2 * self.weights_
            grad_b = float(error.mean())
            self.weights_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        return self

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[NUMERIC_FEATURES].fillna(self.numeric_medians_).astype(float)
        numeric = ((numeric - self.numeric_means_) / self.numeric_stds_).replace([np.inf, -np.inf], 0).fillna(0)
        arrays = [numeric.to_numpy()]
        for col in CATEGORICAL_FEATURES:
            values = frame[col].fillna("missing").astype(str)
            for level in self.category_levels_[col]:
                arrays.append(values.eq(level).astype(float).to_numpy().reshape(-1, 1))
        return np.hstack(arrays)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._transform(frame)
        logits = np.clip(x @ self.weights_ + self.intercept_, -30, 30)
        p1 = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p1, p1])


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _candidate_models() -> dict[str, NumpyCreditRiskModel]:
    return {
        "logistic_l2_0_01": NumpyCreditRiskModel(l2=0.01, learning_rate=0.08, epochs=900),
        "logistic_l2_0_10": NumpyCreditRiskModel(l2=0.10, learning_rate=0.08, epochs=900),
    }


def _roc_auc(y_true: pd.Series, score: np.ndarray) -> float:
    y = y_true.astype(int).to_numpy()
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(y_true: pd.Series, score: np.ndarray) -> float:
    y = y_true.astype(int).to_numpy()
    order = np.argsort(-score)
    sorted_y = y[order]
    positives = sorted_y.sum()
    if positives == 0:
        return 0.0
    precision_at_k = np.cumsum(sorted_y) / (np.arange(len(sorted_y)) + 1)
    return float((precision_at_k * sorted_y).sum() / positives)


def _metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    prediction = (proba >= 0.5).astype(int)
    return {
        "auc": _roc_auc(y_true, proba),
        "average_precision": _average_precision(y_true, proba),
        "accuracy": float((prediction == y_true.astype(int).to_numpy()).mean()),
    }


def train_and_register_model(cutoff_date: str = TRAIN_CUTOFF_DATE) -> dict:
    _clean_dir(MODEL_BANK_DIR)
    df = load_training_dataset()
    df = df.dropna(subset=["label"]).copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing model feature: {col}")

    train_mask = df["label_snapshot_date"].le(pd.to_datetime(cutoff_date))
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()
    if train_df["label"].nunique() < 2 or test_df.empty or test_df["label"].nunique() < 2:
        train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
        test_df = df.drop(train_df.index)

    leaderboard = []
    fitted_models = {}
    for model_name, pipeline in _candidate_models().items():
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["label"])
        proba = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        row = {"model_name": model_name, **_metrics(test_df["label"], proba)}
        leaderboard.append(row)
        fitted_models[model_name] = pipeline

    leaderboard = sorted(leaderboard, key=lambda row: row["auc"], reverse=True)
    best_name = leaderboard[0]["model_name"]
    best_model = fitted_models[best_name]
    best_model_path = MODEL_BANK_DIR / "best_model.pkl"
    with open(best_model_path, "wb") as file:
        pickle.dump(best_model, file)

    reference_scores = best_model.predict_proba(train_df[FEATURE_COLUMNS])[:, 1]
    pd.DataFrame(
        {
            "score": reference_scores,
            "label": train_df["label"].to_numpy(),
            "label_snapshot_date": train_df["label_snapshot_date"].dt.strftime("%Y-%m-%d"),
        }
    ).to_csv(MODEL_BANK_DIR / "reference_scores.csv", index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model_path": str(best_model_path.relative_to(PROJECT_ROOT)),
        "best_model": best_name,
        "selection_metric": "auc",
        "leaderboard": leaderboard,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_cutoff_date": cutoff_date,
        "label_distribution": df["label"].value_counts().sort_index().astype(int).to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "governance": {
            "refresh_trigger": "Retrain monthly or earlier if AUC drops below 0.70, score PSI exceeds 0.20, or observed default rate moves more than 25% from reference.",
            "approval": "Champion model is promoted only after validation metrics and monitoring dashboard are reviewed by data science and risk stakeholders.",
            "deployment_option": "Batch Airflow scoring is the default; the same serialized Python model object can be wrapped by a REST service for real-time underwriting.",
        },
    }
    (MODEL_BANK_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(leaderboard).to_csv(MODEL_BANK_DIR / "leaderboard.csv", index=False)
    return metadata


def load_best_model():
    model_path = MODEL_BANK_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Best model is missing. Run train_and_register_model first.")
    with open(model_path, "rb") as file:
        return pickle.load(file)
