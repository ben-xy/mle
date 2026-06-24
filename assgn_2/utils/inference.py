import json

import pandas as pd

from utils.config import DATAMART_DIR, FEATURE_COLUMNS
from utils.data_processing import load_training_dataset
from utils.modeling import load_best_model


def run_batch_inference(execution_date: str | None = None) -> dict:
    df = load_training_dataset()
    model = load_best_model()
    scores = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    predictions = df[
        [
            "loan_id",
            "Customer_ID",
            "loan_start_date",
            "feature_snapshot_date",
            "label_snapshot_date",
            "label",
        ]
    ].copy()
    predictions["prediction_score"] = scores
    predictions["prediction"] = (predictions["prediction_score"] >= 0.5).astype(int)
    predictions["score_band"] = pd.cut(
        predictions["prediction_score"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["00-20", "20-40", "40-60", "60-80", "80-100"],
    )

    if execution_date is not None:
        target_month = pd.to_datetime(execution_date).to_period("M")
        predictions = predictions.loc[
            predictions["label_snapshot_date"].dt.to_period("M") == target_month
        ].copy()

    output_dir = DATAMART_DIR / "gold" / "model_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions_path = output_dir / "gold_model_predictions_all.csv"
    if all_predictions_path.exists():
        existing = pd.read_csv(
            all_predictions_path,
            parse_dates=["loan_start_date", "feature_snapshot_date", "label_snapshot_date"],
        )
        predictions = pd.concat([existing, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(
            subset=["loan_id", "label_snapshot_date"],
            keep="last",
        )

    for date_value, date_df in predictions.groupby("label_snapshot_date"):
        token = pd.to_datetime(date_value).strftime("%Y_%m_%d")
        date_df.to_csv(output_dir / f"gold_model_predictions_{token}.csv", index=False)
    predictions.to_csv(output_dir / "gold_model_predictions_all.csv", index=False)

    summary = {
        "rows": int(len(predictions)),
        "prediction_months": int(predictions["label_snapshot_date"].nunique()),
        "average_score": float(predictions["prediction_score"].mean()),
        "predicted_default_rate": float(predictions["prediction"].mean()),
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
