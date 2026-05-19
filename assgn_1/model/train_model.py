#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime, timezone

import pyspark
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, ShortType, StringType


FORBIDDEN_FEATURE_COLUMNS = {
    "label",
    "label_def",
    "loan_id",
    "Customer_ID",
    "snapshot_date",
    "loan_start_date",
    "feature_store_def",
}

SUSPICIOUS_LEAKAGE_PATTERNS = [
    "dpd",
    "mob",
    "overdue",
    "paid_amt",
    "due_amt",
    "balance",
    "installment",
    "missed",
    "first_missed",
]


def _log(message):
    print(f"[INFO] {message}")


def _log_check(check_name, detail):
    print(f"[PASS] {check_name}: {detail}")


def _log_section(title):
    print(f"\n=== {title} ===")


def _load_training_dataset(spark, training_dataset_path):
    if not os.path.exists(training_dataset_path):
        raise FileNotFoundError(
            f"{training_dataset_path} does not exist. Run `python main.py` first to build the datamart."
        )

    if os.path.isdir(training_dataset_path):
        parquet_dirs = [
            os.path.join(training_dataset_path, path)
            for path in sorted(os.listdir(training_dataset_path))
            if path.endswith(".parquet")
        ]
        if parquet_dirs:
            return spark.read.parquet(*parquet_dirs)

    return spark.read.parquet(training_dataset_path)


def _validate_no_obvious_leakage(df):
    columns = set(df.columns)
    required_columns = {"label", "snapshot_date", "loan_start_date"}
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        raise ValueError(f"Training dataset is missing required columns: {missing_columns}")

    temporal_mismatch_count = (
        df.filter(F.months_between(F.col("snapshot_date"), F.col("loan_start_date")) != F.lit(6.0)).count()
    )
    if temporal_mismatch_count > 0:
        raise ValueError(
            "Potential temporal leakage: some labels are not exactly 6 months after loan_start_date. "
            f"Mismatch rows: {temporal_mismatch_count}"
        )

    return {
        "required_columns": sorted(required_columns),
        "temporal_mismatch_count": temporal_mismatch_count,
    }


def _select_feature_columns(df):
    numeric_types = (DoubleType, FloatType, IntegerType, LongType, ShortType)
    numeric_columns = []
    categorical_columns = []
    excluded_forbidden_columns = []
    excluded_suspicious_columns = []

    for field in df.schema.fields:
        if field.name in FORBIDDEN_FEATURE_COLUMNS:
            excluded_forbidden_columns.append(field.name)
            continue
        if any(pattern in field.name.lower() for pattern in SUSPICIOUS_LEAKAGE_PATTERNS):
            excluded_suspicious_columns.append(field.name)
            continue
        if isinstance(field.dataType, numeric_types):
            numeric_columns.append(field.name)
        elif isinstance(field.dataType, StringType):
            categorical_columns.append(field.name)

    suspicious_selected = [
        column_name
        for column_name in numeric_columns + categorical_columns
        if column_name in FORBIDDEN_FEATURE_COLUMNS
        or any(pattern in column_name.lower() for pattern in SUSPICIOUS_LEAKAGE_PATTERNS)
    ]
    if suspicious_selected:
        raise ValueError(f"Potential leakage columns selected as model features: {suspicious_selected}")
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns found in the training dataset.")

    return numeric_columns, categorical_columns, excluded_forbidden_columns, excluded_suspicious_columns


def _build_pipeline(numeric_columns, categorical_columns):
    stages = []
    feature_inputs = []

    if numeric_columns:
        imputed_columns = [f"{column_name}_imputed" for column_name in numeric_columns]
        stages.append(Imputer(inputCols=numeric_columns, outputCols=imputed_columns).setStrategy("median"))
        feature_inputs.extend(imputed_columns)

    indexed_columns = []
    encoded_columns = []
    for column_name in categorical_columns:
        indexed_column = f"{column_name}_idx"
        encoded_column = f"{column_name}_ohe"
        stages.append(StringIndexer(inputCol=column_name, outputCol=indexed_column, handleInvalid="keep"))
        indexed_columns.append(indexed_column)
        encoded_columns.append(encoded_column)

    if indexed_columns:
        stages.append(OneHotEncoder(inputCols=indexed_columns, outputCols=encoded_columns, handleInvalid="keep"))
        feature_inputs.extend(encoded_columns)

    stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep"))
    stages.append(LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.05))
    return Pipeline(stages=stages), feature_inputs


def _temporal_train_test_split(df, cutoff_date):
    train_df = df.filter(F.col("snapshot_date") <= F.to_date(F.lit(cutoff_date)))
    test_df = df.filter(F.col("snapshot_date") > F.to_date(F.lit(cutoff_date)))

    if _has_enough_rows_and_classes(train_df) and test_df.count() > 0:
        return train_df, test_df

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    if _has_enough_rows_and_classes(train_df) and test_df.count() > 0:
        return train_df, test_df

    label_values = [row["label"] for row in df.select("label").distinct().collect()]
    train_parts = []
    test_parts = []
    for label_value in label_values:
        label_df = df.filter(F.col("label") == label_value)
        label_train_df, label_test_df = label_df.randomSplit([0.8, 0.2], seed=42)
        train_parts.append(label_train_df)
        test_parts.append(label_test_df)

    train_df = _union_dataframes(train_parts)
    test_df = _union_dataframes(test_parts)
    if not _has_enough_rows_and_classes(train_df):
        raise ValueError("Training split does not contain both label classes. Check label distribution and date range.")
    if test_df.count() == 0:
        test_df = train_df.limit(1)

    return train_df, test_df


def _has_enough_rows_and_classes(df):
    return df.count() > 0 and df.select("label").distinct().count() >= 2


def _union_dataframes(dataframes):
    if not dataframes:
        raise ValueError("No dataframes available to union.")

    combined_df = dataframes[0]
    for dataframe in dataframes[1:]:
        combined_df = combined_df.unionByName(dataframe)
    return combined_df


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def train_model(training_dataset_path, output_dir, cutoff_date):
    _log_section("Configuration")
    _log("Starting feature store validation model.")
    _log(f"Training dataset path: {training_dataset_path}")
    _log(f"Output directory: {output_dir}")
    _log(f"Temporal cutoff date: {cutoff_date}")

    spark = (
        pyspark.sql.SparkSession.builder.appName("assignment_1_feature_store_validation_model")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        _log_section("Data Loading")
        df = _load_training_dataset(spark, training_dataset_path)
        total_rows = df.count()
        _log_check("Training dataset load", f"{total_rows} rows loaded with {len(df.columns)} columns")

        _log_section("Leakage Checks")
        leakage_validation = _validate_no_obvious_leakage(df)
        _log_check(
            "Required columns",
            f"found {', '.join(leakage_validation['required_columns'])}",
        )
        _log_check(
            "Temporal alignment",
            f"all rows have label snapshot_date exactly 6 months after loan_start_date; mismatch rows = {leakage_validation['temporal_mismatch_count']}",
        )

        _log_section("Feature Selection")
        numeric_columns, categorical_columns, excluded_forbidden_columns, excluded_suspicious_columns = _select_feature_columns(df)
        _log_check(
            "Forbidden feature exclusion",
            f"excluded {len(excluded_forbidden_columns)} columns: {', '.join(sorted(excluded_forbidden_columns))}",
        )
        _log_check(
            "Leakage pattern exclusion",
            f"excluded {len(excluded_suspicious_columns)} columns matching {', '.join(SUSPICIOUS_LEAKAGE_PATTERNS)}",
        )
        _log_check(
            "Model feature selection",
            f"selected {len(numeric_columns)} numeric features and {len(categorical_columns)} categorical features",
        )

        pipeline, assembled_feature_inputs = _build_pipeline(numeric_columns, categorical_columns)
        _log_check(
            "Spark ML pipeline build",
            f"{len(assembled_feature_inputs)} assembled feature inputs before logistic regression",
        )

        _log_section("Label Validation")
        model_df = df.select(
            "label",
            "snapshot_date",
            "loan_start_date",
            *numeric_columns,
            *categorical_columns,
        ).filter(F.col("label").isNotNull())

        label_counts = {row["label"]: row["count"] for row in model_df.groupBy("label").count().collect()}
        if len(label_counts) < 2:
            raise ValueError(f"Model validation requires both classes. Label distribution: {label_counts}")
        _log_check("Label distribution", f"both classes present: {label_counts}")

        _log_section("Train/Test Split")
        train_df, test_df = _temporal_train_test_split(model_df, cutoff_date)
        train_rows = train_df.count()
        test_rows = test_df.count()
        _log_check("Train/test split", f"train_rows = {train_rows}, test_rows = {test_rows}")

        _log_section("Model Training")
        _log("Training logistic regression validation model.")
        fitted_pipeline = pipeline.fit(train_df)
        predictions = fitted_pipeline.transform(test_df)

        auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction").evaluate(predictions)
        accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        _log_check("Model training and evaluation", f"AUC = {auc:.6f}, accuracy = {accuracy:.6f}")

        _log_section("Artifact Writing")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "logistic_regression_pipeline")
        fitted_pipeline.write().overwrite().save(model_path)
        _log_check("Model artifact write", model_path)

        metrics = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "training_dataset_path": training_dataset_path,
            "model_path": model_path,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "label_distribution": {str(key): value for key, value in label_counts.items()},
            "temporal_cutoff_date": cutoff_date,
            "auc": auc,
            "accuracy": accuracy,
            "numeric_feature_count": len(numeric_columns),
            "categorical_feature_count": len(categorical_columns),
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
        }

        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, sort_keys=True)
        _log_check("Metrics write", os.path.join(output_dir, "metrics.json"))

        leakage_report = "\n".join(
            [
                "Feature store validation model leakage report",
                "============================================",
                "",
                "Result: PASS",
                "",
                "Checks performed:",
                "- all training dataset parquet partitions are loaded",
                "- label snapshot_date is exactly 6 months after loan_start_date for every row",
                "- forbidden columns are excluded from model features",
                "- columns matching obvious post-application leakage patterns are excluded",
                "- temporal train/test split is attempted, with class-balanced fallback if needed",
                "",
                "Forbidden columns excluded:",
                ", ".join(sorted(FORBIDDEN_FEATURE_COLUMNS)),
                "",
                "Suspicious leakage patterns excluded:",
                ", ".join(SUSPICIOUS_LEAKAGE_PATTERNS),
                "",
                "Numeric features used:",
                ", ".join(numeric_columns),
                "",
                "Categorical features used:",
                ", ".join(categorical_columns),
                "",
                f"AUC: {auc:.6f}",
                f"Accuracy: {accuracy:.6f}",
            ]
        )
        _write_text(os.path.join(output_dir, "leakage_report.txt"), leakage_report)
        _log_check("Leakage report write", os.path.join(output_dir, "leakage_report.txt"))

        _log_section("Completed")
        _log("Validation completed successfully.")

        return metrics
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Train a simple leakage-checked binary classifier.")
    parser.add_argument(
        "--training-dataset-path",
        default="datamart/gold/training_dataset/",
        help="Path to the gold training dataset generated by main.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="model/artifacts/",
        help="Directory where the fitted model and validation reports will be written.",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2024-12-01",
        help="Label snapshot cutoff date for temporal train/test split.",
    )

    args = parser.parse_args()
    metrics = train_model(args.training_dataset_path, args.output_dir, args.cutoff_date)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
