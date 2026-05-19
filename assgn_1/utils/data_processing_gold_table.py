import os
from functools import reduce

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType


def _read_silver(snapshot_date_str, source_name, silver_directory, spark):
    if not isinstance(silver_directory, (str, bytes, os.PathLike)):
        raise TypeError(f"silver_directory for {source_name} must be a path string, got {type(silver_directory).__name__}")
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)
    return spark.read.parquet(filepath)


def process_features_gold_table(snapshot_date_str, silver_directories, gold_directory, spark):
    clickstream_df = _read_silver(snapshot_date_str, "clickstream", silver_directories["clickstream"], spark)
    attributes_df = _read_silver(snapshot_date_str, "attributes", silver_directories["attributes"], spark)
    financials_df = _read_silver(snapshot_date_str, "financials", silver_directories["financials"], spark)

    clickstream_features = [f"fe_{idx}" for idx in range(1, 21)]
    clickstream_df = clickstream_df.withColumn(
        "clickstream_mean", reduce(lambda left, right: left + right, [col(c) for c in clickstream_features]) / F.lit(len(clickstream_features))
    ).withColumn(
        "clickstream_min", F.least(*[col(c) for c in clickstream_features])
    ).withColumn(
        "clickstream_max", F.greatest(*[col(c) for c in clickstream_features])
    )

    df = (
        clickstream_df.join(attributes_df, ["Customer_ID", "snapshot_date"], "inner")
        .join(financials_df, ["Customer_ID", "snapshot_date"], "inner")
    )

    df = df.withColumn("age_bucket", F.when(col("Age") < 25, "18_24")
                       .when(col("Age") < 35, "25_34")
                       .when(col("Age") < 50, "35_49")
                       .otherwise("50_plus"))
    df = df.withColumn("debt_to_income_ratio", col("Outstanding_Debt") / F.when(col("Annual_Income") != 0, col("Annual_Income")))
    df = df.withColumn("emi_to_salary_ratio", col("Total_EMI_per_month") / F.when(col("Monthly_Inhand_Salary") != 0, col("Monthly_Inhand_Salary")))
    df = df.withColumn("available_cash_after_emi", col("Monthly_Inhand_Salary") - col("Total_EMI_per_month"))
    df = df.withColumn("feature_store_def", F.lit("application_snapshot_v1").cast(StringType()))

    selected_columns = [
        "Customer_ID",
        "snapshot_date",
        "feature_store_def",
        "Age",
        "age_bucket",
        "Occupation",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "credit_history_months",
        "Payment_of_Min_Amount",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Payment_Behaviour",
        "Monthly_Balance",
        "has_type_of_loan",
        "debt_to_income_ratio",
        "emi_to_salary_ratio",
        "available_cash_after_emi",
        "clickstream_mean",
        "clickstream_min",
        "clickstream_max",
    ] + clickstream_features

    df = df.select(*selected_columns)

    os.makedirs(gold_directory, exist_ok=True)
    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)
    return df


def process_labels_gold_table(snapshot_date_str, silver_directory, gold_directory, spark, dpd=30, mob=6):
    lms_df = _read_silver(snapshot_date_str, "lms", silver_directory, spark)

    df = (
        lms_df.filter(col("mob") == mob)
        .withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
        .withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
        .select("loan_id", "Customer_ID", "loan_start_date", "label", "label_def", "snapshot_date")
    )

    os.makedirs(gold_directory, exist_ok=True)
    partition_name = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)
    return df


def process_training_gold_table(snapshot_date_str, gold_feature_directory, gold_label_directory, gold_training_directory, spark):
    label_df = spark.read.parquet(
        os.path.join(gold_label_directory, f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet")
    )
    feature_snapshot_date = label_df.select(F.add_months(col("snapshot_date"), -6).alias("feature_snapshot_date")).first()[0]
    feature_snapshot_str = feature_snapshot_date.strftime("%Y_%m_%d")

    feature_df = spark.read.parquet(os.path.join(gold_feature_directory, f"gold_feature_store_{feature_snapshot_str}.parquet"))
    df = (
        label_df.join(
            feature_df,
            (label_df.Customer_ID == feature_df.Customer_ID) & (label_df.loan_start_date == feature_df.snapshot_date),
            "inner",
        )
        .drop(feature_df.Customer_ID)
        .drop(feature_df.snapshot_date)
    )

    os.makedirs(gold_training_directory, exist_ok=True)
    partition_name = f"gold_training_dataset_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_training_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)
    return df
