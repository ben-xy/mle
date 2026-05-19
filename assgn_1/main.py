#!/usr/bin/env python3

import glob
import os

import pyspark

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_gold_table import (
    process_features_gold_table,
    process_labels_gold_table,
    process_training_gold_table,
)
from utils.data_processing_silver_table import (
    process_attributes_silver_table,
    process_clickstream_silver_table,
    process_financials_silver_table,
    process_lms_silver_table,
)
from utils.date_utils import generate_first_of_month_dates

def _count_parquet_folder(spark, folder_path):
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        return 0
    return spark.read.parquet(*files).count()

def main():
    print("\n\nstarting data pipeline\n\n")

    # initialize Spark session
    spark = (
        pyspark.sql.SparkSession.builder.appName("assignment_1_feature_store")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    feature_dates = generate_first_of_month_dates("2023-01-01", "2024-12-01")
    lms_dates = generate_first_of_month_dates("2023-01-01", "2025-11-01")
    label_dates = generate_first_of_month_dates("2023-07-01", "2025-06-01")

    # create datalake directories
    bronze_directories = {
        "clickstream": "datamart/bronze/clickstream/",
        "attributes": "datamart/bronze/attributes/",
        "financials": "datamart/bronze/financials/",
        "lms": "datamart/bronze/lms/",
    }
    silver_directories = {
        "clickstream": "datamart/silver/clickstream/",
        "attributes": "datamart/silver/attributes/",
        "financials": "datamart/silver/financials/",
        "lms": "datamart/silver/lms/",
    }
    # gold directories are created in the respective processing functions
    gold_feature_directory = "datamart/gold/feature_store/"
    gold_label_directory = "datamart/gold/label_store/"
    gold_training_directory = "datamart/gold/training_dataset/"

    # run pipeline
    for source_name in ["clickstream", "attributes", "financials"]:
        for date_str in feature_dates:
            process_bronze_table(date_str, source_name, bronze_directories[source_name], spark)
    
    # run lms processing with different date range
    for date_str in lms_dates:
        process_bronze_table(date_str, "lms", bronze_directories["lms"], spark)

    # run silver processing for features and lms
    for date_str in feature_dates:
        process_clickstream_silver_table(date_str, bronze_directories["clickstream"], silver_directories["clickstream"], spark)
        process_attributes_silver_table(date_str, bronze_directories["attributes"], silver_directories["attributes"], spark)
        process_financials_silver_table(date_str, bronze_directories["financials"], silver_directories["financials"], spark)

    # run silver processing for lms with different date range
    for date_str in lms_dates:
        process_lms_silver_table(date_str, bronze_directories["lms"], silver_directories["lms"], spark)

    # run gold processing for features, labels, and training dataset
    for date_str in feature_dates:
        process_features_gold_table(date_str, silver_directories, gold_feature_directory, spark)
    
    # run gold processing for labels and training dataset with different date range
    for date_str in label_dates:
        process_labels_gold_table(date_str, silver_directories["lms"], gold_label_directory, spark, dpd=30, mob=6)
        process_training_gold_table(date_str, gold_feature_directory, gold_label_directory, gold_training_directory, spark)

    print("\n\n---pipeline summary---")
    print("gold feature store row count:", _count_parquet_folder(spark, gold_feature_directory))
    print("gold label store row count:", _count_parquet_folder(spark, gold_label_directory))
    print("gold training dataset row count:", _count_parquet_folder(spark, gold_training_directory))

    spark.stop()
    print("\n\ncompleted data pipeline\n\n")


if __name__ == "__main__":
    main()
