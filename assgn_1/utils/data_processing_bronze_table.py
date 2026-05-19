import os
from datetime import datetime

from pyspark.sql.functions import col


SOURCE_TABLES = {
    "clickstream": "feature_clickstream.csv",
    "attributes": "features_attributes.csv",
    "financials": "features_financials.csv",
    "lms": "lms_loan_daily.csv",
}


def process_bronze_table(snapshot_date_str, source_name, bronze_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
    csv_file_path = os.path.join("data", SOURCE_TABLES[source_name])

    df = (
        spark.read.csv(csv_file_path, header=True, inferSchema=True)
        .filter(col("snapshot_date").cast("date") == snapshot_date)
    )

    os.makedirs(bronze_directory, exist_ok=True)
    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f"bronze {source_name} {snapshot_date_str} row count: {df.count()}")
    print("saved to:", filepath)

    return df
