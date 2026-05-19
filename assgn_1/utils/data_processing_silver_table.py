import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType


def _read_bronze(snapshot_date_str, source_name, bronze_directory, spark):
    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    return spark.read.csv(filepath, header=True, inferSchema=True)


def _save_silver(df, snapshot_date_str, source_name, silver_directory):
    os.makedirs(silver_directory, exist_ok=True)
    partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)
    return filepath


def _clean_numeric(column_name):
    return F.regexp_replace(col(column_name).cast(StringType()), "[^0-9.\\-]", "").cast(DoubleType())


def _credit_history_months(column_name):
    years = F.regexp_extract(col(column_name).cast(StringType()), r"(\d+)\s+Years?", 1).cast(IntegerType())
    months = F.regexp_extract(col(column_name).cast(StringType()), r"(\d+)\s+Months?", 1).cast(IntegerType())
    return F.coalesce(years, F.lit(0)) * F.lit(12) + F.coalesce(months, F.lit(0))


def process_clickstream_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, "clickstream", bronze_directory, spark)

    for idx in range(1, 21):
        df = df.withColumn(f"fe_{idx}", col(f"fe_{idx}").cast(DoubleType()))

    df = (
        df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
        .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
        .dropDuplicates(["Customer_ID", "snapshot_date"])
    )

    _save_silver(df, snapshot_date_str, "clickstream", silver_directory)
    return df


def process_attributes_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, "attributes", bronze_directory, spark)

    df = (
        df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
        .withColumn("Age", _clean_numeric("Age").cast(IntegerType()))
        .withColumn("Occupation", F.when(col("Occupation") == "_", None).otherwise(col("Occupation")).cast(StringType()))
        .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
        .select("Customer_ID", "Age", "Occupation", "snapshot_date")
        .dropDuplicates(["Customer_ID", "snapshot_date"])
    )

    _save_silver(df, snapshot_date_str, "attributes", silver_directory)
    return df


def process_financials_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, "financials", bronze_directory, spark)

    double_columns = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    integer_columns = [
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
    ]

    for column_name in double_columns:
        df = df.withColumn(column_name, _clean_numeric(column_name))
    for column_name in integer_columns:
        df = df.withColumn(column_name, _clean_numeric(column_name).cast(IntegerType()))

    df = (
        df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
        .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
        .withColumn("credit_history_months", _credit_history_months("Credit_History_Age"))
        .withColumn("Credit_Mix", F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))
        .withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount") == "NM", None).otherwise(col("Payment_of_Min_Amount")))
        .withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour")))
        .withColumn("has_type_of_loan", F.when(F.length(F.trim(col("Type_of_Loan"))) > 0, 1).otherwise(0))
        .select(
            "Customer_ID",
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
            "snapshot_date",
        )
        .dropDuplicates(["Customer_ID", "snapshot_date"])
    )

    _save_silver(df, snapshot_date_str, "financials", silver_directory)
    return df


def process_lms_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    df = _read_bronze(snapshot_date_str, "lms", bronze_directory, spark)

    df = (
        df.withColumn("loan_id", col("loan_id").cast(StringType()))
        .withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
        .withColumn("loan_start_date", col("loan_start_date").cast(DateType()))
        .withColumn("tenure", col("tenure").cast(IntegerType()))
        .withColumn("installment_num", col("installment_num").cast(IntegerType()))
        .withColumn("loan_amt", col("loan_amt").cast(DoubleType()))
        .withColumn("due_amt", col("due_amt").cast(DoubleType()))
        .withColumn("paid_amt", col("paid_amt").cast(DoubleType()))
        .withColumn("overdue_amt", col("overdue_amt").cast(DoubleType()))
        .withColumn("balance", col("balance").cast(DoubleType()))
        .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    )

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()),
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()),
    )

    _save_silver(df, snapshot_date_str, "lms", silver_directory)
    return df
