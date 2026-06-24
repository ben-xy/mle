from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATAMART_DIR = PROJECT_ROOT / "datamart"
MODEL_BANK_DIR = PROJECT_ROOT / "model_bank"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
LABEL_DPD_THRESHOLD = 30
LABEL_MOB = 6
TRAIN_CUTOFF_DATE = "2024-12-01"
BACKFILL_START_DATE = "2023-07-01"
BACKFILL_END_DATE = "2025-06-01"

ID_COLUMNS = [
    "loan_id",
    "Customer_ID",
    "loan_start_date",
    "label_snapshot_date",
    "feature_snapshot_date",
]

CATEGORICAL_FEATURES = [
    "age_bucket",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

NUMERIC_FEATURES = [
    "Age",
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
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "credit_history_months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "has_type_of_loan",
    "debt_to_income_ratio",
    "emi_to_salary_ratio",
    "available_cash_after_emi",
    "clickstream_mean",
    "clickstream_min",
    "clickstream_max",
] + [f"fe_{idx}" for idx in range(1, 21)]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
