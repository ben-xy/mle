import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import (
    DATA_DIR,
    DATAMART_DIR,
    LABEL_DPD_THRESHOLD,
    LABEL_MOB,
    NUMERIC_FEATURES,
)


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _get_target_month(execution_date: str | None) -> pd.Period | None:
    if execution_date is None:
        return None
    return pd.to_datetime(execution_date).to_period("M")


def _get_feature_month(label_month: pd.Period) -> pd.Period:
    return label_month - LABEL_MOB


def _filter_by_month(df: pd.DataFrame, date_col: str, target_month: pd.Period) -> pd.DataFrame:
    return df.loc[pd.to_datetime(df[date_col]).dt.to_period("M") == target_month]


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def _credit_history_months(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    years = text.str.extract(r"(\d+)\s+Years?", expand=False).fillna(0).astype(int)
    months = text.str.extract(r"(\d+)\s+Months?", expand=False).fillna(0).astype(int)
    return years * 12 + months


def _write_partitioned_csv(df: pd.DataFrame, folder: Path, stem: str, date_col: str = "snapshot_date") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for date_value, date_df in df.groupby(date_col):
        date_token = pd.to_datetime(date_value).strftime("%Y_%m_%d")
        date_df.to_csv(folder / f"{stem}_{date_token}.csv", index=False)


def _read_raw() -> dict[str, pd.DataFrame]:
    clickstream = pd.read_csv(DATA_DIR / "feature_clickstream.csv", parse_dates=["snapshot_date"])
    attributes = pd.read_csv(DATA_DIR / "features_attributes.csv", parse_dates=["snapshot_date"])
    financials = pd.read_csv(DATA_DIR / "features_financials.csv", parse_dates=["snapshot_date"])
    lms = pd.read_csv(DATA_DIR / "lms_loan_daily.csv", parse_dates=["loan_start_date", "snapshot_date"])
    return {
        "clickstream": clickstream,
        "attributes": attributes,
        "financials": financials,
        "lms": lms,
    }


def _build_silver(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    clickstream = raw["clickstream"].copy()
    for col in [f"fe_{idx}" for idx in range(1, 21)]:
        clickstream[col] = pd.to_numeric(clickstream[col], errors="coerce")
    clickstream = clickstream.drop_duplicates(["Customer_ID", "snapshot_date"])

    attributes = raw["attributes"][["Customer_ID", "Age", "Occupation", "snapshot_date"]].copy()
    attributes["Age"] = _clean_numeric(attributes["Age"]).astype("Int64")
    attributes["Occupation"] = attributes["Occupation"].replace({"_": np.nan})
    attributes = attributes.drop_duplicates(["Customer_ID", "snapshot_date"])

    financials = raw["financials"].copy()
    numeric_columns = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
    ]
    for col in numeric_columns:
        financials[col] = _clean_numeric(financials[col])
    financials["credit_history_months"] = _credit_history_months(financials["Credit_History_Age"])
    financials["Credit_Mix"] = financials["Credit_Mix"].replace({"_": np.nan})
    financials["Payment_of_Min_Amount"] = financials["Payment_of_Min_Amount"].replace({"NM": np.nan})
    financials["Payment_Behaviour"] = financials["Payment_Behaviour"].replace({"!@9#%8": np.nan})
    financials["has_type_of_loan"] = financials["Type_of_Loan"].fillna("").str.strip().ne("").astype(int)
    financials = financials[
        [
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
        ]
    ].drop_duplicates(["Customer_ID", "snapshot_date"])

    lms = raw["lms"].copy()
    lms["mob"] = pd.to_numeric(lms["installment_num"], errors="coerce").fillna(0).astype(int)
    lms["installments_missed"] = np.ceil(
        lms["overdue_amt"].fillna(0) / lms["due_amt"].replace(0, np.nan)
    ).fillna(0).astype(int)
    lms["first_missed_date"] = lms.apply(
        lambda row: row["snapshot_date"] - pd.DateOffset(months=int(row["installments_missed"]))
        if row["installments_missed"] > 0
        else pd.NaT,
        axis=1,
    )
    lms["dpd"] = np.where(
        lms["overdue_amt"].fillna(0) > 0,
        (lms["snapshot_date"] - lms["first_missed_date"]).dt.days.fillna(0),
        0,
    ).astype(int)
    return {
        "clickstream": clickstream,
        "attributes": attributes,
        "financials": financials,
        "lms": lms,
    }


def _build_gold(silver: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    clickstream = silver["clickstream"].copy()
    click_cols = [f"fe_{idx}" for idx in range(1, 21)]
    clickstream["clickstream_mean"] = clickstream[click_cols].mean(axis=1)
    clickstream["clickstream_min"] = clickstream[click_cols].min(axis=1)
    clickstream["clickstream_max"] = clickstream[click_cols].max(axis=1)

    feature_store = (
        clickstream.merge(silver["attributes"], on=["Customer_ID", "snapshot_date"], how="inner")
        .merge(silver["financials"], on=["Customer_ID", "snapshot_date"], how="inner")
    )
    feature_store["age_bucket"] = pd.cut(
        feature_store["Age"].astype(float),
        bins=[0, 24, 34, 49, np.inf],
        labels=["18_24", "25_34", "35_49", "50_plus"],
    ).astype(str)
    feature_store["debt_to_income_ratio"] = feature_store["Outstanding_Debt"] / feature_store["Annual_Income"].replace(0, np.nan)
    feature_store["emi_to_salary_ratio"] = feature_store["Total_EMI_per_month"] / feature_store["Monthly_Inhand_Salary"].replace(0, np.nan)
    feature_store["available_cash_after_emi"] = feature_store["Monthly_Inhand_Salary"] - feature_store["Total_EMI_per_month"]
    feature_store["feature_store_def"] = "application_snapshot_v2_pandas"

    labels = silver["lms"].loc[silver["lms"]["mob"].eq(LABEL_MOB)].copy()
    labels["label"] = labels["dpd"].ge(LABEL_DPD_THRESHOLD).astype(int)
    labels["label_def"] = f"{LABEL_DPD_THRESHOLD}dpd_{LABEL_MOB}mob"
    labels = labels.rename(columns={"snapshot_date": "label_snapshot_date"})[
        ["loan_id", "Customer_ID", "loan_start_date", "label", "label_def", "label_snapshot_date"]
    ]

    training_dataset = labels.merge(
        feature_store,
        left_on=["Customer_ID", "loan_start_date"],
        right_on=["Customer_ID", "snapshot_date"],
        how="inner",
    )
    training_dataset = training_dataset.rename(columns={"snapshot_date": "feature_snapshot_date"})
    training_dataset["feature_snapshot_date"] = pd.to_datetime(training_dataset["feature_snapshot_date"])
    training_dataset["label_snapshot_date"] = pd.to_datetime(training_dataset["label_snapshot_date"])
    training_dataset = training_dataset.drop(columns=["Name", "SSN"], errors="ignore")
    for col in NUMERIC_FEATURES:
        if col in training_dataset.columns:
            training_dataset[col] = pd.to_numeric(training_dataset[col], errors="coerce")

    return {
        "feature_store": feature_store,
        "label_store": labels,
        "training_dataset": training_dataset,
    }


def build_datamart(execution_date: str | None = None) -> dict:
    target_month = _get_target_month(execution_date)
    if target_month is None:
        _clean_dir(DATAMART_DIR)

    raw = _read_raw()
    if target_month is not None:
        feature_month = _get_feature_month(target_month)
        raw = {
            "clickstream": _filter_by_month(raw["clickstream"], "snapshot_date", feature_month),
            "attributes": _filter_by_month(raw["attributes"], "snapshot_date", feature_month),
            "financials": _filter_by_month(raw["financials"], "snapshot_date", feature_month),
            "lms": _filter_by_month(raw["lms"], "snapshot_date", target_month),
        }

    for source_name, df in raw.items():
        _write_partitioned_csv(df, DATAMART_DIR / "bronze" / source_name, f"bronze_{source_name}")

    silver = _build_silver(raw)
    for source_name, df in silver.items():
        _write_partitioned_csv(df, DATAMART_DIR / "silver" / source_name, f"silver_{source_name}")

    gold = _build_gold(silver)
    gold_root = DATAMART_DIR / "gold"
    _write_partitioned_csv(gold["feature_store"], gold_root / "feature_store", "gold_feature_store")
    _write_partitioned_csv(gold["label_store"], gold_root / "label_store", "gold_label_store", "label_snapshot_date")
    _write_partitioned_csv(gold["training_dataset"], gold_root / "training_dataset", "gold_training_dataset", "label_snapshot_date")

    summary = {
        "bronze_sources": {name: int(len(df)) for name, df in raw.items()},
        "silver_sources": {name: int(len(df)) for name, df in silver.items()},
        "gold_tables": {name: int(len(df)) for name, df in gold.items()},
        "label_distribution": gold["training_dataset"]["label"].value_counts().sort_index().astype(int).to_dict(),
    }
    (DATAMART_DIR / "gold" / "datamart_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_training_dataset() -> pd.DataFrame:
    files = sorted((DATAMART_DIR / "gold" / "training_dataset").glob("*.csv"))
    if not files:
        raise FileNotFoundError("Gold training dataset is missing. Run build_datamart first.")
    return pd.concat((pd.read_csv(path, parse_dates=["loan_start_date", "label_snapshot_date", "feature_snapshot_date"]) for path in files), ignore_index=True)
