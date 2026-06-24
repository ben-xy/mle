import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_processing import build_datamart
from utils.inference import run_batch_inference
from utils.modeling import train_and_register_model
from utils.monitoring import monitor_model
from utils.config import BACKFILL_END_DATE, BACKFILL_START_DATE


default_args = {"owner": "mle", "retries": 0}

with DAG(
    dag_id="assignment_2_end_to_end_ml_pipeline",
    description="Train, score, and monitor a loan default ML pipeline with monthly backfill support.",
    default_args=default_args,
    start_date=datetime.strptime(BACKFILL_START_DATE, "%Y-%m-%d"),
    end_date=datetime.strptime(BACKFILL_END_DATE, "%Y-%m-%d"),
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["mle", "assignment_2", "loan_risk"],
) as dag:
    build_datamart_task = PythonOperator(
        task_id="build_feature_label_datamart",
        python_callable=build_datamart,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    train_model_task = PythonOperator(
        task_id="train_evaluate_register_best_model",
        python_callable=train_and_register_model,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    inference_task = PythonOperator(
        task_id="batch_inference_to_gold_predictions",
        python_callable=run_batch_inference,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    monitoring_task = PythonOperator(
        task_id="monitor_performance_and_stability",
        python_callable=monitor_model,
        op_kwargs={"execution_date": "{{ ds }}"},
    )

    build_datamart_task >> train_model_task >> inference_task >> monitoring_task
