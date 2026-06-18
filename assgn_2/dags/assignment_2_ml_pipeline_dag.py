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
from utils.package_submission import package_code_artifacts


default_args = {"owner": "mle", "retries": 0}

with DAG(
    dag_id="assignment_2_end_to_end_ml_pipeline",
    description="Train, score, monitor, and package a loan default ML pipeline.",
    default_args=default_args,
    start_date=datetime(2026, 6, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mle", "assignment_2", "loan_risk"],
) as dag:
    build_datamart_task = PythonOperator(
        task_id="build_feature_label_datamart",
        python_callable=build_datamart,
    )

    train_model_task = PythonOperator(
        task_id="train_evaluate_register_best_model",
        python_callable=train_and_register_model,
    )

    inference_task = PythonOperator(
        task_id="batch_inference_to_gold_predictions",
        python_callable=run_batch_inference,
    )

    monitoring_task = PythonOperator(
        task_id="monitor_performance_and_stability",
        python_callable=monitor_model,
    )

    package_task = PythonOperator(
        task_id="package_code_artifacts",
        python_callable=package_code_artifacts,
        op_kwargs={"project_root": PROJECT_ROOT},
    )

    build_datamart_task >> train_model_task >> inference_task >> monitoring_task >> package_task
