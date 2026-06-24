FROM apache/airflow:2.9.3-python3.11

USER root
RUN mkdir -p /opt/airflow/project && chown -R airflow:0 /opt/airflow/project /opt/airflow

USER airflow
WORKDIR /opt/airflow/project
COPY requirements.txt /tmp/assignment_2_requirements.txt
RUN pip install --no-cache-dir -r /tmp/assignment_2_requirements.txt
