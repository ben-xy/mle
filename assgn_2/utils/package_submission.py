import shutil
from pathlib import Path


def _ignore(_dir, names):
    ignored = []
    for name in names:
        if name in {"__pycache__", ".DS_Store", "airflow_logs", "submit", "temp"}:
            ignored.append(name)
        elif name.endswith((".pyc", ".pyo")):
            ignored.append(name)
    return ignored


def package_code_artifacts(project_root: Path | None = None) -> str:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    submit_dir = project_root / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)
    zip_base = submit_dir / "assignment_2_code_artifacts"
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()

    include = [
        "dags",
        "utils",
        "data",
        "datamart",
        "model_bank",
        "reports",
        "Dockerfile",
        "docker-compose.yaml",
        "requirements.txt",
        "Readme.txt",
    ]
    staging = submit_dir / "assignment_2_code_artifacts"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in include:
        source = project_root / name
        if source.is_dir():
            shutil.copytree(source, staging / name, ignore=_ignore)
        elif source.exists():
            shutil.copy2(source, staging / name)
    shutil.make_archive(str(zip_base), "zip", root_dir=submit_dir, base_dir="assignment_2_code_artifacts")
    shutil.rmtree(staging)
    return str(zip_base.with_suffix(".zip"))
