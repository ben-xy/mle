# Feature Store Validation Model

This folder contains a simple Spark ML binary classification model used to validate that the gold feature store can train with the gold labels.

Run the notebook from JupyterLab:

```text
model/train_model.ipynb
```

Alternatively, run the script from the `assgn_1` folder after the data pipeline has created `datamart/`:

```bash
python model/train_model.py
```

or:

```bash
./model/train_model.py
```

The script:

- reads `datamart/gold/training_dataset/`
- excludes label, loan identifier, customer identifier, date, and post-application servicing columns from model features
- checks that each label snapshot is exactly 6 months after the loan application snapshot
- trains a logistic regression classifier with a temporal train/test split
- writes the fitted model and reports to `model/artifacts/`

Outputs:

- `model/artifacts/logistic_regression_pipeline/`
- `model/artifacts/metrics.json`
- `model/artifacts/leakage_report.txt`
