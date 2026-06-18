import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import DATAMART_DIR, MODEL_BANK_DIR, REPORTS_DIR

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - local fallback when matplotlib is unavailable.
    plt = None


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        cuts = np.linspace(0, 1, bins + 1)
    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = y_true.astype(int).to_numpy()
    values = score.to_numpy()
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    pos = y == 1
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _load_predictions() -> pd.DataFrame:
    path = DATAMART_DIR / "gold" / "model_predictions" / "gold_model_predictions_all.csv"
    if not path.exists():
        raise FileNotFoundError("Prediction gold table is missing. Run batch inference first.")
    return pd.read_csv(path, parse_dates=["loan_start_date", "feature_snapshot_date", "label_snapshot_date"])


def _write_monitoring_plot(monthly: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        _write_monitoring_svg(monthly, output.with_suffix(".svg"))
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(monthly["label_month"], monthly["auc"], marker="o", color="#155E75", label="AUC")
    axes[0].plot(monthly["label_month"], monthly["accuracy"], marker="o", color="#6D28D9", label="Accuracy")
    axes[0].axhline(0.70, color="#DC2626", linestyle="--", linewidth=1.2, label="AUC guardrail")
    axes[0].set_ylabel("Performance")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].legend(loc="lower left", ncol=3)

    axes[1].bar(monthly["label_month"], monthly["score_psi"], color="#F97316", alpha=0.85, label="Score PSI")
    axes[1].axhline(0.20, color="#DC2626", linestyle="--", linewidth=1.2, label="PSI guardrail")
    axes[1].set_ylabel("Stability")
    axes[1].set_xlabel("Label snapshot month")
    axes[1].legend(loc="upper left")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Monthly model performance and score stability", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_monitoring_svg(monthly: pd.DataFrame, output: Path) -> None:
    width, height = 1120, 680
    margin_left, margin_top, plot_w, plot_h = 90, 80, 940, 460
    months = monthly["label_month"].tolist()
    xs = np.linspace(margin_left, margin_left + plot_w, len(months))

    def y_perf(value: float) -> float:
        return margin_top + (1 - (value - 0.45) / 0.55) * 210

    max_psi = max(float(monthly["score_psi"].max()), 0.25)

    def y_psi(value: float) -> float:
        return margin_top + 280 + (1 - value / max_psi) * 180

    auc_points = " ".join(f"{x:.1f},{y_perf(v):.1f}" for x, v in zip(xs, monthly["auc"].fillna(0.45)))
    acc_points = " ".join(f"{x:.1f},{y_perf(v):.1f}" for x, v in zip(xs, monthly["accuracy"]))
    bars = []
    for x, psi in zip(xs, monthly["score_psi"]):
        y = y_psi(float(psi))
        bars.append(f'<rect x="{x - 14:.1f}" y="{y:.1f}" width="28" height="{margin_top + 460 - y:.1f}" fill="#F97316" opacity="0.82"/>')
    labels = []
    for idx, (x, month) in enumerate(zip(xs, months)):
        if idx % 2 == 0 or idx == len(months) - 1:
            labels.append(f'<text x="{x:.1f}" y="620" transform="rotate(45 {x:.1f},620)" font-size="12" fill="#334155">{month}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#F8FAFC"/>
<text x="70" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#0F172A">Monthly model performance and score stability</text>
<line x1="{margin_left}" y1="{margin_top + 210}" x2="{margin_left + plot_w}" y2="{margin_top + 210}" stroke="#CBD5E1"/>
<line x1="{margin_left}" y1="{margin_top + 460}" x2="{margin_left + plot_w}" y2="{margin_top + 460}" stroke="#CBD5E1"/>
<line x1="{margin_left}" y1="{y_perf(0.70):.1f}" x2="{margin_left + plot_w}" y2="{y_perf(0.70):.1f}" stroke="#DC2626" stroke-dasharray="6,6"/>
<line x1="{margin_left}" y1="{y_psi(0.20):.1f}" x2="{margin_left + plot_w}" y2="{y_psi(0.20):.1f}" stroke="#DC2626" stroke-dasharray="6,6"/>
<polyline points="{auc_points}" fill="none" stroke="#155E75" stroke-width="4"/>
<polyline points="{acc_points}" fill="none" stroke="#6D28D9" stroke-width="4"/>
{''.join(bars)}
<text x="70" y="105" font-family="Arial" font-size="13" fill="#155E75">AUC</text>
<text x="130" y="105" font-family="Arial" font-size="13" fill="#6D28D9">Accuracy</text>
<text x="220" y="105" font-family="Arial" font-size="13" fill="#F97316">Score PSI</text>
<text x="20" y="185" font-family="Arial" font-size="13" fill="#475569">Performance</text>
<text x="20" y="430" font-family="Arial" font-size="13" fill="#475569">Stability</text>
{''.join(labels)}
</svg>'''
    output.write_text(svg, encoding="utf-8")


def monitor_model() -> dict:
    predictions = _load_predictions()
    reference = pd.read_csv(MODEL_BANK_DIR / "reference_scores.csv")
    predictions["label_month"] = predictions["label_snapshot_date"].dt.strftime("%Y-%m")

    rows = []
    reference_scores = reference["score"].to_numpy()
    for month, month_df in predictions.groupby("label_month"):
        y_true = month_df["label"]
        score = month_df["prediction_score"]
        auc = np.nan
        if y_true.nunique() > 1:
            auc = _roc_auc(y_true, score)
        rows.append(
            {
                "label_month": month,
                "rows": int(len(month_df)),
                "observed_default_rate": float(y_true.mean()),
                "predicted_default_rate": float(month_df["prediction"].mean()),
                "average_score": float(score.mean()),
                "auc": float(auc) if not np.isnan(auc) else None,
                "accuracy": float((y_true.astype(int).to_numpy() == month_df["prediction"].astype(int).to_numpy()).mean()),
                "score_psi": _psi(reference_scores, score.to_numpy()),
            }
        )

    monthly = pd.DataFrame(rows).sort_values("label_month")
    output_dir = DATAMART_DIR / "gold" / "model_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "gold_model_monitoring_monthly.csv", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_monitoring_plot(monthly, REPORTS_DIR / "model_monitoring.png")

    summary = {
        "months_monitored": int(len(monthly)),
        "latest_month": str(monthly.iloc[-1]["label_month"]),
        "latest_auc": None if pd.isna(monthly.iloc[-1]["auc"]) else float(monthly.iloc[-1]["auc"]),
        "latest_score_psi": float(monthly.iloc[-1]["score_psi"]),
        "max_score_psi": float(monthly["score_psi"].max()),
        "min_auc": float(monthly["auc"].dropna().min()),
        "guardrail_breaches": {
            "auc_below_0_70": monthly.loc[monthly["auc"].fillna(1) < 0.70, "label_month"].tolist(),
            "score_psi_above_0_20": monthly.loc[monthly["score_psi"] > 0.20, "label_month"].tolist(),
        },
    }
    (output_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
