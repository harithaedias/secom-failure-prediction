"""Exploratory look at SECOM: missing values, feature distributions, class imbalance.

Run:  python -m secom.explore
Prints a text summary and writes figures to reports/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from secom.data import REPORTS_DIR, load


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def class_balance(y: pd.Series, ts: pd.Series) -> None:
    section("Class balance")
    n_fail, n_pass = int(y.sum()), int((y == 0).sum())
    print(f"Wafers: {len(y)}   pass: {n_pass}   fail: {n_fail}")
    print(f"Failure rate: {n_fail / len(y):.2%}   imbalance ratio (pass:fail): {n_pass / n_fail:.1f}:1")
    print(f"Accuracy of always predicting 'pass': {n_pass / len(y):.2%}")

    monthly = y.groupby(ts.dt.to_period("M")).agg(["size", "sum", "mean"])
    monthly.columns = ["wafers", "fails", "fail_rate"]
    print(f"\nPeriod: {ts.min()} -> {ts.max()}")
    print(monthly.to_string(float_format=lambda v: f"{v:.2%}"))


def missing_values(X: pd.DataFrame) -> None:
    section("Missing values")
    col_missing = X.isna().mean()
    row_missing = X.isna().mean(axis=1)
    print(f"Overall missing cells: {X.isna().values.mean():.2%}")
    print(f"Columns with any missing value: {(col_missing > 0).sum()} / {X.shape[1]}")
    print(f"Rows with any missing value:    {(row_missing > 0).sum()} / {X.shape[0]}")
    print(f"Per-row missing share: median {row_missing.median():.2%}, max {row_missing.max():.2%}")

    buckets = pd.cut(
        col_missing,
        bins=[-0.001, 0, 0.01, 0.05, 0.2, 0.5, 1.0],
        labels=["0%", "0-1%", "1-5%", "5-20%", "20-50%", ">50%"],
    )
    print("\nColumns by missing share:")
    print(buckets.value_counts().sort_index().to_string())
    print("\n10 most incomplete columns:")
    print(col_missing.sort_values(ascending=False).head(10).to_string(float_format=lambda v: f"{v:.1%}"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(col_missing[col_missing > 0] * 100, bins=40, color="#3b6ea5")
    ax.set(xlabel="% missing in column", ylabel="number of columns",
           title="Missingness across columns with any missing value")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "missingness.png", dpi=130)
    plt.close(fig)


def feature_distributions(X: pd.DataFrame, y: pd.Series) -> None:
    section("Feature distributions")
    nunique = X.nunique()
    constant = nunique[nunique <= 1].index
    print(f"Constant columns (no information): {len(constant)}")
    print(f"Near-constant columns (<= 5 distinct values): {((nunique > 1) & (nunique <= 5)).sum()}")

    varying = X.drop(columns=constant)
    desc = varying.describe().T
    skew = varying.skew()
    z = (varying - varying.mean()) / varying.std()
    outlier_share = (z.abs() > 5).mean()
    print(f"Scale: column std ranges from {desc['std'].min():.2g} to {desc['std'].max():.2g}")
    print(f"Skewness: median |skew| {skew.abs().median():.2f}; "
          f"{(skew.abs() > 2).sum()} columns with |skew| > 2")
    print(f"Columns with any reading beyond 5 std: {(outlier_share > 0).sum()}")

    corr = varying.corr().abs().values
    np.fill_diagonal(corr, 0)
    redundant = (np.triu(corr) > 0.95).any(axis=0).sum()
    print(f"Columns correlated > 0.95 with an earlier column: {redundant}")

    # Univariate pass-vs-fail separation (rank-based, robust to the heavy tails above).
    rows = []
    for col in varying.columns:
        a, b = varying.loc[y == 1, col].dropna(), varying.loc[y == 0, col].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append((col, u / (len(a) * len(b)), p, a.median(), b.median()))
    uni = pd.DataFrame(rows, columns=["sensor", "auc", "p_value", "median_fail", "median_pass"])
    uni["separation"] = (uni["auc"] - 0.5).abs()
    uni = uni.sort_values("separation", ascending=False).set_index("sensor")
    n_tests = len(uni)
    print(f"\nUnivariate separation: {(uni['p_value'] < 0.05 / n_tests).sum()} of {n_tests} "
          f"sensors differ between pass/fail after Bonferroni correction")
    print("Top 10 sensors by single-feature AUC (0.5 = no signal):")
    print(uni.head(10)[["auc", "p_value", "median_fail", "median_pass"]].to_string(
        float_format=lambda v: f"{v:.4g}"))

    top = uni.head(9).index
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), top):
        lo, hi = varying[col].quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(varying.loc[y == 0, col].dropna(), bins=bins, density=True, alpha=0.6,
                color="#3b6ea5", label="pass")
        ax.hist(varying.loc[y == 1, col].dropna(), bins=bins, density=True, alpha=0.6,
                color="#c8553d", label="fail")
        ax.set_title(f"{col} (AUC {uni.loc[col, 'auc']:.2f})", fontsize=10)
    axes[0, 0].legend()
    fig.suptitle("Sensors with the strongest univariate pass/fail separation (1st-99th pct)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "top_univariate_sensors.png", dpi=130)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    X, y, ts = load()
    section("Shape")
    print(f"{X.shape[0]} wafers x {X.shape[1]} sensor signals, all numeric: "
          f"{all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)}")
    class_balance(y, ts)
    missing_values(X)
    feature_distributions(X, y)
    print(f"\nFigures written to {REPORTS_DIR.name}/")


if __name__ == "__main__":
    main()
