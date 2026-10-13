"""Train and evaluate a LightGBM wafer-failure classifier on SECOM.

Run:  python -m secom.train [--strategy class_weight|smote|both] [--split random|time] [--beta 1.0]

For each imbalance strategy:
  1. 80/20 train / hold-out split (stratified random, or chronological with --split time)
  2. 5-fold stratified CV on the training part -> out-of-fold probabilities
  3. decision threshold chosen on those out-of-fold probabilities (max F-beta)
  4. refit on the full training part, score once on the hold-out
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from secom.data import REPORTS_DIR, load
from secom.pipeline import STRATEGIES, build_pipeline, gain_importance

SEED = 42
TOP_K_STABLE = 20


def best_threshold(y_true, proba, beta: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta**2 * precision + recall
    fbeta = np.where(denom > 0, (1 + beta**2) * precision * recall / np.maximum(denom, 1e-12), 0)
    return float(thresholds[fbeta.argmax()])


def score(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "accuracy": (tp + tn) / len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def cross_validate(pipe, X, y, n_splits: int = 5):
    """Out-of-fold probabilities plus per-fold gain importances."""
    oof = np.zeros(len(y))
    fold_importances = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for train_idx, val_idx in cv.split(X, y):
        fold_pipe = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fold_pipe.predict_proba(X.iloc[val_idx])[:, 1]
        fold_importances.append(gain_importance(fold_pipe))
    return oof, pd.DataFrame(fold_importances).fillna(0.0)


def run_strategy(strategy, X_train, y_train, X_test, y_test, beta):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipe = build_pipeline(strategy, pos_weight=pos_weight, seed=SEED)

    oof, fold_imp = cross_validate(pipe, X_train, y_train)
    threshold = best_threshold(y_train, oof, beta)

    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "strategy": strategy,
        "pipe": pipe,
        "threshold": threshold,
        "fold_importances": fold_imp,
        "cv": score(y_train, oof, threshold),
        "test_default": score(y_test, test_proba, 0.5),
        "test_tuned": score(y_test, test_proba, threshold),
    }


def print_metrics(result, n_test_fail: int) -> None:
    print(f"\n--- {result['strategy']} ---")
    rows = {
        "CV out-of-fold @ tuned thr": result["cv"],
        "Hold-out @ 0.50": result["test_default"],
        f"Hold-out @ tuned thr ({result['threshold']:.3f})": result["test_tuned"],
    }
    table = pd.DataFrame(rows).T
    table[["tp", "fp", "fn", "tn"]] = table[["tp", "fp", "fn", "tn"]].astype(int)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    t = result["test_tuned"]
    print(f"Hold-out at tuned threshold: catches {t['tp']} of {n_test_fail} failures, "
          f"raising {t['fp']} false alarms.")


def importance_table(result, X, y, top_n: int = 10) -> pd.DataFrame:
    final = gain_importance(result["pipe"])
    folds = result["fold_importances"]
    in_top = (folds.rank(axis=1, ascending=False) <= TOP_K_STABLE).sum()
    table = pd.DataFrame({"gain_share": final}).sort_values("gain_share", ascending=False).head(top_n)
    table["cv_gain_share"] = folds.mean().reindex(table.index).fillna(0.0)
    table[f"folds_in_top{TOP_K_STABLE}"] = in_top.reindex(table.index).fillna(0).astype(int)
    table["median_fail"] = X.loc[y == 1, table.index].median()
    table["median_pass"] = X.loc[y == 0, table.index].median()
    table["pct_missing"] = X[table.index].isna().mean() * 100
    table["n_distinct"] = X[table.index].nunique()
    table.index.name = "sensor"
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=STRATEGIES + ("both",), default="both")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="F-beta used to pick the threshold; >1 favours recall (default 1.0)")
    parser.add_argument("--split", choices=("random", "time"), default="random",
                        help="random: stratified shuffle; time: train on the earliest wafers, "
                             "test on the latest (how the model would actually be deployed)")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y, timestamps = load()
    if args.split == "time":
        order = timestamps.sort_values(kind="stable").index
        cut = int(round(len(order) * (1 - args.test_size)))
        X_train, X_test = X.loc[order[:cut]], X.loc[order[cut:]]
        y_train, y_test = y.loc[order[:cut]], y.loc[order[cut:]]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=SEED
        )
    print(f"Split: {args.split}")
    print(f"Train: {len(y_train)} wafers ({y_train.sum()} fail)   "
          f"Hold-out: {len(y_test)} wafers ({y_test.sum()} fail)")
    print(f"Baseline: always-pass accuracy {1 - y_test.mean():.3f}, "
          f"random-ranking PR-AUC {y_test.mean():.3f}")

    strategies = STRATEGIES if args.strategy == "both" else (args.strategy,)
    results = [run_strategy(s, X_train, y_train, X_test, y_test, args.beta) for s in strategies]
    for r in results:
        print_metrics(r, int(y_test.sum()))

    # Importances come from the class-weighted model whenever it was trained: it only ever
    # sees real wafers. SMOTE interpolates between failures, which puts synthetic rows at
    # values a discrete or zero-inflated sensor never takes, and the trees then earn split
    # gain just by spotting those rows.
    by_name = {r["strategy"]: r for r in results}
    best = by_name.get("class_weight", results[0])

    table = importance_table(best, X_train, y_train)
    print(f"\nTop 10 sensor signals driving failure predictions ({best['strategy']} model)")
    print("gain_share = share of total split gain in the final model; "
          f"folds_in_top{TOP_K_STABLE} = of 5 CV refits, how many rank it in the top {TOP_K_STABLE}")
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    if best["strategy"] == "smote":
        print("Caution: SMOTE inflates the importance of sensors with few distinct values "
              "(see n_distinct); cross-check with --strategy class_weight.")

    REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORTS_DIR / "top_sensors.csv")
    pd.DataFrame(
        {(r["strategy"], k): r[k] for r in results for k in ("cv", "test_default", "test_tuned")}
    ).T.to_csv(REPORTS_DIR / "metrics.csv")
    print(f"\nSaved {REPORTS_DIR.name}/top_sensors.csv and {REPORTS_DIR.name}/metrics.csv")


if __name__ == "__main__":
    main()
