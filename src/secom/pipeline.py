"""Preprocessing + model pipelines for SECOM.

Every step is fitted inside the pipeline, so under cross-validation the column
filter, imputer, scaler and SMOTE only ever see the training fold.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

STRATEGIES = ("class_weight", "smote")


class ColumnFilter(BaseEstimator, TransformerMixin):
    """Drop columns that are mostly missing or constant; remember the survivors."""

    def __init__(self, max_missing: float = 0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        keep = (X.isna().mean() <= self.max_missing) & (X.nunique() > 1)
        self.kept_columns_ = list(X.columns[keep])
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.kept_columns_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


def build_pipeline(strategy: str, pos_weight: float = 1.0, seed: int = 42) -> Pipeline:
    """strategy='class_weight': up-weight failures in the loss by `pos_weight`.
    strategy='smote': synthesise minority samples to 1:1 on the training fold.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    # Small, wide, noisy data: shallow trees, heavy column subsampling, strong regularisation.
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.3,
        reg_lambda=5.0,
        scale_pos_weight=pos_weight if strategy == "class_weight" else 1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    steps = [
        ("filter", ColumnFilter(max_missing=0.5)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if strategy == "smote":
        # SMOTE interpolates between nearest neighbours, so features must share a scale.
        steps += [("scale", StandardScaler()), ("smote", SMOTE(k_neighbors=5, random_state=seed))]
    steps.append(("model", model))
    return Pipeline(steps)


def gain_importance(pipe: Pipeline) -> pd.Series:
    """Share of total split gain per sensor (sums to 1) for a fitted pipeline."""
    gain = pipe.named_steps["model"].booster_.feature_importance(importance_type="gain")
    names = pipe.named_steps["filter"].get_feature_names_out()
    return pd.Series(gain / gain.sum(), index=names)
