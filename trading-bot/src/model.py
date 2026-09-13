"""ML model wrapper: predicts probability that price rises the next day."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_COLUMNS


class TradingModel:
    """Thin wrapper around a scikit-learn classifier for up/down prediction."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 5, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, dataset: pd.DataFrame) -> "TradingModel":
        X = dataset[FEATURE_COLUMNS]
        y = dataset["target"]
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba_up(self, dataset: pd.DataFrame) -> pd.Series:
        """Probability the next day's close is higher than today's, per row."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet")
        X = dataset[FEATURE_COLUMNS]
        proba = self.clf.predict_proba(X)
        up_idx = list(self.clf.classes_).index(1)
        return pd.Series(proba[:, up_idx], index=dataset.index, name="proba_up")

    def save(self, path: str | Path) -> None:
        joblib.dump(self.clf, path)

    def load(self, path: str | Path) -> "TradingModel":
        self.clf = joblib.load(path)
        self._fitted = True
        return self
