# beet/fusion/training.py
"""Utilities for training and persisting the EBM fusion model."""
import pickle
from pathlib import Path
from typing import Any

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    import numpy as np
    _HAS_INTERPRET = True
except ImportError:
    _HAS_INTERPRET = False


def train_ebm(X: list[dict], y: list[int], feature_names: list[str] | None = None) -> Any:
    """
    Train EBM on assembled feature vectors.
    X: list of feature dicts (from FeatureAssembler.assemble)
    y: list of labels (1=LLM, 0=human)
    Returns: trained EBM model
    """
    if not _HAS_INTERPRET:
        raise ImportError("interpretML not installed. Run: pip install 'beet[fusion]'")

    if feature_names is None:
        feature_names = list(X[0].keys())

    X_arr = np.array([[row.get(f, float("nan")) for f in feature_names] for row in X])
    X_arr = np.where(np.isnan(X_arr), 0.5, X_arr)
    y_arr = np.array(y)

    model = ExplainableBoostingClassifier(
        feature_names=feature_names,
        max_rounds=300,
        interactions=5,
        random_state=42,
    )
    model.fit(X_arr, y_arr)
    model._beet_feature_names = feature_names  # attach for later use
    return model


def save_model(model: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
