import logging
import os
from typing import Dict, Callable, Any

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .io_utils import save_yaml

logger = logging.getLogger(__name__)


def get_baseline_models(cfg: Dict) -> Dict[str, Callable[[], Any]]:
    """
    Return a mapping {key -> factory()} for baseline models.
    Notes:
      - LinearSVM is wrapped in CalibratedClassifierCV to provide calibrated probabilities.
      - Defaults align with Option 2 param grid mid-points.
    """
    return {
        "lr": lambda: LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=1000),
        "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3, method="sigmoid"),
        "dt": lambda: DecisionTreeClassifier(max_depth=12, min_samples_split=10, min_samples_leaf=5),
    }


def get_ensemble_models(cfg: Dict) -> Dict[str, Callable[[], Any]]:
    """
    Return a mapping {key -> factory()} for ensemble models.
    Defaults align with Option 2 param grid mid-points for fair comparison.
    """
    return {
        "rf": lambda: RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=3, n_jobs=-1
        ),
        "xgb": lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.075, subsample=0.9,
            min_child_weight=2, colsample_bytree=1.0, tree_method="hist",
            eval_metric="logloss", n_jobs=-1
        ),
        "lgbm": lambda: lgb.LGBMClassifier(
            n_estimators=600, num_leaves=47, learning_rate=0.075,
            min_child_samples=75, subsample=0.9, n_jobs=-1
        ),
    }


def get_param_grids(cfg: Dict) -> Dict[str, Dict]:
    """
    Pass through param grids from config; return empty dict if absent.
    """
    return cfg.get("param_grids", {})


def fit_with_grid(model, param_grid: Dict, X, y, scoring="f1_macro", cv=3, model_name: str = "model"):
    """
    Fit a model with GridSearchCV if a param grid is provided.

    Args:
        model: Estimator to train
        param_grid: Hyperparameter grid (empty dict = no search)
        X: Training features
        y: Training labels
        scoring: Metric for GridSearchCV (default: f1_macro)
        cv: Number of cross-validation folds
        model_name: Model identifier for logging

    Returns:
        Tuple[estimator, best_params]:
            - best_estimator_: Trained model (refit on full data if grid used)
            - best_params: Selected hyperparameters (empty dict if no grid)
    """
    if not param_grid:
        logger.info("[%s] No param grid - fitting with default params", model_name)
        model.fit(X, y)
        return model, {}

    try:
        # Calculate grid size for logging
        n_combos = np.prod([len(v) for v in param_grid.values()])
        logger.info("[%s] Grid search: %d combinations × %d folds = %d fits",
                    model_name, n_combos, cv, n_combos * cv)

        gs = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X, y)

        logger.info("[%s] Best params: %s (score: %.4f)",
                    model_name, gs.best_params_, gs.best_score_)
        return gs.best_estimator_, gs.best_params_

    except Exception as e:
        logger.exception("[%s] Grid search failed: %s", model_name, e)
        raise


def fit_model(estimator, X, y):
    """
    Direct fit helper for consistency with ensemble/baseline code.
    """
    estimator.fit(X, y)
    return estimator


def predict_proba_safely(model, X):
    """
    Return probabilities if available; otherwise fall back to:
      - decision_function -> min-max scaling to [0,1]
      - predict() -> cast to float (as a last resort)
    For calibrated SVM, predict_proba will exist and be used.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.full((len(scores),), 0.5)
        return (scores - s_min) / (s_max - s_min)
    else:
        preds = model.predict(X)
        return preds.astype(float)


def load_or_init_metadata(meta_path: str) -> dict:
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_and_save_metadata(meta_path: str, new_metrics: dict, run_dir: str, stage_name: str) -> None:
    # Merge into shared metadata
    prev = load_or_init_metadata(meta_path)
    prev.update(new_metrics)
    save_yaml(prev, meta_path)

    # Also save stage-specific snapshot for reproducibility
    save_yaml(new_metrics, os.path.join(run_dir, f"{stage_name}.yaml"))


def persist_model(model, path: str) -> None:
    """
    Save a trained model to disk with joblib; logs errors explicitly.
    """
    try:
        joblib.dump(model, path)
        logger.info("Saved model to %s", path)
    except Exception as e:
        logger.exception("Failed to persist model: %s", e)
        raise
