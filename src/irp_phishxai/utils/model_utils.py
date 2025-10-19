from typing import Dict, Callable, Any
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)

def get_baseline_models(cfg: Dict) -> Dict[str, Callable[[], Any]]:
    """
    Return a mapping {key -> factory()} for baseline models.
    Notes:
      - LinearSVM is wrapped in CalibratedClassifierCV to provide calibrated probabilities,
        making ROC/PR comparisons fair vs. tree ensembles.
    """
    return {
        "lr": lambda: LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=1000),
        "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=1.0), cv=3, method="sigmoid"),
        "dt": lambda: DecisionTreeClassifier(),
    }

def get_ensemble_models(cfg: Dict) -> Dict[str, Callable[[], Any]]:
    """
    Return a mapping {key -> factory()} for ensemble models.
    Defaults are small and fast; hyperparams will be refined by optional grid search.
    """
    return {
        "rf": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight=None),
        "xgb": lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=1.0,
            colsample_bytree=1.0, tree_method="hist", eval_metric="logloss", n_jobs=-1
        ),
        "lgbm": lambda: lgb.LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.1, n_jobs=-1
        ),
    }

def get_param_grids(cfg: Dict) -> Dict[str, Dict]:
    """
    Pass through param grids from config; return empty dict if absent.
    """
    return cfg.get("param_grids", {})

def fit_with_grid(model, param_grid: Dict, X, y, scoring="f1_macro", cv=3):
    """
    Fit a model with GridSearchCV if a param grid is provided.
    Returns:
      - best_estimator_: trained estimator
      - best_params: dictionary of chosen hyperparameters (or empty if no grid)
    """
    if not param_grid:
        model.fit(X, y)
        return model, {}
    try:
        gs = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_
    except Exception as e:
        logger.exception("Grid search failed: %s", e)
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
