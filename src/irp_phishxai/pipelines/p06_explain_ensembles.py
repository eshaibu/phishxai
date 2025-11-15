import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ..config import load_config
from ..utils.io_utils import read_csv_safely, write_csv
from ..utils.lime_utils import generate_lime_explanation
from ..utils.logging_utils import setup_logging
from ..utils.model_utils import predict_proba_safely
from ..utils.shap_utils import generate_shap_explanations

logger = logging.getLogger(__name__)


def select_hard_instance(model, X: pd.DataFrame, y: np.ndarray) -> int:
    """
    Select a challenging instance for local explanation.

    Strategy:
      1. If misclassifications exist, pick the first error
      2. Otherwise, pick the most uncertain prediction (prob closest to 0.5)

    Args:
        model: Trained classifier
        X: Test features (DataFrame)
        y: True labels

    Returns:
        Index of selected instance
    """
    preds = model.predict(X)
    errors = np.where(preds != y)[0]

    if len(errors) > 0:
        return int(errors[0])

    # No errors - find most uncertain prediction
    probs = predict_proba_safely(model, X)
    pos_probs = probs[:, 1] if probs.ndim == 2 else probs
    return int(np.argmin(np.abs(pos_probs - 0.5)))


def explain_single_model(
        model_key: str,
        cfg: dict,
        test: pd.DataFrame,
        feature_names: list,
        output_dir: str
) -> Optional[dict]:
    """
    Generate all explanations for a single model.

    Returns:
        Metadata dict if successful, None otherwise
    """
    model_path = os.path.join(cfg["paths"]["models"], f"{model_key}.joblib")
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_key)
        return None

    model = joblib.load(model_path)

    # Prepare data
    X = test[feature_names]
    y = test["label"].values

    # Subsample for SHAP global plots
    n_samples = min(cfg["shap"]["sample_size"], len(test))
    rng = np.random.default_rng(cfg["seed"])
    sample_idx = rng.choice(len(test), size=n_samples, replace=False)
    test_sample = test.iloc[sample_idx][feature_names]

    # Select challenging instance
    selected_idx = select_hard_instance(model, X, y)

    # Generate explanations
    shap_success = generate_shap_explanations(
        model, model_key, test_sample, test, feature_names, selected_idx, output_dir
    )

    lime_success = generate_lime_explanation(
        model, model_key, X, test, feature_names, selected_idx, output_dir
    )

    return {
        "model": model_key,
        "shap_sample_size": int(n_samples),
        "lime_idx": int(selected_idx),
        "shap_success": shap_success,
        "lime_success": lime_success
    }


def main(cfg_path: str, models: Optional[list[str]] = None):
    """
    Generate SHAP and LIME explanations for trained models.

    Args:
        cfg_path: Path to configuration file
        models: Optional list of model keys to explain (defaults to config)
    """
    setup_logging()
    cfg = load_config(cfg_path)

    test = read_csv_safely(os.path.join(cfg["paths"]["processed"], "test_features.csv"))
    feature_names = [c for c in test.columns if c != "label"]

    explain_models = models or cfg["models"]["explain"]
    output_dir = os.path.join(cfg["paths"]["reports"], "figures")

    # Generate explanations for each model
    results = []
    for model_key in explain_models:
        metadata = explain_single_model(model_key, cfg, test, feature_names, output_dir)
        if metadata:
            results.append(metadata)

    # Save metadata
    if results:
        output_table = os.path.join(cfg["paths"]["reports"], "tables", "feature_rankings.csv")
        write_csv(pd.DataFrame(results), output_table)
        logger.info("Generated explanations for %d models", len(results))
    else:
        logger.warning("No explanations generated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate model explanations")
    parser.add_argument("--config", default="experiments/configs/starter.yaml")
    parser.add_argument("--models", nargs="*", help="Models to explain (defaults to config)")
    args = parser.parse_args()

    main(args.config, args.models)
