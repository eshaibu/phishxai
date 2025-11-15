import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)

CLASS_NAMES = ["benign", "phish"]


def explain_instance_lime(model, X_train, feature_names, class_names, x_row, outpath: str):
    """
    Produce a LIME explanation figure for a single instance.

    Args:
        model: Trained sklearn model
        X_train: Training features (DataFrame or array)
        feature_names: List of feature column names
        class_names: List of class labels
        x_row: Single instance to explain (DataFrame row or NumPy array)
        outpath: Path to save explanation plot
    """
    try:
        # Convert X_train to NumPy for LIME explainer
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train

        # Convert x_row to NumPy for LIME explain_instance
        x_row_array = x_row.values if hasattr(x_row, 'values') else x_row

        explainer = LimeTabularExplainer(
            X_train_array,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True
        )

        def predict_fn(X):
            """Convert LIME's NumPy arrays to DataFrame for sklearn models."""
            X_df = pd.DataFrame(X, columns=feature_names)

            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_df)
            pred = model.predict(X_df)
            return np.vstack([1 - pred, pred]).T

        exp = explainer.explain_instance(
            x_row_array.ravel(),  # LIME expects 1D array for single instance
            predict_fn,
            num_features=min(10, len(feature_names))
        )
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)
    except Exception as e:
        logger.exception("Failed LIME explanation: %s", e)
        raise


def generate_lime_explanation(
        model,
        model_key: str,
        X_train: pd.DataFrame,
        test_full: pd.DataFrame,
        feature_names: list,
        selected_idx: int,
        output_dir: str
) -> bool:
    """
    Generate LIME local explanation.

    Returns:
        True if successful, False otherwise
    """
    try:
        explain_instance_lime(
            model,
            X_train,
            feature_names,
            CLASS_NAMES,
            test_full.iloc[selected_idx][feature_names],
            os.path.join(output_dir, f"lime_{model_key}.png")
        )
        return True
    except Exception as e:
        logger.warning("LIME explanation failed for %s: %s", model_key, e)
        return False
