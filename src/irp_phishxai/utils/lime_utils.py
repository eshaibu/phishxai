import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from .io_utils import write_csv, save_json

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

    Returns:
        exp: LIME Explanation object
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

        return exp

    except Exception as e:
        logger.exception("Failed LIME explanation: %s", e)
        raise


def save_lime_weights(exp, outpath: str, predicted_class: int = 1):
    """
    Save LIME feature weights to CSV with metadata JSON.

    Args:
        exp: LIME Explanation object
        outpath: CSV output path (e.g., 'lime_weights_rf_idx42.csv')
        predicted_class: Class to extract weights for (1 for phishing)
    """
    try:
        # Extract feature weights for predicted class
        weights = exp.as_list(label=predicted_class)

        lime_df = pd.DataFrame(weights, columns=['feature_condition', 'weight'])
        lime_df = lime_df.sort_values('weight', key=abs, ascending=False).reset_index(drop=True)

        # Save CSV using utility function
        write_csv(lime_df, outpath)

        # Save prediction probabilities as separate JSON
        metadata = {
            'prediction_proba_class_0': float(exp.predict_proba[0]),
            'prediction_proba_class_1': float(exp.predict_proba[1]),
            'explained_class': int(predicted_class)
        }
        metadata_path = outpath.replace('.csv', '_metadata. json')
        save_json(metadata, metadata_path)

        logger.info("Saved LIME weights to %s", outpath)

    except Exception as e:
        logger.exception("Failed to save LIME weights: %s", e)
        raise


def generate_lime_explanation_with_values(
        model,
        model_key: str,
        X_train: pd.DataFrame,
        test_full: pd.DataFrame,
        feature_names: list,
        selected_idx: int,
        figures_dir: str,
        tables_dir: str
) -> bool:
    """
    Generate LIME local explanation with quantitative outputs.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate explanation
        exp = explain_instance_lime(
            model,
            X_train,
            feature_names,
            CLASS_NAMES,
            test_full.iloc[selected_idx][feature_names],
            os.path.join(figures_dir, f"lime_{model_key}.png")
        )

        # Save weights
        save_lime_weights(
            exp,
            os.path.join(tables_dir, f"lime_weights_{model_key}_idx{selected_idx}.csv")
        )

        return True

    except Exception as e:
        logger.warning("LIME explanation failed for %s: %s", model_key, e)
        return False
