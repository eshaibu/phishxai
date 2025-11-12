import logging

import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)


def compute_treeshap_values(model, X_sample):
    """
    Compute SHAP values for tree-based models using TreeExplainer.
    Notes:
      - Subsample X before calling this to control runtime/memory.
    """
    try:
        explainer = shap.TreeExplainer(model)
        return explainer(X_sample)
    except Exception as e:
        logger.exception("Failed to compute SHAP values: %s", e)
        raise


def select_positive_class_values(shap_values):
    """
    Extract SHAP values for the positive class (index 1) in binary classification.
    Handles TreeExplainer output variations across different model types.

    Args:
        shap_values: SHAP Explanation object

    Returns:
        shap.Explanation: Single-output explanation for positive class
    """
    if not hasattr(shap_values, 'values'):
        return shap_values

    vals = shap_values.values

    # Already single-output (RandomForest, etc.)
    if vals.ndim <= 2:
        return shap_values

    # Multi-output: extract class 1 (XGBoost, LightGBM)
    if vals.ndim == 3 and vals.shape[2] == 2:
        base_vals = shap_values.base_values

        # Extract positive class base values
        if isinstance(base_vals, np.ndarray):
            if base_vals.ndim == 2:
                new_base = base_vals[:, 1]
            elif base_vals.size == 2:
                new_base = float(base_vals.flat[1])
            else:
                new_base = base_vals
        else:
            new_base = base_vals

        return shap.Explanation(
            values=vals[:, :, 1],
            base_values=new_base,
            data=shap_values.data,
            feature_names=shap_values.feature_names
        )

    logger.warning("Unexpected SHAP shape %s, returning original", vals.shape)
    return shap_values


def plot_global_importance(shap_values, feature_names, outpath: str):
    """
    Save a bar plot of mean |SHAP| values per feature.
    """
    try:
        shap_values_pos = select_positive_class_values(shap_values)
        shap.plots.bar(shap_values_pos, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        logger.exception("Failed SHAP bar plot: %s", e)
        raise


def plot_beeswarm(shap_values, X_sample, outpath: str):
    """
    Save a SHAP beeswarm plot to show distribution of impacts per feature.
    """
    try:
        shap_values_pos = select_positive_class_values(shap_values)
        shap.plots.beeswarm(shap_values_pos, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        logger.exception("Failed SHAP beeswarm plot: %s", e)
        raise


def plot_local_waterfall(shap_values, outpath: str):
    """
    Save a SHAP waterfall plot for a single instance.

    Automatically handles:
      - Multi-class output (extracts positive class)
      - Batch format (extracts first instance)
      - Base value scalar conversion

    Args:
        shap_values: SHAP Explanation (batch or single instance)
        outpath: Path to save plot (e.g., 'waterfall_rf.png')

    Raises:
        Exception: If SHAP plot generation fails
    """

    try:
        # Extract positive class values
        shap_values_pos = select_positive_class_values(shap_values)

        # Extract single instance if in batch format
        if hasattr(shap_values_pos, 'values') and shap_values_pos.values.ndim > 1:
            instance = shap_values_pos[0]
        else:
            instance = shap_values_pos

        shap.plots.waterfall(instance, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        logger.exception("Failed SHAP waterfall plot: %s", e)
        raise
