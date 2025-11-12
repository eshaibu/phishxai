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
    Extract SHAP values for the positive class in binary classification.
    Handles multiple output formats from SHAP TreeExplainer.

    Returns:
        shap.Explanation: Single-output explanation for the positive class
    """
    if not hasattr(shap_values, 'values'):
        return shap_values

    vals = shap_values.values
    base_vals = shap_values.base_values

    # Already single-output
    if vals.ndim <= 2:
        return shap_values

    # Multi-output: extract positive class (index 1)
    try:
        # Handle 3D case: (n_samples, n_features, n_classes)
        if vals.ndim == 3 and vals.shape[2] == 2:
            # Extract base_values correctly for different shapes
            if isinstance(base_vals, np.ndarray):
                if base_vals.ndim == 2:
                    new_base = base_vals[:, 1]  # (n_samples, n_classes) -> (n_samples,)
                elif base_vals.ndim == 1 and len(base_vals) == 2:
                    new_base = float(base_vals[1])  # (n_classes,) -> scalar
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

        # Fallback: try ellipsis indexing for unknown shapes
        return shap_values[..., 1]

    except (IndexError, TypeError) as e:
        logger.warning("Could not extract positive class from shape %s: %s. Returning original.", vals.shape, e)
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
    Save a SHAP waterfall for a single prediction explaining feature contributions.
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
