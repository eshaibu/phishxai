import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .io_utils import write_csv, save_json

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


def save_global_importance_values(shap_values, feature_names, outpath: str) -> pd.DataFrame:
    """
    Save quantitative global feature importance (mean |SHAP|) to CSV.

    Args:
        shap_values: SHAP Explanation object
        feature_names: List of feature names
        outpath: CSV output path (e.g., 'shap_importance_rf.csv')

    Returns:
        DataFrame with feature importance rankings
    """
    try:
        shap_values_pos = select_positive_class_values(shap_values)

        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values_pos.values).mean(axis=0)

        # Create DataFrame sorted by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        # Add rank column
        importance_df['rank'] = range(1, len(importance_df) + 1)

        write_csv(importance_df, outpath)
        logger.info("Saved SHAP importance values to %s", outpath)
        return importance_df

    except Exception as e:
        logger.exception("Failed to save SHAP importance values: %s", e)
        raise


def save_local_shap_values(shap_values, feature_names, instance_idx: int, outpath: str):
    """
    Save SHAP values for a single instance to CSV with metadata JSON.

    Args:
        shap_values: SHAP Explanation object (should be single instance)
        feature_names: List of feature names
        instance_idx: Index of instance being explained
        outpath: CSV output path (e.g., 'shap_local_rf_idx42.csv')
    """
    try:
        shap_values_pos = select_positive_class_values(shap_values)

        # Extract single instance
        if shap_values_pos.values.ndim > 1:
            instance_shap = shap_values_pos.values[0]
            instance_data = shap_values_pos.data[0] if shap_values_pos.data is not None else None
            base_value = shap_values_pos.base_values[0] if hasattr(shap_values_pos.base_values,
                                                                   '__len__') else shap_values_pos.base_values
        else:
            instance_shap = shap_values_pos.values
            instance_data = shap_values_pos.data
            base_value = shap_values_pos.base_values

        # Create DataFrame with feature values and SHAP contributions
        local_data = {
            'feature': feature_names,
            'shap_value': instance_shap
        }

        if instance_data is not None:
            local_data['feature_value'] = instance_data

        local_df = pd.DataFrame(local_data)
        local_df = local_df.sort_values('shap_value', key=abs, ascending=False).reset_index(drop=True)

        # Save CSV using utility function
        write_csv(local_df, outpath)

        # Save metadata as separate JSON file
        metadata = {
            'instance_idx': int(instance_idx),
            'base_value': float(base_value),
            'shap_sum': float(instance_shap.sum()),
            'prediction': float(base_value + instance_shap.sum())
        }
        metadata_path = outpath.replace('.csv', '_metadata.json')
        save_json(metadata, metadata_path)

        logger.info("Saved local SHAP values to %s", outpath)

    except Exception as e:
        logger.exception("Failed to save local SHAP values: %s", e)
        raise


def generate_shap_explanations_with_values(
        model,
        model_key: str,
        test_sample: pd.DataFrame,
        test_full: pd.DataFrame,
        feature_names: list,
        selected_idx: int,
        output_dir: str
) -> tuple:
    """
    Generate SHAP global and local explanations with quantitative outputs.

    Returns:
        (success: bool, importance_df: pd.DataFrame or None)
    """
    try:
        # Create values subdirectory
        values_dir = os.path.join(output_dir, "values")
        os.makedirs(values_dir, exist_ok=True)

        # Global explanations
        sv = compute_treeshap_values(model, test_sample)
        plot_global_importance(sv, feature_names, os.path.join(output_dir, f"shap_bar_{model_key}.png"))
        plot_beeswarm(sv, test_sample, os.path.join(output_dir, f"shap_beeswarm_{model_key}.png"))

        # Save global importance values
        importance_df = save_global_importance_values(
            sv,
            feature_names,
            os.path.join(values_dir, f"shap_importance_{model_key}.csv")
        )

        # Local waterfall
        sv_local = compute_treeshap_values(model, test_full.iloc[[selected_idx]][feature_names])
        plot_local_waterfall(sv_local, os.path.join(output_dir, f"shap_waterfall_{model_key}.png"))

        # Save local SHAP values
        save_local_shap_values(
            sv_local,
            feature_names,
            selected_idx,
            os.path.join(values_dir, f"shap_local_{model_key}_idx{selected_idx}.csv")
        )

        return True, importance_df

    except Exception as e:
        logger.warning("SHAP explanations failed for %s: %s", model_key, e)
        return False, None
