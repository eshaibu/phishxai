import shap
import matplotlib.pyplot as plt
import logging

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

def plot_global_importance(shap_values, feature_names, outpath: str):
    """
    Save a bar plot of mean |SHAP| values per feature.
    """
    try:
        shap.plots.bar(shap_values, show=False)
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
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        logger.exception("Failed SHAP beeswarm plot: %s", e)
        raise

def plot_local_waterfall(shap_values_row, outpath: str):
    """
    Save a SHAP waterfall for a single prediction explaining feature contributions.
    """
    try:
        shap.plots.waterfall(shap_values_row, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        logger.exception("Failed SHAP waterfall plot: %s", e)
        raise
