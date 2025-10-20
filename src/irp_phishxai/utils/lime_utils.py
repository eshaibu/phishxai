import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import logging

from ..utils.model_utils import ensure_named_frame

logger = logging.getLogger(__name__)

def explain_instance_lime(model, X_train, feature_names, class_names, x_row, outpath: str):
    """
    Produce a LIME explanation figure for a single instance:
      - Uses predict_proba if available; falls back to predicted labels otherwise.
      - Saves a PNG plot of feature contributions near the decision boundary.
    """
    try:
        explainer = LimeTabularExplainer(
            X_train, feature_names=feature_names, class_names=class_names, discretize_continuous=True
        )

        def predict_fn(X):
            X_named = ensure_named_frame(X, feature_names)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            pred = model.predict(X)
            return np.vstack([1 - pred, pred]).T

        exp = explainer.explain_instance(x_row, predict_fn, num_features=min(10, len(feature_names)))
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)
    except Exception as e:
        logger.exception("Failed LIME explanation: %s", e)
        raise
