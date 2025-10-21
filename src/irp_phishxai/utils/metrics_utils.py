import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)


def _ensure_probs(y_prob: np.ndarray) -> np.ndarray:
    """
    Normalize probability-like outputs:
      - If 2D (n,2), return positive class column.
      - If 1D already, pass through unchanged.
    """
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return y_prob[:, 1]
    return y_prob


def fpr_at_tpr(y_true: np.ndarray, y_prob: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    Compute FPR at a given TPR target using the ROC curve.
    Returns NaN if no threshold achieves the requested TPR.
    """
    y_prob = _ensure_probs(y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return float("nan")
    return float(fpr[idx[0]])


def compute_classification_metrics(y_true, y_pred, y_prob) -> Dict:
    """
    Compute core metrics for this project:
      - Accuracy, Macro-F1, ROC-AUC, PR-AUC, FPR@TPR=0.95, Confusion Matrix.
    """
    y_prob = _ensure_probs(y_prob)
    try:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "fpr_at_tpr_0_95": fpr_at_tpr(y_true, y_prob, target_tpr=0.95),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        return metrics
    except Exception as e:
        logger.exception("Failed computing metrics: %s", e)
        raise


def plot_roc_curve(y_true, y_prob, outpath: str) -> None:
    """
    Save a ROC curve PNG for quick visual comparison.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    y_prob = _ensure_probs(y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_pr_curve(y_true, y_prob, outpath: str) -> None:
    """
    Save a Precision–Recall curve PNG (especially informative for positive/rare class).
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    y_prob = _ensure_probs(y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
