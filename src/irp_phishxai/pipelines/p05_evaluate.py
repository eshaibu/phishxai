import joblib
import logging
import os
import time
import yaml

import pandas as pd

from ..config import load_config
from ..utils.io_utils import read_csv_safely, write_csv
from ..utils.logging_utils import setup_logging
from ..utils.metrics_utils import compute_classification_metrics, plot_roc_curve, plot_pr_curve
from ..utils.model_utils import predict_proba_safely

logger = logging.getLogger(__name__)


def main(cfg_path: str, models: list[str] | None = None):
    """
    Stage p05: Evaluate selected models on the test set.
    Outputs:
      - reports/tables/metrics.csv (including timings and model sizes)
      - reports/figures/roc_*.png, pr_*.png
    """
    setup_logging()
    cfg = load_config(cfg_path)

    # Load test data and separate features.
    test = read_csv_safely(os.path.join(cfg["paths"]["processed"], "test_features.csv"))
    X = test.drop(columns=["label"])
    y = test["label"].values

    # Discover available models; optionally filter by CLI list.
    model_dir = cfg["paths"]["models"]
    candidates = [os.path.splitext(f)[0] for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if models:
        candidates = [m for m in candidates if m in models]

    # Load training metadata for fit_time/model_size.
    meta_train = {}
    meta_path = os.path.join(cfg["paths"]["models"], "metadata.yaml")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_train = yaml.safe_load(f) or {}

    rows = []
    for key in candidates:
        model_path = os.path.join(model_dir, f"{key}.joblib")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.exception("Failed loading model %s: %s", model_path, e)
            continue

        # Inference timing (ms per 1k URLs; deterministic & easy to interpret).
        t0 = time.time()
        y_pred = model.predict(X)
        infer_time = time.time() - t0
        ms_per_1k = (infer_time * 1000.0) / max(1, (len(X) / 1000.0))

        # Get probabilities (handles calibrated SVM etc.).
        y_prob = predict_proba_safely(model, X)

        # Compute metrics & merge training metadata.
        m = compute_classification_metrics(y, y_pred, y_prob)
        m["model"] = key
        if key in meta_train:
            m["fit_time_s"] = meta_train[key].get("fit_time_s")
            m["model_size_kb"] = meta_train[key].get("model_size_kb")
        else:
            m["model_size_kb"] = os.path.getsize(model_path) / 1024.0
        m["infer_ms_per_1k"] = round(ms_per_1k, 2)
        rows.append(m)

        # Generate ROC/PR plots (fail-soft: log and continue if an error occurs).
        try:
            plot_roc_curve(y, y_prob, os.path.join(cfg["paths"]["figures"], f"roc_{key}.png"))
            plot_pr_curve(y, y_prob, os.path.join(cfg["paths"]["figures"], f"pr_{key}.png"))
        except Exception as e:
            logger.warning("Plotting error for %s: %s", key, e)

    metrics_df = pd.DataFrame(rows)
    write_csv(metrics_df, os.path.join(cfg["paths"]["tables"], "metrics.csv"))
    logger.info("[evaluate] wrote reports/tables/metrics.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    main(args.config, args.models)
