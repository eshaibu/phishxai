import logging
import os
import time
import yaml

from ..config import load_config
from ..utils.io_utils import read_csv_safely, save_yaml, timestamped_run_dir
from ..utils.logging_utils import setup_logging
from ..utils.model_utils import get_baseline_models, get_param_grids, fit_with_grid, persist_model

logger = logging.getLogger(__name__)


def main(cfg_path: str, models: list[str] | None = None):
    """
    Stage p03: Train the baseline (single) models and record:
      - Best hyperparameters (if grid used)
      - Fit time (seconds)
      - Model size (KB)
    Also merges training metadata into models/metadata.yaml for later evaluation.
    """
    setup_logging()
    cfg = load_config(cfg_path)
    run_dir = timestamped_run_dir("experiments/runs")
    save_yaml(cfg, os.path.join(run_dir, "config_snapshot.yaml"))

    # Load training set and split it into X and y.
    train = read_csv_safely(os.path.join(cfg["paths"]["processed"], "train_features.csv"))
    X = train.drop(columns=["label"]).values
    y = train["label"].values

    all_baselines = get_baseline_models(cfg)
    grids = get_param_grids(cfg)
    metrics = {}

    for key, maker in all_baselines.items():
        if models and key not in models:
            continue
        logger.info("[train-baseline] %s", key)
        model = maker()

        # Time the fit (or grid-search fit).
        t0 = time.time()
        best, params = fit_with_grid(model, grids.get(key, {}), X, y, scoring="f1_macro", cv=3)
        fit_time = time.time() - t0

        # Persist model and capture its on-disk size.
        out_path = os.path.join(cfg["paths"]["models"], f"{key}.joblib")
        persist_model(best, out_path)
        size_kb = os.path.getsize(out_path) / 1024.0

        # Store per-model training metadata.
        metrics[key] = {
            "best_params": params,
            "model_path": out_path,
            "fit_time_s": round(fit_time, 4),
            "model_size_kb": round(size_kb, 2),
        }

    # Merge metadata into a common models/metadata.yaml for evaluation stage.
    meta_path = os.path.join(cfg["paths"]["models"], "metadata.yaml")
    prev = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            prev = yaml.safe_load(f) or {}
    prev.update(metrics)
    save_yaml(prev, meta_path)
    save_yaml(metrics, os.path.join(run_dir, "baseline_training.yaml"))
    logger.info("[train-baseline] done. run_dir=%s", run_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    main(args.config, args.models)
