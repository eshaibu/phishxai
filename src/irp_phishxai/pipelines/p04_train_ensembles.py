import logging
import os
import time
import warnings

from ..config import load_config
from ..utils.io_utils import read_csv_safely, save_yaml, timestamped_run_dir
from ..utils.logging_utils import setup_logging
from ..utils.model_utils import get_ensemble_models, get_param_grids, fit_with_grid, merge_and_save_metadata, persist_model

warnings.filterwarnings('ignore', message='.*No further splits.*')

logger = logging.getLogger(__name__)


def main(cfg_path: str, models: list[str] | None = None):
    """
    Stage p04: Train ensemble models with optional grid search and record:
      - Fit time (seconds), the best parameters, model size (KB)
    Append results to models/metadata.yaml.
    """
    setup_logging()
    cfg = load_config(cfg_path)
    run_dir = timestamped_run_dir("experiments/runs")
    save_yaml(cfg, os.path.join(run_dir, "config_snapshot.yaml"))

    train = read_csv_safely(os.path.join(cfg["paths"]["processed"], "train_features.csv"))
    X = train.drop(columns=["label"])
    y = train["label"].values
    feature_names = list(X.columns)

    all_ens = get_ensemble_models(cfg)
    grids = get_param_grids(cfg)
    metrics = {}

    for key, maker in all_ens.items():
        if models and key not in models:
            continue
        logger.info("[train-ensemble] %s", key)
        model = maker()

        t0 = time.time()
        best, params = fit_with_grid(model, grids.get(key, {}), X, y, scoring="f1_macro", cv=3)
        fit_time = time.time() - t0

        persist_model(best, os.path.join(cfg["paths"]["models"], f"{key}.joblib"))
        size_kb = os.path.getsize(os.path.join(cfg["paths"]["models"], f"{key}.joblib")) / 1024.0

        metrics[key] = {
            "best_params": params,
            "model_path": f"{key}.joblib",
            "fit_time_s": round(fit_time, 4),
            "model_size_kb": round(size_kb, 2),
            "feature_names": feature_names,
        }

    meta_path = os.path.join(cfg["paths"]["models"], "metadata.yaml")
    merge_and_save_metadata(meta_path, metrics, run_dir, "ensemble_training")
    logger.info("[train-ensemble] done. run_dir=%s", run_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    main(args.config, args.models)
