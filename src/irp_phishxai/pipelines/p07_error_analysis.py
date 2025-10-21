import os, joblib, logging
from ..config import load_config
from ..utils.io_utils import read_csv_safely, write_csv
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main(cfg_path: str, model_key: str | None = None):
    """
    Stage p07: Simple error analysis for one or all models.
    If model_key is None, runs for all models in cfg["models"]["train"].
    """
    setup_logging()
    cfg = load_config(cfg_path)

    # Load test set
    test = read_csv_safely(os.path.join(cfg["paths"]["processed"], "test_features.csv"))
    feature_names = [c for c in test.columns if c != "label"]
    X = test[feature_names].values
    y = test["label"].values

    # Determine which models to analyze
    if model_key is None:
        model_keys = cfg["models"]["train"]
        logger.info("[error] Running error analysis for all models: %s", model_keys)
    else:
        model_keys = [model_key]
        logger.info("[error] Running error analysis for %s only", model_key)

    # Loop through each model
    for key in model_keys:
        model_path = os.path.join(cfg["paths"]["models"], f"{key}.joblib")
        if not os.path.exists(model_path):
            logger.warning("[error] Skipping %s (model not found).", key)
            continue

        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.exception("[error] Failed to load model %s: %s", key, e)
            continue

        y_pred = model.predict(X)
        df_err = test.copy()
        df_err["pred"] = y_pred
        df_err["err"] = (df_err["pred"] != df_err["label"]).astype(int)

        agg = (
            df_err.groupby("label")["err"]
            .mean()
            .reset_index()
            .rename(columns={"err": "error_rate"})
        )

        out_path = os.path.join(cfg["paths"]["reports"], "tables", f"errors_{key}.csv")
        write_csv(agg, out_path)
        logger.info("[error] wrote error summary for %s -> %s", key, out_path)

    logger.info("[error] all done.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--model_key", default="xgb")
    args = ap.parse_args()
    main(args.config, args.model_key)
