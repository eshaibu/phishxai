import os, joblib, logging
import pandas as pd
from ..config import load_config
from ..utils.io_utils import read_csv_safely, write_csv
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def main(cfg_path: str, model_key: str = "xgb"):
    """
    Stage p07: Simple error analysis for a chosen model.
    Output:
      - reports/tables/errors_<model>.csv with error rate per true label
    """
    setup_logging()
    cfg = load_config(cfg_path)

    test = read_csv_safely(os.path.join(cfg["paths"]["processed"], "test_features.csv"))
    feature_names = [c for c in test.columns if c != "label"]
    X = test[feature_names].values
    y = test["label"].values

    model_path = os.path.join(cfg["paths"]["models"], f"{model_key}.joblib")
    model = joblib.load(model_path)

    y_pred = model.predict(X)
    df_err = test.copy()
    df_err["pred"] = y_pred
    df_err["err"] = (df_err["pred"] != df_err["label"]).astype(int)

    agg = df_err.groupby("label")["err"].mean().reset_index().rename(columns={"err":"error_rate"})
    write_csv(agg, os.path.join(cfg["paths"]["reports"], "tables", f"errors_{model_key}.csv"))
    logger.info("[error] wrote error summary for %s", model_key)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--model_key", default="xgb")
    args = ap.parse_args()
    main(args.config, args.model_key)
