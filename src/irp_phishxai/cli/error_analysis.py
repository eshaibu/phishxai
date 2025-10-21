from ..pipelines.p07_error_analysis import main as run

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml",
                    help="Path to YAML configuration file.")
    ap.add_argument("--model_key", default="xgb",
                    help="Key of the model to analyze (e.g., rf, xgb, lgbm).")
    args = ap.parse_args()
    run(args.config, args.model_key)
