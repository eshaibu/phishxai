from ..pipelines.p05_evaluate import main as evaluate

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    evaluate(args.config, args.models)
