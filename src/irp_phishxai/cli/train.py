from ..pipelines.p03_train_baselines import main as train_baselines
from ..pipelines.p04_train_ensembles import main as train_ensembles

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--which", choices=["baselines", "ensembles", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    # Allow training families independently or together.
    if args.which in ["baselines", "all"]:
        train_baselines(args.config, args.models)
    if args.which in ["ensembles", "all"]:
        train_ensembles(args.config, args.models)
