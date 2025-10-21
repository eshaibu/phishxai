from ..pipelines.p00_fetch_data import main as run

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    args = ap.parse_args()
    run(args.config)
