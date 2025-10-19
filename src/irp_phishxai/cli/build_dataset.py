from ..pipelines.p00_fetch_data import main as s0
from ..pipelines.p01_clean_align import main as s1
from ..pipelines.p02_features import main as s2
from ..pipelines.p02_split import main as s3

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    args = ap.parse_args()
    # Sequentially execute the data-build pipeline.
    s0(args.config)
    s1(args.config)
    s2(args.config)
    s3(args.config)
