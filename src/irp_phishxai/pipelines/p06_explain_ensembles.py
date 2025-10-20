import os, logging, joblib, numpy as np, pandas as pd
from ..config import load_config
from ..utils.io_utils import read_csv_safely, write_csv
from ..utils.shap_utils import compute_treeshap_values, plot_global_importance, plot_beeswarm, plot_local_waterfall
from ..utils.lime_utils import explain_instance_lime
from ..utils.model_utils import predict_proba_safely, ensure_named_frame
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def main(cfg_path: str, models: list[str] | None = None):
    """
    Stage p06: Generate SHAP (global + local) and LIME (local) explanations for ensembles.
    Strategy for selecting an instance to explain:
      1) Try a misclassified point (error-driven).
      2) If all classified correctly, choose the most uncertain (prob ~ 0.5).
    """
    setup_logging()
    cfg = load_config(cfg_path)
    test = read_csv_safely(os.path.join(cfg["paths"]["processed"], "test_features.csv"))

    feature_names = [c for c in test.columns if c != "label"]
    X = test[feature_names].values
    y = test["label"].values
    X_named = ensure_named_frame(X, feature_names)

    explain_models = models or cfg["models"]["explain"]
    out_fig = os.path.join(cfg["paths"]["reports"], "figures")
    out_tbl = os.path.join(cfg["paths"]["reports"], "tables")

    rows = []
    for key in explain_models:
        model_path = os.path.join(cfg["paths"]["models"], f"{key}.joblib")
        if not os.path.exists(model_path):
            logger.warning("[explain] skip %s, model not found.", key)
            continue
        model = joblib.load(model_path)

        # Subsample test rows for SHAP global visualizations (speed control).
        n = min(cfg["shap"]["sample_size"], len(test))
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(test), size=n, replace=False)
        Xs = test.iloc[idx][feature_names].values

        # Compute SHAP & save global plots (fail-soft).
        try:
            sv = compute_treeshap_values(model, Xs)
            plot_global_importance(sv, feature_names, os.path.join(out_fig, f"shap_bar_{key}.png"))
            plot_beeswarm(sv, Xs, os.path.join(out_fig, f"shap_beeswarm_{key}.png"))
        except Exception as e:
            logger.warning("SHAP global plots failed for %s: %s", key, e)

        # Select a "hard" instance for local SHAP & LIME:
        preds = model.predict(X_named)
        errs = np.where(preds != y)[0]
        if len(errs) > 0:
            ix = int(errs[0])
        else:
            probs = predict_proba_safely(model, X_named)
            pos = probs[:, 1] if (hasattr(probs, "shape") and probs.ndim == 2 and probs.shape[1] == 2) else probs
            ix = int(np.argmin(np.abs(pos - 0.5)))

        # SHAP local (waterfall) and LIME plot (fail-soft individually).
        try:
            sv_local = compute_treeshap_values(model, test.iloc[[ix]][feature_names].values)
            plot_local_waterfall(sv_local[0], os.path.join(out_fig, f"shap_waterfall_{key}.png"))
        except Exception as e:
            logger.warning("SHAP local waterfall failed for %s: %s", key, e)

        try:
            explain_instance_lime(model, X, feature_names, ["benign","phish"],
                                  test.iloc[ix][feature_names].values, os.path.join(out_fig, f"lime_{key}.png"))
        except Exception as e:
            logger.warning("LIME explanation failed for %s: %s", key, e)

        rows.append({"model": key, "shap_sample_size": int(n), "lime_idx": int(ix)})

    if rows:
        write_csv(pd.DataFrame(rows), os.path.join(out_tbl, "feature_rankings.csv"))
    logger.info("[explain] SHAP/LIME artifacts saved.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/starter.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    main(args.config, args.models)
