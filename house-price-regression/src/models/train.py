"""
Trains and compares Linear Regression, Ridge, Lasso, and ElasticNet on the
house-price dataset, then serializes the winning production pipeline.

Design choices worth calling out (also documented in the README):
  1. Hyperparameter search cross-validates the FULL pipeline (feature
     engineering -> preprocessing -> model), not just the model on
     pre-processed data, so scaling/imputation statistics never leak from
     validation folds into training folds.
  2. The search happens in log-price space (matches the modeling target);
     the FINAL fitted pipeline wraps the model in a TransformedTargetRegressor
     so callers (the API, tests, anyone) always get dollar-scale predictions
     with no manual inverse-transform bookkeeping required.
  3. alpha is chosen with the "1-SE rule" (Hastie/Tibshirani/Friedman, ESL
     7.10): not the alpha with the single best CV score, but the most
     regularized alpha whose CV score is still within one standard error of
     the best — a simpler, more conservative model that's likely to
     generalize at least as well.

Run:
    python -m src.models.train
"""
from __future__ import annotations

import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import get_config, resolve
from src.features.build_features import (
    RAW_CATEGORICAL_FEATURES,
    FeatureEngineer,
    build_preprocessor,
    get_clean_feature_names,
    load_raw_features_and_target,
)
from src.visualization import visualize as viz

CFG = get_config()
SEED = CFG["random_seed"]
CURRENT_YEAR = CFG["current_year"]


# ----------------------------------------------------------------------------
# Pipeline construction
# ----------------------------------------------------------------------------
def make_search_pipeline(regressor) -> Pipeline:
    """feature_engineer -> preprocessor -> regressor, fit directly on
    log-target during CV search (no TransformedTargetRegressor here -- that
    wrapping is added only for the final production pipeline)."""
    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer(current_year=CURRENT_YEAR)),
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def make_production_pipeline(regressor) -> Pipeline:
    """Same as above but the regressor is wrapped so .fit/.predict operate
    in dollar space directly -- this is what gets saved and served."""
    ttr = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer(current_year=CURRENT_YEAR)),
            ("preprocessor", build_preprocessor()),
            ("regressor", ttr),
        ]
    )


# ----------------------------------------------------------------------------
# Alpha search + 1-SE rule
# ----------------------------------------------------------------------------
def cv_alpha_search(model_cls, alphas, X, y_log, cv, **model_kwargs) -> pd.DataFrame:
    records = []
    for a in alphas:
        pipe = make_search_pipeline(model_cls(alpha=a, **model_kwargs))
        scores = cross_validate(pipe, X, y_log, cv=cv, scoring="neg_root_mean_squared_error")
        fold_rmse = -scores["test_score"]
        records.append(
            {
                "alpha": a,
                "mean_rmse": fold_rmse.mean(),
                "std_rmse": fold_rmse.std(),
                "se_rmse": fold_rmse.std() / np.sqrt(len(fold_rmse)),
            }
        )
    return pd.DataFrame(records)


def select_alpha_1se(results: pd.DataFrame) -> tuple[float, float]:
    best_idx = results["mean_rmse"].idxmin()
    alpha_min = results.loc[best_idx, "alpha"]
    threshold = results.loc[best_idx, "mean_rmse"] + results.loc[best_idx, "se_rmse"]
    candidates = results[results["mean_rmse"] <= threshold]
    alpha_1se = candidates["alpha"].max()  # most regularized alpha still "good enough"
    return float(alpha_min), float(alpha_1se)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_on_test(fitted_pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = fitted_pipe.predict(X_test)
    n, p = len(y_test), len(fitted_pipe.named_steps["preprocessor"].get_feature_names_out())
    r2 = r2_score(y_test, y_pred)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2),
        "test_adj_r2": float(1 - (1 - r2) * (n - 1) / (n - p - 1)),
        "test_mape_pct": float(mean_absolute_percentage_error(y_test, y_pred) * 100),
    }


def get_inner_coefficients(fitted_pipe: Pipeline) -> pd.Series:
    """Works for both a plain regressor step and a TransformedTargetRegressor
    wrapping one."""
    reg = fitted_pipe.named_steps["regressor"]
    reg = reg.regressor_ if hasattr(reg, "regressor_") else reg
    names = get_clean_feature_names(fitted_pipe.named_steps["preprocessor"])
    return pd.Series(reg.coef_, index=names)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    figures_dir = resolve(CFG["paths"]["figures_dir"])
    models_dir = resolve(CFG["paths"]["models_dir"])
    metrics_path = resolve(CFG["paths"]["metrics_path"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{resolve('mlruns.db')}")
    mlflow.set_experiment("house-price-regression")

    # ---- Load + split -------------------------------------------------
    df = pd.read_csv(resolve(CFG["data"]["raw_path"]))
    X, y = load_raw_features_and_target(df, target_col=CFG["target"]["column"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG["data"]["test_size"], random_state=SEED
    )
    y_train_log = np.log1p(y_train)
    cv = KFold(n_splits=CFG["cv"]["n_folds"], shuffle=CFG["cv"]["shuffle"], random_state=SEED)
    alphas = np.logspace(CFG["alpha_search"]["min_exp"], CFG["alpha_search"]["max_exp"],
                          CFG["alpha_search"]["n_values"])
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ---- EDA figures (computed on train only) --------------------------
    fe = FeatureEngineer(current_year=CURRENT_YEAR)
    X_train_eng = fe.fit_transform(X_train)
    numeric_for_corr = [c for c in X_train_eng.columns if X_train_eng[c].dtype != object and
                         X_train_eng[c].dtype != bool]
    viz.plot_target_distribution(y_train, figures_dir / "01_target_distribution.png")
    viz.plot_correlation_heatmap(
        X_train_eng[numeric_for_corr].assign(sale_price=y_train.values),
        figures_dir / "02_correlation_heatmap.png",
    )

    from src.features.build_features import ENGINEERED_FEATURES, RAW_NUMERIC_FEATURES
    vif_table = viz.compute_vif_table(
        X_train, fe, RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES, RAW_CATEGORICAL_FEATURES
    )
    vif_table.to_csv(resolve("reports", "vif_table.csv"), index=False)
    viz.plot_vif(vif_table, figures_dir / "03_vif.png")
    print("\nTop-8 VIF (post feature engineering):")
    print(vif_table.head(8).to_string(index=False))

    # ---- Ridge / Lasso alpha search + 1-SE rule ------------------------
    alpha_choices = {}
    for name, ModelCls, kwargs in [("Ridge", Ridge, {}), ("Lasso", Lasso, {"max_iter": 20000})]:
        t0 = time.time()
        results = cv_alpha_search(ModelCls, alphas, X_train, y_train_log, cv, **kwargs)
        a_min, a_1se = select_alpha_1se(results)
        alpha_choices[name] = {"alpha_min": a_min, "alpha_1se": a_1se}
        viz.plot_cv_curve(results, name, figures_dir / f"04_{name.lower()}_cv_curve.png", a_min, a_1se)
        print(f"{name}: alpha_min={a_min:.4g}  alpha_1se={a_1se:.4g}  ({time.time()-t0:.1f}s)")

    # ---- ElasticNet: 2D grid (alpha x l1_ratio), plain best-CV choice ---
    en_param_grid = {
        "regressor__alpha": alphas[::4],  # coarser grid, 2D search is more expensive
        "regressor__l1_ratio": CFG["elastic_net"]["l1_ratios"],
    }
    en_search = GridSearchCV(
        make_search_pipeline(ElasticNet(max_iter=20000)),
        en_param_grid, cv=cv, scoring="neg_root_mean_squared_error",
    )
    en_search.fit(X_train, y_train_log)
    en_alpha = en_search.best_params_["regressor__alpha"]
    en_l1_ratio = en_search.best_params_["regressor__l1_ratio"]
    print(f"ElasticNet: alpha={en_alpha:.4g}  l1_ratio={en_l1_ratio}")

    # ---- Fit final production pipelines (dollar-scale via TTR) ---------
    final_specs = {
        "Linear": LinearRegression(),
        "Ridge (1-SE)": Ridge(alpha=alpha_choices["Ridge"]["alpha_1se"]),
        "Ridge (min)": Ridge(alpha=alpha_choices["Ridge"]["alpha_min"]),
        "Lasso (1-SE)": Lasso(alpha=alpha_choices["Lasso"]["alpha_1se"], max_iter=20000),
        "Lasso (min)": Lasso(alpha=alpha_choices["Lasso"]["alpha_min"], max_iter=20000),
        "ElasticNet": ElasticNet(alpha=en_alpha, l1_ratio=en_l1_ratio, max_iter=20000),
    }

    fitted, metrics_rows = {}, []
    for name, reg in final_specs.items():
        with mlflow.start_run(run_name=name):
            pipe = make_production_pipeline(reg)
            pipe.fit(X_train, y_train)
            fitted[name] = pipe
            m = evaluate_on_test(pipe, X_test, y_test)
            coef = get_inner_coefficients(pipe)
            m["model"] = name
            m["n_nonzero_coefs"] = int((coef.abs() > 1e-8).sum())
            m["n_total_coefs"] = len(coef)
            metrics_rows.append(m)

            mlflow.log_params({"model_type": name})
            if hasattr(reg, "alpha"):
                mlflow.log_param("alpha", reg.alpha)
            if hasattr(reg, "l1_ratio"):
                mlflow.log_param("l1_ratio", reg.l1_ratio)
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        print(f"{name:15s} test_rmse=${m['test_rmse']:,.0f}  test_r2={m['test_r2']:.4f}  "
              f"non-zero coefs={m['n_nonzero_coefs']}/{m['n_total_coefs']}")

    metrics_df = pd.DataFrame(metrics_rows).set_index("model").loc[list(final_specs.keys())].reset_index()
    metrics_df.to_csv(resolve("reports", "model_comparison.csv"), index=False)
    viz.plot_model_comparison_bars(metrics_df, figures_dir / "05_model_comparison.png")

    # ---- Which raw features did Lasso actually zero out? ---------------
    lasso_coef = get_inner_coefficients(fitted["Lasso (1-SE)"])
    zeroed = sorted(lasso_coef[lasso_coef.abs() <= 1e-8].index.tolist())
    kept = sorted(lasso_coef[lasso_coef.abs() > 1e-8].index.tolist())
    print(f"\nLasso (1-SE) zeroed {len(zeroed)}/{len(lasso_coef)} features:")
    print(" ", zeroed)

    # ---- Regularization paths (on train, standardized design matrix) ---
    prep_only = fitted["Ridge (min)"].named_steps["preprocessor"]
    Xt_train = np.asarray(prep_only.transform(fe.transform(X_train)))
    feat_names = get_clean_feature_names(prep_only)
    path_alphas = np.logspace(CFG["alpha_search"]["min_exp"], 2.2, 45)  # trimmed for a readable plot
    viz.plot_regularization_paths(Xt_train, y_train_log.values, feat_names, path_alphas,
                                   figures_dir / "06_regularization_paths.png")

    # ---- Bootstrap coefficient-stability vs. alpha (the "why" plot) ----
    # gr_liv_area shows the textbook shrink-toward-zero story cleanly. Its
    # correlated sibling total_sf actually GROWS with alpha instead (Ridge's
    # "grouping effect" pulling correlated coefficients toward each other,
    # not toward zero) -- documented separately in the README/notebook as a
    # deliberate, more advanced finding rather than folded in here.
    boot_alphas = np.concatenate([[0], np.logspace(-1, 3, 12)])
    viz.plot_bootstrap_stability_vs_alpha(
        Xt_train, y_train_log.values, feat_names.index("gr_liv_area"), "gr_liv_area", boot_alphas,
        figures_dir / "07_bootstrap_stability.png", n_boot=CFG["bootstrap"]["n_iterations"],
    )

    # Grouping-effect evidence: how the three correlated "size" coefficients
    # move relative to EACH OTHER (not toward 0) as alpha grows.
    grouping_effect = pd.DataFrame(
        {"alpha": a,
         **{f: Ridge(alpha=a).fit(Xt_train, y_train_log.values).coef_[feat_names.index(f)]
            for f in ["gr_liv_area", "total_bsmt_sf", "total_sf"]}}
        for a in [0.01, 1, 10, 65, 200, 1000]
    )
    grouping_effect.to_csv(resolve("reports", "ridge_grouping_effect.csv"), index=False)

    # ---- Residuals + predicted-vs-actual for the chosen model (Ridge 1SE) ---
    best_name = "Ridge (1-SE)"
    y_pred_best = fitted[best_name].predict(X_test)
    viz.plot_predicted_vs_actual(y_test.values, y_pred_best, figures_dir / "08_pred_vs_actual.png",
                                  f"{best_name}: Predicted vs. Actual (test set)")
    viz.plot_residual_diagnostics(y_test.values, y_pred_best, figures_dir / "09_residuals.png",
                                   f"{best_name}: Residual diagnostics (test set)")

    # ---- Learning curves ------------------------------------------------
    curves = {
        "Linear": make_search_pipeline(LinearRegression()),
        "Ridge": make_search_pipeline(Ridge(alpha=alpha_choices["Ridge"]["alpha_1se"])),
        "Lasso": make_search_pipeline(Lasso(alpha=alpha_choices["Lasso"]["alpha_1se"], max_iter=20000)),
    }
    viz.plot_learning_curves(curves, X_train, y_train_log, figures_dir / "10_learning_curves.png", cv=cv)

    # ---- Feature importance for the chosen model ------------------------
    best_coef = get_inner_coefficients(fitted[best_name])
    viz.plot_feature_importance(best_coef, figures_dir / "11_feature_importance.png",
                                 f"{best_name}: standardized coefficients")

    # ---- Persist everything ---------------------------------------------
    for name, pipe in fitted.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        joblib.dump(pipe, models_dir / f"{fname}.joblib")
    joblib.dump(fitted[best_name], models_dir / "production_model.joblib")

    summary = {
        "best_model": best_name,
        "alpha_choices": alpha_choices,
        "elastic_net": {"alpha": en_alpha, "l1_ratio": en_l1_ratio},
        "lasso_zeroed_features": zeroed,
        "lasso_kept_features": kept,
        "metrics": metrics_df.to_dict(orient="records"),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features_raw": X_train.shape[1],
        "n_features_final": len(feat_names),
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")
    print(f"Saved {len(fitted)} model pipelines -> {models_dir}")
    return summary


if __name__ == "__main__":
    main()
