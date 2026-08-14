"""
All plotting + diagnostic-table functions live here, kept separate from
modeling logic so notebooks and scripts can both import them without
duplicating plotting code.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless environment — never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.02)
PALETTE = {
    "linear": "#546E7A",
    "ridge": "#1E88E5",
    "lasso": "#E64A19",
    "elasticnet": "#43A047",
    "accent": "#8E24AA",
}
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
def plot_target_distribution(y: pd.Series, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(y, kde=True, ax=axes[0], color=PALETTE["linear"])
    axes[0].set_title(f"sale_price  (skew={y.skew():.2f})")
    axes[0].set_xlabel("Sale price ($)")

    y_log = np.log1p(y)
    sns.histplot(y_log, kde=True, ax=axes[1], color=PALETTE["ridge"])
    axes[1].set_title(f"log1p(sale_price)  (skew={y_log.skew():.2f})")
    axes[1].set_xlabel("log1p(Sale price)")
    fig.suptitle("Target distribution: raw vs. log-transformed", y=1.03, fontsize=13)
    _save(fig, out_path)


def plot_correlation_heatmap(df_numeric: pd.DataFrame, out_path: Path):
    corr = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax,
        annot=corr.shape[0] <= 14, fmt=".2f",
    )
    ax.set_title("Raw feature correlation matrix", fontsize=13)
    _save(fig, out_path)


# ----------------------------------------------------------------------------
# Multicollinearity
# ----------------------------------------------------------------------------
def compute_vif_table(X_raw: pd.DataFrame, feature_engineer, numeric_cols: Sequence[str],
                       categorical_cols: Sequence[str]) -> pd.DataFrame:
    """Correctly-specified VIF: engineered features included, categoricals
    encoded with drop='first' (avoids the 'dummy variable trap' inflating
    VIF for one-hot columns for a reason that has nothing to do with the
    numeric multicollinearity we actually care about), plus an explicit
    intercept, as the VIF formula assumes.
    """
    X_eng = feature_engineer.transform(X_raw)
    num = X_eng[list(numeric_cols)].apply(lambda c: c.fillna(c.median()))

    if categorical_cols:
        enc = OneHotEncoder(drop="first", sparse_output=False)
        cat = pd.DataFrame(
            enc.fit_transform(X_eng[list(categorical_cols)].fillna("missing")),
            columns=enc.get_feature_names_out(categorical_cols),
            index=X_eng.index,
        )
        design = pd.concat([num, cat], axis=1)
    else:
        design = num

    design = add_constant(design, has_constant="add")
    vifs = []
    for i, col in enumerate(design.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(design.values, i)
        except (ZeroDivisionError, np.linalg.LinAlgError):
            v = np.inf
        vifs.append((col, v))
    out = pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
    return out.reset_index(drop=True)


def plot_vif(vif_df: pd.DataFrame, out_path: Path, top_n: int = 15):
    plot_df = vif_df.head(top_n).copy()
    finite_cap = plot_df.loc[np.isfinite(plot_df["VIF"]), "VIF"].max()
    display_val = plot_df["VIF"].replace(np.inf, finite_cap * 1.35 if pd.notna(finite_cap) else 50)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.36 * len(plot_df))))
    colors = ["#C62828" if v > 10 else ("#F9A825" if v > 5 else "#546E7A") for v in plot_df["VIF"]]
    bars = ax.barh(plot_df["feature"][::-1], display_val[::-1], color=colors[::-1])
    for bar, raw_v in zip(bars, plot_df["VIF"][::-1]):
        label = "inf" if np.isinf(raw_v) else f"{raw_v:.1f}"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=9)
    ax.axvline(5, color="#F9A825", ls="--", lw=1, label="VIF = 5 (moderate)")
    ax.axvline(10, color="#C62828", ls="--", lw=1, label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("Multicollinearity after feature engineering (top offenders)")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_path)


# ----------------------------------------------------------------------------
# Regularization paths
# ----------------------------------------------------------------------------
def plot_regularization_paths(Xt: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
                               alphas: np.ndarray, out_path: Path, top_k: int = 12):
    """Classic ISLR-style plot: coefficient value vs. alpha, one line per
    feature, for both Ridge (shrink-but-nonzero) and Lasso (shrink-to-zero).
    Only the top_k highest-|coef| features (at the smallest alpha) are
    labeled/colored; the rest are drawn in light gray for context.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    for ax, Model, title in [(axes[0], Ridge, "Ridge"), (axes[1], Lasso, "Lasso")]:
        coefs = np.array([Model(alpha=a, max_iter=20000).fit(Xt, y).coef_ for a in alphas])
        ref_order = np.argsort(-np.abs(coefs[0]))
        top_idx = ref_order[:top_k]
        cmap = plt.get_cmap("tab20")
        for j in range(coefs.shape[1]):
            if j in top_idx:
                rank = list(top_idx).index(j)
                ax.plot(alphas, coefs[:, j], lw=2, color=cmap(rank % 20),
                         label=feature_names[j])
            else:
                ax.plot(alphas, coefs[:, j], lw=0.6, color="lightgray", zorder=0)
        ax.set_xscale("log")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("alpha (log scale)")
        ax.set_ylabel("Coefficient value")
        ax.set_title(f"{title} coefficient paths")

    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
                    title="Top features", frameon=False)
    fig.suptitle("Regularization paths: Ridge shrinks toward zero, Lasso hits zero", y=1.03, fontsize=13)
    _save(fig, out_path)


def plot_cv_curve(alpha_results: pd.DataFrame, model_name: str, out_path: Path,
                   alpha_min: float, alpha_1se: float):
    """CV RMSE (mean +/- 1SE) vs. alpha, with alpha_min and the more
    conservative alpha_1se both marked -- the classic glmnet-style plot."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(alpha_results["alpha"], alpha_results["mean_rmse"], color=PALETTE["ridge"], lw=2)
    ax.fill_between(
        alpha_results["alpha"],
        alpha_results["mean_rmse"] - alpha_results["se_rmse"],
        alpha_results["mean_rmse"] + alpha_results["se_rmse"],
        alpha=0.2, color=PALETTE["ridge"],
    )
    ax.axvline(alpha_min, color="#2E7D32", ls="--", lw=1.4, label=f"alpha_min = {alpha_min:.3g}")
    ax.axvline(alpha_1se, color="#8E24AA", ls="--", lw=1.4, label=f"alpha_1se = {alpha_1se:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Cross-validated RMSE (log-price)")
    ax.set_title(f"{model_name}: CV error vs. regularization strength")
    ax.legend(fontsize=9)
    _save(fig, out_path)


def plot_bootstrap_stability_vs_alpha(Xt: np.ndarray, y: np.ndarray, feature_idx: int,
                                       feature_name: str, alphas: np.ndarray,
                                       out_path: Path, n_boot: int = 200, seed: int = 42):
    """The core 'why does Ridge help' evidence: bootstrap the coefficient for
    one highly-collinear feature at increasing alpha (alpha=0 == OLS) and
    show its variance falls monotonically as regularization increases."""
    rng = np.random.default_rng(seed)
    n = Xt.shape[0]
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]

    stds, means = [], []
    for a in alphas:
        model = LinearRegression() if a == 0 else Ridge(alpha=a)
        vals = np.array([model.fit(Xt[idx], y[idx]).coef_[feature_idx] for idx in boot_idx])
        stds.append(vals.std())
        means.append(vals.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(alphas, stds, marker="o", color=PALETTE["lasso"])
    axes[0].set_xscale("symlog", linthresh=alphas[1] if alphas[0] == 0 else 0.01)
    axes[0].set_xlabel("alpha  (0 = OLS)")
    axes[0].set_ylabel(f"Bootstrap std of '{feature_name}' coefficient")
    axes[0].set_title("Coefficient variance shrinks as alpha grows")

    axes[1].plot(alphas, means, marker="o", color=PALETTE["ridge"])
    axes[1].set_xscale("symlog", linthresh=alphas[1] if alphas[0] == 0 else 0.01)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("alpha  (0 = OLS)")
    axes[1].set_ylabel(f"Bootstrap mean of '{feature_name}' coefficient")
    axes[1].set_title("...at the cost of shrinking the estimate toward 0 (bias)")

    fig.suptitle(f"Bias-variance tradeoff in action: '{feature_name}' (bootstrap n={n_boot})",
                 y=1.03, fontsize=13)
    _save(fig, out_path)


# ----------------------------------------------------------------------------
# Model evaluation
# ----------------------------------------------------------------------------
def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=PALETTE["ridge"], edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="#C62828", lw=1.5, ls="--", label="Perfect prediction")
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_dollar_k_formatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_dollar_k_formatter))
    _save(fig, out_path)


def _dollar_k_formatter(x, _pos=None):
    return f"${x/1000:,.0f}k"


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].scatter(y_pred, residuals, alpha=0.35, s=16, color=PALETTE["linear"], edgecolor="none")
    axes[0].axhline(0, color="#C62828", lw=1.4, ls="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(_dollar_k_formatter))
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Residuals vs. fitted")

    sns.histplot(residuals, kde=True, ax=axes[1], color=PALETTE["ridge"])
    axes[1].set_title(f"Residual distribution (skew={pd.Series(residuals).skew():.2f})")

    from scipy import stats as _stats
    _stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q plot")
    axes[2].get_lines()[0].set(markerfacecolor=PALETTE["accent"], markeredgecolor="none", markersize=4)
    axes[2].get_lines()[1].set(color="#C62828")

    fig.suptitle(title, y=1.04, fontsize=13)
    _save(fig, out_path)


def plot_learning_curves(estimators: dict, X: np.ndarray, y: np.ndarray, out_path: Path,
                          cv=5, train_sizes=None):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, est in estimators.items():
        sizes, _train_scores, val_scores = learning_curve(
            est, X, y, cv=cv, train_sizes=train_sizes,
            scoring="neg_root_mean_squared_error", shuffle=True, random_state=42,
        )
        val_rmse = -val_scores.mean(axis=1)
        ax.plot(sizes, val_rmse, marker="o", label=name,
                color=PALETTE.get(name.lower().replace(" ", ""), None))
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Cross-validated RMSE (log-price)")
    ax.set_title("Learning curves: validation error vs. training set size")
    ax.legend(fontsize=9)
    _save(fig, out_path)


def plot_feature_importance(coef: pd.Series, out_path: Path, title: str, top_n: int = 15):
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(top_n)
    colors = [PALETTE["ridge"] if v > 0 else PALETTE["lasso"] for v in top]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.36 * len(top))))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized coefficient (log-price scale)")
    ax.set_title(title)
    _save(fig, out_path)


def plot_model_comparison_bars(metrics_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    metric_cols = [("test_rmse", "Test RMSE ($)"), ("test_mae", "Test MAE ($)"), ("test_r2", "Test R²")]
    colors = [PALETTE.get(m.lower().replace(" ", "").split("(")[0], "#888") for m in metrics_df["model"]]
    for ax, (col, label) in zip(axes, metric_cols):
        ax.bar(metrics_df["model"], metrics_df[col], color=colors)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(metrics_df[col]):
            ax.text(i, v, f"{v:,.3f}" if col == "test_r2" else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model comparison on held-out test set", y=1.03, fontsize=13)
    _save(fig, out_path)
