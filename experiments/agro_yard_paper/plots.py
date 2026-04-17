
"""Plotting helpers used by the paper notebook and the pipeline wrappers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .common import ConvexHull, METHOD_LABELS, METHOD_ORDER, PAPER_METHOD_ORDER


def plot_method_delta(performance: pd.DataFrame) -> plt.Figure:
    summary = (
        performance.groupby(["method_name", "scale_code"], as_index=False)["delta_vs_fifo_p95_flow_pct"]
        .mean()
        .pivot(index="method_name", columns="scale_code", values="delta_vs_fifo_p95_flow_pct")
        .reindex(index=PAPER_METHOD_ORDER, columns=["XS", "S", "M", "L"])
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(summary * 100.0, annot=True, fmt=".1f", cmap="RdYlGn_r", center=0.0, ax=ax, cbar_kws={"label": "% vs FIFO"})
    ax.set_title("Ganho relativo em p95_flow vs FIFO")
    ax.set_xlabel("Escala")
    ax.set_ylabel("Método")
    fig.tight_layout()
    return fig

def plot_method_runtime(performance: pd.DataFrame) -> plt.Figure:
    runtime = (
        performance.pivot(index="instance_id", columns="method_name", values="runtime_sec")
        .reindex(columns=PAPER_METHOD_ORDER)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(np.log1p(runtime), cmap="mako", ax=ax, cbar_kws={"label": "log(1 + runtime_sec)"})
    ax.set_title("Runtime por instância × método")
    fig.tight_layout()
    return fig

def plot_umap_best_method(umap_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=umap_frame,
        x="umap_x",
        y="umap_y",
        hue="best_method",
        style="regime_code",
        s=110,
        ax=ax,
    )
    ax.set_title("UMAP do espaço de instâncias colorido pelo melhor método")
    fig.tight_layout()
    return fig

def plot_hdbscan_clusters(umap_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=umap_frame,
        x="umap_x",
        y="umap_y",
        hue="cluster_label",
        style="difficulty",
        s=110,
        palette="tab10",
        ax=ax,
    )
    ax.set_title("HDBSCAN sobre o embedding UMAP")
    fig.tight_layout()
    return fig

def plot_solver_footprints(
    umap_frame: pd.DataFrame,
    performance: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=umap_frame, x="umap_x", y="umap_y", color="#cbd5e1", s=55, ax=ax)
    footprint_members = performance.loc[performance["footprint_member"]].merge(
        umap_frame[["instance_id", "umap_x", "umap_y"]], on="instance_id", how="left"
    )
    palette = sns.color_palette("Set2", n_colors=len(METHOD_ORDER))
    for color, method_name in zip(palette, METHOD_ORDER):
        subset = footprint_members.loc[footprint_members["method_name"].eq(method_name)].copy()
        if subset.empty:
            continue
        sns.scatterplot(
            data=subset,
            x="umap_x",
            y="umap_y",
            s=130,
            color=color,
            label=METHOD_LABELS[method_name],
            ax=ax,
        )
        if len(subset) >= 3:
            points = subset[["umap_x", "umap_y"]].to_numpy()
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=1.6, alpha=0.85)
    ax.set_title("Solver footprints no espaço UMAP")
    fig.tight_layout()
    return fig

def plot_selector_shap(shap_frame: pd.DataFrame) -> plt.Figure:
    top = shap_frame[["feature_name", "mean_abs_shap"]].drop_duplicates().nlargest(12, "mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, y="feature_name", x="mean_abs_shap", color="#2563eb", ax=ax)
    ax.set_title("SHAP médio absoluto do seletor")
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig

def plot_e2_tradeoff(e2_table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(
        data=e2_table,
        x="replan_count",
        y="delta_vs_fifo_p95_flow_pct",
        hue="method_name",
        style="regime_code",
        size="runtime_sec",
        sizes=(40, 240),
        ax=axes[0],
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("E2: ganho vs FIFO por custo de replanning")
    axes[0].set_xlabel("replan_count médio")
    axes[0].set_ylabel("delta_vs_fifo_p95_flow_pct")

    pivot = (
        e2_table.groupby(["method_name", "regime_code"], as_index=False)["utility"]
        .mean()
        .pivot(index="method_name", columns="regime_code", values="utility")
        .reindex(index=["M2_PERIODIC_15", "M2_PERIODIC_30", "M3_EVENT_REACTIVE"])
    )
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", ax=axes[1], cbar_kws={"label": "utility"})
    axes[1].set_title("E2: utilidade média por regime")
    axes[1].set_xlabel("regime")
    axes[1].set_ylabel("método")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def plot_performance_profile(profile_frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=profile_frame,
        x="tau",
        y="share_instances",
        hue="method_name",
        hue_order=PAPER_METHOD_ORDER,
        ax=ax,
    )
    ax.set_title("Performance profile sobre utility")
    ax.set_xlabel("tau")
    ax.set_ylabel("fração de instâncias")
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
