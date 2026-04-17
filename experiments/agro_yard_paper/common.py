
"""Shared constants, imports, types and low-level helpers for the Agro Yard paper package."""
from __future__ import annotations

import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from umap import UMAP


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "instances").exists()
            and (candidate / "catalog").exists()
            and (candidate / "tools").exists()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate repository root from current working directory."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())

TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

GUROBI_DIR = REPO_ROOT / "gurobi"
if str(GUROBI_DIR) not in sys.path:
    sys.path.insert(0, str(GUROBI_DIR))

import create_observed_noise_layer as observed_release_builder
import exact_solver_smoke as solver_smoke
import instance_analysis_repl as repl
import online_replay_policy as online_replay
from load_instance import build_gurobi_views, load_instance

observed_release_builder = importlib.reload(observed_release_builder)
solver_smoke = importlib.reload(solver_smoke)
repl = importlib.reload(repl)
online_replay = importlib.reload(online_replay)

SEED = repl.SEED
STAGE_ORDER = repl.STAGE_ORDER
REGIME_ORDER = repl.REGIME_ORDER
SCALE_ORDER = repl.SCALE_ORDER

METHOD_ORDER = [
    "M0_FIFO_OFFICIAL",
    "M0_CUSTOM_FIFO_REPLAY",
    "M1_WEIGHTED_SLACK",
    "M2_PERIODIC_15",
    "M2_PERIODIC_30",
    "M3_EVENT_REACTIVE",
    "Mref_EXACT_XS_S",
    "M4_METAHEURISTIC_L",
]
METHOD_LABELS = {
    "M0_FIFO_OFFICIAL": "M0 FIFO oficial",
    "M0_CUSTOM_FIFO_REPLAY": "M0 FIFO replay",
    "M1_WEIGHTED_SLACK": "M1 Weighted Slack",
    "M2_PERIODIC_15": "M2 Periódico Δ=15",
    "M2_PERIODIC_30": "M2 Periódico Δ=30",
    "M3_EVENT_REACTIVE": "M3 Evento reativo",
    "Mref_EXACT_XS_S": "Mref exato XS/S",
    "M4_METAHEURISTIC_L": "M4 Metaheurística L",
}
PAPER_METHOD_ORDER = [
    "M0_FIFO_OFFICIAL",
    "M1_WEIGHTED_SLACK",
    "M2_PERIODIC_15",
    "M2_PERIODIC_30",
    "M3_EVENT_REACTIVE",
    "Mref_EXACT_XS_S",
    "M4_METAHEURISTIC_L",
]
UTILITY_WEIGHTS = {
    "flow_p95": 0.45,
    "flow_mean": 0.25,
    "makespan": 0.15,
    "weighted_tardiness": 0.10,
    "runtime_sec": 0.05,
}
FIGURE_NAMES = (
    "method_delta_vs_fifo",
    "method_runtime_heatmap",
    "umap_best_method",
    "hdbscan_clusters",
    "solver_footprints",
    "selector_shap",
)


@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    family: str
    delta_min: int | None = None
    supported_scales: tuple[str, ...] | None = None


METHOD_SPECS = {
    "M1_WEIGHTED_SLACK": MethodSpec("M1_WEIGHTED_SLACK", "static"),
    "M2_PERIODIC_15": MethodSpec("M2_PERIODIC_15", "periodic", delta_min=15),
    "M2_PERIODIC_30": MethodSpec("M2_PERIODIC_30", "periodic", delta_min=30),
    "M3_EVENT_REACTIVE": MethodSpec("M3_EVENT_REACTIVE", "event"),
    "Mref_EXACT_XS_S": MethodSpec("Mref_EXACT_XS_S", "exact_reference", supported_scales=("XS", "S")),
    "M4_METAHEURISTIC_L": MethodSpec("M4_METAHEURISTIC_L", "metaheuristic", supported_scales=("L",)),
}


@dataclass(frozen=True)
class ReleaseBundle:
    root: Path
    artifact_dir: Path
    ctx: dict[str, Any]
    summary: dict[str, Any]
    validation_observed: pd.DataFrame
    validation_nominal_style: pd.DataFrame
    g2milp_contract: dict[str, Any]
    params: pd.DataFrame
    catalog: pd.DataFrame
    family_summary: pd.DataFrame
    observed_noise_manifest: dict[str, Any]
    manifest: dict[str, Any]
    jobs: pd.DataFrame
    jobs_enriched: pd.DataFrame
    operations: pd.DataFrame
    eligible: pd.DataFrame
    machines: pd.DataFrame
    precedences: pd.DataFrame
    downtimes: pd.DataFrame
    events: pd.DataFrame
    schedule: pd.DataFrame
    job_metrics: pd.DataFrame
    due_audit: pd.DataFrame
    proc_audit: pd.DataFrame
    proc_audit_enriched: pd.DataFrame
    congestion: pd.DataFrame
    structural_report: pd.DataFrame
    event_report: pd.DataFrame
    audit_reconciliation: pd.DataFrame
    regime_checks: pd.DataFrame
    fifo_schema_report: pd.DataFrame
    release_consistency_report: pd.DataFrame
    utilization: pd.DataFrame
    instance_space_features: pd.DataFrame
    instance_space_pairs: pd.DataFrame
    instance_space_summary: pd.DataFrame
    instance_space_knn_profile: pd.DataFrame
    instance_space_knn_regime_composition: pd.DataFrame
    instance_space_knn_scale_composition: pd.DataFrame
    diagnostics: dict[str, Any] | pd.Series
    unload: pd.DataFrame


@dataclass(frozen=True)
class PaperPipelineResults:
    feature_frame: pd.DataFrame
    performance: pd.DataFrame
    performance_test: pd.DataFrame
    protocol_summary: pd.DataFrame
    umap_frame: pd.DataFrame
    selector_report: pd.DataFrame
    shap_frame: pd.DataFrame
    scorecard: pd.DataFrame
    aslib_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    schedules: dict[str, pd.DataFrame]


def _attach_protocol_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if {"scale_code", "regime_code"}.issubset(enriched.columns):
        enriched["family_id"] = enriched["scale_code"].astype(str) + "__" + enriched["regime_code"].astype(str)
    if "replicate" in enriched.columns:
        replicate_int = enriched["replicate"].astype(int)
        enriched["protocol_role"] = np.where(replicate_int.eq(1), "calibration", "test")
        enriched["protocol_fold"] = np.where(replicate_int.eq(1), "R01", "R02_R03")
    return enriched

def build_protocol_split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = _attach_protocol_columns(frame)
    base_cols = ["protocol_role", "protocol_fold"]
    summary = (
        enriched[["instance_id", "family_id", *base_cols]]
        .drop_duplicates()
        .groupby(base_cols, as_index=False)
        .agg(
            n_instances=("instance_id", "nunique"),
            n_families=("family_id", "nunique"),
        )
        .sort_values(base_cols)
        .reset_index(drop=True)
    )
    return summary

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _load_instance_tables(root: Path, instance_id: str) -> dict[str, pd.DataFrame]:
    inst_dir = root / "instances" / instance_id
    return {
        "jobs": pd.read_csv(inst_dir / "jobs.csv"),
        "operations": pd.read_csv(inst_dir / "operations.csv"),
        "eligible": pd.read_csv(inst_dir / "eligible_machines.csv"),
        "machines": pd.read_csv(inst_dir / "machines.csv"),
        "downtimes": pd.read_csv(inst_dir / "machine_downtimes.csv"),
        "events": pd.read_csv(inst_dir / "events.csv"),
        "fifo_schedule": pd.read_csv(inst_dir / "fifo_schedule.csv"),
        "fifo_job_metrics": pd.read_csv(inst_dir / "fifo_job_metrics.csv"),
        "fifo_summary": pd.DataFrame([json.loads((inst_dir / "fifo_summary.json").read_text(encoding="utf-8"))]),
    }

def _entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    if probs.empty:
        return 0.0
    return float(-(probs * np.log2(probs + 1e-12)).sum())
