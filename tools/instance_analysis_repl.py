#!/usr/bin/env python3
"""REPL-friendly analysis bootstrap for the official observed benchmark release.

Usage:
    python -i tools/instance_analysis_repl.py

After the session starts, the main loaded objects are available as globals:
    SUMMARY
    PARAMS
    CATALOG
    FAMILY_SUMMARY
    JOBS
    JOBS_ENRICHED
    OPERATIONS
    ELIGIBLE
    MACHINES
    EVENTS
    SCHEDULE
    JOB_METRICS
    DUE_AUDIT
    PROC_AUDIT
    STRUCTURAL_REPORT
    EVENT_REPORT
    AUDIT_RECONCILIATION
    REGIME_CHECKS
    FIFO_SCHEMA_REPORT
    UTILIZATION
    DIAGNOSTICS

Main helper functions:
    inventory_tables()
    plot_inventory_overview()
    validation_tables()
    plot_validation_overview()
    plot_observational_layer()
    plot_operational_sanity()
    plot_instance_drilldown("GO_XS_DISRUPTED_01")
    export_all_artifacts()
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from validate_observed_release import diagnostics as release_diagnostics
from validate_observed_release import validate_instance


SEED = 7
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="talk")

STAGE_ORDER = ["WEIGH_IN", "SAMPLE_CLASSIFY", "UNLOAD", "WEIGH_OUT"]
REGIME_ORDER = ["balanced", "peak", "disrupted"]
SCALE_ORDER = ["XS", "S", "M", "L"]
PRIORITY_ORDER = ["URGENT", "CONTRACTED", "REGULAR"]
INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD = 2.0
KNN_K_VALUES = [1, 3, 5]
CORE_DIGEST_FILES = [
    "jobs.csv",
    "operations.csv",
    "eligible_machines.csv",
    "machines.csv",
    "precedences.csv",
    "machine_downtimes.csv",
    "events.csv",
]


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "instances").exists() and (candidate / "catalog").exists() and (candidate / "tools").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from current working directory.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "output" / "repl-analysis-artifacts"


def iter_instance_dirs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "instances").iterdir() if p.is_dir())


def load_instance_csv(root: Path, file_name: str) -> pd.DataFrame:
    frames = []
    for inst_dir in iter_instance_dirs(root):
        frame = pd.read_csv(inst_dir / file_name)
        frame["instance_id"] = inst_dir.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_params_frame(root: Path) -> pd.DataFrame:
    rows = []
    for inst_dir in iter_instance_dirs(root):
        payload = json.loads((inst_dir / "params.json").read_text(encoding="utf-8"))
        rows.append(payload)
    return pd.DataFrame(rows)


def add_instance_context(frame: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    keep = ["instance_id", "scale_code", "regime_code", "replicate", "dataset_version"]
    return frame.merge(params[keep], on="instance_id", how="left")


def compute_due_lower_bounds(jobs: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    lower_bounds = (
        eligible.groupby(["instance_id", "job_id", "op_seq"], as_index=False)["proc_time_min"]
        .min()
        .groupby(["instance_id", "job_id"], as_index=False)["proc_time_min"]
        .sum()
        .rename(columns={"proc_time_min": "nominal_lb_min"})
    )
    merged = jobs.merge(lower_bounds, on=["instance_id", "job_id"], how="left")
    merged["due_slack_min"] = merged["completion_due_min"] - merged["arrival_time_min"]
    merged["due_margin_over_lb_min"] = merged["due_slack_min"] - merged["nominal_lb_min"]
    merged["reveal_lead_min"] = merged["arrival_time_min"] - merged["reveal_time_min"]
    return merged


def build_structural_report(root: Path) -> pd.DataFrame:
    rows = [validate_instance(inst_dir) for inst_dir in iter_instance_dirs(root)]
    return pd.DataFrame(rows)


def event_consistency_frame(jobs: pd.DataFrame, events: pd.DataFrame, downtimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance_id, jobs_g in jobs.groupby("instance_id"):
        events_g = events[events["instance_id"] == instance_id]
        downs_g = downtimes[downtimes["instance_id"] == instance_id]
        visible = events_g[events_g["event_type"] == "JOB_VISIBLE"].groupby("entity_id").size()
        arrival = events_g[events_g["event_type"] == "JOB_ARRIVAL"].groupby("entity_id").size()
        visible_mismatch = int((visible.reindex(jobs_g["job_id"]).fillna(0) != 1).sum())
        arrival_mismatch = int((arrival.reindex(jobs_g["job_id"]).fillna(0) != 1).sum())

        down_rows = events_g[events_g["event_type"] == "MACHINE_DOWN"][["entity_id", "event_time_min"]]
        up_rows = events_g[events_g["event_type"] == "MACHINE_UP"][["entity_id", "event_time_min"]]
        down_expected = downs_g[["machine_id", "start_min"]].rename(columns={"machine_id": "entity_id", "start_min": "event_time_min"})
        up_expected = downs_g[["machine_id", "end_min"]].rename(columns={"machine_id": "entity_id", "end_min": "event_time_min"})
        down_missing = int(len(pd.merge(down_expected, down_rows, on=["entity_id", "event_time_min"], how="left", indicator=True).query('_merge != "both"')))
        up_missing = int(len(pd.merge(up_expected, up_rows, on=["entity_id", "event_time_min"], how="left", indicator=True).query('_merge != "both"')))
        rows.append(
            {
                "instance_id": instance_id,
                "job_visible_mismatch": visible_mismatch,
                "job_arrival_mismatch": arrival_mismatch,
                "machine_down_missing": down_missing,
                "machine_up_missing": up_missing,
            }
        )
    return pd.DataFrame(rows)


def build_audit_reconciliation(jobs: pd.DataFrame, eligible: pd.DataFrame, due_audit: pd.DataFrame, proc_audit: pd.DataFrame) -> pd.DataFrame:
    due_check = jobs[["instance_id", "job_id", "completion_due_min"]].merge(
        due_audit[["instance_id", "job_id", "completion_due_observed_min"]],
        on=["instance_id", "job_id"],
        how="left",
    )
    due_check["due_matches_audit"] = due_check["completion_due_min"].eq(due_check["completion_due_observed_min"])

    proc_check = eligible[["instance_id", "job_id", "op_seq", "machine_id", "proc_time_min"]].merge(
        proc_audit[["instance_id", "job_id", "op_seq", "machine_id", "proc_time_observed_min"]],
        on=["instance_id", "job_id", "op_seq", "machine_id"],
        how="left",
    )
    proc_check["proc_matches_audit"] = proc_check["proc_time_min"].eq(proc_check["proc_time_observed_min"])

    due_summary = due_check.groupby("instance_id", as_index=False)["due_matches_audit"].mean().rename(columns={"due_matches_audit": "due_match_share"})
    proc_summary = proc_check.groupby("instance_id", as_index=False)["proc_matches_audit"].mean().rename(columns={"proc_matches_audit": "proc_match_share"})
    return due_summary.merge(proc_summary, on="instance_id", how="outer")


def build_regime_behavior_checks(
    family_summary: pd.DataFrame,
    job_metrics: pd.DataFrame,
    jobs_enriched: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    queue_summary = job_metrics.groupby(["scale_code", "regime_code"], as_index=False)["queue_time_min"].mean()
    congestion_summary = jobs_enriched.groupby(["scale_code", "regime_code"], as_index=False)["arrival_congestion_score"].mean()
    for scale_code, group in family_summary.groupby("scale_code"):
        flow_group = group.set_index("regime_code")
        queue_group = queue_summary[queue_summary["scale_code"] == scale_code].set_index("regime_code")
        congestion_group = congestion_summary[congestion_summary["scale_code"] == scale_code].set_index("regime_code")
        mean_order = (
            flow_group.loc["balanced", "avg_fifo_mean_flow_min"]
            < flow_group.loc["peak", "avg_fifo_mean_flow_min"]
            < flow_group.loc["disrupted", "avg_fifo_mean_flow_min"]
        )
        p95_order = (
            flow_group.loc["balanced", "avg_fifo_p95_flow_min"]
            < flow_group.loc["peak", "avg_fifo_p95_flow_min"]
            < flow_group.loc["disrupted", "avg_fifo_p95_flow_min"]
        )
        mean_queue_order = (
            queue_group.loc["balanced", "queue_time_min"]
            < queue_group.loc["peak", "queue_time_min"]
            < queue_group.loc["disrupted", "queue_time_min"]
        )
        congestion_mean_order = (
            congestion_group.loc["balanced", "arrival_congestion_score"]
            < congestion_group.loc["peak", "arrival_congestion_score"]
            < congestion_group.loc["disrupted", "arrival_congestion_score"]
        )
        rows.append(
            {
                "scale_code": scale_code,
                "mean_flow_order_ok": bool(mean_order),
                "p95_flow_order_ok": bool(p95_order),
                "mean_queue_order_ok": bool(mean_queue_order),
                "mean_congestion_order_ok": bool(congestion_mean_order),
            }
        )
    return pd.DataFrame(rows).sort_values("scale_code")


def build_fifo_schema_report(
    operations: pd.DataFrame,
    eligible: pd.DataFrame,
    precedences: pd.DataFrame,
    schedule: pd.DataFrame,
    downtimes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for instance_id in sorted(schedule["instance_id"].unique()):
        ops_g = operations[operations["instance_id"] == instance_id].copy()
        elig_g = eligible[eligible["instance_id"] == instance_id].copy()
        prec_g = precedences[precedences["instance_id"] == instance_id].copy()
        sched_g = schedule[schedule["instance_id"] == instance_id].copy().sort_values(["machine_id", "start_min", "end_min"])
        downs_g = downtimes[downtimes["instance_id"] == instance_id].copy()

        eligible_keys = set(map(tuple, elig_g[["job_id", "op_seq", "machine_id"]].drop_duplicates().itertuples(index=False, name=None)))
        sched_keys = list(map(tuple, sched_g[["job_id", "op_seq", "machine_id"]].itertuples(index=False, name=None)))
        ineligible_assignments = sum(1 for key in sched_keys if key not in eligible_keys)

        sched_ops = sched_g.merge(
            ops_g[["job_id", "op_seq", "release_time_min"]],
            on=["job_id", "op_seq"],
            how="left",
        )
        release_time_violations = int((sched_ops["start_min"] < sched_ops["release_time_min"]).sum())

        sched_index = sched_g.set_index(["job_id", "op_seq"])
        precedence_violations = 0
        for row in prec_g.itertuples(index=False):
            pred_end = int(sched_index.loc[(row.job_id, row.pred_op_seq), "end_min"])
            succ_start = int(sched_index.loc[(row.job_id, row.succ_op_seq), "start_min"])
            if succ_start < pred_end + int(row.min_lag_min):
                precedence_violations += 1

        overlap_violations = 0
        for _, machine_sched in sched_g.groupby("machine_id"):
            prev_end = None
            for row in machine_sched.itertuples(index=False):
                if prev_end is not None and int(row.start_min) < prev_end:
                    overlap_violations += 1
                prev_end = int(row.end_min)

        downtime_violations = 0
        downtime_groups = {machine_id: frame for machine_id, frame in downs_g.groupby("machine_id")}
        for row in sched_g.itertuples(index=False):
            machine_downs = downtime_groups.get(row.machine_id)
            if machine_downs is None:
                continue
            if ((machine_downs["start_min"] < row.end_min) & (machine_downs["end_min"] > row.start_min)).any():
                downtime_violations += 1

        rows.append(
            {
                "instance_id": instance_id,
                "eligible_assignment_ok": ineligible_assignments == 0,
                "release_time_ok": release_time_violations == 0,
                "precedence_ok": precedence_violations == 0,
                "machine_overlap_ok": overlap_violations == 0,
                "downtime_ok": downtime_violations == 0,
                "ineligible_assignments": ineligible_assignments,
                "release_time_violations": release_time_violations,
                "precedence_violations": precedence_violations,
                "machine_overlap_violations": overlap_violations,
                "downtime_violations": downtime_violations,
            }
        )
    return pd.DataFrame(rows)


def build_release_consistency_report(
    root: Path,
    manifest: dict[str, Any],
    observed_noise_manifest: dict[str, Any],
    params: pd.DataFrame,
) -> pd.DataFrame:
    param_versions = sorted(params["dataset_version"].dropna().astype(str).unique().tolist())
    parent_versions = sorted(params["parent_dataset_version"].dropna().astype(str).unique().tolist())
    noise_model_ids = sorted(params["observational_noise_model_id"].dropna().astype(str).unique().tolist())

    rows = [
        {
            "check_name": "manifest_dataset_version_matches_instance_params",
            "expected": manifest.get("dataset_version"),
            "observed": ", ".join(param_versions),
            "pass": param_versions == [manifest.get("dataset_version")],
        },
        {
            "check_name": "manifest_parent_dataset_version_matches_instance_params",
            "expected": manifest.get("parent_dataset_version"),
            "observed": ", ".join(parent_versions),
            "pass": parent_versions == [manifest.get("parent_dataset_version")],
        },
        {
            "check_name": "manifest_noise_model_id_matches_instance_params",
            "expected": manifest.get("observational_noise_model_id"),
            "observed": ", ".join(noise_model_ids),
            "pass": noise_model_ids == [manifest.get("observational_noise_model_id")],
        },
        {
            "check_name": "noise_manifest_model_id_matches_root_manifest",
            "expected": manifest.get("observational_noise_model_id"),
            "observed": observed_noise_manifest.get("model_id"),
            "pass": observed_noise_manifest.get("model_id") == manifest.get("observational_noise_model_id"),
        },
        {
            "check_name": "noise_manifest_repository_url_matches_root_manifest",
            "expected": manifest.get("repository_url"),
            "observed": observed_noise_manifest.get("official_release_repository_url"),
            "pass": observed_noise_manifest.get("official_release_repository_url") == manifest.get("repository_url"),
        },
        {
            "check_name": "noise_manifest_release_root_matches_repo_root",
            "expected": str(root),
            "observed": observed_noise_manifest.get("official_release_root"),
            "pass": observed_noise_manifest.get("official_release_root") == str(root),
        },
        {
            "check_name": "noise_manifest_generated_with_matches_root_manifest",
            "expected": manifest.get("generated_with"),
            "observed": observed_noise_manifest.get("generated_with"),
            "pass": observed_noise_manifest.get("generated_with") == manifest.get("generated_with"),
        },
    ]
    return pd.DataFrame(rows)


def build_core_instance_digest_frame(root: Path) -> pd.DataFrame:
    rows = []
    for inst_dir in iter_instance_dirs(root):
        digest = hashlib.sha256()
        for file_name in CORE_DIGEST_FILES:
            digest.update(file_name.encode("utf-8"))
            digest.update((inst_dir / file_name).read_bytes())
        rows.append({"instance_id": inst_dir.name, "core_instance_digest": digest.hexdigest()})
    return pd.DataFrame(rows)


def _standardize_feature_matrix(feature_frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    matrix = feature_frame[feature_cols].astype(float).to_numpy()
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def _build_knn_tables(
    feature_frame: pd.DataFrame,
    distance_matrix: np.ndarray,
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_k = max(k_values)
    neighbor_order = np.argsort(distance_matrix, axis=1)[:, :max_k]
    regimes = feature_frame["regime_code"].to_numpy()
    scales = feature_frame["scale_code"].to_numpy()
    instance_ids = feature_frame["instance_id"].to_numpy()

    profile_rows = []
    regime_rows = []
    scale_rows = []
    for row_idx in range(len(feature_frame)):
        src_regime = regimes[row_idx]
        src_scale = scales[row_idx]
        neighbors = neighbor_order[row_idx]
        neighbor_regimes = regimes[neighbors]
        neighbor_scales = scales[neighbors]
        neighbor_distances = distance_matrix[row_idx, neighbors]
        for k in k_values:
            subset_regimes = neighbor_regimes[:k]
            subset_scales = neighbor_scales[:k]
            subset_distances = neighbor_distances[:k]
            profile_rows.append(
                {
                    "instance_id": instance_ids[row_idx],
                    "scale_code": src_scale,
                    "regime_code": src_regime,
                    "k": int(k),
                    "mean_knn_distance": float(np.mean(subset_distances)),
                    "max_knn_distance": float(np.max(subset_distances)),
                    "same_regime_neighbor_share": float(np.mean(subset_regimes == src_regime)),
                    "same_scale_neighbor_share": float(np.mean(subset_scales == src_scale)),
                }
            )
            if k == max_k:
                for target_regime in REGIME_ORDER:
                    regime_rows.append(
                        {
                            "source_regime": src_regime,
                            "neighbor_regime": target_regime,
                            "k": int(k),
                            "share": float(np.mean(subset_regimes == target_regime)),
                        }
                    )
                for target_scale in SCALE_ORDER:
                    scale_rows.append(
                        {
                            "source_scale": src_scale,
                            "neighbor_scale": target_scale,
                            "k": int(k),
                            "share": float(np.mean(subset_scales == target_scale)),
                        }
                    )

    knn_profile = pd.DataFrame(profile_rows)
    regime_composition = (
        pd.DataFrame(regime_rows)
        .groupby(["source_regime", "neighbor_regime", "k"], as_index=False)["share"]
        .mean()
    )
    scale_composition = (
        pd.DataFrame(scale_rows)
        .groupby(["source_scale", "neighbor_scale", "k"], as_index=False)["share"]
        .mean()
    )
    return knn_profile, regime_composition, scale_composition


def build_instance_space_validation(
    root: Path,
    params: pd.DataFrame,
    jobs_enriched: pd.DataFrame,
    eligible: pd.DataFrame,
    machines: pd.DataFrame,
    downtimes: pd.DataFrame,
    job_metrics: pd.DataFrame,
    utilization: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_digests = build_core_instance_digest_frame(root)

    job_agg = (
        jobs_enriched.groupby("instance_id", as_index=False)
        .agg(
            n_jobs=("job_id", "count"),
            appointment_share=("appointment_flag", "mean"),
            urgent_share=("priority_class", lambda s: s.eq("URGENT").mean()),
            contracted_share=("priority_class", lambda s: s.eq("CONTRACTED").mean()),
            commodity_nunique=("commodity", "nunique"),
            moisture_nunique=("moisture_class", "nunique"),
            load_mean=("load_tons", "mean"),
            load_std=("load_tons", "std"),
            due_margin_mean=("due_margin_over_lb_min", "mean"),
            due_margin_std=("due_margin_over_lb_min", "std"),
            arrival_mean=("arrival_time_min", "mean"),
            arrival_std=("arrival_time_min", "std"),
            congestion_mean=("arrival_congestion_score", "mean"),
            congestion_std=("arrival_congestion_score", "std"),
        )
    )

    eligible_per_operation = (
        eligible.groupby(["instance_id", "job_id", "op_seq"], as_index=False)["machine_id"]
        .count()
        .rename(columns={"machine_id": "eligible_machine_count"})
    )
    eligible_summary = eligible_per_operation.groupby("instance_id", as_index=False).agg(
        eligible_machine_mean=("eligible_machine_count", "mean"),
        eligible_machine_std=("eligible_machine_count", "std"),
        eligible_machine_min=("eligible_machine_count", "min"),
        eligible_machine_max=("eligible_machine_count", "max"),
    )

    eligible_rows = eligible.groupby("instance_id", as_index=False).size().rename(columns={"size": "eligible_rows"})
    proc_summary = eligible.groupby("instance_id", as_index=False).agg(
        proc_time_mean=("proc_time_min", "mean"),
        proc_time_std=("proc_time_min", "std"),
        proc_time_min=("proc_time_min", "min"),
        proc_time_max=("proc_time_min", "max"),
    )

    machine_counts = (
        machines.groupby(["instance_id", "machine_family"], as_index=False)
        .size()
        .pivot(index="instance_id", columns="machine_family", values="size")
        .fillna(0)
        .reset_index()
    )
    machine_counts = machine_counts.rename(
        columns={
            "WB": "wb_machine_count",
            "LAB": "lab_machine_count",
            "HOP": "hop_machine_count",
        }
    )
    for column in ["wb_machine_count", "lab_machine_count", "hop_machine_count"]:
        if column not in machine_counts.columns:
            machine_counts[column] = 0
    machine_counts = machine_counts[
        ["instance_id", "wb_machine_count", "lab_machine_count", "hop_machine_count"]
    ]
    machine_counts["machine_count"] = (
        machine_counts["wb_machine_count"]
        + machine_counts["lab_machine_count"]
        + machine_counts["hop_machine_count"]
    )

    downtime_summary = (
        downtimes.assign(downtime_duration_min=downtimes["end_min"] - downtimes["start_min"])
        .groupby("instance_id", as_index=False)
        .agg(
            downtime_count=("machine_id", "count"),
            downtime_total_min=("downtime_duration_min", "sum"),
        )
    )

    metric_summary = (
        job_metrics.groupby("instance_id", as_index=False)
        .agg(
            flow_mean=("flow_time_min", "mean"),
            flow_p95=("flow_time_min", lambda s: float(np.quantile(s, 0.95))),
            queue_mean=("queue_time_min", "mean"),
            queue_p95=("queue_time_min", lambda s: float(np.quantile(s, 0.95))),
        )
    )

    utilization_by_family = (
        utilization.groupby(["instance_id", "machine_family"], as_index=False)["utilization_share"]
        .mean()
        .pivot(index="instance_id", columns="machine_family", values="utilization_share")
        .fillna(0.0)
        .reset_index()
    )
    utilization_by_family = utilization_by_family.rename(
        columns={"WB": "wb_utilization_mean", "LAB": "lab_utilization_mean", "HOP": "hop_utilization_mean"}
    )
    for column in ["wb_utilization_mean", "lab_utilization_mean", "hop_utilization_mean"]:
        if column not in utilization_by_family.columns:
            utilization_by_family[column] = 0.0
    utilization_by_family = utilization_by_family[
        ["instance_id", "wb_utilization_mean", "lab_utilization_mean", "hop_utilization_mean"]
    ]

    feature_frame = (
        params[["instance_id", "scale_code", "regime_code", "replicate", "planning_horizon_min"]]
        .merge(core_digests, on="instance_id", how="left")
        .merge(job_agg, on="instance_id", how="left")
        .merge(eligible_summary, on="instance_id", how="left")
        .merge(eligible_rows, on="instance_id", how="left")
        .merge(proc_summary, on="instance_id", how="left")
        .merge(machine_counts, on="instance_id", how="left")
        .merge(downtime_summary, on="instance_id", how="left")
        .merge(metric_summary, on="instance_id", how="left")
        .merge(utilization_by_family, on="instance_id", how="left")
        .sort_values(["scale_code", "regime_code", "replicate"])
        .reset_index(drop=True)
    )

    numeric_cols = [
        "planning_horizon_min",
        "n_jobs",
        "appointment_share",
        "urgent_share",
        "contracted_share",
        "commodity_nunique",
        "moisture_nunique",
        "load_mean",
        "load_std",
        "due_margin_mean",
        "due_margin_std",
        "arrival_mean",
        "arrival_std",
        "congestion_mean",
        "congestion_std",
        "eligible_machine_mean",
        "eligible_machine_std",
        "eligible_machine_min",
        "eligible_machine_max",
        "eligible_rows",
        "proc_time_mean",
        "proc_time_std",
        "proc_time_min",
        "proc_time_max",
        "wb_machine_count",
        "lab_machine_count",
        "hop_machine_count",
        "machine_count",
        "downtime_count",
        "downtime_total_min",
        "flow_mean",
        "flow_p95",
        "queue_mean",
        "queue_p95",
        "wb_utilization_mean",
        "lab_utilization_mean",
        "hop_utilization_mean",
    ]
    feature_frame[numeric_cols] = (
        feature_frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    )

    standardized = _standardize_feature_matrix(feature_frame, numeric_cols)
    _, singular_values, right_vectors = np.linalg.svd(standardized, full_matrices=False)
    explained_variance_ratio = (singular_values ** 2) / np.sum(singular_values ** 2)
    pcs = standardized @ right_vectors[:2].T
    feature_frame["pc1"] = pcs[:, 0]
    feature_frame["pc2"] = pcs[:, 1]

    distance_matrix = np.sqrt(((standardized[:, None, :] - standardized[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distance_matrix, np.inf)
    nearest_neighbor_idx = distance_matrix.argmin(axis=1)
    feature_frame["nearest_neighbor_instance_id"] = feature_frame.loc[nearest_neighbor_idx, "instance_id"].to_numpy()
    feature_frame["nearest_neighbor_distance"] = distance_matrix[np.arange(len(feature_frame)), nearest_neighbor_idx]
    knn_profile, regime_composition, scale_composition = _build_knn_tables(
        feature_frame=feature_frame,
        distance_matrix=distance_matrix,
        k_values=KNN_K_VALUES,
    )

    digest_counts = feature_frame["core_instance_digest"].value_counts()
    feature_frame["exact_core_duplicate"] = feature_frame["core_instance_digest"].map(digest_counts).gt(1)

    rounded_features = pd.DataFrame(np.round(standardized, 6))
    feature_frame["exact_feature_duplicate"] = rounded_features.duplicated(keep=False)
    feature_frame["duplicate_like_candidate"] = (
        feature_frame["nearest_neighbor_distance"] < INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD
    )

    pair_rows = []
    for i in range(len(feature_frame)):
        for j in range(i + 1, len(feature_frame)):
            pair_rows.append(
                {
                    "instance_a": feature_frame.loc[i, "instance_id"],
                    "instance_b": feature_frame.loc[j, "instance_id"],
                    "scale_a": feature_frame.loc[i, "scale_code"],
                    "scale_b": feature_frame.loc[j, "scale_code"],
                    "regime_a": feature_frame.loc[i, "regime_code"],
                    "regime_b": feature_frame.loc[j, "regime_code"],
                    "distance": float(distance_matrix[i, j]),
                    "duplicate_like_under_threshold": bool(
                        distance_matrix[i, j] < INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD
                    ),
                }
            )
    closest_pairs = pd.DataFrame(pair_rows).sort_values("distance").reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "instance_count": int(len(feature_frame)),
                "feature_count": int(len(numeric_cols)),
                "pca_pc1_explained_variance_ratio": float(explained_variance_ratio[0]),
                "pca_pc2_explained_variance_ratio": float(explained_variance_ratio[1]),
                "pca_pc1_pc2_explained_variance_ratio": float(explained_variance_ratio[:2].sum()),
                "exact_core_duplicate_count": int(feature_frame["exact_core_duplicate"].sum()),
                "exact_feature_duplicate_count": int(feature_frame["exact_feature_duplicate"].sum()),
                "duplicate_like_threshold": float(INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD),
                "duplicate_like_candidate_count": int(feature_frame["duplicate_like_candidate"].sum()),
                "nearest_neighbor_distance_min": float(feature_frame["nearest_neighbor_distance"].min()),
                "nearest_neighbor_distance_median": float(feature_frame["nearest_neighbor_distance"].median()),
                "nearest_neighbor_distance_max": float(feature_frame["nearest_neighbor_distance"].max()),
                "knn_same_regime_share_k5_mean": float(
                    knn_profile.loc[knn_profile["k"] == max(KNN_K_VALUES), "same_regime_neighbor_share"].mean()
                ),
                "knn_same_scale_share_k5_mean": float(
                    knn_profile.loc[knn_profile["k"] == max(KNN_K_VALUES), "same_scale_neighbor_share"].mean()
                ),
            }
        ]
    )
    return feature_frame, closest_pairs, summary, knn_profile, regime_composition, scale_composition


def machine_utilization_frame(schedule: pd.DataFrame, machines: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    busy = (
        schedule.assign(busy_min=schedule["end_min"] - schedule["start_min"])
        .groupby(["instance_id", "machine_id"], as_index=False)["busy_min"]
        .sum()
    )
    frame = machines.merge(busy, on=["instance_id", "machine_id"], how="left").fillna({"busy_min": 0})
    frame = frame.merge(params[["instance_id", "planning_horizon_min"]], on="instance_id", how="left")
    frame["utilization_share"] = frame["busy_min"] / frame["planning_horizon_min"]
    return frame


def _machine_sort_key(machine_id: str) -> tuple[int, int, str]:
    if machine_id.startswith("WB"):
        family_rank = 0
    elif machine_id.startswith("LAB"):
        family_rank = 1
    elif machine_id.startswith("HOP"):
        family_rank = 2
    else:
        family_rank = 99
    suffix = "".join(ch for ch in machine_id if ch.isdigit())
    return (family_rank, int(suffix) if suffix else 0, machine_id)


def schedule_plot(instance_id: str, schedule: pd.DataFrame, downtimes: pd.DataFrame):
    plot_df = schedule[schedule["instance_id"] == instance_id].copy().sort_values(["machine_id", "start_min", "end_min"])
    downs = downtimes[downtimes["instance_id"] == instance_id].copy()
    machine_order = sorted(plot_df["machine_id"].unique(), key=_machine_sort_key)
    ypos = {machine: idx for idx, machine in enumerate(machine_order)}
    fig, ax = plt.subplots(figsize=(15, max(4.8, 0.9 * len(machine_order) + 1.2)))
    palette = {
        "WEIGH_IN": "#62c2a8",
        "SAMPLE_CLASSIFY": "#ff9b6a",
        "UNLOAD": "#8ea2cf",
        "WEIGH_OUT": "#e186c8",
    }
    label_min_width = 32

    for idx, machine_id in enumerate(machine_order):
        lane_color = "#f7f9fc" if idx % 2 == 0 else "#eef3f8"
        ax.axhspan(idx - 0.42, idx + 0.42, color=lane_color, zorder=0)

    for row in downs.itertuples(index=False):
        ax.barh(
            ypos[row.machine_id],
            row.end_min - row.start_min,
            left=row.start_min,
            height=0.76,
            color="#d95f5f",
            alpha=0.16,
            edgecolor="#d95f5f",
            hatch="///",
            linewidth=0.0,
            zorder=1,
        )

    for row in plot_df.itertuples(index=False):
        width = row.end_min - row.start_min
        ax.barh(
            ypos[row.machine_id],
            width,
            left=row.start_min,
            height=0.58,
            color=palette.get(row.stage_name, "#4c78a8"),
            edgecolor="white",
            linewidth=1.1,
            alpha=0.96,
            zorder=2,
        )
        if width >= label_min_width:
            ax.text(
                row.start_min + width / 2,
                ypos[row.machine_id],
                row.job_id,
                va="center",
                ha="center",
                fontsize=7,
                color="#1f2937",
                zorder=3,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    ax.invert_yaxis()
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Machine")
    ax.set_title(f"FIFO schedule by machine: {instance_id}", fontsize=16, weight="semibold", loc="left", pad=12)
    ax.grid(axis="x", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")
    legend_handles = [Patch(facecolor=palette[s], edgecolor="white", label=s) for s in STAGE_ORDER]
    legend_handles.append(Patch(facecolor="#d95f5f", edgecolor="#d95f5f", hatch="///", alpha=0.16, label="DOWNTIME"))
    ax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        title="Stage",
        bbox_to_anchor=(1.02, 0.98),
        loc="upper left",
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    return fig


def load_context(root: Path = REPO_ROOT, artifact_dir: Path | None = None) -> dict[str, Any]:
    artifact_dir = artifact_dir or (DEFAULT_ARTIFACT_DIR if root == REPO_ROOT else root / "output" / "repl-analysis-artifacts")
    params = load_params_frame(root).sort_values(["scale_code", "regime_code", "replicate"]).reset_index(drop=True)
    catalog = pd.read_csv(root / "catalog" / "benchmark_catalog.csv")
    family_summary = pd.read_csv(root / "catalog" / "instance_family_summary.csv")
    validation_report_observed = pd.read_csv(root / "catalog" / "validation_report_observed.csv")
    observed_noise_manifest = json.loads((root / "catalog" / "observed_noise_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    jobs = add_instance_context(load_instance_csv(root, "jobs.csv"), params)
    operations = add_instance_context(load_instance_csv(root, "operations.csv"), params)
    eligible = add_instance_context(load_instance_csv(root, "eligible_machines.csv"), params)
    machines = add_instance_context(load_instance_csv(root, "machines.csv"), params)
    precedences = add_instance_context(load_instance_csv(root, "precedences.csv"), params)
    downtimes = add_instance_context(load_instance_csv(root, "machine_downtimes.csv"), params)
    events = add_instance_context(load_instance_csv(root, "events.csv"), params)
    schedule = add_instance_context(load_instance_csv(root, "fifo_schedule.csv"), params)
    job_metrics = add_instance_context(load_instance_csv(root, "fifo_job_metrics.csv"), params)
    due_audit = add_instance_context(load_instance_csv(root, "job_noise_audit.csv"), params)
    proc_audit = add_instance_context(load_instance_csv(root, "proc_noise_audit.csv"), params)
    congestion = add_instance_context(load_instance_csv(root, "job_congestion_proxy.csv"), params)

    jobs_enriched = compute_due_lower_bounds(jobs, eligible)
    structural_report = build_structural_report(root).merge(params[["instance_id", "scale_code", "regime_code"]], on="instance_id", how="left")
    event_report = event_consistency_frame(jobs, events, downtimes).merge(params[["instance_id", "scale_code", "regime_code"]], on="instance_id", how="left")
    audit_reconciliation = build_audit_reconciliation(jobs, eligible, due_audit, proc_audit).merge(
        params[["instance_id", "scale_code", "regime_code"]], on="instance_id", how="left"
    )
    regime_checks = build_regime_behavior_checks(family_summary, job_metrics, jobs_enriched)
    fifo_schema_report = build_fifo_schema_report(operations, eligible, precedences, schedule, downtimes).merge(
        params[["instance_id", "scale_code", "regime_code"]],
        on="instance_id",
        how="left",
    )
    release_consistency_report = build_release_consistency_report(root, manifest, observed_noise_manifest, params)
    utilization = machine_utilization_frame(schedule, machines, params)
    (
        instance_space_features,
        instance_space_pairs,
        instance_space_summary,
        instance_space_knn_profile,
        instance_space_knn_regime_composition,
        instance_space_knn_scale_composition,
    ) = build_instance_space_validation(
        root=root,
        params=params,
        jobs_enriched=jobs_enriched,
        eligible=eligible,
        machines=machines,
        downtimes=downtimes,
        job_metrics=job_metrics,
        utilization=utilization,
    )
    diagnostics = release_diagnostics(root)

    unload = (
        eligible.merge(
            operations[["instance_id", "job_id", "op_seq", "stage_name"]],
            on=["instance_id", "job_id", "op_seq"],
            how="left",
        )
        .merge(
            jobs[["instance_id", "job_id", "load_tons", "moisture_class", "commodity", "arrival_congestion_score"]],
            on=["instance_id", "job_id"],
            how="left",
        )
    )
    unload = unload[unload["stage_name"] == "UNLOAD"].copy()

    proc_audit_enriched = proc_audit.copy()
    proc_audit_enriched["proc_multiplier"] = proc_audit_enriched["proc_time_observed_min"] / proc_audit_enriched["proc_time_nominal_min"]

    summary = {
        "dataset_version": manifest["dataset_version"],
        "instance_count": int(params["instance_id"].nunique()),
        "job_count": int(len(jobs)),
        "operation_count": int(len(operations)),
        "eligible_rows": int(len(eligible)),
        "machine_rows": int(len(machines)),
        "structural_pass_rate": float((structural_report["status"] == "PASS").mean()),
        "release_consistency_checks_pass": bool(release_consistency_report["pass"].all()),
        "due_audit_match_share": float(audit_reconciliation["due_match_share"].mean()),
        "proc_audit_match_share": float(audit_reconciliation["proc_match_share"].mean()),
        "r2_due_slack_vs_priority": float(diagnostics["r2_due_slack_vs_priority"]),
        "r2_unload_proc_vs_load_machine_moisture": float(diagnostics["r2_unload_proc_vs_load_machine_moisture"]),
        "fifo_schema_checks_pass": bool(
            fifo_schema_report[
                [
                    "eligible_assignment_ok",
                    "release_time_ok",
                    "precedence_ok",
                    "machine_overlap_ok",
                    "downtime_ok",
                ]
            ].all(axis=None)
        ),
        "flow_regime_order_checks_pass": bool(regime_checks["mean_flow_order_ok"].all() and regime_checks["p95_flow_order_ok"].all()),
        "queue_regime_order_checks_pass": bool(regime_checks["mean_queue_order_ok"].all()),
        "congestion_mean_regime_order_checks_pass": bool(regime_checks["mean_congestion_order_ok"].all()),
        "instance_space_exact_duplicate_checks_pass": bool(
            instance_space_summary.loc[0, "exact_core_duplicate_count"] == 0
            and instance_space_summary.loc[0, "exact_feature_duplicate_count"] == 0
        ),
        "instance_space_duplicate_like_checks_pass": bool(
            instance_space_summary.loc[0, "duplicate_like_candidate_count"] == 0
        ),
        "instance_space_nearest_neighbor_distance_min": float(
            instance_space_summary.loc[0, "nearest_neighbor_distance_min"]
        ),
        "g2milp_role": manifest.get("official_dataset_role", ""),
    }

    return {
        "root": root,
        "artifact_dir": artifact_dir,
        "params": params,
        "catalog": catalog,
        "family_summary": family_summary,
        "validation_report_observed": validation_report_observed,
        "observed_noise_manifest": observed_noise_manifest,
        "manifest": manifest,
        "jobs": jobs,
        "jobs_enriched": jobs_enriched,
        "operations": operations,
        "eligible": eligible,
        "machines": machines,
        "precedences": precedences,
        "downtimes": downtimes,
        "events": events,
        "schedule": schedule,
        "job_metrics": job_metrics,
        "due_audit": due_audit,
        "proc_audit": proc_audit,
        "proc_audit_enriched": proc_audit_enriched,
        "congestion": congestion,
        "structural_report": structural_report,
        "event_report": event_report,
        "audit_reconciliation": audit_reconciliation,
        "regime_checks": regime_checks,
        "fifo_schema_report": fifo_schema_report,
        "release_consistency_report": release_consistency_report,
        "utilization": utilization,
        "instance_space_features": instance_space_features,
        "instance_space_pairs": instance_space_pairs,
        "instance_space_summary": instance_space_summary,
        "instance_space_knn_profile": instance_space_knn_profile,
        "instance_space_knn_regime_composition": instance_space_knn_regime_composition,
        "instance_space_knn_scale_composition": instance_space_knn_scale_composition,
        "diagnostics": diagnostics,
        "unload": unload,
        "summary": summary,
    }


CTX = load_context()

SUMMARY = CTX["summary"]
PARAMS = CTX["params"]
CATALOG = CTX["catalog"]
FAMILY_SUMMARY = CTX["family_summary"]
VALIDATION_REPORT_OBSERVED = CTX["validation_report_observed"]
OBSERVED_NOISE_MANIFEST = CTX["observed_noise_manifest"]
MANIFEST = CTX["manifest"]
JOBS = CTX["jobs"]
JOBS_ENRICHED = CTX["jobs_enriched"]
OPERATIONS = CTX["operations"]
ELIGIBLE = CTX["eligible"]
MACHINES = CTX["machines"]
PRECEDENCES = CTX["precedences"]
DOWNTIMES = CTX["downtimes"]
EVENTS = CTX["events"]
SCHEDULE = CTX["schedule"]
JOB_METRICS = CTX["job_metrics"]
DUE_AUDIT = CTX["due_audit"]
PROC_AUDIT = CTX["proc_audit"]
PROC_AUDIT_ENRICHED = CTX["proc_audit_enriched"]
CONGESTION = CTX["congestion"]
STRUCTURAL_REPORT = CTX["structural_report"]
EVENT_REPORT = CTX["event_report"]
AUDIT_RECONCILIATION = CTX["audit_reconciliation"]
REGIME_CHECKS = CTX["regime_checks"]
FIFO_SCHEMA_REPORT = CTX["fifo_schema_report"]
RELEASE_CONSISTENCY_REPORT = CTX["release_consistency_report"]
UTILIZATION = CTX["utilization"]
INSTANCE_SPACE_FEATURES = CTX["instance_space_features"]
INSTANCE_SPACE_PAIRS = CTX["instance_space_pairs"]
INSTANCE_SPACE_SUMMARY = CTX["instance_space_summary"]
INSTANCE_SPACE_KNN_PROFILE = CTX["instance_space_knn_profile"]
INSTANCE_SPACE_KNN_REGIME_COMPOSITION = CTX["instance_space_knn_regime_composition"]
INSTANCE_SPACE_KNN_SCALE_COMPOSITION = CTX["instance_space_knn_scale_composition"]
DIAGNOSTICS = CTX["diagnostics"]
UNLOAD = CTX["unload"]


def inventory_tables(ctx: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    ctx = ctx or CTX
    inventory = pd.DataFrame([ctx["summary"]])
    machine_family = (
        ctx["machines"].groupby(["machine_family"], as_index=False)["machine_id"]
        .count()
        .rename(columns={"machine_id": "machine_rows"})
        .sort_values("machine_rows", ascending=False)
    )
    return {
        "inventory": inventory,
        "catalog": ctx["catalog"].sort_values(["scale_code", "regime_code", "replicate"]),
        "family_summary": ctx["family_summary"].sort_values(["scale_code", "regime_code"]),
        "machine_family": machine_family,
    }


def validation_tables(ctx: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    ctx = ctx or CTX
    due_margin_summary = (
        ctx["jobs_enriched"].groupby(["scale_code", "regime_code"], as_index=False)["due_margin_over_lb_min"]
        .agg(["mean", "min", "median", "max"])
        .round(2)
        .reset_index()
    )
    return {
        "structural_report": ctx["structural_report"].sort_values(["scale_code", "regime_code", "instance_id"]),
        "event_report": ctx["event_report"].sort_values(["scale_code", "regime_code", "instance_id"]),
        "audit_reconciliation": ctx["audit_reconciliation"].sort_values(["scale_code", "regime_code", "instance_id"]),
        "fifo_schema_report": ctx["fifo_schema_report"].sort_values(["scale_code", "regime_code", "instance_id"]),
        "release_consistency_report": ctx["release_consistency_report"].copy(),
        "due_margin_summary": due_margin_summary,
    }


def instance_space_tables(ctx: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    ctx = ctx or CTX
    return {
        "instance_space_features": ctx["instance_space_features"].sort_values(
            ["nearest_neighbor_distance", "scale_code", "regime_code", "instance_id"]
        ),
        "instance_space_pairs": ctx["instance_space_pairs"].copy(),
        "instance_space_summary": ctx["instance_space_summary"].copy(),
        "instance_space_knn_profile": ctx["instance_space_knn_profile"].sort_values(
            ["k", "mean_knn_distance", "instance_id"]
        ),
        "instance_space_knn_regime_composition": ctx["instance_space_knn_regime_composition"].copy(),
        "instance_space_knn_scale_composition": ctx["instance_space_knn_scale_composition"].copy(),
    }


def _annotate_category_medians(
    ax,
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    order: list[str],
    fmt: str = "{:.0f}",
    pad_share: float = 0.02,
) -> None:
    medians = data.groupby(category_col)[value_col].median()
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * pad_share
    for idx, category in enumerate(order):
        if category in medians.index:
            ax.text(
                idx,
                medians[category] + pad,
                fmt.format(medians[category]),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color="#334155",
            )


def _label_bars(ax, fmt: str = "{:.1f}", suffix: str = "", pad_share: float = 0.015) -> None:
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * pad_share
    for patch in ax.patches:
        value = patch.get_height()
        if abs(value) < 1e-9:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + pad,
            f"{fmt.format(value)}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
        )


def plot_inventory_overview(ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    jobs_heatmap = ctx["family_summary"].pivot(index="scale_code", columns="regime_code", values="avg_n_jobs").reindex(index=SCALE_ORDER, columns=REGIME_ORDER)
    sns.heatmap(jobs_heatmap, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[0], cbar_kws={"label": "Jobs médios"})
    axes[0].set_title("Cobertura do release\nCada célula resume as 3 réplicas da família", fontsize=13)
    axes[0].set_xlabel("Regime")
    axes[0].set_ylabel("Escala")

    machine_family = inventory_tables(ctx)["machine_family"]
    sns.barplot(data=machine_family, x="machine_family", y="machine_rows", hue="machine_family", dodge=False, legend=False, ax=axes[1], palette="crest")
    axes[1].set_title("Volume de linhas por família de máquina", fontsize=13)
    axes[1].set_xlabel("Família")
    axes[1].set_ylabel("Linhas no release")
    axes[1].tick_params(axis="x", rotation=20)
    _label_bars(axes[1], fmt="{:.0f}")

    fig.suptitle("Inventário do dataset oficial", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.95, "Leitura rápida: o release cobre 4 escalas, 3 regimes e mantém a estrutura de recursos esperada.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / "inventory_overview.png", dpi=160, bbox_inches="tight")
    return fig


def plot_validation_overview(ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    fig, axes = plt.subplot_mosaic(
        [["issues", "audit"], ["margin", "margin"]],
        figsize=(15.8, 10.0),
        gridspec_kw={"height_ratios": [1.0, 1.1]},
    )
    issue_heatmap = ctx["structural_report"].pivot_table(
        index="scale_code",
        columns="regime_code",
        values="issue_count",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=SCALE_ORDER, columns=REGIME_ORDER)
    issue_annotations = issue_heatmap.map(lambda value: "PASS\n0 issues" if value == 0 else f"{int(value)}\nissues")
    sns.heatmap(
        issue_heatmap,
        annot=issue_annotations,
        fmt="",
        cmap=sns.light_palette("#15803d", as_cmap=True),
        ax=axes["issues"],
        cbar=False,
        linewidths=1.5,
        linecolor="white",
    )
    axes["issues"].set_title("Integridade estrutural\nCada célula deve ficar em PASS", fontsize=13)
    axes["issues"].set_xlabel("Regime")
    axes["issues"].set_ylabel("Escala")

    audit_summary = pd.DataFrame(
        {
            "check": ["Prazo vs audit", "Proc_time vs audit"],
            "match_share": [
                float(ctx["audit_reconciliation"]["due_match_share"].mean()),
                float(ctx["audit_reconciliation"]["proc_match_share"].mean()),
            ],
        }
    )
    sns.barplot(
        data=audit_summary,
        x="check",
        y="match_share",
        hue="check",
        dodge=False,
        legend=False,
        ax=axes["audit"],
        palette=["#2a9d8f", "#457b9d"],
    )
    axes["audit"].axhline(1.0, color="#0f172a", linewidth=1.0, linestyle="--", alpha=0.7)
    axes["audit"].set_ylim(0.995, 1.001)
    axes["audit"].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axes["audit"].set_title("Reconciliação auditável\nMeta: 100% das linhas", fontsize=13)
    axes["audit"].set_xlabel("")
    axes["audit"].set_ylabel("Linhas reconciliadas")
    _label_bars(axes["audit"], fmt="{:.1%}", pad_share=0.001)

    margin_order = REGIME_ORDER
    sns.boxplot(
        data=ctx["jobs_enriched"],
        x="regime_code",
        y="due_margin_over_lb_min",
        order=margin_order,
        hue="regime_code",
        dodge=False,
        legend=False,
        ax=axes["margin"],
        palette="flare",
    )
    _annotate_category_medians(axes["margin"], ctx["jobs_enriched"], "regime_code", "due_margin_over_lb_min", margin_order)
    axes["margin"].set_title("Prazo acima do lower bound físico\nMediana anotada em cada regime", fontsize=13)
    axes["margin"].set_xlabel("Regime")
    axes["margin"].set_ylabel("Margem sobre LB (min)")

    fig.suptitle("Validação estrutural e auditabilidade", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.93,
        "Leitura rápida: as 36 instâncias passam sem issues e os dois audits reconciliam 100% das linhas centrais.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0, w_pad=1.8)
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / "structural_validation_and_auditability.png", dpi=160, bbox_inches="tight")
    return fig


def plot_observational_layer(ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    sns.boxplot(
        data=ctx["jobs_enriched"],
        x="priority_class",
        y="due_slack_min",
        order=PRIORITY_ORDER,
        hue="priority_class",
        dodge=False,
        legend=False,
        ax=axes[0, 0],
        palette="viridis",
    )
    _annotate_category_medians(axes[0, 0], ctx["jobs_enriched"], "priority_class", "due_slack_min", PRIORITY_ORDER)
    axes[0, 0].set_title(
        f"Folga observada ainda segue prioridade\nR²(priority -> slack) = {ctx['diagnostics']['r2_due_slack_vs_priority']:.3f}",
        fontsize=13,
    )
    axes[0, 0].set_xlabel("Classe de prioridade")
    axes[0, 0].set_ylabel("Folga observada (min)")

    appointment_df = ctx["jobs_enriched"].assign(
        appointment_label=lambda frame: np.where(frame["appointment_flag"].eq(1), "Com appointment", "Sem appointment")
    )
    appointment_order = ["Sem appointment", "Com appointment"]
    sns.boxplot(
        data=appointment_df,
        x="appointment_label",
        y="reveal_lead_min",
        order=appointment_order,
        hue="appointment_label",
        dodge=False,
        legend=False,
        ax=axes[0, 1],
        palette="coolwarm",
    )
    _annotate_category_medians(axes[0, 1], appointment_df, "appointment_label", "reveal_lead_min", appointment_order)
    axes[0, 1].set_title("Appointments antecipam a visibilidade dos jobs\nSem appointment a mediana fica essencialmente em 0 min", fontsize=13)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Chegada - revelação (min)")

    unload_trend = (
        ctx["unload"].groupby(["regime_code", "load_tons"], as_index=False)["proc_time_min"].median()
    )
    sns.lineplot(
        data=unload_trend,
        x="load_tons",
        y="proc_time_min",
        hue="regime_code",
        hue_order=REGIME_ORDER,
        marker="o",
        linewidth=2.2,
        ax=axes[1, 0],
        palette="deep",
    )
    axes[1, 0].set_title(
        f"UNLOAD cresce com carga e severidade do regime\nR²(load + machine + moisture -> proc) = {ctx['diagnostics']['r2_unload_proc_vs_load_machine_moisture']:.3f}",
        fontsize=13,
    )
    axes[1, 0].set_xlabel("Carga do job (t)")
    axes[1, 0].set_ylabel("Proc_time mediano em UNLOAD (min)")

    sns.boxplot(
        data=ctx["proc_audit_enriched"],
        x="stage_name",
        y="proc_multiplier",
        order=STAGE_ORDER,
        hue="stage_name",
        dodge=False,
        legend=False,
        ax=axes[1, 1],
        palette="Set3",
    )
    axes[1, 1].axhline(1.0, color="#475569", linewidth=1.0, linestyle="--", alpha=0.8)
    _annotate_category_medians(axes[1, 1], ctx["proc_audit_enriched"], "stage_name", "proc_multiplier", STAGE_ORDER, fmt="{:.2f}", pad_share=0.015)
    axes[1, 1].set_title("Multiplicador observado/nominal por estágio\nWEIGH_* fica perto de 1; SAMPLE/UNLOAD concentram a variação", fontsize=13)
    axes[1, 1].set_xlabel("Estágio")
    axes[1, 1].set_ylabel("Observed / nominal")
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.suptitle("Comportamento da camada observacional", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.95,
        "Leitura rápida: a prioridade continua relevante, mas o dataset deixa de ser rigidamente determinístico e introduz variação estruturada.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / "observational_layer_behavior.png", dpi=160, bbox_inches="tight")
    return fig


def plot_congestion_diagnostics(ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    congestion_vs_proc = ctx["proc_audit_enriched"].copy()
    congestion_vs_proc["congestion_decile"] = pd.qcut(
        congestion_vs_proc["arrival_congestion_score"],
        q=10,
        labels=range(1, 11),
        duplicates="drop",
    )
    trend = (
        congestion_vs_proc.groupby(["stage_name", "congestion_decile"], as_index=False, observed=False)["proc_multiplier"]
        .median()
        .dropna(subset=["congestion_decile"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    sns.lineplot(
        data=trend,
        x="congestion_decile",
        y="proc_multiplier",
        hue="stage_name",
        style="stage_name",
        markers=True,
        linewidth=2.0,
        ax=axes[0],
        palette="Set2",
    )
    axes[0].axhline(1.0, color="#475569", linewidth=1.0, linestyle="--", alpha=0.8)
    axes[0].set_title("Mais congestionamento tende a inflar proc_time\nCada ponto é a mediana em um decil de congestionamento", fontsize=13)
    axes[0].set_xlabel("Decil do congestionamento na chegada (1 = baixo, 10 = alto)")
    axes[0].set_ylabel("Observed / nominal")

    sns.boxplot(
        data=ctx["jobs_enriched"],
        x="regime_code",
        y="arrival_congestion_score",
        order=REGIME_ORDER,
        hue="regime_code",
        dodge=False,
        legend=False,
        ax=axes[1],
        palette="mako",
    )
    _annotate_category_medians(axes[1], ctx["jobs_enriched"], "regime_code", "arrival_congestion_score", REGIME_ORDER, fmt="{:.2f}")
    axes[1].set_title("Regimes mais severos concentram congestionamento maior\nAs medianas sobem de balanced para disrupted", fontsize=13)
    axes[1].set_xlabel("Regime")
    axes[1].set_ylabel("Arrival congestion score")

    fig.suptitle("Proxy de congestionamento", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.95,
        "Leitura rápida: o congestionamento não é ruído aleatório; ele se associa ao multiplicador de proc_time e é mais alto nos regimes severos.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / "congestion_diagnostics.png", dpi=160, bbox_inches="tight")
    return fig


def plot_operational_sanity(ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    mean_heatmap = ctx["family_summary"].pivot(index="scale_code", columns="regime_code", values="avg_fifo_mean_flow_min").reindex(index=SCALE_ORDER, columns=REGIME_ORDER)
    sns.heatmap(mean_heatmap, annot=True, fmt=".1f", cmap="YlOrBr", ax=axes[0, 0], cbar_kws={"label": "Minutos"})
    axes[0, 0].set_title("Flow médio FIFO\nDeve piorar de balanced -> peak -> disrupted", fontsize=13)
    axes[0, 0].set_xlabel("Regime")
    axes[0, 0].set_ylabel("Escala")

    p95_heatmap = ctx["family_summary"].pivot(index="scale_code", columns="regime_code", values="avg_fifo_p95_flow_min").reindex(index=SCALE_ORDER, columns=REGIME_ORDER)
    sns.heatmap(p95_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0, 1], cbar_kws={"label": "Minutos"})
    axes[0, 1].set_title("Flow p95 FIFO\nA cauda também deve piorar monotonicamente", fontsize=13)
    axes[0, 1].set_xlabel("Regime")
    axes[0, 1].set_ylabel("Escala")

    sns.boxplot(data=ctx["job_metrics"], x="regime_code", y="flow_time_min", order=REGIME_ORDER, hue="regime_code", dodge=False, legend=False, ax=axes[1, 0], palette="Spectral")
    _annotate_category_medians(axes[1, 0], ctx["job_metrics"], "regime_code", "flow_time_min", REGIME_ORDER)
    axes[1, 0].set_title("Distribuição de flow time por regime\nMedianas anotadas para leitura rápida", fontsize=13)
    axes[1, 0].set_xlabel("Regime")
    axes[1, 0].set_ylabel("Flow time (min)")

    util_plot = ctx["utilization"].groupby(["machine_family", "regime_code"], as_index=False)["utilization_share"].mean()
    sns.barplot(data=util_plot, x="machine_family", y="utilization_share", hue="regime_code", hue_order=REGIME_ORDER, ax=axes[1, 1], palette="deep")
    axes[1, 1].set_title("Utilização média por família de máquina\nBarras mais altas indicam recursos mais pressionados", fontsize=13)
    axes[1, 1].set_xlabel("Família de máquina")
    axes[1, 1].set_ylabel("Utilization share")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1, 1].tick_params(axis="x", rotation=20)
    _label_bars(axes[1, 1], fmt="{:.0%}", pad_share=0.01)

    fig.suptitle("Sanidade operacional por regime", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.95,
        "Leitura rápida: em todas as escalas, balanced < peak < disrupted tanto no flow médio quanto na cauda p95.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / "operational_performance_and_regime_sanity.png", dpi=160, bbox_inches="tight")
    return fig


def plot_instance_space_coverage(ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    feature_frame = ctx["instance_space_features"].copy()
    closest_pairs = ctx["instance_space_pairs"].head(8).copy()
    summary = ctx["instance_space_summary"].iloc[0]
    knn_profile = ctx["instance_space_knn_profile"].copy()
    regime_composition = ctx["instance_space_knn_regime_composition"].copy()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.4, 10.2),
        gridspec_kw={"width_ratios": [1.0, 1.18], "height_ratios": [1.0, 1.03]},
    )
    palette = {"balanced": "#2a9d8f", "peak": "#e9c46a", "disrupted": "#e76f51"}
    markers = {"XS": "o", "S": "s", "M": "D", "L": "^"}

    sns.scatterplot(
        data=feature_frame,
        x="pc1",
        y="pc2",
        hue="regime_code",
        hue_order=REGIME_ORDER,
        style="scale_code",
        style_order=SCALE_ORDER,
        markers=markers,
        palette=palette,
        s=92,
        edgecolor="white",
        linewidth=0.8,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        "PCA do espaço de instâncias\nPC1 + PC2 resumem a dispersão multivariada",
        fontsize=13,
    )
    axes[0, 0].set_xlabel(f"PC1 ({summary['pca_pc1_explained_variance_ratio']:.1%} da variância)")
    axes[0, 0].set_ylabel(f"PC2 ({summary['pca_pc2_explained_variance_ratio']:.1%} da variância)")
    axes[0, 0].axhline(0, color="#cbd5e1", linewidth=0.9, zorder=0)
    axes[0, 0].axvline(0, color="#cbd5e1", linewidth=0.9, zorder=0)
    axes[0, 0].grid(alpha=0.28)
    regime_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[regime],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=regime,
        )
        for regime in REGIME_ORDER
    ]
    scale_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[scale],
            linestyle="",
            color="#334155",
            markerfacecolor="#94a3b8",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=scale,
        )
        for scale in SCALE_ORDER
    ]
    regime_legend = axes[0, 0].legend(
        handles=regime_handles,
        title="Regime",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=10,
        borderpad=0.6,
    )
    axes[0, 0].add_artist(regime_legend)
    axes[0, 0].legend(
        handles=scale_handles,
        title="Escala",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=10,
        borderpad=0.6,
    )

    sns.boxplot(
        data=knn_profile,
        x="k",
        y="mean_knn_distance",
        hue="scale_code",
        hue_order=SCALE_ORDER,
        palette="Set2",
        showfliers=False,
        width=0.68,
        ax=axes[0, 1],
    )
    sns.stripplot(
        data=knn_profile,
        x="k",
        y="mean_knn_distance",
        color="#334155",
        alpha=0.32,
        size=3.2,
        jitter=0.16,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(
        "Perfil de distância kNN\nA distância média cresce quando ampliamos a vizinhança",
        fontsize=13,
    )
    axes[0, 1].set_xlabel("k vizinhos mais próximos")
    axes[0, 1].set_ylabel("Distância média aos k vizinhos")
    axes[0, 1].axhline(
        INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD,
        color="#475569",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    axes[0, 1].text(
        0.03,
        0.04,
        f"limiar duplicate-like = {INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD:.1f}",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    legend = axes[0, 1].get_legend()
    if legend is not None:
        legend.set_title("Escala")
        legend.get_title().set_fontsize(10)
        for text in legend.get_texts():
            text.set_fontsize(10)
        legend.get_frame().set_alpha(0.95)
    axes[0, 1].grid(axis="y", alpha=0.25)

    regime_heatmap = (
        regime_composition[regime_composition["k"] == max(KNN_K_VALUES)]
        .pivot(index="source_regime", columns="neighbor_regime", values="share")
        .reindex(index=REGIME_ORDER, columns=REGIME_ORDER)
    )
    sns.heatmap(
        regime_heatmap,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        ax=axes[1, 0],
        annot_kws={"fontsize": 11},
        cbar_kws={"label": "Share médio entre os 5-NN", "shrink": 0.9},
    )
    axes[1, 0].set_title(
        "Pureza de vizinhança por regime (k=5)\nValores altos na diagonal indicam separação estrutural",
        fontsize=13,
    )
    axes[1, 0].set_xlabel("Regime do vizinho")
    axes[1, 0].set_ylabel("Regime da instância fonte")

    def _short_instance_label(label: str) -> str:
        return (
            label.replace("GO_", "")
            .replace("BALANCED", "BAL")
            .replace("DISRUPTED", "DISR")
            .replace("_", "-")
        )

    closest_pairs["pair_label"] = closest_pairs["instance_a"].map(_short_instance_label) + " ↔ " + closest_pairs["instance_b"].map(_short_instance_label)
    closest_pairs_plot = closest_pairs.sort_values("distance", ascending=True).copy()
    sns.barplot(
        data=closest_pairs_plot,
        x="distance",
        y="pair_label",
        color="#8ecae6",
        ax=axes[1, 1],
    )
    axes[1, 1].axvline(
        INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD,
        color="#475569",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    axes[1, 1].set_title("Pares mais próximos do release\nNenhum par cai na zona de suspeita", fontsize=13)
    axes[1, 1].set_xlabel("Distância Euclidiana em features padronizadas")
    axes[1, 1].set_ylabel("")
    axes[1, 1].tick_params(axis="y", labelsize=10)
    axes[1, 1].grid(axis="x", alpha=0.25)
    x_max = max(float(closest_pairs_plot["distance"].max()) * 1.12, INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD * 1.4)
    axes[1, 1].set_xlim(0, x_max)
    for patch, (_, row) in zip(axes[1, 1].patches, closest_pairs_plot.iterrows()):
        axes[1, 1].text(
            min(float(row["distance"]) + x_max * 0.018, x_max * 0.97),
            patch.get_y() + patch.get_height() / 2,
            f"{float(row['distance']):.2f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#334155",
        )
    axes[1, 1].text(
        0.98,
        0.04,
        "0 pares abaixo do limiar",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )

    fig.suptitle("Cobertura do espaço de instâncias", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.93,
        (
            "Leitura rápida: o release não tem duplicatas exatas e o vizinho mais próximo mais apertado ainda fica "
            f"em {summary['nearest_neighbor_distance_min']:.2f}, bem acima do limiar heurístico {INSTANCE_SPACE_DUPLICATE_LIKE_THRESHOLD:.1f}."
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0, w_pad=2.0)
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / "instance_space_coverage.png", dpi=160, bbox_inches="tight")
    return fig


def plot_instance_drilldown(instance_id: str = "GO_XS_DISRUPTED_01", ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    fig = schedule_plot(instance_id, ctx["schedule"], ctx["downtimes"])
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / f"{instance_id.lower()}_fifo_schedule.png", dpi=160, bbox_inches="tight")
    return fig


def plot_job_level_views(instance_id: str = "GO_XS_DISRUPTED_01", ctx: dict[str, Any] | None = None, save: bool = False):
    ctx = ctx or CTX
    sample_jobs = ctx["jobs_enriched"][ctx["jobs_enriched"]["instance_id"] == instance_id]
    sample_metrics = ctx["job_metrics"][ctx["job_metrics"]["instance_id"] == instance_id]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    sns.scatterplot(
        data=sample_jobs,
        x="arrival_time_min",
        y="completion_due_min",
        hue="priority_class",
        hue_order=PRIORITY_ORDER,
        palette="viridis",
        ax=axes[0],
        s=75,
        alpha=0.85,
    )
    axes[0].set_title(f"{instance_id}: chegada vs prazo\nCada ponto representa um job e a cor indica prioridade", fontsize=13)
    axes[0].set_xlabel("Arrival time (min)")
    axes[0].set_ylabel("Completion due (min)")

    top_flow = sample_metrics.sort_values("flow_time_min", ascending=False).head(12).sort_values("flow_time_min", ascending=True)
    sns.barplot(
        data=top_flow,
        x="flow_time_min",
        y="job_id",
        hue="job_id",
        dodge=False,
        legend=False,
        ax=axes[1],
        palette="rocket",
        orient="h",
    )
    axes[1].set_title(f"{instance_id}: 12 maiores flow times\nAs barras mais longas identificam os jobs mais críticos", fontsize=13)
    axes[1].set_xlabel("Flow time (min)")
    axes[1].set_ylabel("Job")
    for patch in axes[1].patches:
        value = patch.get_width()
        axes[1].text(value + 1.5, patch.get_y() + patch.get_height() / 2, f"{value:.0f}", va="center", ha="left", fontsize=8, color="#334155")

    fig.suptitle("Drilldown de jobs", x=0.02, y=1.03, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.95,
        "Leitura rápida: o cronograma respeita a ordem de máquina, e os jobs críticos aparecem tanto no Gantt quanto no ranking de flow time.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if save:
        _ensure_artifact_dir(ctx)
        fig.savefig(ctx["artifact_dir"] / f"{instance_id.lower()}_job_level_views.png", dpi=160, bbox_inches="tight")
    return fig


def _ensure_artifact_dir(ctx: dict[str, Any]) -> None:
    ctx["artifact_dir"].mkdir(parents=True, exist_ok=True)


def export_all_artifacts(ctx: dict[str, Any] | None = None, instance_id: str = "GO_XS_DISRUPTED_01") -> Path:
    ctx = ctx or CTX
    _ensure_artifact_dir(ctx)
    inventory_tables(ctx)["inventory"].to_csv(ctx["artifact_dir"] / "inventory_summary.csv", index=False)
    validation_tables(ctx)["structural_report"].to_csv(ctx["artifact_dir"] / "structural_report.csv", index=False)
    validation_tables(ctx)["event_report"].to_csv(ctx["artifact_dir"] / "event_report.csv", index=False)
    validation_tables(ctx)["audit_reconciliation"].to_csv(ctx["artifact_dir"] / "audit_reconciliation.csv", index=False)
    validation_tables(ctx)["fifo_schema_report"].to_csv(ctx["artifact_dir"] / "fifo_schema_report.csv", index=False)
    validation_tables(ctx)["release_consistency_report"].to_csv(ctx["artifact_dir"] / "release_consistency_report.csv", index=False)
    validation_tables(ctx)["due_margin_summary"].to_csv(ctx["artifact_dir"] / "due_margin_summary.csv", index=False)
    instance_space_tables(ctx)["instance_space_features"].to_csv(ctx["artifact_dir"] / "instance_space_features.csv", index=False)
    instance_space_tables(ctx)["instance_space_pairs"].to_csv(ctx["artifact_dir"] / "instance_space_pairs.csv", index=False)
    instance_space_tables(ctx)["instance_space_summary"].to_csv(ctx["artifact_dir"] / "instance_space_summary.csv", index=False)
    instance_space_tables(ctx)["instance_space_knn_profile"].to_csv(ctx["artifact_dir"] / "instance_space_knn_profile.csv", index=False)
    instance_space_tables(ctx)["instance_space_knn_regime_composition"].to_csv(
        ctx["artifact_dir"] / "instance_space_knn_regime_composition.csv", index=False
    )
    instance_space_tables(ctx)["instance_space_knn_scale_composition"].to_csv(
        ctx["artifact_dir"] / "instance_space_knn_scale_composition.csv", index=False
    )
    pd.DataFrame([ctx["summary"]]).to_csv(ctx["artifact_dir"] / "repl_summary.csv", index=False)
    plot_inventory_overview(ctx, save=True)
    plot_validation_overview(ctx, save=True)
    plot_observational_layer(ctx, save=True)
    plot_congestion_diagnostics(ctx, save=True)
    plot_operational_sanity(ctx, save=True)
    plot_instance_space_coverage(ctx, save=True)
    plot_instance_drilldown(instance_id=instance_id, ctx=ctx, save=True)
    plot_job_level_views(instance_id=instance_id, ctx=ctx, save=True)
    return ctx["artifact_dir"]


def repl_help() -> None:
    print("Loaded the observed benchmark into REPL globals.")
    print("")
    print("Quick start:")
    print("  SUMMARY")
    print("  inventory_tables()['inventory']")
    print("  validation_tables()['structural_report'].head()")
    print("  plot_inventory_overview()")
    print("  plot_validation_overview()")
    print("  plot_observational_layer()")
    print("  plot_congestion_diagnostics()")
    print("  plot_operational_sanity()")
    print("  plot_instance_space_coverage()")
    print("  plot_instance_drilldown('GO_XS_DISRUPTED_01')")
    print("  plot_job_level_views('GO_XS_DISRUPTED_01')")
    print("  export_all_artifacts()")


if __name__ == "__main__":
    print("Observed benchmark REPL loaded.")
    print(json.dumps(SUMMARY, indent=2, ensure_ascii=False))
    print("")
    repl_help()
