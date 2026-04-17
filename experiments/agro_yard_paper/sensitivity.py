
"""Computational sensitivity helpers and cache-aware wrappers for the paper notebook."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import Any, METHOD_SPECS, UTILITY_WEIGHTS, _attach_protocol_columns, math, np
from .pipeline import (
    _build_job_order,
    _exact_subset_job_cap,
    _load_instance_tables,
    _optimize_job_order,
    _schedule_jobs_from_order,
    _solve_exact_reference_schedule,
    _summarize_schedule,
    online_replay,
)


def run_budgeted_method(
    root: Path,
    instance_id: str,
    method_name: str,
    budget_sec: float,
    n_workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    instance = _load_instance_tables(root, instance_id)
    jobs = instance["jobs"].copy()
    if method_name == "Mref_EXACT_XS_S":
        scale_code = pd.read_csv(root / "catalog" / "benchmark_catalog.csv").loc[
            lambda df: df["instance_id"].eq(instance_id), "scale_code"
        ].iloc[0]
        cap = _exact_subset_job_cap(str(scale_code))
        exact_schedule, exact_info = _solve_exact_reference_schedule(
            root=root,
            instance_id=instance_id,
            time_limit_sec=budget_sec,
            max_jobs=cap,
            method_name=method_name,
        )
        remaining_order = [job for job in _build_job_order(instance, "M3_EVENT_REACTIVE") if job not in set(exact_schedule["job_id"])]
        schedule, _ = _schedule_jobs_from_order(
            instance_id=instance_id,
            method_name=method_name,
            instance=instance,
            job_order=remaining_order,
            blocked_rows=exact_schedule,
        )
        _, summary = _summarize_schedule(
            instance_id=instance_id,
            method_name=method_name,
            jobs=jobs,
            schedule=schedule,
            runtime_sec=exact_info["runtime_sec"],
            solver_status=exact_info["solver_status"],
            replan_count=1,
            extra={
                "n_workers": int(n_workers),
                "budget_sec": float(budget_sec),
                "threads": int(n_workers),
                "exact_job_count": exact_info["exact_job_count"],
            },
        )
        return schedule, summary

    if method_name == "M4_METAHEURISTIC_L":
        seed_order = _build_job_order(instance, "M3_EVENT_REACTIVE")
        schedule, summary = _optimize_job_order(
            root=root,
            instance_id=instance_id,
            method_name=method_name,
            seed_order=seed_order,
            time_budget_sec=budget_sec,
            n_workers=n_workers,
        )
    elif method_name in {"M2_PERIODIC_15", "M2_PERIODIC_30", "M3_EVENT_REACTIVE"}:
        schedule, online_info = online_replay.run_online_policy(
            instance_id=instance_id,
            method_name=method_name,
            instance=instance,
            delta_min=METHOD_SPECS[method_name].delta_min,
        )
        _, summary = _summarize_schedule(
            instance_id=instance_id,
            method_name=method_name,
            jobs=jobs,
            schedule=schedule,
            runtime_sec=online_info["runtime_sec"],
            solver_status=online_info["solver_status"],
            replan_count=online_info["replan_count"],
        )
    else:
        raise ValueError(f"Unsupported budgeted method {method_name}")

    summary["n_workers"] = int(n_workers)
    summary["threads"] = int(n_workers)
    summary["budget_sec"] = float(budget_sec)
    return schedule, summary

def _schedule_signature(schedule: pd.DataFrame) -> pd.Series:
    ordered = schedule.sort_values(["job_id", "op_seq"]).copy()
    return ordered["machine_id"].astype(str) + "|" + ordered["start_min"].round(3).astype(str)

def run_compute_sensitivity(
    root: Path,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    protocol_catalog = _attach_protocol_columns(catalog)
    protocol_catalog = protocol_catalog.sort_values([
        "protocol_role",
        "scale_code",
        "regime_code",
        "replicate",
        "instance_id",
    ]).reset_index(drop=True)
    budget_map = {"short": 1.0, "medium": 2.5, "long": 5.0}
    exact_budget_map = {"short": 3.0, "medium": 6.0, "long": 9.0}
    rows = []
    baseline_schedules: dict[tuple[str, str], pd.DataFrame] = {}
    for instance_row in protocol_catalog.itertuples(index=False):
        methods = ["M2_PERIODIC_15", "M3_EVENT_REACTIVE"]
        if instance_row.scale_code in {"XS", "S"}:
            methods.append("Mref_EXACT_XS_S")
        if instance_row.scale_code == "L":
            methods.append("M4_METAHEURISTIC_L")
        for method_name in methods:
            for budget_label, budget_sec in budget_map.items():
                actual_budget = exact_budget_map[budget_label] if method_name == "Mref_EXACT_XS_S" else budget_sec
                for n_workers in [1, 2]:
                    schedule, summary = run_budgeted_method(
                        root=root,
                        instance_id=instance_row.instance_id,
                        method_name=method_name,
                        budget_sec=actual_budget,
                        n_workers=n_workers,
                    )
                    key = (instance_row.instance_id, method_name)
                    if budget_label == "medium" and n_workers == 1:
                        baseline_schedules[key] = schedule.copy()
                        stability = 0.0
                    else:
                        baseline = baseline_schedules.get(key)
                        if baseline is None:
                            stability = np.nan
                        else:
                            merged = pd.DataFrame(
                                {
                                    "base": _schedule_signature(baseline).to_numpy(),
                                    "curr": _schedule_signature(schedule).to_numpy(),
                                }
                            )
                            stability = float((merged["base"] != merged["curr"]).mean())
                    rows.append(
                        {
                            "instance_id": instance_row.instance_id,
                            "scale_code": instance_row.scale_code,
                            "regime_code": instance_row.regime_code,
                            "replicate": int(instance_row.replicate),
                            "family_id": instance_row.family_id,
                            "protocol_role": instance_row.protocol_role,
                            "protocol_fold": instance_row.protocol_fold,
                            "method_name": method_name,
                            "budget_label": budget_label,
                            "budget_sec": actual_budget,
                            "n_workers": n_workers,
                            "threads": n_workers,
                            "runtime_sec": float(summary["runtime_sec"]),
                            "replan_count": float(summary["replan_count"]),
                            "solver_status": summary["solver_status"],
                            "flow_mean": float(summary["flow_mean"]),
                            "flow_p95": float(summary["flow_p95"]),
                            "makespan": float(summary["makespan"]),
                            "weighted_tardiness": float(summary["weighted_tardiness"]),
                            "plan_instability": stability,
                        }
                    )
    sensitivity = pd.DataFrame(rows)
    utility_rows = []
    for instance_id, group in sensitivity.groupby("instance_id"):
        group = group.copy()
        for metric_name in ["flow_p95", "flow_mean", "makespan", "weighted_tardiness", "runtime_sec"]:
            values = group[metric_name].astype(float)
            min_v = float(values.min())
            max_v = float(values.max())
            if math.isclose(min_v, max_v):
                group[f"{metric_name}_norm"] = 0.0
            else:
                group[f"{metric_name}_norm"] = (values - min_v) / (max_v - min_v)
        group["utility"] = sum(group[f"{metric}_norm"] * weight for metric, weight in UTILITY_WEIGHTS.items())
        utility_rows.append(group)
    sensitivity = pd.concat(utility_rows, ignore_index=True)
    calibration_choice = (
        sensitivity.loc[sensitivity["protocol_role"].eq("calibration")]
        .groupby(["scale_code", "method_name", "budget_label", "n_workers"], as_index=False)
        .agg(
            median_utility=("utility", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .sort_values(["scale_code", "method_name", "median_utility", "median_runtime_sec", "budget_label", "n_workers"])
        .groupby(["scale_code", "method_name"], as_index=False)
        .first()
        .assign(selected_config=True)
    )
    sensitivity = sensitivity.merge(
        calibration_choice[["scale_code", "method_name", "budget_label", "n_workers", "selected_config"]],
        on=["scale_code", "method_name", "budget_label", "n_workers"],
        how="left",
    )
    sensitivity["selected_config"] = sensitivity["selected_config"].fillna(False)
    return sensitivity.sort_values([
        "protocol_role",
        "scale_code",
        "regime_code",
        "replicate",
        "instance_id",
        "method_name",
        "budget_label",
        "n_workers",
    ]).reset_index(drop=True)


def load_or_run_compute_sensitivity(root: Path, catalog: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    sensitivity_catalog = catalog.copy()
    expected_roles = {"calibration", "test"}
    expected_instances = sensitivity_catalog["instance_id"].nunique()
    needs_sensitivity_recompute = not out_path.exists()

    if not needs_sensitivity_recompute:
        sensitivity = pd.read_csv(out_path)
        if "replicate" not in sensitivity.columns:
            sensitivity = sensitivity.merge(
                sensitivity_catalog,
                on=["instance_id", "scale_code", "regime_code"],
                how="left",
            )
        sensitivity = _attach_protocol_columns(sensitivity)
        observed_roles = set(sensitivity["protocol_role"].dropna().unique())
        observed_instances = sensitivity["instance_id"].nunique()
        needs_sensitivity_recompute = observed_roles != expected_roles or observed_instances != expected_instances

    if needs_sensitivity_recompute:
        sensitivity = run_compute_sensitivity(root=root, catalog=sensitivity_catalog)
        sensitivity.to_csv(out_path, index=False)
        return sensitivity

    if "selected_config" not in sensitivity.columns:
        calibration_configs = (
            sensitivity.loc[sensitivity["protocol_role"].eq("calibration")]
            .groupby(["scale_code", "method_name", "budget_label", "n_workers"], as_index=False)
            .agg(utility=("utility", "median"), runtime_sec=("runtime_sec", "median"))
            .sort_values(["scale_code", "method_name", "utility", "runtime_sec"], ascending=[True, True, True, True])
            .groupby(["scale_code", "method_name"], as_index=False)
            .head(1)[["scale_code", "method_name", "budget_label", "n_workers"]]
            .assign(selected_config=True)
        )
        sensitivity = sensitivity.merge(
            calibration_configs,
            on=["scale_code", "method_name", "budget_label", "n_workers"],
            how="left",
        )
        sensitivity["selected_config"] = sensitivity["selected_config"].fillna(False)
        sensitivity.to_csv(out_path, index=False)
    return sensitivity
