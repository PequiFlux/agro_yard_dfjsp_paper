#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    "model_id": "pequiflux_observed_noise_v1_1",
    "global_seed": 20260327,
    "due_model": {
        "base_slack_by_priority_min": {"REGULAR": 300, "CONTRACTED": 270, "URGENT": 240},
        "fixed_effect_min": {
            "appointment_flag": {"0": 6, "1": -8},
            "commodity": {"SOY": 10, "CORN": 0, "SORGHUM": 5},
            "moisture_class": {"DRY": -5, "NORMAL": 0, "WET": 9},
            "shift_bucket": {"EARLY": 8, "MID": 0, "LATE": -12, "OVERNIGHT": -20},
            "regime_code": {"balanced": 0, "peak": 8, "disrupted": 18},
        },
        "instance_random_sd_min": 8.0,
        "shift_random_sd_min": 4.0,
        "idio_t_df": 5,
        "idio_scale_min_by_regime": {"balanced": 8.0, "peak": 10.0, "disrupted": 12.0},
        "min_buffer_over_nominal_lb_min": 18,
        "max_extra_over_base_min": 120,
    },
    "proc_model": {
        "stage_min_proc_min": {
            "WEIGH_IN": 4,
            "SAMPLE_CLASSIFY": 6,
            "UNLOAD": 10,
            "WEIGH_OUT": 4,
        },
        "sigma_machine": {
            "WEIGH_IN": 0.030,
            "SAMPLE_CLASSIFY": 0.045,
            "UNLOAD": 0.060,
            "WEIGH_OUT": 0.025,
        },
        "sigma_shift": {
            "WEIGH_IN": 0.012,
            "SAMPLE_CLASSIFY": 0.025,
            "UNLOAD": 0.030,
            "WEIGH_OUT": 0.010,
        },
        "sigma_instance": {
            "WEIGH_IN": 0.010,
            "SAMPLE_CLASSIFY": 0.020,
            "UNLOAD": 0.025,
            "WEIGH_OUT": 0.008,
        },
        "sigma_idio": {
            "WEIGH_IN": 0.020,
            "SAMPLE_CLASSIFY": 0.030,
            "UNLOAD": 0.040,
            "WEIGH_OUT": 0.018,
        },
        "regime_log_effect": {
            "balanced": {"WEIGH_IN": 0.000, "SAMPLE_CLASSIFY": 0.000, "UNLOAD": 0.000, "WEIGH_OUT": 0.000},
            "peak":     {"WEIGH_IN": 0.010, "SAMPLE_CLASSIFY": 0.018, "UNLOAD": 0.022, "WEIGH_OUT": 0.008},
            "disrupted": {"WEIGH_IN": 0.020, "SAMPLE_CLASSIFY": 0.030, "UNLOAD": 0.040, "WEIGH_OUT": 0.015},
        },
        "beta_congestion": {
            "WEIGH_IN": 0.020,
            "SAMPLE_CLASSIFY": 0.035,
            "UNLOAD": 0.050,
            "WEIGH_OUT": 0.015,
        },
        "commodity_log_effect": {
            "WEIGH_IN": {"SOY": 0.000, "CORN": 0.000, "SORGHUM": 0.000},
            "SAMPLE_CLASSIFY": {"SOY": 0.020, "CORN": 0.000, "SORGHUM": 0.010},
            "UNLOAD": {"SOY": 0.010, "CORN": 0.000, "SORGHUM": 0.015},
            "WEIGH_OUT": {"SOY": 0.000, "CORN": 0.000, "SORGHUM": 0.000},
        },
        "moisture_log_effect": {
            "WEIGH_IN": {"DRY": 0.000, "NORMAL": 0.000, "WET": 0.000},
            "SAMPLE_CLASSIFY": {"DRY": -0.020, "NORMAL": 0.000, "WET": 0.045},
            "UNLOAD": {"DRY": -0.010, "NORMAL": 0.000, "WET": 0.028},
            "WEIGH_OUT": {"DRY": 0.000, "NORMAL": 0.000, "WET": 0.000},
        },
        "pause_prob": {
            "balanced": {"WEIGH_IN": 0.03, "SAMPLE_CLASSIFY": 0.05, "UNLOAD": 0.08, "WEIGH_OUT": 0.02},
            "peak": {"WEIGH_IN": 0.05, "SAMPLE_CLASSIFY": 0.08, "UNLOAD": 0.13, "WEIGH_OUT": 0.04},
            "disrupted": {"WEIGH_IN": 0.08, "SAMPLE_CLASSIFY": 0.12, "UNLOAD": 0.18, "WEIGH_OUT": 0.06},
        },
        "pause_max_min": {"WEIGH_IN": 2, "SAMPLE_CLASSIFY": 3, "UNLOAD": 5, "WEIGH_OUT": 2},
        "log_mult_clip": (-0.28, 0.42),
    },
}


def stable_seed(parts: Iterable[str | int]) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return int(digest, 16) % (2**32 - 1)


def shift_bucket(arrival_min: int) -> str:
    if arrival_min < 240:
        return "EARLY"
    if arrival_min < 480:
        return "MID"
    if arrival_min < 720:
        return "LATE"
    return "OVERNIGHT"


def triangular_congestion(arrivals: np.ndarray, bandwidth: float = 60.0) -> np.ndarray:
    n = len(arrivals)
    scores = np.zeros(n, dtype=float)
    for i, a in enumerate(arrivals):
        dist = np.abs(arrivals - a)
        contrib = np.maximum(0.0, 1.0 - dist / bandwidth)
        contrib[i] = 0.0
        scores[i] = contrib.sum()
    if np.all(scores == 0):
        return scores
    denom = np.percentile(scores, 95)
    if denom <= 0:
        denom = scores.max()
    if denom <= 0:
        return np.zeros_like(scores)
    return np.clip(scores / denom, 0.0, 1.5)


def sample_student_t(rng: np.random.Generator, df: float, scale: float) -> float:
    return float(rng.standard_t(df) * scale)


def earliest_nonoverlap_start(candidate_start: int, duration: int, busy: List[Tuple[int, int]], downtimes: List[Tuple[int, int]]) -> int:
    intervals = sorted(list(busy) + list(downtimes), key=lambda x: (x[0], x[1]))
    t = candidate_start
    changed = True
    while changed:
        changed = False
        for s, e in intervals:
            if t + duration <= s or t >= e:
                continue
            t = e
            changed = True
            break
    return t


def fifo_schedule(instance_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    jobs = pd.read_csv(instance_dir / "jobs.csv")
    ops = pd.read_csv(instance_dir / "operations.csv")
    prec = pd.read_csv(instance_dir / "precedences.csv")
    elig = pd.read_csv(instance_dir / "eligible_machines.csv")
    machines = pd.read_csv(instance_dir / "machines.csv")
    downtimes = pd.read_csv(instance_dir / "machine_downtimes.csv")

    job_order = jobs.sort_values(["arrival_time_min", "job_id"])["job_id"].tolist()
    ops_by_job = {j: g.sort_values("op_seq").copy() for j, g in ops.groupby("job_id")}
    elig_by_op = {(r.job_id, int(r.op_seq)): [] for r in elig.itertuples(index=False)}
    for r in elig.itertuples(index=False):
        elig_by_op[(r.job_id, int(r.op_seq))].append((r.machine_id, int(r.proc_time_min)))
    for key in elig_by_op:
        elig_by_op[key] = sorted(elig_by_op[key], key=lambda x: x[0])

    machine_busy: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    machine_down: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for r in downtimes.itertuples(index=False):
        machine_down[r.machine_id].append((int(r.start_min), int(r.end_min)))
    for m in machine_down:
        machine_down[m].sort()

    end_times: Dict[Tuple[str, int], int] = {}
    rows = []

    min_lag = {(r.job_id, int(r.pred_op_seq), int(r.succ_op_seq)): int(r.min_lag_min) for r in prec.itertuples(index=False)}
    release = {(r.job_id, int(r.op_seq)): int(r.release_time_min) for r in ops.itertuples(index=False)}
    stage = {(r.job_id, int(r.op_seq)): r.stage_name for r in ops.itertuples(index=False)}

    for j in job_order:
        prev_end = None
        for r in ops_by_job[j].itertuples(index=False):
            o = int(r.op_seq)
            est = release[(j, o)]
            if prev_end is not None:
                est = max(est, prev_end + min_lag.get((j, o - 1, o), 0))
            best = None
            for m_id, p in elig_by_op[(j, o)]:
                cand = earliest_nonoverlap_start(est, p, machine_busy[m_id], machine_down.get(m_id, []))
                end = cand + p
                key = (end, cand, m_id)
                if best is None or key < best[:3]:
                    best = (end, cand, m_id, p)
            if best is None:
                raise RuntimeError(f"No eligible machine for {(j,o)}")
            end, start, m_id, p = best
            machine_busy[m_id].append((start, end))
            machine_busy[m_id].sort()
            rows.append({
                "job_id": j,
                "op_seq": o,
                "stage_name": stage[(j, o)],
                "machine_id": m_id,
                "start_min": int(start),
                "end_min": int(end),
                "wait_before_stage_min": int(start - est),
            })
            end_times[(j, o)] = int(end)
            prev_end = int(end)

    sched = pd.DataFrame(rows).sort_values(["job_id", "op_seq"]).reset_index(drop=True)
    metrics_rows = []
    jobs_idx = jobs.set_index("job_id")
    for j, g in sched.groupby("job_id"):
        completion = int(g["end_min"].max())
        arrival = int(jobs_idx.loc[j, "arrival_time_min"])
        flow = completion - arrival
        queue_time = int(g["wait_before_stage_min"].sum())
        overwait = max(0, flow - int(jobs_idx.loc[j, "statutory_wait_limit_min"]))
        metrics_rows.append({
            "job_id": j,
            "completion_min": completion,
            "flow_time_min": int(flow),
            "queue_time_min": int(queue_time),
            "overwait_min": int(overwait),
        })
    metrics = pd.DataFrame(metrics_rows).sort_values("job_id").reset_index(drop=True)
    makespan = int(sched["end_min"].max())
    summary = {
        "makespan_min": makespan,
        "mean_flow_min": round(float(metrics["flow_time_min"].mean()), 2),
        "p95_flow_min": int(np.percentile(metrics["flow_time_min"], 95, method="linear")),
        "overwait_share": round(float((metrics["overwait_min"] > 0).mean()), 4),
    }
    return sched, metrics, summary


def update_catalog(out_root: Path) -> None:
    instances_dir = out_root / "instances"
    rows = []
    for inst_dir in sorted(p for p in instances_dir.iterdir() if p.is_dir()):
        params = json.load(open(inst_dir / "params.json", "r", encoding="utf-8"))
        jobs = pd.read_csv(inst_dir / "jobs.csv")
        machines = pd.read_csv(inst_dir / "machines.csv")
        elig = pd.read_csv(inst_dir / "eligible_machines.csv")
        ops = pd.read_csv(inst_dir / "operations.csv")
        downs = pd.read_csv(inst_dir / "machine_downtimes.csv")
        fifo_summary = json.load(open(inst_dir / "fifo_summary.json", "r", encoding="utf-8"))
        unload = elig.merge(ops[["job_id", "op_seq", "stage_name"]], on=["job_id", "op_seq"])
        unload = unload[unload["stage_name"] == "UNLOAD"]
        rows.append({
            "instance_id": params["instance_id"],
            "relative_path": f"instances/{params['instance_id']}",
            "scale_code": params["scale_code"],
            "regime_code": params["regime_code"],
            "replicate": params["replicate"],
            "random_seed": params["random_seed"],
            "n_jobs": len(jobs),
            "n_machines": len(machines),
            "n_scales": int((machines["machine_family"] == "WEIGHBRIDGE").sum()),
            "n_labs": int((machines["machine_family"] == "LAB").sum()),
            "n_hoppers": int((machines["machine_family"] == "HOPPER").sum()),
            "n_breakdowns": len(downs),
            "share_urgent": round(float((jobs["priority_class"] == "URGENT").mean()), 4),
            "share_appointment": round(float(jobs["appointment_flag"].mean()), 4),
            "avg_load_tons": round(float(jobs["load_tons"].mean()), 2),
            "avg_unload_proc_time_min": round(float(unload["proc_time_min"].mean()), 2),
            "fifo_makespan_min": int(fifo_summary["makespan_min"]),
            "fifo_mean_flow_min": float(fifo_summary["mean_flow_min"]),
            "fifo_p95_flow_min": int(fifo_summary["p95_flow_min"]),
            "fifo_overwait_share": float(fifo_summary["overwait_share"]),
            "recommended_solver_track": "exact" if params["scale_code"] in {"XS", "S"} else ("hybrid" if params["scale_code"] == "M" else "metaheuristic"),
        })
    cat = pd.DataFrame(rows)
    cat.to_csv(out_root / "catalog" / "benchmark_catalog.csv", index=False)
    fam = cat.groupby(["scale_code", "regime_code"], as_index=False).agg(
        instance_count=("instance_id", "count"),
        avg_n_jobs=("n_jobs", "mean"),
        avg_fifo_makespan_min=("fifo_makespan_min", "mean"),
        avg_fifo_mean_flow_min=("fifo_mean_flow_min", "mean"),
        avg_fifo_p95_flow_min=("fifo_p95_flow_min", "mean"),
        avg_fifo_overwait_share=("fifo_overwait_share", "mean"),
    )
    for c in ["avg_n_jobs", "avg_fifo_makespan_min", "avg_fifo_mean_flow_min", "avg_fifo_p95_flow_min", "avg_fifo_overwait_share"]:
        fam[c] = fam[c].round(2)
    fam.to_csv(out_root / "catalog" / "instance_family_summary.csv", index=False)


def regression_r2(y: np.ndarray, X: np.ndarray) -> float:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def diagnostics_for_root(root: Path) -> Dict[str, float]:
    jobs_all = []
    unload_all = []
    for inst_dir in sorted((root / "instances").iterdir()):
        if not inst_dir.is_dir():
            continue
        jobs = pd.read_csv(inst_dir / "jobs.csv")
        jobs["slack_min"] = jobs["completion_due_min"] - jobs["arrival_time_min"]
        jobs_all.append(jobs)
        elig = pd.read_csv(inst_dir / "eligible_machines.csv")
        ops = pd.read_csv(inst_dir / "operations.csv")[["job_id", "op_seq", "stage_name"]]
        jobsj = jobs[["job_id", "load_tons", "moisture_class"]]
        unload = elig.merge(ops, on=["job_id", "op_seq"]).merge(jobsj, on="job_id")
        unload = unload[unload["stage_name"] == "UNLOAD"]
        unload_all.append(unload)
    jobs = pd.concat(jobs_all, ignore_index=True)
    unload = pd.concat(unload_all, ignore_index=True)

    X_due = pd.get_dummies(jobs[["priority_class"]], drop_first=False).astype(float)
    X_due = np.c_[np.ones(len(X_due)), X_due.values]
    r2_due = regression_r2(jobs["slack_min"].astype(float).values, X_due)

    X_un = pd.concat([
        unload[["load_tons"]].reset_index(drop=True),
        pd.get_dummies(unload["machine_id"], prefix="m", drop_first=False).reset_index(drop=True),
        pd.get_dummies(unload["moisture_class"], prefix="moist", drop_first=False).reset_index(drop=True),
    ], axis=1).astype(float)
    X_un = np.c_[np.ones(len(X_un)), X_un.values]
    r2_un = regression_r2(unload["proc_time_min"].astype(float).values, X_un)

    return {
        "r2_due_slack_vs_priority": round(r2_due, 4),
        "r2_unload_proc_vs_load_machine_moisture": round(r2_un, 4),
    }


def validate_instance(instance_dir: Path) -> Dict[str, object]:
    jobs = pd.read_csv(instance_dir / "jobs.csv")
    ops = pd.read_csv(instance_dir / "operations.csv")
    prec = pd.read_csv(instance_dir / "precedences.csv")
    elig = pd.read_csv(instance_dir / "eligible_machines.csv")
    due_audit = pd.read_csv(instance_dir / "job_noise_audit.csv")
    proc_audit = pd.read_csv(instance_dir / "proc_noise_audit.csv")
    machines = pd.read_csv(instance_dir / "machines.csv")
    downs = pd.read_csv(instance_dir / "machine_downtimes.csv")
    events = pd.read_csv(instance_dir / "events.csv")
    sched = pd.read_csv(instance_dir / "fifo_schedule.csv")
    metrics = pd.read_csv(instance_dir / "fifo_job_metrics.csv")

    issues = []
    # exactly 4 operations and 3 precedences per job
    op_counts = ops.groupby("job_id")["op_seq"].count()
    if not (op_counts == 4).all():
        issues.append("not_all_jobs_have_4_operations")
    pred_counts = prec.groupby("job_id")["succ_op_seq"].count()
    if not (pred_counts == 3).all():
        issues.append("not_all_jobs_have_3_precedences")
    # all ops eligible
    op_keys = set(map(tuple, ops[["job_id", "op_seq"]].itertuples(index=False, name=None)))
    elig_keys = set(map(tuple, elig[["job_id", "op_seq"]].drop_duplicates().itertuples(index=False, name=None)))
    if not op_keys.issubset(elig_keys):
        issues.append("ops_without_eligible_machine")
    # due lower bound
    due_audit_idx = due_audit.drop_duplicates(subset=["job_id"]).set_index("job_id")
    merged = jobs.set_index("job_id").join(
        due_audit_idx[["completion_due_observed_min", "nominal_processing_lb_min"]]
    )
    if merged[["completion_due_observed_min", "nominal_processing_lb_min"]].isna().any().any():
        issues.append("job_due_audit_missing_fields")
    if not merged["completion_due_min"].eq(merged["completion_due_observed_min"]).all():
        issues.append("job_due_audit_mismatch")
    if not (
        (merged["completion_due_min"] - merged["arrival_time_min"])
        >= (merged["nominal_processing_lb_min"] + 18)
    ).all():
        issues.append("due_below_nominal_lb_plus_buffer")
    proc_obs = proc_audit.rename(columns={"proc_time_observed_min": "proc_time_audit_min"})
    proc_merge = elig.merge(
        proc_obs[["job_id", "op_seq", "machine_id", "proc_time_audit_min"]],
        on=["job_id", "op_seq", "machine_id"],
        how="left",
    )
    if proc_merge["proc_time_audit_min"].isna().any():
        issues.append("proc_audit_missing_rows")
    if not proc_merge["proc_time_min"].eq(proc_merge["proc_time_audit_min"]).all():
        issues.append("proc_audit_mismatch")
    # events consistency
    vis = events[events["event_type"] == "JOB_VISIBLE"].groupby("entity_id")["event_time_min"].count()
    arr = events[events["event_type"] == "JOB_ARRIVAL"].groupby("entity_id")["event_time_min"].count()
    if not all(vis.reindex(jobs["job_id"]).fillna(0).eq(1)):
        issues.append("job_visible_event_count_mismatch")
    if not all(arr.reindex(jobs["job_id"]).fillna(0).eq(1)):
        issues.append("job_arrival_event_count_mismatch")
    # schedule non-overlap by machine
    for m_id, g in sched.sort_values(["machine_id", "start_min", "end_min"]).groupby("machine_id"):
        prev_end = -10**9
        for row in g.itertuples(index=False):
            if int(row.start_min) < prev_end:
                issues.append(f"machine_overlap_{m_id}")
                break
            prev_end = int(row.end_min)
    # schedule consistent with proc time
    proc_map = {(r.job_id, int(r.op_seq), r.machine_id): int(r.proc_time_min) for r in elig.itertuples(index=False)}
    for r in sched.itertuples(index=False):
        if int(r.end_min) - int(r.start_min) != proc_map[(r.job_id, int(r.op_seq), r.machine_id)]:
            issues.append("sched_proc_mismatch")
            break
    # metrics consistency
    jobs_idx = jobs.set_index("job_id")
    for r in metrics.itertuples(index=False):
        comp = int(sched[sched["job_id"] == r.job_id]["end_min"].max())
        if comp != int(r.completion_min):
            issues.append("metric_completion_mismatch")
            break
        flow = comp - int(jobs_idx.loc[r.job_id, "arrival_time_min"])
        if flow != int(r.flow_time_min):
            issues.append("metric_flow_mismatch")
            break
    return {
        "instance_id": instance_dir.name,
        "issue_count": len(issues),
        "issues": ";".join(issues),
        "status": "PASS" if not issues else "FAIL",
    }


def main(src_root: Path, out_root: Path) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(src_root, out_root)

    # update root-level readme note if exists
    readme = out_root / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        prepend = (
            "# Observed-noise derivative release\n\n"
            "This derivative benchmark adds a structured observational noise layer on top of the nominal v1.0.0 release. "
            "Core schema is preserved for Gurobi compatibility; audit files trace nominal values, latent effects, and observed values.\n\n"
        )
        readme.write_text(prepend + content, encoding="utf-8")

    root_manifest = {
        "parent_dataset": str(src_root),
        "derived_dataset": str(out_root),
        "model_id": CONFIG["model_id"],
        "global_seed": CONFIG["global_seed"],
        "description": "Structured observational noise layer over nominal Agro Yard D-FJSP benchmark.",
        "config": CONFIG,
    }
    (out_root / "catalog").mkdir(parents=True, exist_ok=True)
    with open(out_root / "catalog" / "observed_noise_manifest.json", "w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2, ensure_ascii=False)

    for inst_dir in sorted((out_root / "instances").iterdir()):
        if not inst_dir.is_dir():
            continue
        params = json.load(open(inst_dir / "params.json", "r", encoding="utf-8"))
        regime = params["regime_code"]
        base_seed = stable_seed([CONFIG["global_seed"], params["instance_id"], params["random_seed"]])
        rng = np.random.default_rng(base_seed)

        jobs = pd.read_csv(inst_dir / "jobs.csv")
        ops = pd.read_csv(inst_dir / "operations.csv")
        elig = pd.read_csv(inst_dir / "eligible_machines.csv")
        machines = pd.read_csv(inst_dir / "machines.csv")
        events = pd.read_csv(inst_dir / "events.csv")

        jobs["shift_bucket"] = jobs["arrival_time_min"].map(shift_bucket)
        jobs = jobs.sort_values(["arrival_time_min", "job_id"]).reset_index(drop=True)
        jobs["arrival_congestion_score"] = triangular_congestion(jobs["arrival_time_min"].to_numpy())

        # instance-level latent effects
        due_inst = rng.normal(0.0, CONFIG["due_model"]["instance_random_sd_min"])
        due_shift_latent = {b: rng.normal(0.0, CONFIG["due_model"]["shift_random_sd_min"]) for b in ["EARLY", "MID", "LATE", "OVERNIGHT"]}
        proc_stage_inst = {stage: rng.normal(0.0, CONFIG["proc_model"]["sigma_instance"][stage]) for stage in CONFIG["proc_model"]["sigma_instance"]}
        proc_machine_latent = {}
        for r in machines.itertuples(index=False):
            stages = r.allowed_stages.split(";")
            primary_stage = "UNLOAD" if r.machine_family == "HOPPER" else ("SAMPLE_CLASSIFY" if r.machine_family == "LAB" else "WEIGH_IN")
            proc_machine_latent[r.machine_id] = rng.normal(0.0, CONFIG["proc_model"]["sigma_machine"][primary_stage])
        proc_shift_latent = {(stage, b): rng.normal(0.0, CONFIG["proc_model"]["sigma_shift"][stage]) for stage in CONFIG["proc_model"]["sigma_shift"] for b in ["EARLY", "MID", "LATE", "OVERNIGHT"]}

        # due-date observational layer
        min_proc_lb = elig.groupby(["job_id", "op_seq"])["proc_time_min"].min().reset_index().groupby("job_id")["proc_time_min"].sum()
        due_audit_rows = []
        new_due = {}
        for r in jobs.itertuples(index=False):
            base = CONFIG["due_model"]["base_slack_by_priority_min"][r.priority_class]
            fe = CONFIG["due_model"]["fixed_effect_min"]
            fixed = (
                fe["appointment_flag"][str(int(r.appointment_flag))]
                + fe["commodity"][r.commodity]
                + fe["moisture_class"][r.moisture_class]
                + fe["shift_bucket"][r.shift_bucket]
                + fe["regime_code"][regime]
            )
            shift_lat = due_shift_latent[r.shift_bucket]
            idio = sample_student_t(rng, CONFIG["due_model"]["idio_t_df"], CONFIG["due_model"]["idio_scale_min_by_regime"][regime])
            slack_nom = int(r.completion_due_min - r.arrival_time_min)
            slack_obs = int(round(base + fixed + due_inst + shift_lat + idio))
            lower = int(min_proc_lb.loc[r.job_id] + CONFIG["due_model"]["min_buffer_over_nominal_lb_min"])
            upper = int(base + CONFIG["due_model"]["max_extra_over_base_min"])
            slack_obs = max(lower, min(upper, slack_obs))
            due_obs = int(r.arrival_time_min + slack_obs)
            new_due[r.job_id] = due_obs
            due_audit_rows.append({
                "job_id": r.job_id,
                "arrival_time_min": int(r.arrival_time_min),
                "completion_due_nominal_min": int(r.completion_due_min),
                "completion_due_observed_min": due_obs,
                "due_slack_nominal_min": slack_nom,
                "due_slack_observed_min": slack_obs,
                "base_priority_slack_min": base,
                "fixed_appointment_min": fe["appointment_flag"][str(int(r.appointment_flag))],
                "fixed_commodity_min": fe["commodity"][r.commodity],
                "fixed_moisture_min": fe["moisture_class"][r.moisture_class],
                "fixed_shift_min": fe["shift_bucket"][r.shift_bucket],
                "fixed_regime_min": fe["regime_code"][regime],
                "random_instance_min": round(float(due_inst), 4),
                "random_shift_min": round(float(shift_lat), 4),
                "random_idio_min": round(float(idio), 4),
                "nominal_processing_lb_min": int(min_proc_lb.loc[r.job_id]),
                "shift_bucket": r.shift_bucket,
            })
        jobs["completion_due_min"] = jobs["job_id"].map(new_due).astype(int)
        jobs.to_csv(inst_dir / "jobs.csv", index=False)
        pd.DataFrame(due_audit_rows).to_csv(inst_dir / "job_noise_audit.csv", index=False)

        # proc-time observational layer
        job_attrs = jobs[["job_id", "commodity", "moisture_class", "shift_bucket", "arrival_congestion_score"]].copy()
        ops_small = ops[["job_id", "op_seq", "stage_name"]].copy()
        merged = elig.merge(ops_small, on=["job_id", "op_seq"], how="left").merge(job_attrs, on="job_id", how="left")
        proc_audit = []
        observed_proc = []
        for r in merged.itertuples(index=False):
            stage = r.stage_name
            nominal = int(r.proc_time_min)
            m_eff = proc_machine_latent[r.machine_id]
            s_eff = proc_shift_latent[(stage, r.shift_bucket)]
            i_eff = proc_stage_inst[stage]
            reg_eff = CONFIG["proc_model"]["regime_log_effect"][regime][stage]
            cong_eff = CONFIG["proc_model"]["beta_congestion"][stage] * float(r.arrival_congestion_score)
            com_eff = CONFIG["proc_model"]["commodity_log_effect"][stage][r.commodity]
            moist_eff = CONFIG["proc_model"]["moisture_log_effect"][stage][r.moisture_class]
            idio = sample_student_t(rng, 6, CONFIG["proc_model"]["sigma_idio"][stage])
            pause = int(rng.integers(1, CONFIG["proc_model"]["pause_max_min"][stage] + 1)) if rng.random() < CONFIG["proc_model"]["pause_prob"][regime][stage] else 0
            log_mult = m_eff + s_eff + i_eff + reg_eff + cong_eff + com_eff + moist_eff + idio
            log_mult = float(np.clip(log_mult, CONFIG["proc_model"]["log_mult_clip"][0], CONFIG["proc_model"]["log_mult_clip"][1]))
            observed = int(max(CONFIG["proc_model"]["stage_min_proc_min"][stage], round(nominal * math.exp(log_mult) + pause)))
            observed_proc.append(observed)
            proc_audit.append({
                "job_id": r.job_id,
                "op_seq": int(r.op_seq),
                "machine_id": r.machine_id,
                "stage_name": stage,
                "proc_time_nominal_min": nominal,
                "proc_time_observed_min": observed,
                "machine_latent_log": round(float(m_eff), 5),
                "shift_latent_log": round(float(s_eff), 5),
                "instance_latent_log": round(float(i_eff), 5),
                "regime_log": round(float(reg_eff), 5),
                "congestion_log": round(float(cong_eff), 5),
                "commodity_log": round(float(com_eff), 5),
                "moisture_log": round(float(moist_eff), 5),
                "idio_log": round(float(idio), 5),
                "additive_pause_min": int(pause),
                "arrival_congestion_score": round(float(r.arrival_congestion_score), 5),
                "shift_bucket": r.shift_bucket,
                "commodity": r.commodity,
                "moisture_class": r.moisture_class,
            })
        merged["proc_time_min"] = observed_proc
        merged[["job_id", "op_seq", "machine_id", "proc_time_min", "setup_included"]].to_csv(inst_dir / "eligible_machines.csv", index=False)
        pd.DataFrame(proc_audit).to_csv(inst_dir / "proc_noise_audit.csv", index=False)
        jobs[["job_id", "arrival_time_min", "arrival_congestion_score", "shift_bucket"]].to_csv(inst_dir / "job_congestion_proxy.csv", index=False)

        # update params
        old_version = params.get("dataset_version", "1.0.0")
        params["dataset_version"] = "1.1.0-observed"
        params["parent_dataset_version"] = old_version
        params["observational_noise_model_id"] = CONFIG["model_id"]
        params["observational_noise_seed"] = int(base_seed)
        params["notes"] = (
            "Structured observational-noise derivative of the synthetic calibrated benchmark. "
            "Nominal values remain traceable via audit CSVs."
        )
        with open(inst_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        # regenerate FIFO baseline and summaries
        sched, metrics, summary = fifo_schedule(inst_dir)
        sched.to_csv(inst_dir / "fifo_schedule.csv", index=False)
        metrics.to_csv(inst_dir / "fifo_job_metrics.csv", index=False)
        with open(inst_dir / "fifo_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # instance-level noise manifest
        inst_manifest = {
            "instance_id": params["instance_id"],
            "observational_noise_model_id": CONFIG["model_id"],
            "observational_noise_seed": int(base_seed),
            "due_instance_random_min": round(float(due_inst), 5),
            "due_shift_random_min": {k: round(float(v), 5) for k, v in due_shift_latent.items()},
            "proc_stage_instance_log": {k: round(float(v), 5) for k, v in proc_stage_inst.items()},
            "proc_machine_latent_log": {k: round(float(v), 5) for k, v in proc_machine_latent.items()},
        }
        with open(inst_dir / "noise_manifest.json", "w", encoding="utf-8") as f:
            json.dump(inst_manifest, f, indent=2, ensure_ascii=False)

        # refresh events order if jobs were reordered in memory: values unchanged except due not reflected in events
        events.to_csv(inst_dir / "events.csv", index=False)

    update_catalog(out_root)

    # validation and diagnostics
    val_rows = [validate_instance(p) for p in sorted((out_root / "instances").iterdir()) if p.is_dir()]
    pd.DataFrame(val_rows).to_csv(out_root / "catalog" / "validation_report_observed.csv", index=False)
    diag = diagnostics_for_root(out_root)
    with open(out_root / "catalog" / "noise_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a structured observational noise layer to the Agro Yard D-FJSP benchmark.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    main(args.src_root, args.out_root)
