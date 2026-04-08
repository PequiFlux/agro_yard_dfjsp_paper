#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def validate_instance(instance_dir: Path):
    jobs = pd.read_csv(instance_dir / "jobs.csv")
    ops = pd.read_csv(instance_dir / "operations.csv")
    prec = pd.read_csv(instance_dir / "precedences.csv")
    elig = pd.read_csv(instance_dir / "eligible_machines.csv")
    due_audit = pd.read_csv(instance_dir / "job_noise_audit.csv")
    proc_audit = pd.read_csv(instance_dir / "proc_noise_audit.csv")
    events = pd.read_csv(instance_dir / "events.csv")
    sched = pd.read_csv(instance_dir / "fifo_schedule.csv")
    metrics = pd.read_csv(instance_dir / "fifo_job_metrics.csv")
    issues = []
    op_counts = ops.groupby("job_id")["op_seq"].count()
    if not (op_counts == 4).all():
        issues.append("not_all_jobs_have_4_operations")
    pred_counts = prec.groupby("job_id")["succ_op_seq"].count()
    if not (pred_counts == 3).all():
        issues.append("not_all_jobs_have_3_precedences")
    op_keys = set(map(tuple, ops[["job_id", "op_seq"]].itertuples(index=False, name=None)))
    elig_keys = set(map(tuple, elig[["job_id", "op_seq"]].drop_duplicates().itertuples(index=False, name=None)))
    if not op_keys.issubset(elig_keys):
        issues.append("ops_without_eligible_machine")
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
    vis = events[events["event_type"] == "JOB_VISIBLE"].groupby("entity_id")["event_time_min"].count()
    arr = events[events["event_type"] == "JOB_ARRIVAL"].groupby("entity_id")["event_time_min"].count()
    if not all(vis.reindex(jobs["job_id"]).fillna(0).eq(1)):
        issues.append("job_visible_event_count_mismatch")
    if not all(arr.reindex(jobs["job_id"]).fillna(0).eq(1)):
        issues.append("job_arrival_event_count_mismatch")
    for m_id, g in sched.sort_values(["machine_id", "start_min", "end_min"]).groupby("machine_id"):
        prev_end = -10**9
        for row in g.itertuples(index=False):
            if int(row.start_min) < prev_end:
                issues.append(f"machine_overlap_{m_id}")
                break
            prev_end = int(row.end_min)
    proc_map = {(r.job_id, int(r.op_seq), r.machine_id): int(r.proc_time_min) for r in elig.itertuples(index=False)}
    for r in sched.itertuples(index=False):
        if int(r.end_min) - int(r.start_min) != proc_map[(r.job_id, int(r.op_seq), r.machine_id)]:
            issues.append("sched_proc_mismatch")
            break
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
    return {"instance_id": instance_dir.name, "issue_count": len(issues), "issues": ";".join(issues), "status": "PASS" if not issues else "FAIL"}


def regression_r2(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def diagnostics(root: Path):
    jobs_all, unload_all = [], []
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
    X_un = pd.concat([
        unload[["load_tons"]].reset_index(drop=True),
        pd.get_dummies(unload["machine_id"], prefix="m", drop_first=False).reset_index(drop=True),
        pd.get_dummies(unload["moisture_class"], prefix="moist", drop_first=False).reset_index(drop=True),
    ], axis=1).astype(float)
    X_un = np.c_[np.ones(len(X_un)), X_un.values]
    return {
        "r2_due_slack_vs_priority": round(regression_r2(jobs["slack_min"].astype(float).values, X_due), 4),
        "r2_unload_proc_vs_load_machine_moisture": round(regression_r2(unload["proc_time_min"].astype(float).values, X_un), 4),
    }


def main(root: Path):
    rows = [validate_instance(p) for p in sorted((root / "instances").iterdir()) if p.is_dir()]
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print("\nDiagnostics:")
    print(json.dumps(diagnostics(root), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    main(args.root)
