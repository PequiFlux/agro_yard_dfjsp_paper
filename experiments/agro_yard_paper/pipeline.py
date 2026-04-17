
"""Paper benchmark pipeline, selector logic and cached orchestration wrappers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    Any,
    Bounds,
    ConvexHull,
    DecisionTreeClassifier,
    LGBMClassifier,
    LinearConstraint,
    METHOD_LABELS,
    METHOD_ORDER,
    METHOD_SPECS,
    PAPER_METHOD_ORDER,
    PaperPipelineResults,
    Parallel,
    RandomForestClassifier,
    SEED,
    UMAP,
    UTILITY_WEIGHTS,
    _attach_protocol_columns,
    _ensure_dir,
    _entropy,
    _load_instance_tables,
    accuracy_score,
    balanced_accuracy_score,
    build_gurobi_views,
    coo_matrix,
    delayed,
    f1_score,
    hdbscan,
    json,
    load_instance,
    math,
    milp,
    np,
    online_replay,
    pd,
    plt,
    shap,
    time,
)
from .plots import (
    plot_hdbscan_clusters,
    plot_method_delta,
    plot_method_runtime,
    plot_selector_shap,
    plot_solver_footprints,
    plot_umap_best_method,
)


def build_paper_feature_frame(ctx: dict[str, Any]) -> pd.DataFrame:
    base = ctx["instance_space_features"].copy()
    jobs = ctx["jobs_enriched"].copy()
    events = ctx["events"].copy()
    eligible = ctx["eligible"].copy()
    downtimes = ctx["downtimes"].copy()

    due_stats = jobs.groupby("instance_id", as_index=False).agg(
        due_slack_mean=("due_slack_min", "mean"),
        due_slack_std=("due_slack_min", "std"),
        due_slack_p10=("due_slack_min", lambda s: float(np.quantile(s, 0.10))),
        due_slack_p90=("due_slack_min", lambda s: float(np.quantile(s, 0.90))),
        urgent_share=("priority_class", lambda s: float(s.eq("URGENT").mean())),
        appointment_share=("appointment_flag", "mean"),
        commodity_entropy=("commodity", _entropy),
        moisture_entropy=("moisture_class", _entropy),
        wet_share=("moisture_class", lambda s: float(s.eq("WET").mean())),
    )
    event_stats = (
        events.sort_values(["instance_id", "event_time_min"])
        .groupby("instance_id", as_index=False)
        .agg(
            event_count=("event_type", "count"),
            first_event_min=("event_time_min", "min"),
            last_event_min=("event_time_min", "max"),
            visible_jobs_at_t0=("event_time_min", lambda s: int((s == 0).sum())),
        )
    )
    inter_event = (
        events.sort_values(["instance_id", "event_time_min"])
        .groupby("instance_id")["event_time_min"]
        .apply(lambda s: pd.Series(np.diff(s.to_numpy(dtype=float))).describe())
        .unstack()
        .reset_index()
        .rename(columns={"mean": "inter_event_mean", "std": "inter_event_std"})
    )
    compatibility = (
        eligible.groupby("instance_id", as_index=False)
        .agg(
            compatibility_density=("machine_id", "count"),
            proc_time_cv=("proc_time_min", lambda s: float(s.std(ddof=0) / max(s.mean(), 1e-6))),
        )
    )
    compat_denominator = (base["n_jobs"] * 4 * base["machine_count"]).replace(0, 1.0)
    downtime_by_machine = (
        downtimes.assign(duration_min=downtimes["end_min"] - downtimes["start_min"])
        .groupby(["instance_id", "machine_id"], as_index=False)["duration_min"]
        .sum()
        .groupby("instance_id", as_index=False)
        .agg(
            downtime_total_min=("duration_min", "sum"),
            downtime_max_machine_share=("duration_min", lambda s: float(s.max() / max(s.sum(), 1e-6))),
        )
    )

    frame = (
        base.merge(due_stats, on="instance_id", how="left")
        .merge(event_stats, on="instance_id", how="left")
        .merge(inter_event[["instance_id", "inter_event_mean", "inter_event_std"]], on="instance_id", how="left")
        .merge(compatibility, on="instance_id", how="left")
        .merge(downtime_by_machine, on="instance_id", how="left")
    )
    frame["compatibility_density"] = frame["compatibility_density"] / compat_denominator
    frame["visible_jobs_fraction_t0"] = frame["visible_jobs_at_t0"] / frame["n_jobs"].replace(0, 1.0)
    frame["reveal_span_min"] = frame["last_event_min"] - frame["first_event_min"]
    frame["bottleneck_utilization"] = frame[
        ["wb_utilization_mean", "lab_utilization_mean", "hop_utilization_mean"]
    ].max(axis=1)
    frame["bottleneck_machine_family"] = (
        frame[["wb_utilization_mean", "lab_utilization_mean", "hop_utilization_mean"]]
        .idxmax(axis=1)
        .map(
            {
                "wb_utilization_mean": "WB",
                "lab_utilization_mean": "LAB",
                "hop_utilization_mean": "HOP",
            }
        )
    )
    frame["size_hint"] = frame["scale_code"].map({"XS": 0, "S": 1, "M": 2, "L": 3}).fillna(-1)
    frame["regime_hint"] = frame["regime_code"].map({"balanced": 0, "peak": 1, "disrupted": 2}).fillna(-1)
    frame = frame.fillna(0.0)
    return frame.sort_values(["scale_code", "regime_code", "replicate"]).reset_index(drop=True)

def _build_job_order(instance: dict[str, pd.DataFrame], method_name: str) -> list[str]:
    jobs = instance["jobs"].copy()
    eligible = instance["eligible"].copy()

    nominal_lb = (
        eligible.groupby(["job_id", "op_seq"], as_index=False)["proc_time_min"]
        .min()
        .groupby("job_id", as_index=False)["proc_time_min"]
        .sum()
        .rename(columns={"proc_time_min": "nominal_lb_min"})
    )
    jobs = jobs.merge(nominal_lb, on="job_id", how="left")
    jobs["weighted_slack"] = (
        jobs["completion_due_min"] - jobs["arrival_time_min"] - jobs["nominal_lb_min"]
    ) / jobs["priority_weight"].replace(0.0, 1.0)
    jobs["reactive_urgency"] = (
        jobs["completion_due_min"] - jobs["reveal_time_min"] - jobs["nominal_lb_min"]
    ) / jobs["priority_weight"].replace(0.0, 1.0)

    if method_name == "M0_CUSTOM_FIFO_REPLAY":
        order = jobs.sort_values(
            ["arrival_time_min", "reveal_time_min", "job_id"],
            ascending=[True, True, True],
        )
    elif method_name == "M1_WEIGHTED_SLACK":
        order = jobs.sort_values(
            ["weighted_slack", "priority_weight", "arrival_time_min", "job_id"],
            ascending=[True, False, True, True],
        )
    elif method_name == "M2_PERIODIC_15":
        order = jobs.assign(period_bucket=(jobs["reveal_time_min"] // 15).astype(int)).sort_values(
            ["period_bucket", "weighted_slack", "arrival_congestion_score", "arrival_time_min", "job_id"],
            ascending=[True, True, False, True, True],
        )
    elif method_name == "M2_PERIODIC_30":
        order = jobs.assign(period_bucket=(jobs["reveal_time_min"] // 30).astype(int)).sort_values(
            ["period_bucket", "weighted_slack", "arrival_congestion_score", "arrival_time_min", "job_id"],
            ascending=[True, True, False, True, True],
        )
    elif method_name == "M3_EVENT_REACTIVE":
        order = jobs.sort_values(
            ["reveal_time_min", "reactive_urgency", "arrival_congestion_score", "arrival_time_min", "job_id"],
            ascending=[True, True, False, True, True],
        )
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    return order["job_id"].tolist()

def _build_machine_windows(
    instance: dict[str, pd.DataFrame],
    blocked_rows: pd.DataFrame | None = None,
) -> dict[str, list[tuple[float, float]]]:
    downtimes = instance["downtimes"].copy()
    windows = {
        machine_id: sorted(
            list(
                downtimes.loc[downtimes["machine_id"].eq(machine_id), ["start_min", "end_min"]]
                .astype(float)
                .itertuples(index=False, name=None)
            )
        )
        for machine_id in instance["machines"]["machine_id"]
    }
    if blocked_rows is not None and not blocked_rows.empty:
        for row in blocked_rows.itertuples(index=False):
            windows.setdefault(row.machine_id, []).append((float(row.start_min), float(row.end_min)))
        for machine_id in windows:
            windows[machine_id] = sorted(windows[machine_id], key=lambda item: (item[0], item[1]))
    return windows

def _find_earliest_machine_slot(
    machine_id: str,
    machine_available: dict[str, float],
    machine_windows: dict[str, list[tuple[float, float]]],
    ready_min: float,
    duration_min: float,
    machine_end: dict[str, float],
) -> float | None:
    start = max(float(ready_min), float(machine_available[machine_id]))
    for down_start, down_end in machine_windows.get(machine_id, []):
        if start >= down_end:
            continue
        if start + duration_min <= down_start:
            break
        start = max(start, down_end)
    return float(start)

def _build_job_metrics_from_schedule(
    jobs: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    grouped = schedule.groupby("job_id", as_index=False).agg(
        completion_min=("end_min", "max"),
        queue_time_min=("wait_before_stage_min", "sum"),
    )
    job_metrics = jobs[
        ["job_id", "arrival_time_min", "completion_due_min", "priority_weight", "statutory_wait_limit_min"]
    ].copy().merge(grouped, on="job_id", how="left")
    job_metrics["completion_min"] = job_metrics["completion_min"].fillna(job_metrics["arrival_time_min"])
    job_metrics["queue_time_min"] = job_metrics["queue_time_min"].fillna(0.0)
    job_metrics["flow_time_min"] = job_metrics["completion_min"] - job_metrics["arrival_time_min"]
    job_metrics["overwait_min"] = np.maximum(
        job_metrics["queue_time_min"] - job_metrics["statutory_wait_limit_min"], 0.0
    )
    job_metrics["tardiness_min"] = np.maximum(
        job_metrics["completion_min"] - job_metrics["completion_due_min"], 0.0
    )
    return job_metrics

def _summarize_schedule(
    instance_id: str,
    method_name: str,
    jobs: pd.DataFrame,
    schedule: pd.DataFrame,
    runtime_sec: float,
    solver_status: str,
    replan_count: int,
    extra: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    job_metrics = _build_job_metrics_from_schedule(jobs=jobs, schedule=schedule)
    summary = {
        "instance_id": instance_id,
        "method_name": method_name,
        "runtime_sec": float(runtime_sec),
        "makespan": float(job_metrics["completion_min"].max()),
        "flow_mean": float(job_metrics["flow_time_min"].mean()),
        "flow_p95": float(np.quantile(job_metrics["flow_time_min"], 0.95)),
        "queue_mean": float(job_metrics["queue_time_min"].mean()),
        "queue_p95": float(np.quantile(job_metrics["queue_time_min"], 0.95)),
        "weighted_tardiness": float((job_metrics["tardiness_min"] * job_metrics["priority_weight"]).sum()),
        "replan_count": int(replan_count),
        "solver_status": solver_status,
    }
    if extra:
        summary.update(extra)
    return job_metrics, summary

def _schedule_jobs_from_order(
    instance_id: str,
    method_name: str,
    instance: dict[str, pd.DataFrame],
    job_order: list[str],
    blocked_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = instance["jobs"].copy()
    ops = instance["operations"].copy().sort_values(["job_id", "op_seq"])
    eligible = instance["eligible"].copy()
    machines = instance["machines"].copy()

    blocked_rows = blocked_rows.copy() if blocked_rows is not None else pd.DataFrame()
    machine_available = dict(zip(machines["machine_id"], machines["availability_start_min"].astype(float)))
    machine_windows = _build_machine_windows(instance=instance, blocked_rows=blocked_rows)
    job_index = {job_id: idx for idx, job_id in enumerate(job_order)}
    scheduled_rows: list[dict[str, Any]] = []
    fixed_jobs = set()

    if not blocked_rows.empty:
        blocked_rows = blocked_rows.copy()
        blocked_rows["policy_method"] = method_name
        scheduled_rows.extend(blocked_rows.to_dict(orient="records"))
        fixed_jobs = set(blocked_rows["job_id"].astype(str))

    for job_id in job_order:
        if job_id in fixed_jobs:
            continue
        job_row = jobs.loc[jobs["job_id"].eq(job_id)].iloc[0]
        ready_min = float(max(job_row["arrival_time_min"], job_row["reveal_time_min"]))
        job_ops = ops.loc[ops["job_id"].eq(job_id)].sort_values("op_seq")
        for op_row in job_ops.itertuples(index=False):
            op_ready = max(ready_min, float(op_row.release_time_min))
            eligible_rows = eligible.loc[
                eligible["job_id"].eq(job_id) & eligible["op_seq"].eq(op_row.op_seq)
            ].copy()
            choices = []
            for elig_row in eligible_rows.itertuples(index=False):
                start_min = _find_earliest_machine_slot(
                    machine_id=elig_row.machine_id,
                    machine_available=machine_available,
                    machine_windows=machine_windows,
                    ready_min=op_ready,
                    duration_min=float(elig_row.proc_time_min),
                    machine_end={},
                )
                if start_min is None:
                    continue
                end_min = start_min + float(elig_row.proc_time_min)
                choices.append((end_min, start_min, float(elig_row.proc_time_min), elig_row.machine_id))
            if not choices:
                raise RuntimeError(f"No feasible machine choice for {instance_id} {job_id} op {op_row.op_seq}.")
            _, start_min, proc_time_min, machine_id = min(choices)
            end_min = start_min + proc_time_min
            machine_available[machine_id] = end_min
            scheduled_rows.append(
                {
                    "instance_id": instance_id,
                    "job_id": job_id,
                    "job_rank": job_index[job_id],
                    "op_seq": int(op_row.op_seq),
                    "stage_name": op_row.stage_name,
                    "machine_id": machine_id,
                    "start_min": float(start_min),
                    "end_min": float(end_min),
                    "wait_before_stage_min": float(max(0.0, start_min - op_ready)),
                    "policy_method": method_name,
                }
            )
            ready_min = end_min

    schedule = (
        pd.DataFrame(scheduled_rows)
        .sort_values(["start_min", "end_min", "job_id", "op_seq"])
        .reset_index(drop=True)
    )
    return schedule, _build_job_metrics_from_schedule(jobs=jobs, schedule=schedule)

def _method_objective_tuple(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(summary["weighted_tardiness"]),
        float(summary["flow_p95"]),
        float(summary["flow_mean"]),
        float(summary["makespan"]),
        float(summary["runtime_sec"]),
    )

def _perturb_job_order(order: list[str], rng: np.random.Generator) -> list[str]:
    candidate = list(order)
    if len(candidate) < 2:
        return candidate
    i, j = sorted(rng.choice(len(candidate), size=2, replace=False).tolist())
    if rng.random() < 0.5:
        candidate[i], candidate[j] = candidate[j], candidate[i]
    else:
        value = candidate.pop(j)
        candidate.insert(i, value)
    return candidate

def _evaluate_candidate_order(
    root: Path,
    instance_id: str,
    method_name: str,
    job_order: list[str],
    blocked_rows: pd.DataFrame | None = None,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    instance = _load_instance_tables(root, instance_id)
    started = time.perf_counter()
    schedule, _ = _schedule_jobs_from_order(
        instance_id=instance_id,
        method_name=method_name,
        instance=instance,
        job_order=job_order,
        blocked_rows=blocked_rows,
    )
    _, summary = _summarize_schedule(
        instance_id=instance_id,
        method_name=method_name,
        jobs=instance["jobs"].copy(),
        schedule=schedule,
        runtime_sec=time.perf_counter() - started,
        solver_status="heuristic_feasible",
        replan_count=max(1, instance["jobs"]["reveal_time_min"].nunique()),
    )
    return job_order, schedule, summary

def _optimize_job_order(
    root: Path,
    instance_id: str,
    method_name: str,
    seed_order: list[str],
    time_budget_sec: float,
    n_workers: int = 1,
    blocked_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(SEED + abs(hash((instance_id, method_name))) % 100000)
    _, best_schedule, best_summary = _evaluate_candidate_order(
        root=root,
        instance_id=instance_id,
        method_name=method_name,
        job_order=seed_order,
        blocked_rows=blocked_rows,
    )
    best_order = list(seed_order)
    started = time.perf_counter()
    while time.perf_counter() - started < time_budget_sec:
        batch_orders = [_perturb_job_order(best_order, rng) for _ in range(max(2, 2 * n_workers))]
        if n_workers > 1:
            results = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_evaluate_candidate_order)(
                    root=root,
                    instance_id=instance_id,
                    method_name=method_name,
                    job_order=order,
                    blocked_rows=blocked_rows,
                )
                for order in batch_orders
            )
        else:
            results = [
                _evaluate_candidate_order(
                    root=root,
                    instance_id=instance_id,
                    method_name=method_name,
                    job_order=order,
                    blocked_rows=blocked_rows,
                )
                for order in batch_orders
            ]
        for order, schedule, summary in results:
            if _method_objective_tuple(summary) < _method_objective_tuple(best_summary):
                best_order = list(order)
                best_schedule = schedule
                best_summary = summary
    best_summary["runtime_sec"] = float(time.perf_counter() - started)
    best_summary["solver_status"] = f"metaheuristic_workers_{n_workers}"
    return best_schedule, best_summary

def _status_label_scipy(result: Any) -> str:
    message = str(getattr(result, "message", "")).lower()
    if getattr(result, "status", None) == 0:
        return "optimal"
    if "time limit" in message or "time_limit" in message:
        return "time_limit"
    if "infeasible" in message:
        return "infeasible"
    if getattr(result, "success", False):
        return "feasible"
    return "other"

def _exact_subset_job_cap(scale_code: str) -> int:
    return {"XS": 12, "S": 16}.get(scale_code, 12)

def _build_exact_model_for_jobs(instance_dir: Path, max_jobs: int) -> tuple[np.ndarray, np.ndarray, Bounds, list[LinearConstraint], dict[str, Any], dict[str, dict[Any, int]], dict[str, Any]]:
    raw = load_instance(instance_dir)
    ordered_jobs = sorted(raw["jobs"], key=lambda row: (row["arrival_time_min"], row["job_id"]))[:max_jobs]
    keep_jobs = {row["job_id"] for row in ordered_jobs}
    restricted = dict(raw)
    restricted["jobs"] = [row for row in raw["jobs"] if row["job_id"] in keep_jobs]
    restricted["operations"] = [row for row in raw["operations"] if row["job_id"] in keep_jobs]
    restricted["precedences"] = [row for row in raw["precedences"] if row["job_id"] in keep_jobs]
    restricted["eligible_machines"] = [row for row in raw["eligible_machines"] if row["job_id"] in keep_jobs]
    keep_machines = {row["machine_id"] for row in restricted["eligible_machines"]}
    restricted["machines"] = [row for row in raw["machines"] if row["machine_id"] in keep_machines]
    restricted["machine_downtimes"] = [row for row in raw["machine_downtimes"] if row["machine_id"] in keep_machines]
    data = build_gurobi_views(restricted)
    ops_release = {(row["job_id"], row["op_seq"]): int(row["release_time_min"]) for row in restricted["operations"]}
    lag_by_arc = {
        (row["job_id"], row["pred_op_seq"], row["succ_op_seq"]): int(row["min_lag_min"])
        for row in restricted["precedences"]
    }
    horizon = max(
        int(data["params"]["planning_horizon_min"]),
        max(data["DUE"].values()),
        max((end for downs in data["DOWNTIMES_BY_MACHINE"].values() for _, end, _ in downs), default=0),
    )
    max_proc = max(data["PROC"].values())
    big_m = float(horizon + max_proc + 1)
    ops = list(data["OPS"])
    eligible_keys = list(data["ELIGIBLE_KEYS"])
    last_ops = [(job_id, max(op_seq for j2, op_seq in ops if j2 == job_id)) for job_id in data["J"]]

    x_idx = {key: i for i, key in enumerate(eligible_keys)}
    cursor = len(x_idx)
    s_idx = {(j, o): cursor + i for i, (j, o) in enumerate(ops)}
    cursor += len(s_idx)
    c_idx = {(j, o): cursor + i for i, (j, o) in enumerate(ops)}
    cursor += len(c_idx)
    cmax_idx = cursor
    cursor += 1

    pair_records: list[tuple[tuple[str, int], tuple[str, int], str]] = []
    for machine_id in data["M"]:
        eligible_ops = sorted({(j, o) for (j, o, m) in eligible_keys if m == machine_id})
        for idx_a in range(len(eligible_ops)):
            for idx_b in range(idx_a + 1, len(eligible_ops)):
                pair_records.append((eligible_ops[idx_a], eligible_ops[idx_b], machine_id))
    y_idx = {record: cursor + i for i, record in enumerate(pair_records)}
    cursor += len(y_idx)

    downtime_records: list[tuple[str, int, str, int]] = []
    for (j, o, machine_id) in eligible_keys:
        for dt_idx, _ in enumerate(data["DOWNTIMES_BY_MACHINE"].get(machine_id, [])):
            downtime_records.append((j, o, machine_id, dt_idx))
    q_idx = {record: cursor + i for i, record in enumerate(downtime_records)}
    cursor += len(q_idx)

    n_vars = cursor
    c = np.zeros(n_vars, dtype=float)
    c[cmax_idx] = 1.0
    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, big_m, dtype=float)
    integrality = np.zeros(n_vars, dtype=int)
    for idx in list(x_idx.values()) + list(y_idx.values()) + list(q_idx.values()):
        ub[idx] = 1.0
        integrality[idx] = 1

    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    def add_constraint(coeffs: dict[int, float], lb_value: float, ub_value: float) -> None:
        nonlocal row
        for col, value in coeffs.items():
            row_idx.append(row)
            col_idx.append(col)
            values.append(value)
        lower.append(lb_value)
        upper.append(ub_value)
        row += 1

    for j, o in ops:
        add_constraint({x_idx[(j, o, m)]: 1.0 for m in data["ELIGIBLE_BY_OP"][(j, o)]}, 1.0, 1.0)
    for j, o in ops:
        coeffs = {c_idx[(j, o)]: 1.0, s_idx[(j, o)]: -1.0}
        for m in data["ELIGIBLE_BY_OP"][(j, o)]:
            coeffs[x_idx[(j, o, m)]] = -float(data["PROC"][(j, o, m)])
        add_constraint(coeffs, 0.0, 0.0)
    for j, o in ops:
        add_constraint({s_idx[(j, o)]: 1.0}, float(ops_release[(j, o)]), np.inf)
    for (j, pred, succ) in data["PRED"]:
        lag = float(lag_by_arc[(j, pred, succ)])
        add_constraint({s_idx[(j, succ)]: 1.0, c_idx[(j, pred)]: -1.0}, lag, np.inf)
    for j, o in last_ops:
        add_constraint({cmax_idx: 1.0, c_idx[(j, o)]: -1.0}, 0.0, np.inf)
    for (op_a, op_b, machine_id), y_var in y_idx.items():
        j_a, o_a = op_a
        j_b, o_b = op_b
        x_a = x_idx[(j_a, o_a, machine_id)]
        x_b = x_idx[(j_b, o_b, machine_id)]
        add_constraint({s_idx[(j_b, o_b)]: 1.0, c_idx[(j_a, o_a)]: -1.0, y_var: -big_m, x_a: -big_m, x_b: -big_m}, -3.0 * big_m, np.inf)
        add_constraint({s_idx[(j_a, o_a)]: 1.0, c_idx[(j_b, o_b)]: -1.0, y_var: big_m, x_a: -big_m, x_b: -big_m}, -2.0 * big_m, np.inf)
    for (j, o, machine_id, dt_idx), q_var in q_idx.items():
        down_start, down_end, _ = data["DOWNTIMES_BY_MACHINE"][machine_id][dt_idx]
        x_var = x_idx[(j, o, machine_id)]
        add_constraint({c_idx[(j, o)]: 1.0, q_var: big_m, x_var: -big_m}, -np.inf, float(down_start + big_m))
        add_constraint({s_idx[(j, o)]: 1.0, q_var: big_m, x_var: -big_m}, float(down_end - big_m), np.inf)

    matrix = coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (np.asarray(row_idx, dtype=np.int32), np.asarray(col_idx, dtype=np.int32)),
        ),
        shape=(row, n_vars),
    ).tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    metadata = {
        "job_ids": [row["job_id"] for row in ordered_jobs],
        "ops_release": ops_release,
        "op_stage": {(row["job_id"], row["op_seq"]): row["stage_name"] for row in restricted["operations"]},
        "big_m": big_m,
        "job_count": len(data["J"]),
    }
    mappings = {"x": x_idx, "s": s_idx, "c": c_idx}
    return c, integrality, Bounds(lb, ub), [LinearConstraint(matrix, np.array(lower), np.array(upper))], metadata, mappings, restricted

def _solve_exact_reference_schedule(
    root: Path,
    instance_id: str,
    time_limit_sec: float,
    max_jobs: int,
    method_name: str = "Mref_EXACT_XS_S",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    instance_dir = root / "instances" / instance_id
    c, integrality, bounds, constraints, metadata, mappings, restricted = _build_exact_model_for_jobs(
        instance_dir=instance_dir,
        max_jobs=max_jobs,
    )
    started = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit_sec), "disp": False},
    )
    runtime_sec = time.perf_counter() - started
    solution = getattr(result, "x", None)
    if solution is None or not np.isfinite(getattr(result, "fun", np.nan)):
        raise RuntimeError(f"Exact reference failed for {instance_id} with status={_status_label_scipy(result)}")
    chosen = {}
    for key, idx in mappings["x"].items():
        if solution[idx] >= 0.5:
            chosen[(key[0], key[1])] = key[2]
    rows = []
    for (job_id, op_seq), machine_id in sorted(chosen.items(), key=lambda item: (solution[mappings["s"][(item[0][0], item[0][1])]], item[0][0], item[0][1])):
        start_min = float(solution[mappings["s"][(job_id, op_seq)]])
        end_min = float(solution[mappings["c"][(job_id, op_seq)]])
        rows.append(
            {
                "instance_id": instance_id,
                "job_id": job_id,
                "job_rank": 0,
                "op_seq": int(op_seq),
                "stage_name": metadata["op_stage"][(job_id, op_seq)],
                "machine_id": machine_id,
                "start_min": start_min,
                "end_min": end_min,
                "wait_before_stage_min": 0.0,
                "policy_method": method_name,
            }
        )
    schedule = pd.DataFrame(rows).sort_values(["start_min", "end_min", "job_id", "op_seq"]).reset_index(drop=True)
    for job_id, job_rows in schedule.groupby("job_id"):
        job_rows = job_rows.sort_values("op_seq")
        prev_end = None
        for idx, row in job_rows.iterrows():
            op_release = float(metadata["ops_release"][(job_id, int(row["op_seq"]))])
            op_ready = op_release if prev_end is None else max(op_release, prev_end)
            schedule.loc[idx, "wait_before_stage_min"] = max(0.0, float(row["start_min"]) - op_ready)
            prev_end = float(row["end_min"])
    info = {
        "runtime_sec": float(runtime_sec),
        "solver_status": _status_label_scipy(result),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "exact_job_count": int(metadata["job_count"]),
    }
    return schedule, info

def _schedule_jobs_by_policy(root: Path, instance_id: str, method_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    instance = _load_instance_tables(root, instance_id)
    jobs = instance["jobs"].copy()
    scale_code = str(jobs["job_id"].iloc[0]).split("_")[1] if False else None
    started = time.perf_counter()

    if method_name in {"M0_CUSTOM_FIFO_REPLAY", "M1_WEIGHTED_SLACK"}:
        job_order = _build_job_order(instance, method_name)
        schedule, _ = _schedule_jobs_from_order(
            instance_id=instance_id,
            method_name=method_name,
            instance=instance,
            job_order=job_order,
        )
        _, summary = _summarize_schedule(
            instance_id=instance_id,
            method_name=method_name,
            jobs=jobs,
            schedule=schedule,
            runtime_sec=time.perf_counter() - started,
            solver_status="heuristic_feasible",
            replan_count=1,
        )
        return schedule, summary

    if method_name in {"M2_PERIODIC_15", "M2_PERIODIC_30", "M3_EVENT_REACTIVE"}:
        online_schedule, online_info = online_replay.run_online_policy(
            instance_id=instance_id,
            method_name=method_name,
            instance=instance,
            delta_min=METHOD_SPECS[method_name].delta_min,
        )
        _, summary = _summarize_schedule(
            instance_id=instance_id,
            method_name=method_name,
            jobs=jobs,
            schedule=online_schedule,
            runtime_sec=online_info["runtime_sec"],
            solver_status=online_info["solver_status"],
            replan_count=online_info["replan_count"],
        )
        return online_schedule, summary

    catalog_row = pd.read_csv(root / "catalog" / "benchmark_catalog.csv").loc[
        lambda df: df["instance_id"].eq(instance_id)
    ].iloc[0]
    scale_code = str(catalog_row["scale_code"])

    if method_name == "M4_METAHEURISTIC_L":
        seed_order = _build_job_order(instance, "M3_EVENT_REACTIVE")
        schedule, summary = _optimize_job_order(
            root=root,
            instance_id=instance_id,
            method_name=method_name,
            seed_order=seed_order,
            time_budget_sec=2.5,
            n_workers=4,
        )
        summary["replan_count"] = int(max(1, jobs["reveal_time_min"].nunique()))
        return schedule, summary

    if method_name == "Mref_EXACT_XS_S":
        cap = _exact_subset_job_cap(scale_code)
        exact_schedule, exact_info = _solve_exact_reference_schedule(
            root=root,
            instance_id=instance_id,
            time_limit_sec=6.0,
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
            extra={"mip_gap": exact_info["mip_gap"], "exact_job_count": exact_info["exact_job_count"]},
        )
        return schedule, summary

    raise ValueError(f"Unsupported method: {method_name}")

def build_method_performance_matrix(root: Path, ctx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    catalog = ctx["catalog"][["instance_id", "scale_code", "regime_code", "replicate"]].copy()
    official_job_metrics = ctx["job_metrics"].copy()
    official_schedule = ctx["schedule"].copy()
    official_summary = (
        official_job_metrics.groupby("instance_id", as_index=False)
        .agg(
            flow_mean=("flow_time_min", "mean"),
            flow_p95=("flow_time_min", lambda s: float(np.quantile(s, 0.95))),
            queue_mean=("queue_time_min", "mean"),
            queue_p95=("queue_time_min", lambda s: float(np.quantile(s, 0.95))),
            makespan=("completion_min", "max"),
            weighted_tardiness=("overwait_min", "sum"),
        )
        .assign(runtime_sec=0.0, replan_count=1, solver_status="official_release", method_name="M0_FIFO_OFFICIAL")
    )

    long_rows = [official_summary.merge(catalog, on="instance_id", how="left")]
    schedules = {"M0_FIFO_OFFICIAL": official_schedule.copy()}

    for method_name in METHOD_SPECS:
        summary_rows = []
        schedule_rows = []
        spec = METHOD_SPECS[method_name]
        eligible_instances = catalog.copy()
        if spec.supported_scales:
            eligible_instances = eligible_instances.loc[eligible_instances["scale_code"].isin(spec.supported_scales)]
        for instance_id in eligible_instances["instance_id"].tolist():
            schedule, summary = _schedule_jobs_by_policy(root=root, instance_id=instance_id, method_name=method_name)
            summary_rows.append(summary)
            schedule_rows.append(schedule)
        method_df = pd.DataFrame(summary_rows).merge(catalog, on="instance_id", how="left")
        long_rows.append(method_df)
        schedules[method_name] = pd.concat(schedule_rows, ignore_index=True)

    performance = pd.concat(long_rows, ignore_index=True)
    fifo_baseline = (
        performance.loc[performance["method_name"].eq("M0_FIFO_OFFICIAL"), ["instance_id", "flow_mean", "flow_p95", "makespan"]]
        .rename(
            columns={
                "flow_mean": "fifo_flow_mean",
                "flow_p95": "fifo_flow_p95",
                "makespan": "fifo_makespan",
            }
        )
    )
    performance = performance.merge(fifo_baseline, on="instance_id", how="left")
    performance["delta_vs_fifo_mean_flow_pct"] = (
        performance["flow_mean"] - performance["fifo_flow_mean"]
    ) / performance["fifo_flow_mean"].replace(0.0, np.nan)
    performance["delta_vs_fifo_p95_flow_pct"] = (
        performance["flow_p95"] - performance["fifo_flow_p95"]
    ) / performance["fifo_flow_p95"].replace(0.0, np.nan)
    performance["delta_vs_fifo_makespan_pct"] = (
        performance["makespan"] - performance["fifo_makespan"]
    ) / performance["fifo_makespan"].replace(0.0, np.nan)
    performance = _attach_protocol_columns(performance)
    performance = performance.sort_values([
        "protocol_role",
        "scale_code",
        "regime_code",
        "replicate",
        "instance_id",
        "method_name",
    ]).reset_index(drop=True)
    return performance, schedules

def add_utility_and_difficulty(performance: pd.DataFrame) -> pd.DataFrame:
    frame = performance.copy()
    utility_parts = []
    for instance_id, group in frame.groupby("instance_id"):
        group = group.copy()
        for metric_name in UTILITY_WEIGHTS:
            values = group[metric_name].astype(float)
            min_v = float(values.min())
            max_v = float(values.max())
            if math.isclose(max_v, min_v):
                group[f"{metric_name}_norm"] = 0.0
            else:
                group[f"{metric_name}_norm"] = (values - min_v) / (max_v - min_v)
        group["utility"] = sum(group[f"{metric}_norm"] * weight for metric, weight in UTILITY_WEIGHTS.items())
        best = float(group["utility"].min())
        group["regret"] = group["utility"] - best
        utility_parts.append(group)
    frame = pd.concat(utility_parts, ignore_index=True)
    best_by_instance = (
        frame.sort_values(["instance_id", "utility", "runtime_sec", "method_name"])
        .groupby("instance_id", as_index=False)
        .first()[["instance_id", "method_name", "utility"]]
        .rename(columns={"method_name": "best_method", "utility": "best_utility"})
    )
    frame = frame.merge(best_by_instance, on="instance_id", how="left")
    best_utility_by_instance = frame[["instance_id", "best_utility"]].drop_duplicates()
    q1 = float(best_utility_by_instance["best_utility"].quantile(1 / 3))
    q2 = float(best_utility_by_instance["best_utility"].quantile(2 / 3))
    difficulty_map = {}
    for row in best_utility_by_instance.itertuples(index=False):
        if row.best_utility <= q1:
            difficulty_map[row.instance_id] = "easy"
        elif row.best_utility <= q2:
            difficulty_map[row.instance_id] = "medium"
        else:
            difficulty_map[row.instance_id] = "hard"
    frame["difficulty"] = frame["instance_id"].map(difficulty_map)
    frame["footprint_member"] = frame["regret"] <= 0.15
    return frame

def _numeric_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    blocked = {
        "instance_id",
        "core_instance_digest",
        "nearest_neighbor_instance_id",
        "scale_code",
        "regime_code",
        "bottleneck_machine_family",
    }
    return [
        column
        for column in feature_frame.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]

def build_umap_hdbscan_frame(feature_frame: pd.DataFrame, best_method_df: pd.DataFrame) -> pd.DataFrame:
    frame = feature_frame.copy()
    numeric_cols = _numeric_feature_columns(frame)
    matrix = frame[numeric_cols].astype(float).to_numpy()
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    umap = UMAP(n_neighbors=min(10, len(frame) - 1), min_dist=0.15, random_state=SEED)
    embedding = umap.fit_transform(matrix)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    clusters = clusterer.fit_predict(embedding)
    frame["umap_x"] = embedding[:, 0]
    frame["umap_y"] = embedding[:, 1]
    frame["cluster_label"] = clusters
    frame = frame.merge(best_method_df[["instance_id", "best_method", "difficulty"]], on="instance_id", how="left")
    return frame

def build_selector_results(feature_frame: pd.DataFrame, best_method_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = feature_frame.merge(best_method_df[["instance_id", "best_method", "difficulty"]], on="instance_id", how="left")
    numeric_cols = _numeric_feature_columns(data)
    X = data[numeric_cols].astype(float).to_numpy()
    y = data["best_method"].astype(str).to_numpy()
    classes = sorted(pd.unique(y).tolist())
    class_to_int = {label: idx for idx, label in enumerate(classes)}
    y_int = np.array([class_to_int[label] for label in y], dtype=int)

    estimators: list[tuple[str, Any]] = [
        ("decision_tree", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=SEED, min_samples_leaf=2)),
    ]
    if LGBMClassifier is not None:
        estimators.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    random_state=SEED,
                    verbosity=-1,
                ),
            )
        )

    rows = []
    best_model_name = None
    best_predictions = None
    best_estimator = None
    best_score = -np.inf
    for model_name, estimator in estimators:
        predictions = []
        for idx in range(len(data)):
            train_mask = np.ones(len(data), dtype=bool)
            train_mask[idx] = False
            estimator.fit(X[train_mask], y_int[train_mask])
            pred_idx = int(estimator.predict(X[idx : idx + 1])[0])
            predictions.append(classes[pred_idx])
        predictions = np.array(predictions)
        acc = accuracy_score(y, predictions)
        macro_f1 = f1_score(y, predictions, average="macro")
        bal_acc = balanced_accuracy_score(y, predictions)
        rows.append(
            {
                "model_name": model_name,
                "top1_accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "balanced_accuracy": float(bal_acc),
            }
        )
        composite = acc + macro_f1
        if composite > best_score:
            best_score = composite
            best_model_name = model_name
            best_predictions = predictions
            best_estimator = estimator

    report = pd.DataFrame(rows).sort_values(["top1_accuracy", "macro_f1"], ascending=False).reset_index(drop=True)
    best_estimator.fit(X, y_int)
    explainer = shap.TreeExplainer(best_estimator)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_importance = np.mean([np.abs(value) for value in shap_values], axis=0)
    elif np.asarray(shap_values).ndim == 3:
        shap_importance = np.abs(np.asarray(shap_values)).mean(axis=0)
    else:
        shap_importance = np.abs(np.asarray(shap_values))
    mean_abs = np.asarray(shap_importance).mean(axis=0)
    shap_summary = pd.DataFrame(
        {
            "feature_name": numeric_cols,
            "mean_abs_shap": mean_abs,
            "selected_model": best_model_name,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return report, shap_summary.reset_index(drop=True)

def export_aslib_scenario(feature_frame: pd.DataFrame, performance: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    _ensure_dir(out_dir)
    numeric_cols = _numeric_feature_columns(feature_frame)
    features = feature_frame[["instance_id"] + numeric_cols].copy()
    performance_wide = (
        performance.pivot(index="instance_id", columns="method_name", values="utility")
        .reset_index()
        .rename_axis(columns=None)
    )
    runstatus_wide = (
        performance.assign(runstatus="ok")
        .pivot(index="instance_id", columns="method_name", values="runstatus")
        .reset_index()
        .rename_axis(columns=None)
    )
    feature_costs = pd.DataFrame(
        {
            "feature_name": numeric_cols,
            "cost_sec": [0.001] * len(numeric_cols),
        }
    )
    cv = feature_frame[["instance_id", "replicate", "scale_code", "regime_code"]].copy()
    cv["fold"] = cv["replicate"].astype(int)
    paths = {
        "features": out_dir / "features.csv",
        "performance": out_dir / "performance.csv",
        "runstatus": out_dir / "runstatus.csv",
        "feature_costs": out_dir / "feature_costs.csv",
        "cv": out_dir / "cv.csv",
    }
    features.to_csv(paths["features"], index=False)
    performance_wide.to_csv(paths["performance"], index=False)
    runstatus_wide.to_csv(paths["runstatus"], index=False)
    feature_costs.to_csv(paths["feature_costs"], index=False)
    cv.to_csv(paths["cv"], index=False)
    return paths

def build_scorecard(ctx: dict[str, Any], performance: pd.DataFrame, selector_report: pd.DataFrame) -> pd.DataFrame:
    summary = ctx["summary"]
    best_methods = performance[["instance_id", "best_method"]].drop_duplicates()
    return pd.DataFrame(
        [
            {"metric_name": "structural_pass_rate", "metric_value": summary["structural_pass_rate"]},
            {"metric_name": "release_consistency_checks_pass", "metric_value": float(summary["release_consistency_checks_pass"])},
            {"metric_name": "flow_regime_order_checks_pass", "metric_value": float(summary["flow_regime_order_checks_pass"])},
            {"metric_name": "instance_space_exact_duplicate_checks_pass", "metric_value": float(summary["instance_space_exact_duplicate_checks_pass"])},
            {"metric_name": "method_count", "metric_value": float(performance["method_name"].nunique())},
            {"metric_name": "best_method_diversity", "metric_value": float(best_methods["best_method"].nunique())},
            {"metric_name": "selector_top1_accuracy", "metric_value": float(selector_report.loc[0, "top1_accuracy"])},
            {"metric_name": "selector_macro_f1", "metric_value": float(selector_report.loc[0, "macro_f1"])},
        ]
    )


def render_figures(
    performance: pd.DataFrame,
    umap_frame: pd.DataFrame,
    shap_frame: pd.DataFrame,
    figure_dir: Path,
) -> dict[str, Path]:
    _ensure_dir(figure_dir)
    figures = {
        "method_delta_vs_fifo": plot_method_delta(performance),
        "method_runtime_heatmap": plot_method_runtime(performance),
        "umap_best_method": plot_umap_best_method(umap_frame),
        "hdbscan_clusters": plot_hdbscan_clusters(umap_frame),
        "solver_footprints": plot_solver_footprints(umap_frame, performance),
        "selector_shap": plot_selector_shap(shap_frame),
    }
    paths: dict[str, Path] = {}
    for figure_name, fig in figures.items():
        out_path = figure_dir / f"{figure_name}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[figure_name] = out_path
    return paths


def run_full_pipeline(root: Path, ctx: dict[str, Any], figure_dir: Path) -> PaperPipelineResults:
    feature_frame = build_paper_feature_frame(ctx)
    performance_raw, schedules = build_method_performance_matrix(root=root, ctx=ctx)
    performance = add_utility_and_difficulty(performance_raw)
    performance = _attach_protocol_columns(performance)
    performance_test = performance.loc[performance["protocol_role"].eq("test")].copy().reset_index(drop=True)
    protocol_summary = build_protocol_split_summary(performance)
    best_method_df = performance.sort_values(["instance_id", "utility"]).groupby("instance_id", as_index=False).first()
    umap_frame = build_umap_hdbscan_frame(feature_frame=feature_frame, best_method_df=best_method_df)
    selector_report, shap_frame = build_selector_results(feature_frame=feature_frame, best_method_df=best_method_df)
    scorecard = build_scorecard(ctx=ctx, performance=performance_test, selector_report=selector_report)

    catalog_dir = root / "catalog"
    aslib_dir = catalog_dir / "aslib_scenario"
    feature_frame.to_csv(catalog_dir / "instance_features.csv", index=False)
    performance.to_csv(catalog_dir / "method_performance_matrix.csv", index=False)
    performance_test.to_csv(catalog_dir / "method_performance_test_matrix.csv", index=False)
    protocol_summary.to_csv(catalog_dir / "protocol_split_summary.csv", index=False)
    scorecard.to_csv(catalog_dir / "scorecard_release_sbpo.csv", index=False)
    selector_report.to_csv(catalog_dir / "selector_report.csv", index=False)
    shap_frame[["feature_name", "mean_abs_shap", "selected_model"]].drop_duplicates().to_csv(
        catalog_dir / "selector_shap_summary.csv", index=False
    )
    umap_frame.to_csv(catalog_dir / "instance_umap_hdbscan.csv", index=False)
    aslib_paths = export_aslib_scenario(feature_frame=feature_frame, performance=performance, out_dir=aslib_dir)
    figure_paths = render_figures(
        performance=performance_test,
        umap_frame=umap_frame,
        shap_frame=shap_frame,
        figure_dir=figure_dir,
    )
    return PaperPipelineResults(
        feature_frame=feature_frame,
        performance=performance,
        performance_test=performance_test,
        protocol_summary=protocol_summary,
        umap_frame=umap_frame,
        selector_report=selector_report,
        shap_frame=shap_frame,
        scorecard=scorecard,
        aslib_paths=aslib_paths,
        figure_paths=figure_paths,
        schedules=schedules,
    )


def load_or_run_full_pipeline(
    root: Path,
    ctx: dict[str, Any],
    figure_dir: Path,
    catalog_dir: Path | None = None,
) -> PaperPipelineResults:
    catalog_dir = catalog_dir or (root / "catalog")
    cached_paper_paths = {
        "feature_frame": catalog_dir / "instance_features.csv",
        "performance": catalog_dir / "method_performance_matrix.csv",
        "performance_test": catalog_dir / "method_performance_test_matrix.csv",
        "protocol_summary": catalog_dir / "protocol_split_summary.csv",
        "umap_frame": catalog_dir / "instance_umap_hdbscan.csv",
        "selector_report": catalog_dir / "selector_report.csv",
        "shap_frame": catalog_dir / "selector_shap_summary.csv",
        "scorecard": catalog_dir / "scorecard_release_sbpo.csv",
    }
    use_cached_paper_results = all(path.exists() for path in cached_paper_paths.values())
    if not use_cached_paper_results:
        return run_full_pipeline(root=root, ctx=ctx, figure_dir=figure_dir)

    feature_frame = pd.read_csv(cached_paper_paths["feature_frame"])
    performance = pd.read_csv(cached_paper_paths["performance"])
    performance_test = pd.read_csv(cached_paper_paths["performance_test"])
    protocol_summary = pd.read_csv(cached_paper_paths["protocol_summary"])
    umap_frame = pd.read_csv(cached_paper_paths["umap_frame"])
    selector_report = pd.read_csv(cached_paper_paths["selector_report"])
    shap_frame = pd.read_csv(cached_paper_paths["shap_frame"])
    scorecard = pd.read_csv(cached_paper_paths["scorecard"])

    aslib_dir = catalog_dir / "aslib_scenario"
    aslib_paths = {
        "features": aslib_dir / "features.csv",
        "performance": aslib_dir / "performance.csv",
        "runstatus": aslib_dir / "runstatus.csv",
        "feature_costs": aslib_dir / "feature_costs.csv",
        "cv": aslib_dir / "cv.csv",
    }
    if not all(path.exists() for path in aslib_paths.values()):
        aslib_paths = export_aslib_scenario(feature_frame=feature_frame, performance=performance, out_dir=aslib_dir)

    figure_paths = {name: figure_dir / f"{name}.png" for name in (
        "method_delta_vs_fifo",
        "method_runtime_heatmap",
        "umap_best_method",
        "hdbscan_clusters",
        "solver_footprints",
        "selector_shap",
    )}
    if not all(path.exists() for path in figure_paths.values()):
        figure_paths = render_figures(
            performance=performance_test,
            umap_frame=umap_frame,
            shap_frame=shap_frame,
            figure_dir=figure_dir,
        )

    return PaperPipelineResults(
        feature_frame=feature_frame,
        performance=performance,
        performance_test=performance_test,
        protocol_summary=protocol_summary,
        umap_frame=umap_frame,
        selector_report=selector_report,
        shap_frame=shap_frame,
        scorecard=scorecard,
        aslib_paths=aslib_paths,
        figure_paths=figure_paths,
        schedules={},
    )
