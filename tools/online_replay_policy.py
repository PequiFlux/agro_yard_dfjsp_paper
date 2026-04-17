from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd


EPS = 1e-9
RELEVANT_EVENT_TYPES = {"JOB_VISIBLE", "JOB_ARRIVAL", "MACHINE_DOWN", "MACHINE_UP"}
SCHEDULE_COLUMNS = [
    "instance_id",
    "job_id",
    "job_rank",
    "op_seq",
    "stage_name",
    "machine_id",
    "start_min",
    "end_min",
    "wait_before_stage_min",
    "policy_method",
]


def _empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEDULE_COLUMNS)


def _normalize_schedule(schedule: pd.DataFrame, method_name: str) -> pd.DataFrame:
    if schedule.empty:
        return _empty_schedule()
    frame = schedule.copy()
    for column in SCHEDULE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["policy_method"] = method_name
    return frame[SCHEDULE_COLUMNS].sort_values(
        ["start_min", "end_min", "job_id", "op_seq"]
    ).reset_index(drop=True)


def _build_machine_windows(
    instance: dict[str, pd.DataFrame],
    blocked_rows: pd.DataFrame | None = None,
) -> dict[str, list[tuple[float, float]]]:
    downtimes = instance["downtimes"].copy()
    windows = {
        machine_id: sorted(
            downtimes.loc[downtimes["machine_id"].eq(machine_id), ["start_min", "end_min"]]
            .astype(float)
            .itertuples(index=False, name=None)
        )
        for machine_id in instance["machines"]["machine_id"].astype(str)
    }
    if blocked_rows is not None and not blocked_rows.empty:
        for row in blocked_rows.itertuples(index=False):
            windows.setdefault(str(row.machine_id), []).append(
                (float(row.start_min), float(row.end_min))
            )
        for machine_id in windows:
            windows[machine_id] = sorted(windows[machine_id], key=lambda item: (item[0], item[1]))
    return windows


def _find_earliest_machine_slot(
    machine_id: str,
    machine_available: dict[str, float],
    machine_windows: dict[str, list[tuple[float, float]]],
    ready_min: float,
    duration_min: float,
) -> float | None:
    start = max(float(ready_min), float(machine_available[machine_id]))
    for down_start, down_end in machine_windows.get(machine_id, []):
        if start >= down_end - EPS:
            continue
        if start + duration_min <= down_start + EPS:
            break
        start = max(start, down_end)
    return float(start)


def _build_progress_frame(
    instance: dict[str, pd.DataFrame],
    current_time: float,
    fixed_rows: pd.DataFrame,
) -> pd.DataFrame:
    jobs = instance["jobs"].copy()
    ops = instance["operations"].copy().sort_values(["job_id", "op_seq"])
    eligible = instance["eligible"].copy()
    min_proc = (
        eligible.groupby(["job_id", "op_seq"], as_index=False)["proc_time_min"]
        .min()
        .rename(columns={"proc_time_min": "min_proc_time_min"})
    )
    ops = ops.merge(min_proc, on=["job_id", "op_seq"], how="left")
    fixed_summary = (
        fixed_rows.groupby("job_id", as_index=False)
        .agg(last_fixed_op_seq=("op_seq", "max"), last_fixed_end=("end_min", "max"))
        if not fixed_rows.empty
        else pd.DataFrame(columns=["job_id", "last_fixed_op_seq", "last_fixed_end"])
    )
    rows: list[dict[str, Any]] = []
    for job in jobs.itertuples(index=False):
        job_id = str(job.job_id)
        if float(job.reveal_time_min) > float(current_time) + EPS:
            continue
        fixed_job = fixed_summary.loc[fixed_summary["job_id"].eq(job_id)]
        last_fixed_op_seq = int(fixed_job["last_fixed_op_seq"].iloc[0]) if not fixed_job.empty else 0
        last_fixed_end = float(fixed_job["last_fixed_end"].iloc[0]) if not fixed_job.empty else 0.0
        pending_ops = ops.loc[
            ops["job_id"].eq(job_id) & ops["op_seq"].gt(last_fixed_op_seq)
        ].sort_values("op_seq")
        if pending_ops.empty:
            continue
        next_op = pending_ops.iloc[0]
        base_ready_min = max(
            float(current_time),
            float(job.arrival_time_min),
            float(job.reveal_time_min),
            last_fixed_end,
        )
        dispatch_ready_min = max(base_ready_min, float(next_op["release_time_min"]))
        remaining_lb_min = float(pending_ops["min_proc_time_min"].sum())
        priority_weight = float(getattr(job, "priority_weight", 1.0) or 1.0)
        if math.isclose(priority_weight, 0.0):
            priority_weight = 1.0
        rows.append(
            {
                "job_id": job_id,
                "arrival_time_min": float(job.arrival_time_min),
                "reveal_time_min": float(job.reveal_time_min),
                "completion_due_min": float(job.completion_due_min),
                "arrival_congestion_score": float(
                    getattr(job, "arrival_congestion_score", 0.0) or 0.0
                ),
                "priority_weight": priority_weight,
                "last_fixed_op_seq": last_fixed_op_seq,
                "last_fixed_end": last_fixed_end,
                "next_op_seq": int(next_op["op_seq"]),
                "dispatch_ready_min": dispatch_ready_min,
                "remaining_lb_min": remaining_lb_min,
                "dynamic_weighted_slack": (
                    float(job.completion_due_min) - dispatch_ready_min - remaining_lb_min
                )
                / priority_weight,
                "dynamic_reactive_urgency": (
                    float(job.completion_due_min) - float(current_time) - remaining_lb_min
                )
                / priority_weight,
            }
        )
    return pd.DataFrame(rows)


def build_visible_job_order(
    instance: dict[str, pd.DataFrame],
    method_name: str,
    current_time: float,
    fixed_rows: pd.DataFrame,
) -> list[str]:
    frame = _build_progress_frame(
        instance=instance,
        current_time=current_time,
        fixed_rows=fixed_rows,
    )
    if frame.empty:
        return []
    normalized_method = method_name.replace("_BUDGET", "")
    if normalized_method.startswith("M2_PERIODIC"):
        order = frame.sort_values(
            [
                "dynamic_weighted_slack",
                "priority_weight",
                "arrival_congestion_score",
                "dispatch_ready_min",
                "arrival_time_min",
                "job_id",
            ],
            ascending=[True, False, False, True, True, True],
        )
    elif normalized_method == "M3_EVENT_REACTIVE":
        order = frame.sort_values(
            [
                "dispatch_ready_min",
                "dynamic_reactive_urgency",
                "priority_weight",
                "arrival_congestion_score",
                "arrival_time_min",
                "job_id",
            ],
            ascending=[True, True, False, False, True, True],
        )
    else:
        raise ValueError(f"Unsupported online policy: {method_name}")
    return order["job_id"].astype(str).tolist()


def schedule_visible_jobs(
    instance_id: str,
    method_name: str,
    instance: dict[str, pd.DataFrame],
    current_time: float,
    job_order: list[str],
    fixed_rows: pd.DataFrame,
) -> pd.DataFrame:
    jobs = instance["jobs"].copy()
    ops = instance["operations"].copy().sort_values(["job_id", "op_seq"])
    eligible = instance["eligible"].copy()
    machines = instance["machines"].copy()

    fixed_rows = _normalize_schedule(fixed_rows, method_name=method_name)
    machine_available = {
        str(machine_id): max(float(start_min), float(current_time))
        for machine_id, start_min in zip(
            machines["machine_id"].astype(str),
            machines["availability_start_min"].astype(float),
        )
    }
    if not fixed_rows.empty:
        for row in fixed_rows.itertuples(index=False):
            machine_id = str(row.machine_id)
            machine_available[machine_id] = max(
                machine_available.get(machine_id, float(current_time)),
                float(row.end_min),
            )
    machine_windows = _build_machine_windows(instance=instance, blocked_rows=fixed_rows)
    fixed_summary = (
        fixed_rows.groupby("job_id", as_index=False)
        .agg(last_fixed_op_seq=("op_seq", "max"), last_fixed_end=("end_min", "max"))
        if not fixed_rows.empty
        else pd.DataFrame(columns=["job_id", "last_fixed_op_seq", "last_fixed_end"])
    )
    fixed_lookup = {
        str(row.job_id): (int(row.last_fixed_op_seq), float(row.last_fixed_end))
        for row in fixed_summary.itertuples(index=False)
    }
    scheduled_rows: list[dict[str, Any]] = fixed_rows.to_dict(orient="records")
    job_index = {job_id: idx for idx, job_id in enumerate(job_order)}

    for job_id in job_order:
        job_row = jobs.loc[jobs["job_id"].eq(job_id)].iloc[0]
        last_fixed_op_seq, last_fixed_end = fixed_lookup.get(job_id, (0, 0.0))
        ready_min = max(
            float(current_time),
            float(job_row["arrival_time_min"]),
            float(job_row["reveal_time_min"]),
            float(last_fixed_end),
        )
        job_ops = ops.loc[
            ops["job_id"].eq(job_id) & ops["op_seq"].gt(last_fixed_op_seq)
        ].sort_values("op_seq")
        for op_row in job_ops.itertuples(index=False):
            op_ready = max(ready_min, float(op_row.release_time_min), float(current_time))
            eligible_rows = eligible.loc[
                eligible["job_id"].eq(job_id) & eligible["op_seq"].eq(op_row.op_seq)
            ].copy()
            choices = []
            for elig_row in eligible_rows.itertuples(index=False):
                machine_id = str(elig_row.machine_id)
                start_min = _find_earliest_machine_slot(
                    machine_id=machine_id,
                    machine_available=machine_available,
                    machine_windows=machine_windows,
                    ready_min=op_ready,
                    duration_min=float(elig_row.proc_time_min),
                )
                if start_min is None:
                    continue
                end_min = start_min + float(elig_row.proc_time_min)
                choices.append(
                    (end_min, start_min, float(elig_row.proc_time_min), machine_id)
                )
            if not choices:
                raise RuntimeError(
                    f"No feasible machine choice for {instance_id} {job_id} op {op_row.op_seq}."
                )
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

    return _normalize_schedule(pd.DataFrame(scheduled_rows), method_name=method_name)


def _relevant_event_times(instance: dict[str, pd.DataFrame]) -> list[float]:
    events = instance["events"].copy()
    if events.empty:
        return []
    filtered = events.loc[events["event_type"].isin(RELEVANT_EVENT_TYPES), "event_time_min"]
    return sorted({float(value) for value in filtered.tolist()})


def _next_replan_time(
    method_name: str,
    current_time: float,
    delta_min: int | None,
    event_times: list[float],
) -> float | None:
    normalized_method = method_name.replace("_BUDGET", "")
    candidates: list[float] = []
    if normalized_method.startswith("M2_PERIODIC"):
        if delta_min is None:
            raise ValueError(f"Missing delta_min for periodic method {method_name}")
        candidates.append(float(current_time) + float(delta_min))
    elif normalized_method == "M3_EVENT_REACTIVE":
        for event_time in event_times:
            if event_time > float(current_time) + EPS:
                candidates.append(event_time)
                break
    else:
        raise ValueError(f"Unsupported online policy: {method_name}")
    return min(candidates) if candidates else None


def _freeze_started_rows(schedule: pd.DataFrame, cutoff_min: float) -> pd.DataFrame:
    if schedule.empty:
        return _empty_schedule()
    started = schedule.loc[schedule["start_min"].astype(float) < float(cutoff_min) - EPS].copy()
    return _normalize_schedule(started, method_name=str(schedule["policy_method"].iloc[0]))


def run_online_policy(
    instance_id: str,
    method_name: str,
    instance: dict[str, pd.DataFrame],
    delta_min: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    fixed_rows = _empty_schedule()
    current_time = 0.0
    total_ops = int(len(instance["operations"]))
    event_times = _relevant_event_times(instance)
    replan_count = 0

    while True:
        job_order = build_visible_job_order(
            instance=instance,
            method_name=method_name,
            current_time=current_time,
            fixed_rows=fixed_rows,
        )
        schedule = schedule_visible_jobs(
            instance_id=instance_id,
            method_name=method_name,
            instance=instance,
            current_time=current_time,
            job_order=job_order,
            fixed_rows=fixed_rows,
        )
        replan_count += 1
        next_replan = _next_replan_time(
            method_name=method_name,
            current_time=current_time,
            delta_min=delta_min,
            event_times=event_times,
        )
        if next_replan is None:
            fixed_rows = schedule.copy()
            break
        completion_min = (
            float(schedule["end_min"].max()) if not schedule.empty else float(current_time)
        )
        if completion_min <= float(next_replan) + EPS and len(schedule) == total_ops:
            fixed_rows = schedule.copy()
            break
        if completion_min <= float(next_replan) + EPS:
            fixed_rows = schedule.copy()
            current_time = float(next_replan)
            continue
        fixed_rows = _freeze_started_rows(schedule=schedule, cutoff_min=float(next_replan))
        current_time = float(next_replan)
        if len(fixed_rows) >= total_ops:
            break

    final_schedule = _normalize_schedule(fixed_rows, method_name=method_name)
    info = {
        "runtime_sec": float(time.perf_counter() - started),
        "solver_status": "heuristic_online_replay",
        "replan_count": int(replan_count),
    }
    return final_schedule, info
