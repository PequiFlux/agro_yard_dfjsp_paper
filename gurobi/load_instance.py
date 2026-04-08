#!/usr/bin/env python3
"""
Gurobi-oriented loader for the Agro Yard D-FJSP GO Benchmark.

This module intentionally keeps the raw files normalized and converts them
to tuple-indexed Python dictionaries that are convenient for gurobipy models.
It uses only the Python standard library.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_INT_FIELDS = {
    "machines.csv": {"availability_start_min", "availability_end_min", "setup_min", "rate_tph"},
    "jobs.csv": {"load_tons", "arrival_time_min", "reveal_time_min", "appointment_flag",
                 "statutory_wait_limit_min", "completion_due_min"},
    "operations.csv": {"op_seq", "mandatory", "release_time_min"},
    "precedences.csv": {"pred_op_seq", "succ_op_seq", "min_lag_min"},
    "eligible_machines.csv": {"op_seq", "proc_time_min", "setup_included"},
    "machine_downtimes.csv": {"start_min", "end_min"},
    "events.csv": {"event_time_min"},
    "fifo_schedule.csv": {"op_seq", "start_min", "end_min", "wait_before_stage_min"},
    "fifo_job_metrics.csv": {"completion_min", "flow_time_min", "queue_time_min", "overwait_min"},
}
_FLOAT_FIELDS = {
    "machines.csv": {"speed_factor"},
    "jobs.csv": {"priority_weight", "overwait_cost_rs_per_min"},
}


def _cast_value(file_name: str, field: str, value: str) -> Any:
    if value == "":
        return ""
    if field in _INT_FIELDS.get(file_name, set()):
        return int(value)
    if field in _FLOAT_FIELDS.get(file_name, set()):
        return float(value)
    return value


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    file_name = path.name
    return [{k: _cast_value(file_name, k, v) for k, v in row.items()} for row in rows]


def list_instances(root_dir: Path | str) -> List[str]:
    root = Path(root_dir)
    instances_dir = root / "instances"
    if not instances_dir.exists():
        return []
    return sorted(p.name for p in instances_dir.iterdir() if p.is_dir())


def load_catalog(root_dir: Path | str) -> List[Dict[str, Any]]:
    root = Path(root_dir)
    return _read_csv(root / "catalog" / "benchmark_catalog.csv")


def load_instance(instance_dir: Path | str) -> Dict[str, Any]:
    p = Path(instance_dir)
    if p.is_dir() and (p / "params.json").exists():
        instance_path = p
    else:
        raise FileNotFoundError(f"Could not find instance directory with params.json at: {p}")

    data = {}
    with (instance_path / "params.json").open("r", encoding="utf-8") as f:
        data["params"] = json.load(f)

    for name in [
        "machines.csv",
        "jobs.csv",
        "operations.csv",
        "precedences.csv",
        "eligible_machines.csv",
        "machine_downtimes.csv",
        "events.csv",
        "fifo_schedule.csv",
        "fifo_job_metrics.csv",
    ]:
        data[name[:-4]] = _read_csv(instance_path / name)

    with (instance_path / "fifo_summary.json").open("r", encoding="utf-8") as f:
        data["fifo_summary"] = json.load(f)

    return data


def build_gurobi_views(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw instance into Gurobi-friendly sets and coefficient dictionaries.

    Returned keys are named to work naturally with gurobipy.Model.addVars
    and tuple-indexed dictionaries:

    J                   list[str]                 jobs
    OPS                 list[tuple[str,int]]      (job, op_seq)
    M                   list[str]                 machines
    ELIGIBLE_KEYS       list[tuple[str,int,str]]  (job, op_seq, machine)
    PROC                dict[(j,o,m)] -> int      processing time
    RELEASE             dict[j] -> int            arrival time
    REVEAL              dict[j] -> int            reveal time
    DUE                 dict[j] -> int            soft due / service target
    PRIORITY_W          dict[j] -> float          priority weight
    OVERWAIT_COST       dict[j] -> float          BRL per minute over 300-min wait
    LOAD_TONS           dict[j] -> int            load in tons
    STAGE               dict[(j,o)] -> str        stage name
    FAMILY              dict[(j,o)] -> str        required machine family
    PRED                list[(j,pred,succ)]       precedence arcs
    MACHINES_BY_FAMILY  dict[str] -> list[str]
    ELIGIBLE_BY_OP      dict[(j,o)] -> list[str]
    DOWNTIMES_BY_MACHINE dict[str] -> list[(start,end,reason)]
    """
    jobs = data["jobs"]
    operations = data["operations"]
    machines = data["machines"]
    eligible = data["eligible_machines"]
    precedences = data["precedences"]
    downtimes = data["machine_downtimes"]

    J = [row["job_id"] for row in jobs]
    OPS = [(row["job_id"], row["op_seq"]) for row in operations]
    M = [row["machine_id"] for row in machines]
    ELIGIBLE_KEYS = [(row["job_id"], row["op_seq"], row["machine_id"]) for row in eligible]

    PROC = {(row["job_id"], row["op_seq"], row["machine_id"]): row["proc_time_min"] for row in eligible}
    RELEASE = {row["job_id"]: row["arrival_time_min"] for row in jobs}
    REVEAL = {row["job_id"]: row["reveal_time_min"] for row in jobs}
    DUE = {row["job_id"]: row["completion_due_min"] for row in jobs}
    PRIORITY_W = {row["job_id"]: row["priority_weight"] for row in jobs}
    OVERWAIT_COST = {row["job_id"]: row["overwait_cost_rs_per_min"] for row in jobs}
    LOAD_TONS = {row["job_id"]: row["load_tons"] for row in jobs}
    STAGE = {(row["job_id"], row["op_seq"]): row["stage_name"] for row in operations}
    FAMILY = {(row["job_id"], row["op_seq"]): row["machine_family"] for row in operations}
    PRED = [(row["job_id"], row["pred_op_seq"], row["succ_op_seq"]) for row in precedences]

    MACHINES_BY_FAMILY = defaultdict(list)
    for row in machines:
        MACHINES_BY_FAMILY[row["machine_family"]].append(row["machine_id"])
    MACHINES_BY_FAMILY = dict(MACHINES_BY_FAMILY)

    ELIGIBLE_BY_OP = defaultdict(list)
    for row in eligible:
        ELIGIBLE_BY_OP[(row["job_id"], row["op_seq"])].append(row["machine_id"])
    ELIGIBLE_BY_OP = dict(ELIGIBLE_BY_OP)

    DOWNTIMES_BY_MACHINE = defaultdict(list)
    for row in downtimes:
        DOWNTIMES_BY_MACHINE[row["machine_id"]].append((row["start_min"], row["end_min"], row["reason"]))
    for machine_id in list(DOWNTIMES_BY_MACHINE):
        DOWNTIMES_BY_MACHINE[machine_id].sort(key=lambda x: (x[0], x[1]))
    DOWNTIMES_BY_MACHINE = dict(DOWNTIMES_BY_MACHINE)

    return {
        "params": data["params"],
        "J": J,
        "OPS": OPS,
        "M": M,
        "ELIGIBLE_KEYS": ELIGIBLE_KEYS,
        "PROC": PROC,
        "RELEASE": RELEASE,
        "REVEAL": REVEAL,
        "DUE": DUE,
        "PRIORITY_W": PRIORITY_W,
        "OVERWAIT_COST": OVERWAIT_COST,
        "LOAD_TONS": LOAD_TONS,
        "STAGE": STAGE,
        "FAMILY": FAMILY,
        "PRED": PRED,
        "MACHINES_BY_FAMILY": MACHINES_BY_FAMILY,
        "ELIGIBLE_BY_OP": ELIGIBLE_BY_OP,
        "DOWNTIMES_BY_MACHINE": DOWNTIMES_BY_MACHINE,
    }


def summarize_instance(data: Dict[str, Any]) -> Dict[str, Any]:
    jobs = data["jobs"]
    machines = data["machines"]
    return {
        "instance_id": data["params"]["instance_id"],
        "n_jobs": len(jobs),
        "n_machines": len(machines),
        "n_eligible_rows": len(data["eligible_machines"]),
        "n_downtimes": len(data["machine_downtimes"]),
        "fifo_summary": data["fifo_summary"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and summarize one benchmark instance.")
    parser.add_argument("instance_dir", type=Path, help="Path to an instance directory, e.g. instances/GO_XS_BALANCED_01")
    args = parser.parse_args()
    raw = load_instance(args.instance_dir)
    summary = summarize_instance(raw)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
