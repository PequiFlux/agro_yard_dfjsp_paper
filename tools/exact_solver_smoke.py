#!/usr/bin/env python3
"""Budgeted exact-solver smoke tests for the observed benchmark release.

This module uses ``scipy.optimize.milp`` as a lightweight exact MIP backend
available in this environment. It is intentionally scoped as a smoke test:

- full flexible assignment
- continuous-time disjunctive non-overlap
- downtime avoidance
- makespan minimization

The goal is not to define the final benchmark protocol for the TCC, but to
show that the release is solver-informative:

- small instances load and solve to feasibility/optimality
- larger instances exhibit non-trivial gaps under a fixed time budget
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "gurobi") not in sys.path:
    sys.path.insert(0, str(ROOT / "gurobi"))

from load_instance import build_gurobi_views, load_catalog, load_instance  # noqa: E402


DEFAULT_CASES = [
    {"instance_id": "GO_XS_BALANCED_01", "time_limit_sec": 5.0, "max_jobs": 8},
    {"instance_id": "GO_S_BALANCED_01", "time_limit_sec": 5.0, "max_jobs": 12},
    {"instance_id": "GO_M_BALANCED_01", "time_limit_sec": 5.0, "max_jobs": 18},
    {"instance_id": "GO_L_BALANCED_01", "time_limit_sec": 5.0, "max_jobs": 24},
]


def _catalog_lookup(root: Path) -> pd.DataFrame:
    return pd.DataFrame(load_catalog(root))


def _status_label(result: Any) -> str:
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


def _restrict_raw_to_jobs(raw: dict[str, Any], max_jobs: int | None) -> dict[str, Any]:
    if max_jobs is None:
        return raw
    ordered_jobs = sorted(raw["jobs"], key=lambda row: (row["arrival_time_min"], row["job_id"]))[:max_jobs]
    keep_jobs = {row["job_id"] for row in ordered_jobs}
    restricted = dict(raw)
    restricted["jobs"] = [row for row in raw["jobs"] if row["job_id"] in keep_jobs]
    restricted["operations"] = [row for row in raw["operations"] if row["job_id"] in keep_jobs]
    restricted["precedences"] = [row for row in raw["precedences"] if row["job_id"] in keep_jobs]
    restricted["eligible_machines"] = [row for row in raw["eligible_machines"] if row["job_id"] in keep_jobs]
    keep_machines = {row["machine_id"] for row in restricted["eligible_machines"]}
    restricted["machines"] = [row for row in raw["machines"] if row["machine_id"] in keep_machines]
    restricted["machine_downtimes"] = [
        row for row in raw["machine_downtimes"] if row["machine_id"] in keep_machines
    ]
    restricted["events"] = [
        row
        for row in raw["events"]
        if (row.get("entity_id") in keep_jobs) or (row.get("entity_id") in keep_machines)
    ]
    return restricted


def _build_instance_model(instance_dir: Path, max_jobs: int | None = None) -> tuple[np.ndarray, np.ndarray, Bounds, list[tuple[str, int]], dict[str, Any]]:
    raw = _restrict_raw_to_jobs(load_instance(instance_dir), max_jobs=max_jobs)
    data = build_gurobi_views(raw)
    ops_release = {
        (row["job_id"], row["op_seq"]): int(row["release_time_min"])
        for row in raw["operations"]
    }
    lag_by_arc = {
        (row["job_id"], row["pred_op_seq"], row["succ_op_seq"]): int(row["min_lag_min"])
        for row in raw["precedences"]
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
    for idx in x_idx.values():
        ub[idx] = 1.0
        integrality[idx] = 1
    for idx in y_idx.values():
        ub[idx] = 1.0
        integrality[idx] = 1
    for idx in q_idx.values():
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
        add_constraint(
            {x_idx[(j, o, m)]: 1.0 for m in data["ELIGIBLE_BY_OP"][(j, o)]},
            1.0,
            1.0,
        )

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
        add_constraint(
            {
                s_idx[(j_b, o_b)]: 1.0,
                c_idx[(j_a, o_a)]: -1.0,
                y_var: -big_m,
                x_a: -big_m,
                x_b: -big_m,
            },
            -3.0 * big_m,
            np.inf,
        )
        add_constraint(
            {
                s_idx[(j_a, o_a)]: 1.0,
                c_idx[(j_b, o_b)]: -1.0,
                y_var: big_m,
                x_a: -big_m,
                x_b: -big_m,
            },
            -2.0 * big_m,
            np.inf,
        )

    for (j, o, machine_id, dt_idx), q_var in q_idx.items():
        down_start, down_end, _ = data["DOWNTIMES_BY_MACHINE"][machine_id][dt_idx]
        x_var = x_idx[(j, o, machine_id)]
        add_constraint(
            {c_idx[(j, o)]: 1.0, q_var: big_m, x_var: -big_m},
            -np.inf,
            float(down_start + big_m),
        )
        add_constraint(
            {s_idx[(j, o)]: 1.0, q_var: big_m, x_var: -big_m},
            float(down_end - big_m),
            np.inf,
        )

    # HiGHS in the SciPy build used here expects 32-bit sparse indices.
    matrix = coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(row_idx, dtype=np.int32),
                np.asarray(col_idx, dtype=np.int32),
            ),
        ),
        shape=(row, n_vars),
    ).tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    metadata = {
        "instance_id": data["params"]["instance_id"],
        "subset_job_count": len(data["J"]),
        "job_count": len(data["J"]),
        "op_count": len(ops),
        "eligible_var_count": len(x_idx),
        "machine_pair_binary_count": len(y_idx),
        "downtime_binary_count": len(q_idx),
        "constraint_count": row,
        "big_m": big_m,
    }
    return c, integrality, Bounds(lb, ub), [LinearConstraint(matrix, np.array(lower), np.array(upper))], metadata


def solve_instance(instance_id: str, time_limit_sec: float, root: Path = ROOT, max_jobs: int | None = None) -> dict[str, Any]:
    instance_dir = root / "instances" / instance_id
    c, integrality, bounds, constraints, meta = _build_instance_model(instance_dir, max_jobs=max_jobs)
    started = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit_sec), "disp": False},
    )
    wall_time = time.perf_counter() - started

    catalog = _catalog_lookup(root)
    catalog_row = catalog.loc[catalog["instance_id"] == instance_id].iloc[0]
    objective = float(result.fun) if getattr(result, "fun", None) is not None else np.nan
    dual_bound = float(result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None else np.nan
    gap = float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else np.nan
    node_count = getattr(result, "mip_node_count", 0)
    has_solution = np.isfinite(objective)

    return {
        "instance_id": instance_id,
        "scale_code": catalog_row["scale_code"],
        "regime_code": catalog_row["regime_code"],
        "recommended_solver_track": catalog_row.get("recommended_solver_track", ""),
        "time_limit_sec": float(time_limit_sec),
        "max_jobs": int(max_jobs) if max_jobs is not None else int(meta["job_count"]),
        "status_code": int(result.status),
        "status_label": _status_label(result),
        "success": bool(result.success),
        "has_solution": bool(has_solution),
        "objective_makespan_min": objective,
        "dual_bound_makespan_min": dual_bound,
        "mip_gap": gap,
        "wall_time_sec": float(wall_time),
        "mip_node_count": int(node_count or 0),
        "job_count": meta["job_count"],
        "op_count": meta["op_count"],
        "eligible_var_count": meta["eligible_var_count"],
        "machine_pair_binary_count": meta["machine_pair_binary_count"],
        "downtime_binary_count": meta["downtime_binary_count"],
        "constraint_count": meta["constraint_count"],
    }


def run_smoke_suite(cases: list[dict[str, Any]] | None = None, root: Path = ROOT) -> pd.DataFrame:
    cases = cases or DEFAULT_CASES
    rows = [
        solve_instance(
            case["instance_id"],
            case["time_limit_sec"],
            root=root,
            max_jobs=case.get("max_jobs"),
        )
        for case in cases
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_smoke_suite()
    print(df.to_string(index=False))
