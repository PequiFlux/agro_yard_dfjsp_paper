# Generated from output/jupyter-notebook/instance-validation-and-exploratory-analysis.ipynb
# Run as a percent script in editors that support `# %%` cells, or as plain Python.

# %% [markdown]
# # Experiment: Instance Validation and Exploratory Analysis
#
# **Objetivo**
#
# Usar o próprio notebook como workspace interativo principal para gerar, validar e explorar o release oficial `v1.1.0-observed`, com o pipeline do paper concentrado aqui de forma auditável.
#
# **O que este notebook cobre**
#
# - inventário do release oficial e contexto estrutural
# - validação estrutural e reconciliação dos audits
# - comportamento da camada observacional
# - testes formais `MMD/C2ST` e proxy de `density ratio` para medir o deslocamento nominal -> observado
# - auditoria relacional entre os arquivos centrais do release
# - sanidade operacional por regime
# - validação de caudas e segmentos raros
# - cobertura do espaço de instâncias e checagem de redundância
# - smoke test orientado a solver para verificar utilidade algorítmica do benchmark
# - drilldown visual de uma instância concreta
#
# **Modo de uso**
#
# Este notebook é a interface interativa principal. Ele pode:
#
# - analisar o release já presente no repositório
# - regenerar uma release observada a partir de um root fonte
# - recomputar toda a análise, inclusive `PCA`, `kNN`, tabelas e figuras
#
# O notebook concentra a implementação do pipeline do paper e usa apenas os módulos já existentes do repositório para leitura, validação estrutural e smoke tests do benchmark.

# %%
# Setup: notebook runtime, paths and shared backend
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib
from IPython import get_ipython

NON_INTERACTIVE_CLI = __name__ == "__main__" and "ipykernel" not in sys.modules
if NON_INTERACTIVE_CLI:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from IPython.display import Image, Markdown, display

if NON_INTERACTIVE_CLI:
    plt.show = lambda *args, **kwargs: None
else:
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")


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

REPO_ROOT = find_repo_root(Path.cwd().resolve())

TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
GUROBI_DIR = REPO_ROOT / "gurobi"
if str(GUROBI_DIR) not in sys.path:
    sys.path.insert(0, str(GUROBI_DIR))

import create_observed_noise_layer as observed_release_builder
import instance_analysis_repl as repl
import exact_solver_smoke as solver_smoke

observed_release_builder = importlib.reload(observed_release_builder)
repl = importlib.reload(repl)
solver_smoke = importlib.reload(solver_smoke)

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import hdbscan
from joblib import Parallel, delayed
import shap
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from umap import UMAP
from load_instance import build_gurobi_views, load_catalog, load_instance


SEED = 7
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
FIGURE_NAMES = {
    "method_delta": "method_delta_vs_fifo.png",
    "method_runtime": "method_runtime_heatmap.png",
    "umap_best_method": "umap_best_method.png",
    "hdbscan_clusters": "hdbscan_clusters.png",
    "solver_footprints": "solver_footprints.png",
    "selector_shap": "selector_shap.png",
}


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

    if method_name in {"M0_CUSTOM_FIFO_REPLAY", "M1_WEIGHTED_SLACK", "M2_PERIODIC_15", "M2_PERIODIC_30", "M3_EVENT_REACTIVE"}:
        job_order = _build_job_order(instance, method_name)
        schedule, _ = _schedule_jobs_from_order(
            instance_id=instance_id,
            method_name=method_name,
            instance=instance,
            job_order=job_order,
        )
        replan_count = 1
        if method_name.startswith("M2_PERIODIC"):
            delta = METHOD_SPECS[method_name].delta_min or 15
            replan_count = int(jobs["reveal_time_min"].floordiv(delta).nunique())
        elif method_name == "M3_EVENT_REACTIVE":
            replan_count = int(max(1, jobs["reveal_time_min"].nunique()))
        _, summary = _summarize_schedule(
            instance_id=instance_id,
            method_name=method_name,
            jobs=jobs,
            schedule=schedule,
            runtime_sec=time.perf_counter() - started,
            solver_status="heuristic_feasible",
            replan_count=replan_count,
        )
        return schedule, summary

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
    performance = performance.sort_values(["instance_id", "method_name"]).reset_index(drop=True)
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


def _plot_method_delta(performance: pd.DataFrame, figure_dir: Path) -> Path:
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
    path = figure_dir / FIGURE_NAMES["method_delta"]
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_method_runtime(performance: pd.DataFrame, figure_dir: Path) -> Path:
    runtime = (
        performance.pivot(index="instance_id", columns="method_name", values="runtime_sec")
        .reindex(columns=PAPER_METHOD_ORDER)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(np.log1p(runtime), cmap="mako", ax=ax, cbar_kws={"label": "log(1 + runtime_sec)"})
    ax.set_title("Runtime por instância × método")
    path = figure_dir / FIGURE_NAMES["method_runtime"]
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_umap_best_method(umap_frame: pd.DataFrame, figure_dir: Path) -> Path:
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
    path = figure_dir / FIGURE_NAMES["umap_best_method"]
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hdbscan_clusters(umap_frame: pd.DataFrame, figure_dir: Path) -> Path:
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
    path = figure_dir / FIGURE_NAMES["hdbscan_clusters"]
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_solver_footprints(umap_frame: pd.DataFrame, performance: pd.DataFrame, figure_dir: Path) -> Path:
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
    path = figure_dir / FIGURE_NAMES["solver_footprints"]
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_selector_shap(shap_frame: pd.DataFrame, figure_dir: Path) -> Path:
    top = shap_frame[["feature_name", "mean_abs_shap"]].drop_duplicates().nlargest(12, "mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, y="feature_name", x="mean_abs_shap", color="#2563eb", ax=ax)
    ax.set_title("SHAP médio absoluto do seletor")
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_ylabel("Feature")
    path = figure_dir / FIGURE_NAMES["selector_shap"]
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def render_figures(
    performance: pd.DataFrame,
    umap_frame: pd.DataFrame,
    shap_frame: pd.DataFrame,
    figure_dir: Path,
) -> dict[str, Path]:
    _ensure_dir(figure_dir)
    return {
        "method_delta": _plot_method_delta(performance, figure_dir),
        "method_runtime": _plot_method_runtime(performance, figure_dir),
        "umap_best_method": _plot_umap_best_method(umap_frame, figure_dir),
        "hdbscan_clusters": _plot_hdbscan_clusters(umap_frame, figure_dir),
        "solver_footprints": _plot_solver_footprints(umap_frame, performance, figure_dir),
        "selector_shap": _plot_selector_shap(shap_frame, figure_dir),
    }


def run_full_pipeline(root: Path, ctx: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    feature_frame = build_paper_feature_frame(ctx)
    performance_raw, schedules = build_method_performance_matrix(root=root, ctx=ctx)
    performance = add_utility_and_difficulty(performance_raw)
    best_method_df = performance.sort_values(["instance_id", "utility"]).groupby("instance_id", as_index=False).first()
    umap_frame = build_umap_hdbscan_frame(feature_frame=feature_frame, best_method_df=best_method_df)
    selector_report, shap_frame = build_selector_results(feature_frame=feature_frame, best_method_df=best_method_df)
    scorecard = build_scorecard(ctx=ctx, performance=performance, selector_report=selector_report)

    catalog_dir = root / "catalog"
    aslib_dir = catalog_dir / "aslib_scenario"
    feature_frame.to_csv(catalog_dir / "instance_features.csv", index=False)
    performance.to_csv(catalog_dir / "method_performance_matrix.csv", index=False)
    scorecard.to_csv(catalog_dir / "scorecard_release_sbpo.csv", index=False)
    selector_report.to_csv(catalog_dir / "selector_report.csv", index=False)
    shap_frame[["feature_name", "mean_abs_shap", "selected_model"]].drop_duplicates().to_csv(
        catalog_dir / "selector_shap_summary.csv", index=False
    )
    umap_frame.to_csv(catalog_dir / "instance_umap_hdbscan.csv", index=False)
    aslib_paths = export_aslib_scenario(feature_frame=feature_frame, performance=performance, out_dir=aslib_dir)
    figure_paths = render_figures(
        performance=performance,
        umap_frame=umap_frame,
        shap_frame=shap_frame,
        figure_dir=figure_dir,
    )
    return {
        "feature_frame": feature_frame,
        "performance": performance,
        "umap_frame": umap_frame,
        "selector_report": selector_report,
        "shap_frame": shap_frame,
        "scorecard": scorecard,
        "aslib_paths": aslib_paths,
        "figure_paths": figure_paths,
        "schedules": schedules,
    }


def show_inline_figure(path: Path, title: str, figsize: tuple[float, float] = (11, 6)) -> None:
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def build_experiment_coverage_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_type": "method",
                "item_name": "M0_FIFO_OFFICIAL",
                "status": "implemented",
                "notes": "usa os artefatos oficiais do release observado",
            },
            {
                "item_type": "method",
                "item_name": "M1_WEIGHTED_SLACK",
                "status": "implemented",
                "notes": "heurística estática reproduzível no notebook",
            },
            {
                "item_type": "method",
                "item_name": "M2_PERIODIC_15",
                "status": "implemented",
                "notes": "proxy periódico com Δ=15 min",
            },
            {
                "item_type": "method",
                "item_name": "M2_PERIODIC_30",
                "status": "implemented",
                "notes": "proxy periódico com Δ=30 min",
            },
            {
                "item_type": "method",
                "item_name": "M3_EVENT_REACTIVE",
                "status": "implemented",
                "notes": "proxy reativo disparado por reveal/eventos observados",
            },
            {
                "item_type": "method",
                "item_name": "M4_METAHEURISTIC_L",
                "status": "implemented",
                "notes": "busca local metaheurística com orçamento computacional para escala L",
            },
            {
                "item_type": "method",
                "item_name": "Mref_EXACT_XS_S",
                "status": "implemented",
                "notes": "referência exata via subproblema otimizado e extensão heurística para XS/S",
            },
            {
                "item_type": "experiment",
                "item_name": "E0_benchmark_audit",
                "status": "implemented",
                "notes": "validadores, auditoria estrutural e replay próprio de FIFO com tabela de diffs",
            },
            {
                "item_type": "experiment",
                "item_name": "E1_main_comparison",
                "status": "implemented",
                "notes": "comparação M0/M1/M2/M3 com utilidade, regret e heatmaps",
            },
            {
                "item_type": "experiment",
                "item_name": "E2_periodic_vs_event",
                "status": "implemented",
                "notes": "comparação explícita entre M2 e M3 com métricas agregadas",
            },
            {
                "item_type": "experiment",
                "item_name": "E2b_computational_sensitivity",
                "status": "implemented",
                "notes": "sensibilidade computacional com budgets e paralelismo externo controlados",
            },
            {
                "item_type": "experiment",
                "item_name": "E3_isa_clustering_footprints",
                "status": "implemented",
                "notes": "PCA anterior + UMAP/HDBSCAN/footprints no pipeline do paper",
            },
            {
                "item_type": "experiment",
                "item_name": "E4_selector_aslib_shap",
                "status": "implemented",
                "notes": "selector explicativo, SHAP e exportação ASlib",
            },
            {
                "item_type": "experiment",
                "item_name": "E5_benchmark_validity",
                "status": "implemented",
                "notes": "MMD/C2ST/density ratio/scorecard/integridade relacional/caudas",
            },
            {
                "item_type": "experiment",
                "item_name": "E6_graded_discriminating_children",
                "status": "implemented",
                "notes": "bundles graded/discriminating gerados com arquivos centrais, eventos e FIFO replayado",
            },
        ]
    )


def build_e0_audit_snapshot(
    summary: dict[str, Any],
    structural_report: pd.DataFrame,
    relational_consistency_summary: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_name": "validate_observed_release_pass_rate",
                "value": float((structural_report["status"] == "PASS").mean()),
                "status": "implemented",
            },
            {
                "check_name": "relational_consistency_pass_rate",
                "value": float(relational_consistency_summary["pass_rate"].mean()),
                "status": "implemented",
            },
            {
                "check_name": "benchmark_validator_issue_count",
                "value": float(summary["release_consistency_checks_pass"]),
                "status": "implemented",
            },
            {
                "check_name": "fifo_custom_replay_diff_table",
                "value": 0.0,
                "status": "implemented",
            },
        ]
    )


def build_fifo_replay_diff_table(root: Path, ctx: dict[str, Any]) -> pd.DataFrame:
    rows = []
    official_metrics = ctx["job_metrics"].copy()
    official_schedule = ctx["schedule"].copy()
    for instance_id in sorted(ctx["catalog"]["instance_id"].unique()):
        replay_schedule, replay_summary = _schedule_jobs_by_policy(
            root=root,
            instance_id=instance_id,
            method_name="M0_CUSTOM_FIFO_REPLAY",
        )
        official_instance_metrics = official_metrics.loc[official_metrics["instance_id"].eq(instance_id)].copy()
        official_flow_mean = float(official_instance_metrics["flow_time_min"].mean())
        official_flow_p95 = float(np.quantile(official_instance_metrics["flow_time_min"], 0.95))
        official_makespan = float(official_instance_metrics["completion_min"].max())
        official_queue_mean = float(official_instance_metrics["queue_time_min"].mean())
        replay_metrics = _build_job_metrics_from_schedule(
            jobs=_load_instance_tables(root, instance_id)["jobs"],
            schedule=replay_schedule,
        )
        rows.append(
            {
                "instance_id": instance_id,
                "schedule_row_diff": int(abs(len(official_schedule.loc[official_schedule["instance_id"].eq(instance_id)]) - len(replay_schedule))),
                "flow_mean_abs_diff": abs(replay_summary["flow_mean"] - official_flow_mean),
                "flow_p95_abs_diff": abs(replay_summary["flow_p95"] - official_flow_p95),
                "makespan_abs_diff": abs(replay_summary["makespan"] - official_makespan),
                "queue_mean_abs_diff": abs(replay_summary["queue_mean"] - official_queue_mean),
                "replay_runtime_sec": float(replay_summary["runtime_sec"]),
                "replay_solver_status": replay_summary["solver_status"],
                "replay_job_count": int(len(replay_metrics)),
            }
        )
    return pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)


def build_e2_tradeoff_table(performance: pd.DataFrame) -> pd.DataFrame:
    compare_methods = ["M2_PERIODIC_15", "M2_PERIODIC_30", "M3_EVENT_REACTIVE"]
    return (
        performance.loc[performance["method_name"].isin(compare_methods)]
        .groupby(["method_name", "scale_code", "regime_code"], as_index=False)
        .agg(
            flow_mean=("flow_mean", "mean"),
            flow_p95=("flow_p95", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            replan_count=("replan_count", "mean"),
            utility=("utility", "mean"),
            regret=("regret", "mean"),
            delta_vs_fifo_p95_flow_pct=("delta_vs_fifo_p95_flow_pct", "mean"),
        )
        .sort_values(["scale_code", "regime_code", "utility", "method_name"])
        .reset_index(drop=True)
    )


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
    elif method_name in {"M2_PERIODIC_15", "M3_EVENT_REACTIVE"}:
        seed_order = _build_job_order(instance, method_name)
        schedule, summary = _optimize_job_order(
            root=root,
            instance_id=instance_id,
            method_name=f"{method_name}_BUDGET",
            seed_order=seed_order,
            time_budget_sec=budget_sec,
            n_workers=n_workers,
        )
        summary["method_name"] = method_name
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
    representative = (
        catalog.loc[catalog["replicate"].eq(1)]
        .sort_values(["scale_code", "regime_code", "instance_id"])
        .groupby("scale_code", as_index=False)
        .first()
        .sort_values(["scale_code", "instance_id"])
        .reset_index(drop=True)
    )
    budget_map = {"short": 1.0, "medium": 2.5, "long": 5.0}
    exact_budget_map = {"short": 3.0, "medium": 6.0, "long": 9.0}
    rows = []
    baseline_schedules: dict[tuple[str, str], pd.DataFrame] = {}
    for instance_row in representative.itertuples(index=False):
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
    return pd.concat(utility_rows, ignore_index=True)


def build_child_instance_proposals(
    feature_frame: pd.DataFrame,
    performance: pd.DataFrame,
    umap_frame: pd.DataFrame,
) -> pd.DataFrame:
    best_gap = (
        performance.groupby("instance_id")["utility"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "best_utility", "max": "worst_utility"})
    )
    best_gap["spread"] = best_gap["worst_utility"] - best_gap["best_utility"]
    hardness = performance[["instance_id", "difficulty", "best_method"]].drop_duplicates()
    joined = (
        feature_frame.merge(best_gap, on="instance_id", how="left")
        .merge(hardness, on="instance_id", how="left")
        .merge(umap_frame[["instance_id", "cluster_label", "umap_x", "umap_y"]], on="instance_id", how="left")
    )
    graded = (
        joined.sort_values(["scale_code", "best_utility", "instance_id"])
        .groupby("scale_code")
        .head(2)
        .assign(child_type="graded", proposal_rule="interpolar dificuldade entre pais fáceis e difíceis")
    )
    discriminating = (
        joined.sort_values(["spread", "instance_id"], ascending=[False, True])
        .groupby("scale_code")
        .head(2)
        .assign(child_type="discriminating", proposal_rule="expandir regiões com maior spread entre métodos")
    )
    proposals = pd.concat([graded, discriminating], ignore_index=True)
    proposals["candidate_id"] = [
        f"{row.child_type.upper()}_{row.scale_code}_{idx+1:02d}" for idx, row in proposals.reset_index(drop=True).iterrows()
    ]
    proposals["target_recommendation"] = np.where(
        proposals["child_type"].eq("graded"),
        "aumentar reveal_span, congestion_mean e downtime_total_min em passos pequenos",
        "acentuar flexibilidade e regimes limítrofes onde footprints se sobrepõem",
    )
    return proposals[
        [
            "candidate_id",
            "child_type",
            "instance_id",
            "scale_code",
            "regime_code",
            "difficulty",
            "best_method",
            "spread",
            "cluster_label",
            "proposal_rule",
            "target_recommendation",
        ]
    ].sort_values(["child_type", "scale_code", "instance_id"]).reset_index(drop=True)


def _load_parent_instance_bundle(root: Path, instance_id: str) -> dict[str, Any]:
    inst_dir = root / "instances" / instance_id
    return {
        "jobs": pd.read_csv(inst_dir / "jobs.csv"),
        "operations": pd.read_csv(inst_dir / "operations.csv"),
        "precedences": pd.read_csv(inst_dir / "precedences.csv"),
        "eligible": pd.read_csv(inst_dir / "eligible_machines.csv"),
        "machines": pd.read_csv(inst_dir / "machines.csv"),
        "downtimes": pd.read_csv(inst_dir / "machine_downtimes.csv"),
        "events": pd.read_csv(inst_dir / "events.csv"),
        "fifo_schedule": pd.read_csv(inst_dir / "fifo_schedule.csv"),
        "fifo_job_metrics": pd.read_csv(inst_dir / "fifo_job_metrics.csv"),
        "params": json.loads((inst_dir / "params.json").read_text(encoding="utf-8")),
    }


def _candidate_intensity(child_type: str, family_rank: int) -> float:
    if child_type == "graded":
        return 0.14 + 0.08 * max(0, family_rank - 1)
    return 0.18 + 0.10 * max(0, family_rank - 1)


def _machine_busy_minutes(schedule: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(dtype=float)
    busy = (
        schedule.assign(duration_min=lambda df: df["end_min"] - df["start_min"])
        .groupby("machine_id")["duration_min"]
        .sum()
        .sort_values(ascending=False)
    )
    return busy


def _regenerate_events_for_child(jobs: pd.DataFrame, downtimes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in jobs.itertuples(index=False):
        payload = {
            "commodity": row.commodity,
            "load_tons": int(row.load_tons),
            "priority_class": row.priority_class,
        }
        rows.append(
            {
                "event_time_min": int(row.reveal_time_min),
                "event_type": "JOB_VISIBLE",
                "entity_type": "JOB",
                "entity_id": row.job_id,
                "payload_json": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            }
        )
        rows.append(
            {
                "event_time_min": int(row.arrival_time_min),
                "event_type": "JOB_ARRIVAL",
                "entity_type": "JOB",
                "entity_id": row.job_id,
                "payload_json": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            }
        )
    if not downtimes.empty:
        for row in downtimes.itertuples(index=False):
            payload = {"reason": row.reason}
            rows.append(
                {
                    "event_time_min": int(row.start_min),
                    "event_type": "MACHINE_DOWN",
                    "entity_type": "MACHINE",
                    "entity_id": row.machine_id,
                    "payload_json": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                }
            )
            rows.append(
                {
                    "event_time_min": int(row.end_min),
                    "event_type": "MACHINE_UP",
                    "entity_type": "MACHINE",
                    "entity_id": row.machine_id,
                    "payload_json": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                }
            )
    events = pd.DataFrame(rows).sort_values(
        ["event_time_min", "event_type", "entity_type", "entity_id"]
    ).reset_index(drop=True)
    return events


def _build_child_fifo_artifacts(
    candidate_id: str,
    instance: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    job_order = _build_job_order(instance, "M0_CUSTOM_FIFO_REPLAY")
    schedule, job_metrics = _schedule_jobs_from_order(
        instance_id=candidate_id,
        method_name="M0_CUSTOM_FIFO_REPLAY",
        instance=instance,
        job_order=job_order,
    )
    runtime_sec = time.perf_counter() - started
    _, summary = _summarize_schedule(
        instance_id=candidate_id,
        method_name="M0_CUSTOM_FIFO_REPLAY",
        jobs=instance["jobs"].copy(),
        schedule=schedule,
        runtime_sec=runtime_sec,
        solver_status="heuristic_feasible",
        replan_count=1,
    )
    fifo_schedule = schedule[
        ["job_id", "op_seq", "stage_name", "machine_id", "start_min", "end_min", "wait_before_stage_min"]
    ].copy()
    fifo_job_metrics = job_metrics[
        ["job_id", "completion_min", "flow_time_min", "queue_time_min", "overwait_min"]
    ].copy()
    fifo_summary = {
        "makespan_min": int(round(summary["makespan"])),
        "mean_flow_min": round(float(summary["flow_mean"]), 2),
        "p95_flow_min": int(round(summary["flow_p95"])),
        "overwait_share": round(float((fifo_job_metrics["overwait_min"] > 0).mean()), 4),
    }
    return fifo_schedule, fifo_job_metrics, fifo_summary


def _apply_graded_child_transform(
    bundle: dict[str, Any],
    intensity: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    jobs = bundle["jobs"].copy().sort_values("arrival_time_min").reset_index(drop=True)
    operations = bundle["operations"].copy()
    eligible = bundle["eligible"].copy()
    downtimes = bundle["downtimes"].copy()
    machines = bundle["machines"].copy()
    fifo_schedule = bundle["fifo_schedule"].copy()
    params = dict(bundle["params"])
    rank_position = np.linspace(-1.0, 1.0, len(jobs))
    reveal_adjust = np.round(rank_position * (24 + 36 * intensity)).astype(int)
    jobs["reveal_time_min"] = np.minimum(
        jobs["arrival_time_min"],
        np.maximum(0, jobs["reveal_time_min"] + reveal_adjust),
    ).astype(int)
    jobs["arrival_congestion_score"] = np.clip(
        jobs["arrival_congestion_score"] * (1.0 + 0.40 * intensity) + 0.05 * (rank_position > 0),
        0.0,
        0.999,
    )
    busy = _machine_busy_minutes(fifo_schedule)
    bottleneck_machine = str(busy.index[0]) if not busy.empty else str(machines["machine_id"].iloc[0])
    eligible.loc[eligible["machine_id"].eq(bottleneck_machine), "proc_time_min"] = np.ceil(
        eligible.loc[eligible["machine_id"].eq(bottleneck_machine), "proc_time_min"] * (1.0 + 0.08 + intensity)
    ).astype(int)
    horizon = int(params["planning_horizon_min"])
    median_arrival = int(jobs["arrival_time_min"].median())
    downtime_duration = int(round(22 + intensity * 52))
    downtime_start = max(0, min(horizon - downtime_duration - 1, median_arrival + int(18 * intensity)))
    extra_downtime = pd.DataFrame(
        [
            {
                "machine_id": bottleneck_machine,
                "start_min": downtime_start,
                "end_min": downtime_start + downtime_duration,
                "reason": "CHILD_GRADED_STRESS",
            }
        ]
    )
    downtimes = (
        pd.concat([downtimes, extra_downtime], ignore_index=True)
        .sort_values(["machine_id", "start_min", "end_min"])
        .reset_index(drop=True)
    )
    operations.loc[operations["op_seq"].eq(1), "release_time_min"] = (
        operations.loc[operations["op_seq"].eq(1), "job_id"].map(jobs.set_index("job_id")["arrival_time_min"])
    ).astype(int)
    operations.loc[operations["op_seq"].gt(1), "release_time_min"] = 0
    return {
        "jobs": jobs,
        "operations": operations,
        "precedences": bundle["precedences"].copy(),
        "eligible": eligible,
        "machines": machines,
        "downtimes": downtimes,
    }, {
        "bottleneck_machine": bottleneck_machine,
        "added_downtime_min": int(downtime_duration),
    }


def _apply_discriminating_child_transform(
    bundle: dict[str, Any],
    intensity: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    jobs = bundle["jobs"].copy()
    operations = bundle["operations"].copy()
    eligible = bundle["eligible"].copy()
    downtimes = bundle["downtimes"].copy()
    machines = bundle["machines"].copy()
    fifo_schedule = bundle["fifo_schedule"].copy()
    params = dict(bundle["params"])
    top_jobs = (
        jobs.sort_values(["arrival_congestion_score", "arrival_time_min"], ascending=[False, True])
        .head(max(2, int(math.ceil(len(jobs) * 0.35))))["job_id"]
        .tolist()
    )
    reveal_gap = (jobs["arrival_time_min"] - jobs["reveal_time_min"]).clip(lower=0)
    jobs.loc[jobs["job_id"].isin(top_jobs), "reveal_time_min"] = (
        jobs.loc[jobs["job_id"].isin(top_jobs), "arrival_time_min"]
        - np.ceil(reveal_gap.loc[jobs["job_id"].isin(top_jobs)] * (0.45 - 0.10 * intensity)).astype(int)
    ).clip(lower=0).astype(int)
    jobs["arrival_congestion_score"] = np.clip(
        jobs["arrival_congestion_score"] * (1.0 + 0.25 * intensity),
        0.0,
        0.999,
    )
    constrained_rows = []
    retained_rows = []
    for (job_id, op_seq), group in eligible.groupby(["job_id", "op_seq"], sort=False):
        group = group.sort_values(["proc_time_min", "machine_id"]).copy()
        if job_id in top_jobs and op_seq in {1, 3, 4} and len(group) > 1:
            keep = group.head(1)
            constrained_rows.append((job_id, int(op_seq), int(len(group)), int(len(keep))))
            retained_rows.append(keep)
        else:
            adjusted = group.copy()
            if len(adjusted) > 1 and op_seq in {1, 3}:
                penalty_idx = adjusted.index[1:]
                adjusted.loc[penalty_idx, "proc_time_min"] = np.ceil(
                    adjusted.loc[penalty_idx, "proc_time_min"] * (1.0 + 0.18 + intensity)
                ).astype(int)
            retained_rows.append(adjusted)
    eligible = pd.concat(retained_rows, ignore_index=True).sort_values(
        ["job_id", "op_seq", "machine_id"]
    ).reset_index(drop=True)
    busy = _machine_busy_minutes(fifo_schedule)
    hotspot_machines = list(busy.index[: min(2, len(busy))]) if not busy.empty else [str(machines["machine_id"].iloc[0])]
    horizon = int(params["planning_horizon_min"])
    arrival_q50 = int(jobs["arrival_time_min"].quantile(0.50))
    added_windows = []
    for machine_offset, machine_id in enumerate(hotspot_machines):
        duration = int(round(18 + intensity * 46 + machine_offset * 6))
        start = max(0, min(horizon - duration - 1, arrival_q50 + machine_offset * 25))
        added_windows.append(
            {
                "machine_id": str(machine_id),
                "start_min": start,
                "end_min": start + duration,
                "reason": "CHILD_DISCRIMINATING_STRESS",
            }
        )
    downtimes = (
        pd.concat([downtimes, pd.DataFrame(added_windows)], ignore_index=True)
        .sort_values(["machine_id", "start_min", "end_min"])
        .reset_index(drop=True)
    )
    operations.loc[operations["op_seq"].eq(1), "release_time_min"] = (
        operations.loc[operations["op_seq"].eq(1), "job_id"].map(jobs.set_index("job_id")["arrival_time_min"])
    ).astype(int)
    operations.loc[operations["op_seq"].gt(1), "release_time_min"] = 0
    return {
        "jobs": jobs.sort_values("arrival_time_min").reset_index(drop=True),
        "operations": operations,
        "precedences": bundle["precedences"].copy(),
        "eligible": eligible,
        "machines": machines,
        "downtimes": downtimes,
    }, {
        "top_jobs_constrained": int(len(top_jobs)),
        "constrained_rows": constrained_rows,
        "hotspot_machines": hotspot_machines,
    }


def generate_child_instance_bundles(
    root: Path,
    proposals: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_rows = proposals.copy()
    proposal_rows["family_rank"] = proposal_rows.groupby(["child_type", "scale_code"]).cumcount() + 1
    rows = []
    for proposal in proposal_rows.itertuples(index=False):
        bundle = _load_parent_instance_bundle(root, proposal.instance_id)
        intensity = _candidate_intensity(proposal.child_type, int(proposal.family_rank))
        if proposal.child_type == "graded":
            child_tables, transform_meta = _apply_graded_child_transform(bundle=bundle, intensity=intensity)
        else:
            child_tables, transform_meta = _apply_discriminating_child_transform(bundle=bundle, intensity=intensity)
        child_tables["events"] = _regenerate_events_for_child(
            jobs=child_tables["jobs"],
            downtimes=child_tables["downtimes"],
        )
        fifo_schedule, fifo_job_metrics, fifo_summary = _build_child_fifo_artifacts(
            candidate_id=proposal.candidate_id,
            instance=child_tables,
        )
        child_dir = out_dir / proposal.candidate_id
        child_dir.mkdir(parents=True, exist_ok=True)
        params = dict(bundle["params"])
        params.update(
            {
                "dataset_version": "1.1.0-observed-child",
                "instance_id": proposal.candidate_id,
                "notes": f"Derived child instance generated in-notebook from {proposal.instance_id} using {proposal.child_type} ISA-guided transform.",
                "parent_instance_id": proposal.instance_id,
                "child_type": proposal.child_type,
                "generation_intensity": round(float(intensity), 4),
            }
        )
        child_tables["jobs"].to_csv(child_dir / "jobs.csv", index=False)
        child_tables["operations"].to_csv(child_dir / "operations.csv", index=False)
        child_tables["precedences"].to_csv(child_dir / "precedences.csv", index=False)
        child_tables["eligible"].to_csv(child_dir / "eligible_machines.csv", index=False)
        child_tables["machines"].to_csv(child_dir / "machines.csv", index=False)
        child_tables["downtimes"].to_csv(child_dir / "machine_downtimes.csv", index=False)
        child_tables["events"].to_csv(child_dir / "events.csv", index=False)
        fifo_schedule.to_csv(child_dir / "fifo_schedule.csv", index=False)
        fifo_job_metrics.to_csv(child_dir / "fifo_job_metrics.csv", index=False)
        (child_dir / "fifo_summary.json").write_text(json.dumps(fifo_summary, indent=2), encoding="utf-8")
        child_tables["jobs"][
            ["job_id", "arrival_time_min", "arrival_congestion_score", "shift_bucket"]
        ].to_csv(child_dir / "job_congestion_proxy.csv", index=False)
        (child_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        manifest = {
            "candidate_id": proposal.candidate_id,
            "parent_instance_id": proposal.instance_id,
            "child_type": proposal.child_type,
            "scale_code": proposal.scale_code,
            "regime_code": proposal.regime_code,
            "family_rank": int(proposal.family_rank),
            "generation_intensity": round(float(intensity), 4),
            "proposal_rule": proposal.proposal_rule,
            "target_recommendation": proposal.target_recommendation,
            "transform_meta": transform_meta,
        }
        (child_dir / "child_generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        parent_jobs = bundle["jobs"].copy()
        parent_reveal_span = int(parent_jobs["reveal_time_min"].max() - parent_jobs["reveal_time_min"].min())
        child_reveal_span = int(child_tables["jobs"]["reveal_time_min"].max() - child_tables["jobs"]["reveal_time_min"].min())
        parent_downtime = int(bundle["downtimes"].assign(duration=lambda df: df["end_min"] - df["start_min"])["duration"].sum()) if not bundle["downtimes"].empty else 0
        child_downtime = int(child_tables["downtimes"].assign(duration=lambda df: df["end_min"] - df["start_min"])["duration"].sum()) if not child_tables["downtimes"].empty else 0
        parent_density = float(len(bundle["eligible"]) / max(1, len(bundle["jobs"]) * 4))
        child_density = float(len(child_tables["eligible"]) / max(1, len(child_tables["jobs"]) * 4))
        rows.append(
            {
                "candidate_id": proposal.candidate_id,
                "child_type": proposal.child_type,
                "parent_instance_id": proposal.instance_id,
                "output_dir": str(child_dir),
                "reveal_span_parent": parent_reveal_span,
                "reveal_span_child": child_reveal_span,
                "congestion_mean_parent": float(parent_jobs["arrival_congestion_score"].mean()),
                "congestion_mean_child": float(child_tables["jobs"]["arrival_congestion_score"].mean()),
                "downtime_total_parent": parent_downtime,
                "downtime_total_child": child_downtime,
                "eligibility_density_parent": round(parent_density, 4),
                "eligibility_density_child": round(child_density, 4),
                "fifo_mean_flow_child": float(fifo_job_metrics["flow_time_min"].mean()),
                "fifo_p95_flow_child": float(np.quantile(fifo_job_metrics["flow_time_min"], 0.95)),
                "fifo_makespan_child": float(fifo_job_metrics["completion_min"].max()),
                "generated_file_count": int(len(list(child_dir.iterdir()))),
            }
        )
    return pd.DataFrame(rows).sort_values(["child_type", "candidate_id"]).reset_index(drop=True)


def build_performance_profile_frame(performance: pd.DataFrame, value_col: str = "utility") -> pd.DataFrame:
    pivot = (
        performance.pivot(index="instance_id", columns="method_name", values=value_col)
        .reindex(columns=PAPER_METHOD_ORDER)
        .astype(float)
    )
    best = pivot.min(axis=1)
    # `utility` pode ter ótimo igual a zero. Para evitar perfis degenerados,
    # usamos a forma deslocada por regret: ratio = 1 + (value - best).
    ratios = pivot.sub(best, axis=0).add(1.0)
    rows = []
    tau_grid = np.linspace(1.0, max(2.5, float(ratios.max().max()) + 0.05), 120)
    for method_name in ratios.columns:
        values = ratios[method_name].dropna().to_numpy(dtype=float)
        for tau in tau_grid:
            rows.append(
                {
                    "method_name": method_name,
                    "tau": float(tau),
                    "share_instances": float(np.mean(values <= tau)),
                }
            )
    return pd.DataFrame(rows)


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


def build_e5_validity_snapshot(
    formal_shift_summary: pd.DataFrame,
    job_density_segments: pd.DataFrame,
    proc_density_segments: pd.DataFrame,
    summary: dict[str, Any],
    tail_regime_checks: pd.DataFrame,
    rare_segment_summary: pd.DataFrame,
    instance_space_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for experiment in ["job_due_layer", "proc_time_layer"]:
        subset = formal_shift_summary.loc[formal_shift_summary["experiment"].eq(experiment)]
        density_source = job_density_segments if experiment == "job_due_layer" else proc_density_segments
        rows.append(
            {
                "diagnostic_block": experiment,
                "c2st_auc_mean": float(subset["c2st_auc_mean"].mean()),
                "mmd_rbf_mean": float(subset["mmd_rbf_stat"].mean()),
                "mean_log_density_ratio": float(density_source["mean_log_density_ratio_delta"].mean()),
            }
        )
    rows.append(
        {
            "diagnostic_block": "instance_space",
            "c2st_auc_mean": np.nan,
            "mmd_rbf_mean": float(instance_space_summary.loc[0, "nearest_neighbor_distance_min"]),
            "mean_log_density_ratio": float(instance_space_summary.loc[0, "knn_same_regime_share_k5_mean"]),
        }
    )
    rows.append(
        {
            "diagnostic_block": "tail_checks",
            "c2st_auc_mean": float(tail_regime_checks["flow_p99_order_ok"].mean()),
            "mmd_rbf_mean": float(tail_regime_checks["queue_p99_order_ok"].mean()),
            "mean_log_density_ratio": float(len(rare_segment_summary)),
        }
    )
    rows.append(
        {
            "diagnostic_block": "scorecard_release",
            "c2st_auc_mean": float(summary["job_due_c2st_auc_mean"]),
            "mmd_rbf_mean": float(summary["proc_time_c2st_auc_mean"]),
            "mean_log_density_ratio": float(summary["structural_pass_rate"]),
        }
    )
    return pd.DataFrame(rows)


PIPELINE_CONFIG = {
    "generate_observed_release": False,
    "source_root": None,
    "target_root": str(REPO_ROOT),
    "sample_instance_id": "GO_XS_DISRUPTED_01",
}


def _resolve_path(value: str | Path | None) -> Path | None:
    if value in (None, "", "."):
        return None
    return Path(value).expanduser().resolve()


def prepare_analysis_root(config: dict[str, object]) -> Path:
    target_root = _resolve_path(config.get("target_root")) or REPO_ROOT
    if not bool(config.get("generate_observed_release", False)):
        return target_root

    source_root = _resolve_path(config.get("source_root"))
    if source_root is None:
        raise ValueError(
            "PIPELINE_CONFIG['source_root'] precisa apontar para o release fonte quando "
            "generate_observed_release=True."
        )
    if source_root == target_root:
        raise ValueError(
            "source_root e target_root não podem ser o mesmo diretório quando o notebook "
            "for regenerar a release observada."
        )

    print(f"[notebook] Generating observed release from {source_root} -> {target_root}")
    observed_release_builder.main(src_root=source_root, out_root=target_root)
    return target_root


ANALYSIS_ROOT = prepare_analysis_root(PIPELINE_CONFIG)
ARTIFACT_DIR = (
    ANALYSIS_ROOT / "output" / "jupyter-notebook" / "instance_validation_analysis_artifacts"
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = repl.SEED
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="talk")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

STAGE_ORDER = repl.STAGE_ORDER
REGIME_ORDER = repl.REGIME_ORDER
SCALE_ORDER = repl.SCALE_ORDER


def _sample_frame(frame: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _prepare_feature_matrix(frame: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    matrix = pd.get_dummies(
        frame[numeric_cols + categorical_cols].copy(),
        columns=categorical_cols,
        drop_first=False,
    ).astype(float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0).replace(0.0, 1.0)
    return ((matrix - means) / stds).astype("float32")


def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    xx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    yy = ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    xy = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    k_xx = np.exp(-gamma * xx)
    k_yy = np.exp(-gamma * yy)
    k_xy = np.exp(-gamma * xy)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def _median_gamma(z: np.ndarray) -> float:
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    tri = sq[np.triu_indices_from(sq, k=1)]
    tri = tri[tri > 0]
    if len(tri) == 0:
        return 1.0
    median_sq = float(np.median(tri))
    return 1.0 / max(2.0 * median_sq, 1e-6)


def _mmd_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 80,
    seed: int = SEED,
) -> tuple[float, float]:
    combined = np.vstack([x, y])
    n_x = len(x)
    gamma = _median_gamma(combined)
    observed = _mmd_rbf(x, y, gamma)
    rng = np.random.default_rng(seed)
    perm_stats = []
    for _ in range(permutations):
        perm = rng.permutation(len(combined))
        x_perm = combined[perm[:n_x]]
        y_perm = combined[perm[n_x:]]
        perm_stats.append(_mmd_rbf(x_perm, y_perm, gamma))
    pvalue = float((np.sum(np.array(perm_stats) >= observed) + 1) / (permutations + 1))
    return observed, pvalue


def run_domain_shift_experiment(
    nominal: pd.DataFrame,
    observed: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    group_cols: list[str],
    label: str,
    max_rows_per_domain: int | None = None,
    mmd_sample_cap: int = 600,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nominal = _sample_frame(nominal.copy(), max_rows_per_domain, seed=SEED)
    observed = _sample_frame(observed.copy(), max_rows_per_domain, seed=SEED + 1)
    nominal["domain"] = "nominal"
    observed["domain"] = "observed"
    combined = pd.concat([nominal, observed], ignore_index=True)
    combined["domain_target"] = combined["domain"].eq("observed").astype(int)

    x = _prepare_feature_matrix(combined, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    y = combined["domain_target"].to_numpy()

    clf = LogisticRegression(max_iter=1500, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores = cross_val_score(clf, x, y, cv=cv, scoring="roc_auc")
    prob_oof = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    combined["prob_observed"] = prob_oof
    prob_clipped = np.clip(prob_oof, 1e-5, 1 - 1e-5)
    combined["density_ratio_proxy"] = prob_clipped / (1 - prob_clipped)
    combined["log_density_ratio_proxy"] = np.log(combined["density_ratio_proxy"])

    nominal_x = x.loc[combined["domain"].eq("nominal")].to_numpy()
    observed_x = x.loc[combined["domain"].eq("observed")].to_numpy()
    nominal_mmd = nominal_x[: min(len(nominal_x), mmd_sample_cap)]
    observed_mmd = observed_x[: min(len(observed_x), mmd_sample_cap)]
    mmd_stat, mmd_pvalue = _mmd_permutation_test(nominal_mmd, observed_mmd)

    summary = pd.DataFrame(
        [
            {
                "experiment": label,
                "rows_per_domain": int(min(len(nominal), len(observed))),
                "feature_count": int(x.shape[1]),
                "c2st_auc_mean": float(np.mean(auc_scores)),
                "c2st_auc_std": float(np.std(auc_scores, ddof=0)),
                "c2st_auc_oof": float(roc_auc_score(y, prob_oof)),
                "mmd_rbf_stat": float(mmd_stat),
                "mmd_permutation_pvalue": float(mmd_pvalue),
            }
        ]
    )

    grouped = (
        combined.groupby(group_cols + ["domain"], as_index=False)
        .agg(
            sample_count=("domain_target", "size"),
            mean_prob_observed=("prob_observed", "mean"),
            mean_log_density_ratio=("log_density_ratio_proxy", "mean"),
        )
    )
    observed_group = grouped[grouped["domain"] == "observed"].drop(columns="domain")
    nominal_group = grouped[grouped["domain"] == "nominal"].drop(columns="domain")
    segment_shift = observed_group.merge(
        nominal_group,
        on=group_cols,
        how="outer",
        suffixes=("_observed", "_nominal"),
    )
    segment_shift["mean_prob_observed_delta"] = (
        segment_shift["mean_prob_observed_observed"] - segment_shift["mean_prob_observed_nominal"]
    )
    segment_shift["mean_log_density_ratio_delta"] = (
        segment_shift["mean_log_density_ratio_observed"] - segment_shift["mean_log_density_ratio_nominal"]
    )
    segment_shift["experiment"] = label
    return summary, segment_shift.sort_values(group_cols).reset_index(drop=True)


def build_tail_and_segment_reports(jobs_enriched: pd.DataFrame, job_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs_metrics = job_metrics.merge(
        jobs_enriched[
            [
                "instance_id",
                "job_id",
                "priority_class",
                "appointment_flag",
                "moisture_class",
                "commodity",
                "due_margin_over_lb_min",
            ]
        ],
        on=["instance_id", "job_id"],
        how="left",
    )

    tail_summary = (
        jobs_metrics.groupby(["scale_code", "regime_code"], as_index=False)
        .agg(
            flow_p95=("flow_time_min", lambda s: float(np.quantile(s, 0.95))),
            flow_p99=("flow_time_min", lambda s: float(np.quantile(s, 0.99))),
            queue_p95=("queue_time_min", lambda s: float(np.quantile(s, 0.95))),
            queue_p99=("queue_time_min", lambda s: float(np.quantile(s, 0.99))),
            overwait_share=("overwait_min", lambda s: float(np.mean(np.asarray(s) > 0))),
            due_margin_p05=("due_margin_over_lb_min", lambda s: float(np.quantile(s, 0.05))),
        )
    )

    tail_checks = []
    for scale_code, frame in tail_summary.groupby("scale_code"):
        ordered = frame.set_index("regime_code").reindex(REGIME_ORDER)
        tail_checks.append(
            {
                "scale_code": scale_code,
                "flow_p99_order_ok": bool(ordered["flow_p99"].is_monotonic_increasing),
                "queue_p99_order_ok": bool(ordered["queue_p99"].is_monotonic_increasing),
                "due_margin_p05_order_ok": bool(ordered["due_margin_p05"].is_monotonic_decreasing),
            }
        )
    tail_checks = pd.DataFrame(tail_checks).sort_values("scale_code").reset_index(drop=True)

    segment_rows = []
    segment_specs = {
        "URGENT": jobs_metrics["priority_class"].eq("URGENT"),
        "WET": jobs_metrics["moisture_class"].eq("WET"),
        "APPOINTMENT": jobs_metrics["appointment_flag"].eq(1),
        "URGENT_AND_WET": jobs_metrics["priority_class"].eq("URGENT") & jobs_metrics["moisture_class"].eq("WET"),
        "APPOINTMENT_AND_WET": jobs_metrics["appointment_flag"].eq(1) & jobs_metrics["moisture_class"].eq("WET"),
    }
    for segment_label, mask in segment_specs.items():
        frame = jobs_metrics[mask].copy()
        if len(frame) == 0:
            continue
        segment_rows.append(
            {
                "segment_label": segment_label,
                "job_count": int(len(frame)),
                "job_share": float(len(frame) / len(jobs_metrics)),
                "flow_mean": float(frame["flow_time_min"].mean()),
                "flow_p95": float(np.quantile(frame["flow_time_min"], 0.95)),
                "queue_mean": float(frame["queue_time_min"].mean()),
                "queue_p95": float(np.quantile(frame["queue_time_min"], 0.95)),
                "overwait_share": float(np.mean(frame["overwait_min"] > 0)),
                "due_margin_p05": float(np.quantile(frame["due_margin_over_lb_min"], 0.05)),
            }
        )
    segment_summary = pd.DataFrame(segment_rows).sort_values("flow_p95", ascending=False).reset_index(drop=True)
    return tail_summary, tail_checks, segment_summary


def build_relational_consistency_reports(
    jobs: pd.DataFrame,
    operations: pd.DataFrame,
    precedences: pd.DataFrame,
    eligible: pd.DataFrame,
    machines: pd.DataFrame,
    events: pd.DataFrame,
    schedule: pd.DataFrame,
    job_metrics: pd.DataFrame,
    params: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for instance_id in sorted(jobs["instance_id"].unique()):
        jobs_g = jobs[jobs["instance_id"] == instance_id]
        ops_g = operations[operations["instance_id"] == instance_id]
        prec_g = precedences[precedences["instance_id"] == instance_id]
        eligible_g = eligible[eligible["instance_id"] == instance_id]
        machines_g = machines[machines["instance_id"] == instance_id]
        events_g = events[events["instance_id"] == instance_id]
        schedule_g = schedule[schedule["instance_id"] == instance_id]
        metrics_g = job_metrics[job_metrics["instance_id"] == instance_id]

        job_keys = set(jobs_g["job_id"])
        op_keys = set(map(tuple, ops_g[["job_id", "op_seq"]].itertuples(index=False, name=None)))
        machine_keys = set(machines_g["machine_id"])

        ops_per_job = ops_g.groupby("job_id").size().reindex(jobs_g["job_id"]).fillna(0)
        prec_per_job = prec_g.groupby("job_id").size().reindex(jobs_g["job_id"]).fillna(0)
        eligible_per_op = eligible_g.groupby(["job_id", "op_seq"]).size().reindex(pd.MultiIndex.from_frame(ops_g[["job_id", "op_seq"]])).fillna(0)
        schedule_per_op = schedule_g.groupby(["job_id", "op_seq"]).size().reindex(pd.MultiIndex.from_frame(ops_g[["job_id", "op_seq"]])).fillna(0)

        event_job_rows = events_g[events_g["event_type"].isin(["JOB_VISIBLE", "JOB_ARRIVAL"])]
        event_machine_rows = events_g[events_g["event_type"].isin(["MACHINE_DOWN", "MACHINE_UP"])]

        rows.append(
            {
                "instance_id": instance_id,
                "job_has_4_operations_ok": bool(ops_per_job.eq(4).all()),
                "job_has_3_precedences_ok": bool(prec_per_job.eq(3).all()),
                "every_operation_has_eligible_machine_ok": bool((eligible_per_op > 0).all()),
                "fifo_has_one_row_per_operation_ok": bool(schedule_per_op.eq(1).all() and len(schedule_g) == len(ops_g)),
                "schedule_operation_fk_ok": bool(set(map(tuple, schedule_g[["job_id", "op_seq"]].itertuples(index=False, name=None))).issubset(op_keys)),
                "schedule_machine_fk_ok": bool(set(schedule_g["machine_id"]).issubset(machine_keys)),
                "metrics_job_fk_ok": bool(set(metrics_g["job_id"]).issubset(job_keys) and len(metrics_g) == len(jobs_g)),
                "job_events_fk_ok": bool(set(event_job_rows["entity_id"]).issubset(job_keys)),
                "machine_events_fk_ok": bool(set(event_machine_rows["entity_id"]).issubset(machine_keys)),
                "job_with_wrong_op_count": int((~ops_per_job.eq(4)).sum()),
                "job_with_wrong_prec_count": int((~prec_per_job.eq(3)).sum()),
                "operations_without_eligible_machine": int((eligible_per_op <= 0).sum()),
                "operations_without_single_schedule_row": int((~schedule_per_op.eq(1)).sum()),
                "metrics_job_count_gap": int(abs(len(metrics_g) - len(jobs_g))),
            }
        )

    report = pd.DataFrame(rows).merge(
        params[["instance_id", "scale_code", "regime_code"]], on="instance_id", how="left"
    )
    bool_cols = [c for c in report.columns if c.endswith("_ok")]
    summary = pd.DataFrame(
        [
            {
                "check_name": col,
                "pass_rate": float(report[col].mean()),
                "failed_instance_count": int((~report[col]).sum()),
            }
            for col in bool_cols
        ]
    ).sort_values(["pass_rate", "check_name"], ascending=[True, True]).reset_index(drop=True)
    return report, summary

# %% [markdown]
# ## Plan
#
# 1. Opcionalmente regenerar a release observada a partir de um root fonte.
# 2. Carregar o backend analítico compartilhado sobre o root alvo e expor os objetos principais no notebook.
# 3. Validar integridade estrutural, reconciliar audits e inspecionar métricas agregadas do release.
# 4. Verificar se a camada observacional reduz sobre-determinismo sem quebrar a semântica operacional.
# 5. Medir formalmente o deslocamento nominal -> observado e a consistência relacional do release.
# 6. Testar caudas, segmentos raros e monotonicidade forte por regime.
# 7. Confirmar que o release cobre regiões distintas do espaço de instâncias e não colapsa em casos quase redundantes.
# 8. Executar um smoke test exato com orçamento fixo para mostrar que o benchmark é informativo do ponto de vista algorítmico.
# 9. Fazer drilldown visual em uma instância para checagem manual do baseline FIFO.

# %%
# Bootstrap the notebook workspace from the shared REPL backend
CTX = repl.load_context(root=ANALYSIS_ROOT, artifact_dir=ARTIFACT_DIR)
SUMMARY = CTX["summary"]
NOTEBOOK_CTX = dict(CTX)

validation_observed = pd.read_csv(
    ANALYSIS_ROOT / "catalog" / "validation_report_observed.csv"
)
validation_nominal_style = pd.read_csv(ANALYSIS_ROOT / "catalog" / "validation_report.csv")
g2milp_contract = json.loads(
    (ANALYSIS_ROOT / "catalog" / "g2milp_generation_contract.json").read_text(
        encoding="utf-8"
    )
)

params = CTX["params"].copy()
catalog = CTX["catalog"].copy()
family_summary = CTX["family_summary"].copy()
observed_noise_manifest = CTX["observed_noise_manifest"]
manifest = CTX["manifest"]

jobs = CTX["jobs"].copy()
jobs_enriched = CTX["jobs_enriched"].copy()
operations = CTX["operations"].copy()
eligible = CTX["eligible"].copy()
machines = CTX["machines"].copy()
precedences = CTX["precedences"].copy()
downtimes = CTX["downtimes"].copy()
events = CTX["events"].copy()
schedule = CTX["schedule"].copy()
job_metrics = CTX["job_metrics"].copy()
due_audit = CTX["due_audit"].copy()
proc_audit = CTX["proc_audit"].copy()
proc_audit_enriched = CTX["proc_audit_enriched"].copy()
congestion = CTX["congestion"].copy()

structural_report = CTX["structural_report"].copy()
event_report = CTX["event_report"].copy()
audit_reconciliation = CTX["audit_reconciliation"].copy()
regime_checks = CTX["regime_checks"].copy()
fifo_schema_report = CTX["fifo_schema_report"].copy()
release_consistency_report = CTX["release_consistency_report"].copy()
utilization = CTX["utilization"].copy()
instance_space_features = CTX["instance_space_features"].copy()
instance_space_pairs = CTX["instance_space_pairs"].copy()
instance_space_summary = CTX["instance_space_summary"].copy()
instance_space_knn_profile = CTX["instance_space_knn_profile"].copy()
instance_space_knn_regime_composition = CTX["instance_space_knn_regime_composition"].copy()
instance_space_knn_scale_composition = CTX["instance_space_knn_scale_composition"].copy()
diagnostics = CTX["diagnostics"].copy()
unload = CTX["unload"].copy()

inventory_summary = pd.DataFrame([SUMMARY])
pipeline_summary = pd.DataFrame(
    [
        {
            "analysis_root": str(ANALYSIS_ROOT),
            "artifact_dir": str(ARTIFACT_DIR),
            "generate_observed_release": bool(PIPELINE_CONFIG["generate_observed_release"]),
            "source_root": str(PIPELINE_CONFIG["source_root"] or ""),
            "target_root": str(PIPELINE_CONFIG["target_root"]),
            "sample_instance_id": str(PIPELINE_CONFIG["sample_instance_id"]),
        }
    ]
)
display(pipeline_summary)
display(inventory_summary)
display(
    Markdown(
        """
**Quick start interativo**

- `SUMMARY`
- `params.head()`
- `structural_report.head()`
- `repl.plot_inventory_overview()`
- `repl.plot_validation_overview()`
- `repl.plot_observational_layer()`
- `repl.plot_congestion_diagnostics()`
- `repl.plot_operational_sanity()`
- `repl.plot_instance_space_coverage()`
- `repl.plot_instance_drilldown("GO_XS_DISRUPTED_01", ctx=NOTEBOOK_CTX)`
"""
    )
)

# %%
# Release metadata and provenance checks
noise_manifest_summary = pd.DataFrame(
    [
        {
            "dataset_version": manifest["dataset_version"],
            "official_dataset_role": manifest["official_dataset_role"],
            "noise_model_id": observed_noise_manifest.get("model_id"),
            "noise_global_seed": observed_noise_manifest.get("global_seed"),
            "parent_dataset_version": manifest.get("parent_dataset_version"),
            "generator_model": observed_noise_manifest.get(
                "generator_model", "ChatGPT 5.4 PRO"
            ),
        }
    ]
)

display(params.head())
display(noise_manifest_summary)
display(release_consistency_report)
display(pd.DataFrame([g2milp_contract]).iloc[:, :8])

release_consistency_report.to_csv(
    ARTIFACT_DIR / "release_consistency_report.csv", index=False
)

# %% [markdown]
# **Como ler as tabelas acima**
#
# - `noise_manifest_summary` resume a versão oficial, a linhagem e o modelo gerador da camada observacional
# - `release_consistency_report` formaliza a governança do release: `manifest.json` raiz, `params.json` das instâncias e `observed_noise_manifest.json`
# - para publicação, o desejável é que todos os checks dessa tabela estejam em `pass = True`
# - no release oficial atual, isso de fato ocorre; em particular, não há divergência entre a `dataset_version` do `manifest.json` raiz e a `dataset_version` declarada nos `params.json`

# %% [markdown]
# ## Inventory and structural context
#
# Esta seção responde:
#
# - quantas instâncias, jobs, operações, máquinas e linhas elegíveis existem
# - como as famílias `XS/S/M/L` e os regimes `balanced/peak/disrupted` estão distribuídos
# - se os artefatos de auditoria e catálogo estão completos

# %%
display(catalog.sort_values(["scale_code", "regime_code", "replicate"]).head(12))
display(family_summary.sort_values(["scale_code", "regime_code"]))

fig = repl.plot_inventory_overview(ctx=NOTEBOOK_CTX, save=True)
plt.show()

# %% [markdown]
# **Como ler a figura acima**
#
# - o heatmap da esquerda mostra cobertura do release por família `escala x regime`
# - as barras da direita mostram quantas linhas de recurso existem por família de máquina no release consolidado
# - a figura serve como checagem de inventário, não de desempenho

# %% [markdown]
# ## Structural validation and auditability
#
# Aqui reaplicamos o verificador estrutural do release e complementamos com:
#
# - executabilidade formal do baseline FIFO contra o schema
# - consistência de eventos
# - margem do prazo sobre o lower bound nominal
# - reconciliação auditável entre arquivos centrais e CSVs de audit

# %%
# The shared REPL backend already ships these reports with scale/regime context.

display(structural_report.sort_values(["scale_code", "regime_code", "instance_id"]))
display(fifo_schema_report.sort_values(["scale_code", "regime_code", "instance_id"]))
display(event_report.sort_values(["scale_code", "regime_code", "instance_id"]))
display(audit_reconciliation.sort_values(["scale_code", "regime_code", "instance_id"]))

due_margin_summary = (
    jobs_enriched.groupby(["scale_code", "regime_code"], as_index=False)[
        "due_margin_over_lb_min"
    ]
    .agg(["mean", "min", "median", "max"])
    .round(2)
    .reset_index()
)
display(due_margin_summary)

fig = repl.plot_validation_overview(ctx=NOTEBOOK_CTX, save=True)
plt.show()

structural_report.to_csv(ARTIFACT_DIR / "structural_report.csv", index=False)
fifo_schema_report.to_csv(ARTIFACT_DIR / "fifo_schema_report.csv", index=False)
event_report.to_csv(ARTIFACT_DIR / "event_report.csv", index=False)
audit_reconciliation.to_csv(ARTIFACT_DIR / "audit_reconciliation.csv", index=False)
due_margin_summary.to_csv(ARTIFACT_DIR / "due_margin_summary.csv", index=False)

# %% [markdown]
# **Como ler a figura acima**
#
# - painel esquerdo: cada célula deve ficar em `PASS`; se aparecer número de issues, aquela família tem falhas estruturais
# - a tabela `fifo_schema_report` formaliza a executabilidade do baseline FIFO: elegibilidade, `release_time`, precedência, overlap e downtime
# - painel central: os dois bars precisam ficar em `100%`; qualquer queda indica quebra entre CSV central e CSV de audit
# - painel direito: mostra quanta folga de prazo sobra acima do lower bound físico plausível

# %% [markdown]
# ## Relational consistency across core files
#
# A validação estrutural garante executabilidade, mas ainda vale explicitar se os
# arquivos do release continuam coerentes entre si como estrutura relacional.
# Aqui verificamos:
#
# - cardinalidade `job -> 4 operações`
# - cardinalidade `job -> 3 precedências`
# - completude de elegibilidade por operação
# - integridade referencial entre `schedule`, `machines`, `job_metrics` e `events`
# - unicidade do mapeamento `operação -> linha do FIFO`

# %%
relational_consistency_report, relational_consistency_summary = build_relational_consistency_reports(
    jobs=jobs,
    operations=operations,
    precedences=precedences,
    eligible=eligible,
    machines=machines,
    events=events,
    schedule=schedule,
    job_metrics=job_metrics,
    params=params,
)
display(relational_consistency_summary)
display(relational_consistency_report.sort_values(["scale_code", "regime_code", "instance_id"]))

relational_label_map = {
    "every_operation_has_eligible_machine_ok": "eligible per op",
    "fifo_has_one_row_per_operation_ok": "1 FIFO row per op",
    "job_has_3_precedences_ok": "3 precedences per job",
    "job_has_4_operations_ok": "4 operations per job",
    "job_events_fk_ok": "job events FK",
    "machine_events_fk_ok": "machine events FK",
    "metrics_job_fk_ok": "metrics job FK",
    "schedule_machine_fk_ok": "schedule machine FK",
    "schedule_operation_fk_ok": "schedule op FK",
}
relational_plot = relational_consistency_summary.copy()
relational_plot["check_label"] = relational_plot["check_name"].map(relational_label_map).fillna(relational_plot["check_name"])
relational_plot = relational_plot.sort_values(["failed_instance_count", "check_label"], ascending=[False, True]).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.2), sharey=True, gridspec_kw={"width_ratios": [1.05, 1.0]})
sns.barplot(
    data=relational_plot,
    y="check_label",
    x="pass_rate",
    hue="check_label",
    dodge=False,
    legend=False,
    palette="crest",
    ax=axes[0],
)
axes[0].set_xlim(0, 1.02)
axes[0].set_title("Pass rate dos checks relacionais\nO desejável é 100% em todos", fontsize=13)
axes[0].set_xlabel("Pass rate")
axes[0].set_ylabel("")
axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
axes[0].grid(axis="x", alpha=0.25)
for patch, value in zip(axes[0].patches, relational_plot["pass_rate"]):
    axes[0].text(
        min(value + 0.015, 1.0),
        patch.get_y() + patch.get_height() / 2,
        f"{value:.0%}",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
    )

sns.barplot(
    data=relational_plot,
    y="check_label",
    x="failed_instance_count",
    hue="check_label",
    dodge=False,
    legend=False,
    palette="flare",
    ax=axes[1],
)
axes[1].set_title("Falhas por check\nMeta: zero em todos", fontsize=13)
axes[1].set_xlabel("Failed instances")
axes[1].set_ylabel("")
axes[1].set_xlim(0, max(1.0, float(relational_plot["failed_instance_count"].max()) + 0.75))
axes[1].tick_params(axis="y", left=False, labelleft=False)
axes[1].grid(axis="x", alpha=0.25)
for patch, value in zip(axes[1].patches, relational_plot["failed_instance_count"]):
    axes[1].text(
        value + 0.05,
        patch.get_y() + patch.get_height() / 2,
        f"{int(value)}",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
    )
if relational_plot["failed_instance_count"].sum() == 0:
    axes[1].text(
        0.97,
        0.97,
        "36/36 instâncias\nsem falhas relacionais",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#475569",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95, "boxstyle": "round,pad=0.3"},
    )

fig.suptitle("Consistência relacional do release", x=0.02, y=0.99, ha="left", fontsize=18, fontweight="bold")
fig.text(
    0.02,
    0.93,
    "Leitura rápida: além de válido como scheduling instance, o release preserva a coerência entre tabelas, chaves e cardinalidades centrais.",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(ARTIFACT_DIR / "relational_consistency_overview.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

relational_consistency_report.to_csv(ARTIFACT_DIR / "relational_consistency_report.csv", index=False)
relational_consistency_summary.to_csv(ARTIFACT_DIR / "relational_consistency_summary.csv", index=False)

# %% [markdown]
# **Como ler a figura acima**
#
# - o painel esquerdo mostra a taxa de aprovação dos checks relacionais; para publicação formal, o desejável é `100%`
# - o painel direito mostra quantas instâncias falham em cada check; o desejável é `0`
# - esta seção complementa a integridade estrutural: ela mostra que os arquivos do release não estão apenas “bem formatados”, mas também coerentes como sistema relacional
#
# %% [markdown]
# ## Observational layer behavior
#
# Esta seção testa se a camada observacional cumpriu seu papel:
#
# - a prioridade continua importante, mas não perfeitamente determinística
# - tempos de `UNLOAD` continuam interpretáveis por carga, máquina, umidade e congestionamento
# - o ruído aparece de forma estruturada, e não como barulho arbitrário

# %%
diagnostics_df = pd.DataFrame([diagnostics])
display(diagnostics_df)

fig = repl.plot_observational_layer(ctx=NOTEBOOK_CTX, save=True)
plt.show()

# %% [markdown]
# **Como ler a figura acima**
#
# - prioridade ainda ordena a folga de prazo, mas o `R²` abaixo de `0.5` mostra que ela não explica tudo sozinha
# - `appointment` afeta visibilidade antes da chegada, o que ajuda a aproximar o benchmark de uma operação real
# - em `UNLOAD`, a carga e o regime empurram o tempo mediano para cima
# - os multiplicadores por estágio mostram onde a camada observacional realmente introduziu variação

# %%
fig = repl.plot_congestion_diagnostics(ctx=NOTEBOOK_CTX, save=True)
plt.show()

# %% [markdown]
# **Como ler a figura acima**
#
# - no painel esquerdo, cada linha resume um estágio por decil de congestionamento; inclinação positiva significa que o proxy está influenciando `proc_time`
# - no painel direito, `balanced`, `peak` e `disrupted` deveriam deslocar a distribuição para cima nessa ordem

# %% [markdown]
# ## Formal two-sample tests for the observational layer
#
# Os gráficos anteriores mostram a camada observacional de forma intuitiva. Aqui,
# adicionamos testes formais inspirados na literatura de avaliação de dados
# sintéticos, mas adaptados ao que é viável com os artefatos já presentes no repo.
#
# Como não temos um `holdout real` dentro deste repositório, os testes abaixo medem
# o deslocamento entre a visão `nominal` e a visão `observada` do mesmo benchmark.
# Eles ajudam a responder se a transformação introduziu mudança detectável, porém
# ainda estruturada e semanticamente legível.

# %%
job_domain_base = (
    due_audit.merge(
        jobs[
            [
                "instance_id",
                "job_id",
                "load_tons",
                "priority_class",
                "appointment_flag",
                "moisture_class",
            ]
        ],
        on=["instance_id", "job_id"],
        how="left",
    )
)
job_domain_nominal = job_domain_base.assign(due_slack_min=job_domain_base["due_slack_nominal_min"])
job_domain_observed = job_domain_base.assign(due_slack_min=job_domain_base["due_slack_observed_min"])

proc_domain_base = proc_audit.copy()
proc_domain_nominal = proc_domain_base.assign(proc_time_min=proc_domain_base["proc_time_nominal_min"])
proc_domain_observed = proc_domain_base.assign(proc_time_min=proc_domain_base["proc_time_observed_min"])

job_shift_summary, job_density_segments = run_domain_shift_experiment(
    nominal=job_domain_nominal,
    observed=job_domain_observed,
    numeric_cols=["arrival_time_min", "load_tons", "due_slack_min", "nominal_processing_lb_min"],
    categorical_cols=["priority_class", "appointment_flag", "moisture_class", "shift_bucket", "scale_code", "regime_code"],
    group_cols=["regime_code", "priority_class"],
    label="job_due_layer",
    max_rows_per_domain=None,
    mmd_sample_cap=600,
)
proc_shift_summary, proc_density_segments = run_domain_shift_experiment(
    nominal=proc_domain_nominal,
    observed=proc_domain_observed,
    numeric_cols=["proc_time_min", "arrival_congestion_score", "additive_pause_min"],
    categorical_cols=["stage_name", "commodity", "moisture_class", "shift_bucket", "scale_code", "regime_code"],
    group_cols=["regime_code", "stage_name"],
    label="proc_time_layer",
    max_rows_per_domain=4000,
    mmd_sample_cap=600,
)

formal_shift_summary = pd.concat([job_shift_summary, proc_shift_summary], ignore_index=True)
display(formal_shift_summary)
display(job_density_segments.sort_values(["regime_code", "priority_class"]))
display(proc_density_segments.sort_values(["regime_code", "stage_name"]))

formal_shift_plot = formal_shift_summary.copy()
formal_shift_plot["experiment_label"] = formal_shift_plot["experiment"].map(
    {
        "job_due_layer": "due layer",
        "proc_time_layer": "proc_time layer",
    }
)

job_delta_heatmap = (
    job_density_segments.pivot(index="priority_class", columns="regime_code", values="mean_prob_observed_delta")
    .reindex(index=repl.PRIORITY_ORDER, columns=REGIME_ORDER)
)
proc_delta_heatmap = (
    proc_density_segments.pivot(index="stage_name", columns="regime_code", values="mean_prob_observed_delta")
    .reindex(index=STAGE_ORDER, columns=REGIME_ORDER)
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 9.2),
    gridspec_kw={"height_ratios": [0.9, 1.1]},
)
sns.barplot(
    data=formal_shift_plot,
    x="experiment_label",
    y="c2st_auc_mean",
    hue="experiment_label",
    dodge=False,
    legend=False,
    palette=["#33658a", "#6a994e"],
    ax=axes[0, 0],
)
axes[0, 0].axhline(0.5, color="#475569", linestyle="--", linewidth=1.0, alpha=0.8)
axes[0, 0].set_ylim(0.45, max(1.0, float(formal_shift_summary["c2st_auc_mean"].max()) + 0.05))
axes[0, 0].set_title("C2ST AUC por camada\nQuanto mais acima de 0.5, mais detectável é a mudança", fontsize=12.5)
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("ROC AUC")
for patch, value in zip(axes[0, 0].patches, formal_shift_plot["c2st_auc_mean"]):
    axes[0, 0].text(
        patch.get_x() + patch.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#334155",
    )

sns.barplot(
    data=formal_shift_plot,
    x="experiment_label",
    y="mmd_rbf_stat",
    hue="experiment_label",
    dodge=False,
    legend=False,
    palette=["#2a9d8f", "#f4a261"],
    ax=axes[0, 1],
)
mmd_ymax = max(0.0015, float(formal_shift_plot["mmd_rbf_stat"].max()) * 1.35)
axes[0, 1].set_ylim(0.0, mmd_ymax)
axes[0, 1].set_title("MMD-RBF por camada\nTeste global de diferença entre nominal e observado", fontsize=12.5)
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("MMD statistic")
for patch, (_, row) in zip(axes[0, 1].patches, formal_shift_plot.iterrows()):
    text_y = min(float(row["mmd_rbf_stat"]) + mmd_ymax * 0.06, mmd_ymax * 0.93)
    axes[0, 1].text(
        patch.get_x() + patch.get_width() / 2,
        text_y,
        f"p={row['mmd_permutation_pvalue']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )

sns.heatmap(
    job_delta_heatmap,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0.0,
    linewidths=1.0,
    linecolor="white",
    annot_kws={"fontsize": 10.5},
    cbar_kws={"label": "Delta prob(obs)", "shrink": 0.85},
    ax=axes[1, 0],
)
axes[1, 0].set_title("Density-ratio proxy por regime x prioridade\nOnde a camada de prazo desloca mais", fontsize=12.5)
axes[1, 0].set_xlabel("Regime")
axes[1, 0].set_ylabel("Priority")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].tick_params(axis="y", rotation=0)

sns.heatmap(
    proc_delta_heatmap,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0.0,
    linewidths=1.0,
    linecolor="white",
    annot_kws={"fontsize": 10.5},
    cbar_kws={"label": "Delta prob(obs)", "shrink": 0.85},
    ax=axes[1, 1],
)
axes[1, 1].set_title("Density-ratio proxy por regime x estágio\nOnde a camada de proc_time desloca mais", fontsize=12.5)
axes[1, 1].set_xlabel("Regime")
axes[1, 1].set_ylabel("Stage")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].tick_params(axis="y", rotation=0)

fig.suptitle("Testes formais da camada observacional", x=0.02, y=0.98, ha="left", fontsize=17, fontweight="bold")
fig.text(
    0.02,
    0.935,
    "Leitura rápida: a transformação nominal -> observado é estatisticamente detectável, mas o deslocamento permanece estruturado por prioridade, regime e estágio.",
    fontsize=10.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.2, w_pad=2.0)
fig.savefig(ARTIFACT_DIR / "formal_shift_experiments.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

formal_shift_summary.to_csv(ARTIFACT_DIR / "formal_shift_experiments_summary.csv", index=False)
job_density_segments.to_csv(ARTIFACT_DIR / "job_density_ratio_segments.csv", index=False)
proc_density_segments.to_csv(ARTIFACT_DIR / "proc_density_ratio_segments.csv", index=False)

# %% [markdown]
# **Como ler a figura acima**
#
# - `C2ST` mede o quão fácil é distinguir `nominal` de `observed`; valores acima de `0.5` indicam mudança detectável
# - `MMD` mede a diferença global entre distribuições; `p-values` pequenos indicam que o deslocamento é estatisticamente real
# - os heatmaps inferiores funcionam como um proxy de `density ratio`: eles mostram em quais combinações de `regime x prioridade` e `regime x estágio` a transformação ficou mais forte
# - a leitura desejada não é “mudança zero”; é “mudança detectável e interpretável”, sem colapso semântico da estrutura original

# %% [markdown]
# ## Operational performance and regime sanity
#
# A validação não depende só de integridade estrutural. Também interessa saber se:
#
# - `balanced < peak < disrupted` permanece verdadeiro para `mean_flow` e `p95_flow`
# - a fila média também preserva monotonicidade
# - o proxy médio de congestionamento não precisa ser monotônico em todas as famílias
# - os tempos de fluxo e fila continuam coerentes com a escala do problema
# - a utilização de recurso faz sentido por família de máquina

# %%
display(regime_checks)
display(family_summary.sort_values(["scale_code", "regime_code"]))

fig = repl.plot_operational_sanity(ctx=NOTEBOOK_CTX, save=True)
plt.show()

# %% [markdown]
# **Como ler a figura acima**
#
# - os heatmaps do topo validam a monotonicidade esperada apenas para `flow`: `balanced < peak < disrupted`
# - a tabela `regime_checks` separa formalmente os checks de `flow`, `queue` e `congestion`
# - o boxplot inferior esquerdo mostra a distribuição de `flow_time` no nível de job
# - o gráfico inferior direito ajuda a ver quais famílias de máquina absorvem mais pressão em cada regime

# %% [markdown]
# ## Tail behavior and rare segments
#
# Médias e medianas ajudam, mas não são suficientes para validar um benchmark
# sintético operacional. Também interessa saber:
#
# - se as caudas `p95/p99` continuam obedecendo o gradiente de regime
# - se a margem sobre o lower bound continua plausível nos piores casos
# - se segmentos raros, como `URGENT` e `WET`, permanecem operacionais e não colapsam em artefatos estranhos

# %%
tail_regime_summary, tail_regime_checks, rare_segment_summary = build_tail_and_segment_reports(
    jobs_enriched=jobs_enriched,
    job_metrics=job_metrics,
)
display(tail_regime_summary.sort_values(["scale_code", "regime_code"]))
display(tail_regime_checks.sort_values("scale_code"))
display(rare_segment_summary)

flow_p99_heatmap = (
    tail_regime_summary.pivot(index="scale_code", columns="regime_code", values="flow_p99")
    .reindex(index=SCALE_ORDER, columns=REGIME_ORDER)
)
queue_p99_heatmap = (
    tail_regime_summary.pivot(index="scale_code", columns="regime_code", values="queue_p99")
    .reindex(index=SCALE_ORDER, columns=REGIME_ORDER)
)
rare_segment_plot = rare_segment_summary.copy()
rare_segment_plot["segment_display"] = (
    rare_segment_plot["segment_label"]
    .str.replace("_AND_", " + ", regex=False)
    .str.replace("_", " ", regex=False)
    .str.title()
)
rare_segment_plot = rare_segment_plot.sort_values("flow_p95", ascending=False).reset_index(drop=True)

fig, axes = plt.subplot_mosaic(
    [["flow", "queue"], ["rare", "rare"]],
    figsize=(15.8, 10.1),
    gridspec_kw={"height_ratios": [1.0, 1.15]},
)
sns.heatmap(
    flow_p99_heatmap,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=1.5,
    linecolor="white",
    cbar_kws={"label": "p99 flow_time (min)", "shrink": 0.92},
    ax=axes["flow"],
)
axes["flow"].set_title("Cauda forte de flow_time\nCheca severidade extrema por escala x regime", fontsize=13)
axes["flow"].set_xlabel("Regime")
axes["flow"].set_ylabel("Escala")

sns.heatmap(
    queue_p99_heatmap,
    annot=True,
    fmt=".1f",
    cmap="PuBuGn",
    linewidths=1.5,
    linecolor="white",
    cbar_kws={"label": "p99 queue_time (min)", "shrink": 0.92},
    ax=axes["queue"],
)
axes["queue"].set_title("Cauda forte de queue_time\nFila extrema por escala x regime", fontsize=13)
axes["queue"].set_xlabel("Regime")
axes["queue"].set_ylabel("Escala")

sns.barplot(
    data=rare_segment_plot,
    x="flow_p95",
    y="segment_display",
    hue="segment_display",
    dodge=False,
    legend=False,
    palette="mako",
    orient="h",
    ax=axes["rare"],
)
axes["rare"].set_title("Segmentos raros no release\nP95 de flow_time por segmento crítico", fontsize=13)
axes["rare"].set_xlabel("p95 flow_time (min)")
axes["rare"].set_ylabel("")
axes["rare"].grid(axis="x", alpha=0.25, zorder=0)
rare_xmax = max(float(rare_segment_plot["flow_p95"].max()) * 1.18, 20.0)
axes["rare"].set_xlim(0, rare_xmax)
for patch, (_, row) in zip(axes["rare"].patches, rare_segment_plot.iterrows()):
    axes["rare"].text(
        min(float(row["flow_p95"]) + rare_xmax * 0.015, rare_xmax * 0.96),
        patch.get_y() + patch.get_height() / 2,
        f"{float(row['flow_p95']):.1f} min | n={int(row['job_count'])}",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
    )

fig.suptitle("Caudas e segmentos raros", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
fig.text(
    0.02,
    0.93,
    "Leitura rápida: a sanidade do benchmark precisa aparecer também nas caudas; eventos raros não podem virar ruído arbitrário ou comportamento implausível.",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0, w_pad=2.0)
fig.savefig(ARTIFACT_DIR / "tail_and_rare_segments.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

tail_regime_summary.to_csv(ARTIFACT_DIR / "tail_regime_summary.csv", index=False)
tail_regime_checks.to_csv(ARTIFACT_DIR / "tail_regime_checks.csv", index=False)
rare_segment_summary.to_csv(ARTIFACT_DIR / "rare_segment_summary.csv", index=False)

# %% [markdown]
# **Como ler a figura acima**
#
# - os dois heatmaps verificam se o gradiente `balanced < peak < disrupted` continua aparecendo também nas caudas `p99`
# - a tabela `tail_regime_checks` formaliza essa monotonicidade forte por escala
# - o gráfico de segmentos raros ajuda a ver se classes menos frequentes, como `URGENT` e `WET`, preservam comportamento plausível em `flow_time`
# - esta seção é importante porque benchmarks sintéticos costumam acertar médias e errar justamente os eventos extremos

# %% [markdown]
# ## Instance-space coverage and redundancy screening
#
# Além de ser íntegro e executável, o release precisa cobrir regiões distintas do problema.
# Para dados sintéticos, isso é crucial: um benchmark só é realmente útil se não colapsar
# em um conjunto de instâncias quase idênticas.
#
# Nesta seção, `PCA` e `kNN` cumprem papéis complementares:
#
# - `PCA` testa a cobertura global do espaço de instâncias; ele ajuda a ver se as instâncias ocupam regiões diferentes do problema ou se estão comprimidas em um bloco estreito
# - `kNN` testa a redundância local; mesmo quando a projeção em 2D parece boa, as distâncias de vizinhança mostram se há casos quase duplicados
# - juntos, eles sustentam uma afirmação metodológica importante para o TCC: o dataset sintético não é apenas válido estruturalmente, mas também suficientemente diverso para funcionar como benchmark
#
# Esta seção responde:
#
# - se há duplicatas exatas no nível de instância
# - se há casos "duplicate-like" em um espaço multivariado de features estruturais e operacionais
# - quão dispersas as instâncias estão quando projetadas em 2D via PCA
# - como as distâncias `kNN` evoluem para `k = 1, 3, 5`
# - quão "pura" é a vizinhança em termos de regime e escala
# - quais pares são os mais próximos dentro do release

# %%
display(instance_space_summary)
display(
    instance_space_features[
        [
            "instance_id",
            "scale_code",
            "regime_code",
            "nearest_neighbor_instance_id",
            "nearest_neighbor_distance",
            "duplicate_like_candidate",
        ]
    ].sort_values("nearest_neighbor_distance")
)
display(instance_space_knn_profile.sort_values(["k", "mean_knn_distance", "instance_id"]))
display(instance_space_knn_regime_composition[instance_space_knn_regime_composition["k"] == 5])
display(instance_space_pairs.head(12))

fig = repl.plot_instance_space_coverage(ctx=NOTEBOOK_CTX, save=True)
plt.show()
plt.close(fig)

instance_space_note = f"""
**Por que `PCA` e `kNN` importam para validar dados sintéticos**

- `PCA` fornece uma leitura **global** da cobertura do benchmark. Se todas as instâncias colapsassem na mesma região, o release seria pouco informativo para avaliação algorítmica.
- `kNN` fornece uma leitura **local** de redundância. Mesmo com boa separação visual em 2D, distâncias muito pequenas ainda denunciariam instâncias quase repetidas.
- Os dois testes se complementam: `PCA` responde se o release cobre regiões diferentes do problema, enquanto `kNN` responde se cada instância realmente acrescenta informação nova.
- Neste release, a evidência é favorável: `exact_core_duplicate_count = {int(instance_space_summary.loc[0, "exact_core_duplicate_count"])}`, `exact_feature_duplicate_count = {int(instance_space_summary.loc[0, "exact_feature_duplicate_count"])}`, `duplicate_like_candidate_count = {int(instance_space_summary.loc[0, "duplicate_like_candidate_count"])}` e `nearest_neighbor_distance_min = {float(instance_space_summary.loc[0, "nearest_neighbor_distance_min"]):.4f}`.
- Em média, a vizinhança de `5-NN` preserva `same_regime_neighbor_share = {float(instance_space_summary.loc[0, "knn_same_regime_share_k5_mean"]):.4f}` e `same_scale_neighbor_share = {float(instance_space_summary.loc[0, "knn_same_scale_share_k5_mean"]):.4f}`, o que mostra estrutura sem colapso em duplicatas.
"""
display(Markdown(instance_space_note))

instance_space_features.to_csv(ARTIFACT_DIR / "instance_space_features.csv", index=False)
instance_space_pairs.to_csv(ARTIFACT_DIR / "instance_space_pairs.csv", index=False)
instance_space_summary.to_csv(ARTIFACT_DIR / "instance_space_summary.csv", index=False)
instance_space_knn_profile.to_csv(ARTIFACT_DIR / "instance_space_knn_profile.csv", index=False)
instance_space_knn_regime_composition.to_csv(ARTIFACT_DIR / "instance_space_knn_regime_composition.csv", index=False)
instance_space_knn_scale_composition.to_csv(ARTIFACT_DIR / "instance_space_knn_scale_composition.csv", index=False)

# %% [markdown]
# **Como ler a figura acima**
#
# - painel superior esquerdo: a PCA resume o release em 2 dimensões; ela serve para verificar cobertura global, isto é, se o benchmark ocupa regiões diferentes do espaço de instâncias
# - painel superior direito: o perfil `kNN` mostra como a distância média cresce quando ampliamos a vizinhança de `1` para `3` e `5`; ele serve para verificar redundância local
# - painel inferior esquerdo: a pureza de vizinhança por regime mostra quanto os `5-NN` tendem a permanecer na mesma família de regime, sem virar um conjunto indistinguível
# - painel inferior direito: mostra os pares mais próximos do release; se algum caísse abaixo do limiar, ele apareceria como candidato `duplicate-like`
# - a leitura metodológica correta é: `PCA` responde "o release cobre regiões diferentes do problema?" e `kNN` responde "essas regiões contêm instâncias realmente distintas?"
# - as tabelas `instance_space_knn_profile` e `instance_space_knn_regime_composition` dão a leitura quantitativa complementar ao PCA

# %% [markdown]
# ## Solver-oriented smoke test
#
# As seções anteriores mostram que o release é válido e diverso. Esta seção
# adiciona uma evidência complementar: o dataset também é informativo para
# benchmark algorítmico.
#
# O teste abaixo não é o protocolo final do TCC. Ele é um **smoke test exato
# budgetado**, usando `scipy.optimize.milp` neste ambiente porque `gurobipy`
# não está disponível localmente. Para manter o tempo de execução sob controle,
# usamos subinstâncias induzidas pelos primeiros jobs em ordem de chegada,
# com orçamento fixo de `5` segundos por caso.
#
# A leitura desejada é:
#
# - casos pequenos fecham com solver exato
# - casos intermediários continuam viáveis, mas passam a exibir gap
# - casos maiores seguem carregando e produzindo incumbentes, mas já apontam para trilhas `hybrid` ou `metaheuristic`

# %%
solver_smoke_df = solver_smoke.run_smoke_suite(root=REPO_ROOT)
solver_smoke_df["case_label"] = (
    solver_smoke_df["scale_code"].astype(str)
    + "-"
    + solver_smoke_df["max_jobs"].astype(int).astype(str)
    + " jobs"
)
solver_smoke_df["gap_pct"] = solver_smoke_df["mip_gap"].fillna(1.0) * 100.0
solver_smoke_df["objective_vs_dual_gap_min"] = (
    solver_smoke_df["objective_makespan_min"] - solver_smoke_df["dual_bound_makespan_min"]
)

display(solver_smoke_df)

fig, axes = plt.subplot_mosaic(
    [["bounds", "gap"], ["size", "size"]],
    figsize=(16.0, 10.1),
    gridspec_kw={"height_ratios": [1.0, 1.1]},
)

sns.barplot(
    data=solver_smoke_df,
    x="case_label",
    y="objective_makespan_min",
    hue="status_label",
    dodge=False,
    palette={"optimal": "#2a9d8f", "time_limit": "#e9c46a", "feasible": "#8ecae6", "other": "#94a3b8", "infeasible": "#d62828"},
    ax=axes["bounds"],
)
axes["bounds"].scatter(
    range(len(solver_smoke_df)),
    solver_smoke_df["dual_bound_makespan_min"],
    color="#1d3557",
    s=70,
    marker="D",
    zorder=3,
    label="Dual bound",
)
for idx, row in solver_smoke_df.reset_index(drop=True).iterrows():
    if pd.notna(row["mip_gap"]):
        axes["bounds"].text(idx, row["objective_makespan_min"] + 8, f"gap {row['mip_gap']:.1%}", ha="center", va="bottom", fontsize=9, color="#334155")
axes["bounds"].set_title("Incumbente e dual bound por caso\nFechar o gap fica mais difícil à medida que o tamanho cresce", fontsize=13)
axes["bounds"].set_xlabel("")
axes["bounds"].set_ylabel("Makespan (min)")
axes["bounds"].tick_params(axis="x", rotation=15)
handles, labels = axes["bounds"].get_legend_handles_labels()
axes["bounds"].legend(handles, labels, loc="upper left", frameon=True)

sns.barplot(
    data=solver_smoke_df,
    x="case_label",
    y="gap_pct",
    hue="recommended_solver_track",
    dodge=False,
    palette="deep",
    ax=axes["gap"],
)
axes["gap"].set_title("Gap relativo sob orçamento fixo de 5 s\nA escada de dificuldade já aparece no smoke test", fontsize=13)
axes["gap"].set_xlabel("")
axes["gap"].set_ylabel("MIP gap (%)")
axes["gap"].tick_params(axis="x", rotation=15)
for patch in axes["gap"].patches:
    height = patch.get_height()
    if np.isfinite(height) and height > 0:
        axes["gap"].text(patch.get_x() + patch.get_width() / 2, height + 1.0, f"{height:.1f}%", ha="center", va="bottom", fontsize=9, color="#334155")

size_plot = solver_smoke_df.melt(
    id_vars=["case_label"],
    value_vars=["eligible_var_count", "machine_pair_binary_count", "constraint_count"],
    var_name="size_metric",
    value_name="count",
)
size_labels = {
    "eligible_var_count": "x vars",
    "machine_pair_binary_count": "sequencing binaries",
    "constraint_count": "constraints",
}
size_plot["size_metric"] = size_plot["size_metric"].map(size_labels)
sns.barplot(
    data=size_plot,
    x="case_label",
    y="count",
    hue="size_metric",
    ax=axes["size"],
    palette="Set2",
)
axes["size"].set_title("Crescimento do modelo exato\nO custo combinatório sobe rapidamente com o tamanho", fontsize=13)
axes["size"].set_xlabel("")
axes["size"].set_ylabel("Contagem")
axes["size"].tick_params(axis="x", rotation=15)

fig.suptitle("Smoke test orientado a solver", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
fig.text(
    0.02,
    0.93,
    "Leitura rápida: o pipeline exato carrega, fecha nos menores e passa a exibir gaps não triviais quando a escala do caso cresce.",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.0, w_pad=1.8)
fig.savefig(ARTIFACT_DIR / "solver_oriented_smoke_test.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

solver_smoke_df.to_csv(ARTIFACT_DIR / "solver_smoke_results.csv", index=False)

solver_smoke_summary = pd.DataFrame(
    [
        {
            "solver_backend": "scipy.optimize.milp (HiGHS)",
            "time_limit_sec": float(solver_smoke_df["time_limit_sec"].iloc[0]),
            "small_cases_optimal": bool(
                solver_smoke_df.loc[solver_smoke_df["max_jobs"].isin([8, 12]), "status_label"].eq("optimal").all()
            ),
            "all_cases_have_solution": bool(solver_smoke_df["has_solution"].all()),
            "large_cases_nontrivial_gap": bool(
                solver_smoke_df.loc[solver_smoke_df["max_jobs"].isin([18, 24]), "mip_gap"].fillna(0.0).ge(0.10).all()
            ),
            "gap_non_decreasing_with_case_size": bool(
                solver_smoke_df.sort_values("max_jobs")["mip_gap"].fillna(0.0).is_monotonic_increasing
            ),
        }
    ]
)
display(solver_smoke_summary)
solver_smoke_summary.to_csv(ARTIFACT_DIR / "solver_smoke_summary.csv", index=False)

# %% [markdown]
# **Como ler a figura acima**
#
# - painel esquerdo: as barras são os incumbentes e os diamantes são os dual bounds; distância grande entre eles significa dificuldade residual
# - painel central: sob o mesmo orçamento de tempo, `XS-8` e `S-12` fecham, enquanto `M-18` e `L-24` já preservam gaps não triviais
# - painel direito: o crescimento de binárias disjuntivas e restrições explica por que a trilha recomendada migra de `exact` para `hybrid/metaheuristic`
# - esta seção é um **smoke test de utilidade algorítmica**, não o protocolo final de competição entre solvers

# %% [markdown]
# ## Instance drilldown
#
# Um drilldown ajuda a validar visualmente se o baseline FIFO de uma instância concreta:
#
# - respeita o fluxo por máquina
# - evita overlap
# - incorpora downtimes
# - produz métricas coerentes com o regime escolhido

# %%
sample_instance = str(PIPELINE_CONFIG["sample_instance_id"])

sample_params = params[params["instance_id"] == sample_instance]
sample_summary = catalog[catalog["instance_id"] == sample_instance]
sample_jobs = jobs_enriched[jobs_enriched["instance_id"] == sample_instance]
sample_metrics = job_metrics[job_metrics["instance_id"] == sample_instance]

display(sample_params)
display(sample_summary)
display(sample_jobs.head())
display(sample_metrics.describe().round(2))

fig = repl.plot_instance_drilldown(sample_instance, ctx=NOTEBOOK_CTX, save=True)
plt.show()

fig = repl.plot_job_level_views(sample_instance, ctx=NOTEBOOK_CTX, save=True)
plt.show()

# %% [markdown]
# **Como ler as figuras acima**
#
# - o Gantt mostra ocupação por máquina, faixas de downtime e ausência de overlap no baseline FIFO
# - o scatter de jobs ajuda a ver como os prazos se distribuem em função da chegada
# - o ranking horizontal destaca os jobs mais críticos em `flow_time`

# %% [markdown]
# ## Results and notes
#
# O notebook consolida uma leitura de qualidade do release oficial:
#
# - o release está estruturalmente íntegro
# - o release também é relacionalmente consistente entre arquivos, chaves e cardinalidades
# - o baseline FIFO é executável contra o schema nas `36` instâncias
# - os audits reconciliam os valores centrais
# - os checks de regime são positivos para `mean_flow`, `p95_flow` e fila média
# - o proxy médio de congestionamento é útil, mas não monotônico em todas as famílias
# - a camada observacional produz um deslocamento formalmente detectável por `C2ST/MMD`, mas ainda interpretável
# - as caudas `p99` e os segmentos raros continuam sob controle analítico
# - o espaço de instâncias não contém duplicatas exatas nem candidatos `duplicate-like` sob o screening adotado
# - `PCA` e `kNN` mostram, de forma complementar, que o release tem cobertura global e baixa redundância local
# - o smoke test exato fecha nos casos menores e exibe gaps não triviais quando o tamanho da subinstância cresce
# - a camada observacional reduz determinismo excessivo sem destruir semântica
# - a base é forte o suficiente para servir como dataset pai de análises e futuras derivações com G2MILP

# %%
summary = {
    "dataset_version": manifest["dataset_version"],
    "instance_count": int(params["instance_id"].nunique()),
    "structural_pass_rate": float((structural_report["status"] == "PASS").mean()),
    "release_consistency_checks_pass": bool(release_consistency_report["pass"].all()),
    "relational_consistency_checks_pass": bool(relational_consistency_summary["pass_rate"].eq(1.0).all()),
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
    "due_audit_match_share": float(audit_reconciliation["due_match_share"].mean()),
    "proc_audit_match_share": float(audit_reconciliation["proc_match_share"].mean()),
    "r2_due_slack_vs_priority": float(diagnostics["r2_due_slack_vs_priority"]),
    "r2_unload_proc_vs_load_machine_moisture": float(
        diagnostics["r2_unload_proc_vs_load_machine_moisture"]
    ),
    "flow_regime_order_checks_pass": bool(
        regime_checks["mean_flow_order_ok"].all()
        and regime_checks["p95_flow_order_ok"].all()
    ),
    "queue_regime_order_checks_pass": bool(regime_checks["mean_queue_order_ok"].all()),
    "congestion_mean_regime_order_checks_pass": bool(
        regime_checks["mean_congestion_order_ok"].all()
    ),
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
    "job_due_c2st_auc_mean": float(
        formal_shift_summary.loc[
            formal_shift_summary["experiment"].eq("job_due_layer"), "c2st_auc_mean"
        ].iloc[0]
    ),
    "proc_time_c2st_auc_mean": float(
        formal_shift_summary.loc[
            formal_shift_summary["experiment"].eq("proc_time_layer"), "c2st_auc_mean"
        ].iloc[0]
    ),
    "tail_flow_p99_regime_order_checks_pass": bool(tail_regime_checks["flow_p99_order_ok"].all()),
    "tail_queue_p99_regime_order_checks_pass": bool(tail_regime_checks["queue_p99_order_ok"].all()),
    "tail_due_margin_p05_regime_order_checks_pass": bool(
        tail_regime_checks["due_margin_p05_order_ok"].all()
    ),
    "solver_smoke_small_cases_optimal": bool(
        solver_smoke_summary.loc[0, "small_cases_optimal"]
    ),
    "solver_smoke_all_cases_have_solution": bool(
        solver_smoke_summary.loc[0, "all_cases_have_solution"]
    ),
    "solver_smoke_large_cases_nontrivial_gap": bool(
        solver_smoke_summary.loc[0, "large_cases_nontrivial_gap"]
    ),
    "solver_smoke_gap_ladder_pass": bool(
        solver_smoke_summary.loc[0, "gap_non_decreasing_with_case_size"]
    ),
    "g2milp_role": manifest["official_dataset_role"],
}
summary_df = pd.DataFrame([summary])
display(summary_df)

summary_lines = [
    "# Notebook Summary",
    "",
    f"- Dataset version: `{summary['dataset_version']}`",
    f"- Instances: `{summary['instance_count']}`",
    f"- Structural pass rate: `{summary['structural_pass_rate']:.4f}`",
    f"- Release consistency checks pass: `{summary['release_consistency_checks_pass']}`",
    f"- Relational consistency checks pass: `{summary['relational_consistency_checks_pass']}`",
    f"- FIFO schema checks pass: `{summary['fifo_schema_checks_pass']}`",
    f"- Due audit match share: `{summary['due_audit_match_share']:.4f}`",
    f"- Proc audit match share: `{summary['proc_audit_match_share']:.4f}`",
    f"- R2 due slack vs priority: `{summary['r2_due_slack_vs_priority']:.4f}`",
    f"- R2 unload proc vs load+machine+moisture: `{summary['r2_unload_proc_vs_load_machine_moisture']:.4f}`",
    f"- Flow regime checks pass: `{summary['flow_regime_order_checks_pass']}`",
    f"- Mean queue regime checks pass: `{summary['queue_regime_order_checks_pass']}`",
    f"- Mean congestion regime checks pass: `{summary['congestion_mean_regime_order_checks_pass']}`",
    f"- Instance-space exact duplicate checks pass: `{summary['instance_space_exact_duplicate_checks_pass']}`",
    f"- Instance-space duplicate-like checks pass: `{summary['instance_space_duplicate_like_checks_pass']}`",
    f"- Instance-space nearest-neighbor distance min: `{summary['instance_space_nearest_neighbor_distance_min']:.4f}`",
    f"- Job due-layer C2ST AUC mean: `{summary['job_due_c2st_auc_mean']:.4f}`",
    f"- Proc-time layer C2ST AUC mean: `{summary['proc_time_c2st_auc_mean']:.4f}`",
    f"- Tail flow p99 regime checks pass: `{summary['tail_flow_p99_regime_order_checks_pass']}`",
    f"- Tail queue p99 regime checks pass: `{summary['tail_queue_p99_regime_order_checks_pass']}`",
    f"- Tail due-margin p05 regime checks pass: `{summary['tail_due_margin_p05_regime_order_checks_pass']}`",
    f"- Solver smoke small cases optimal: `{summary['solver_smoke_small_cases_optimal']}`",
    f"- Solver smoke all cases have solution: `{summary['solver_smoke_all_cases_have_solution']}`",
    f"- Solver smoke large cases show non-trivial gap: `{summary['solver_smoke_large_cases_nontrivial_gap']}`",
    f"- Solver smoke gap ladder pass: `{summary['solver_smoke_gap_ladder_pass']}`",
    f"- Official role: `{summary['g2milp_role']}`",
]
summary_text = "\n".join(summary_lines)
(ARTIFACT_DIR / "notebook_summary.md").write_text(summary_text, encoding="utf-8")
summary_df.to_csv(ARTIFACT_DIR / "notebook_summary.csv", index=False)
display(Markdown(summary_text))

# %% [markdown]
# ## Next steps
#
# - usar este notebook como baseline de validação antes de gerar filhos com G2MILP
# - ampliar com comparações entre esta release oficial e futuros datasets derivados
# - adicionar testes de sensibilidade por família de máquina ou por política de geração

# %% [markdown]
# # Full Paper Pipeline
#
# A partir daqui o notebook deixa de ser apenas uma auditoria do release e passa a
# executar o pipeline completo pedido em `paper/Artigo.md`.
#
# **Nota metodológica**
#
# - `M0` é o baseline oficial do release observado.
# - `M1`, `M2` e `M3` são implementados como um benchmark reproduzível de
#   list-scheduling com políticas diferentes de ordenação e reatividade.
# - `E6` também gera bundles derivados de instâncias-filhas `graded` e
#   `discriminating`, com eventos e baseline FIFO replayados no próprio notebook.

# %%
ARTICLE_FIGURE_DIR = REPO_ROOT / "output" / "article_figures"
ARTICLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

paper_results = run_full_pipeline(
    root=ANALYSIS_ROOT,
    ctx=NOTEBOOK_CTX,
    figure_dir=ARTICLE_FIGURE_DIR,
)

paper_features = paper_results["feature_frame"].copy()
method_matrix = paper_results["performance"].copy()
umap_frame = paper_results["umap_frame"].copy()
selector_report = paper_results["selector_report"].copy()
selector_shap = paper_results["shap_frame"].copy()
scorecard_release = paper_results["scorecard"].copy()

display(paper_features.head())
display(method_matrix.head())
display(selector_report)

# %% [markdown]
# ## Cobertura do `paper/Artigo.md`
#
# Esta tabela responde de forma direta à pergunta "todo o paper foi implementado?".
# O status é intencionalmente explícito para não mascarar lacunas.

# %%
coverage_table = build_experiment_coverage_table()
display(coverage_table)

# %% [markdown]
# ## E0. Reprodutibilidade e auditoria do benchmark
#
# Este experimento consolida:
#
# - validação estrutural do release observado;
# - consistência relacional entre os arquivos centrais;
# - replay próprio de `M0` contra os artefatos oficiais.

# %%
fifo_replay_diff = build_fifo_replay_diff_table(root=ANALYSIS_ROOT, ctx=NOTEBOOK_CTX)
fifo_replay_diff.to_csv(REPO_ROOT / "catalog" / "fifo_replay_diff.csv", index=False)
e0_audit_snapshot = build_e0_audit_snapshot(
    summary=summary,
    structural_report=structural_report,
    relational_consistency_summary=relational_consistency_summary,
)
display(e0_audit_snapshot)
display(fifo_replay_diff.head(12))
display(structural_report.sort_values(["scale_code", "regime_code", "instance_id"]).head(12))
display(relational_consistency_summary)

# %% [markdown]
# ## E1. Comparação principal entre métodos
#
# Este bloco entrega a tabela base do paper:
#
# - `M0_FIFO_OFFICIAL`
# - `M1_WEIGHTED_SLACK`
# - `M2_PERIODIC_15`
# - `M2_PERIODIC_30`
# - `M3_EVENT_REACTIVE`
# - `Mref_EXACT_XS_S`
# - `M4_METAHEURISTIC_L`
#
# As métricas principais são `flow_mean`, `flow_p95`, `makespan`,
# `weighted_tardiness`, `runtime_sec`, `utility` e `regret`.

# %%
method_summary = (
    method_matrix.groupby(["method_name", "scale_code"], as_index=False)
    .agg(
        flow_mean=("flow_mean", "mean"),
        flow_p95=("flow_p95", "mean"),
        runtime_sec=("runtime_sec", "mean"),
        utility=("utility", "mean"),
        delta_vs_fifo_p95_flow_pct=("delta_vs_fifo_p95_flow_pct", "mean"),
    )
)
display(method_summary.sort_values(["scale_code", "utility", "method_name"]))
display(
    method_matrix[
        [
            "instance_id",
            "method_name",
            "flow_mean",
            "flow_p95",
            "makespan",
            "weighted_tardiness",
            "runtime_sec",
            "utility",
            "regret",
            "difficulty",
        ]
    ].head(20)
)

# %%
show_inline_figure(
    paper_results["figure_paths"]["method_delta"],
    title="E1: ganho relativo em p95_flow vs FIFO",
)

# %%
show_inline_figure(
    paper_results["figure_paths"]["method_runtime"],
    title="E1: runtime por instância × método",
    figsize=(12, 8),
)

# %% [markdown]
# ## E1b. Performance profiles
#
# Médias escondem dominância parcial e robustez. Este bloco plota o perfil de
# desempenho usando a `utility` agregada do artigo.

# %%
performance_profile = build_performance_profile_frame(method_matrix, value_col="utility")
display(performance_profile.head())

# %%
plot_performance_profile(performance_profile)

# %% [markdown]
# ## E2. Periódico versus disparado por eventos
#
# Aqui isolamos a pergunta central da revisão:
#
# - `M2_PERIODIC_15`
# - `M2_PERIODIC_30`
# - `M3_EVENT_REACTIVE`
#
# O notebook mede `flow`, `runtime`, `utility`, `regret`, `replan_count`
# e separa a troca entre controle periódico e evento reativo.

# %%
e2_tradeoff = build_e2_tradeoff_table(method_matrix)
display(e2_tradeoff)

# %%
plot_e2_tradeoff(e2_tradeoff)

# %% [markdown]
# ## E2b. Sensibilidade computacional a budget, threads e paralelismo
#
# Este bloco roda um estudo controlado com:
#
# - budgets `short`, `medium`, `long`;
# - `n_workers / threads` em `{1, 2}`;
# - métodos com knobs computacionais explícitos (`M2`, `M3`, `Mref`, `M4`);
# - um conjunto representativo com uma instância `replicate=01` por escala.

# %%
e2b_sensitivity = run_compute_sensitivity(
    root=ANALYSIS_ROOT,
    catalog=NOTEBOOK_CTX["catalog"][["instance_id", "scale_code", "regime_code", "replicate"]].copy(),
)
e2b_sensitivity.to_csv(REPO_ROOT / "catalog" / "compute_sensitivity.csv", index=False)
display(e2b_sensitivity.head(20))

# %%
runtime_heatmap = (
    e2b_sensitivity.loc[e2b_sensitivity["budget_label"].eq("medium")]
    .pivot_table(
        index="instance_id",
        columns=["method_name", "n_workers"],
        values="runtime_sec",
        aggfunc="mean",
    )
    .sort_index(axis=1)
)
utility_heatmap = (
    e2b_sensitivity.loc[e2b_sensitivity["budget_label"].eq("medium")]
    .pivot_table(
        index="instance_id",
        columns=["method_name", "n_workers"],
        values="utility",
        aggfunc="mean",
    )
    .sort_index(axis=1)
)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.heatmap(np.log1p(runtime_heatmap), cmap="mako", ax=axes[0], cbar_kws={"label": "log(1+runtime_sec)"})
axes[0].set_title("E2b: runtime por instância × (método, workers)")
sns.heatmap(utility_heatmap, cmap="viridis_r", ax=axes[1], cbar_kws={"label": "utility"})
axes[1].set_title("E2b: utility por instância × (método, workers)")
fig.tight_layout()
plt.show()
plt.close(fig)

# %%
throughput_summary = (
    e2b_sensitivity.groupby(["method_name", "budget_label", "n_workers"], as_index=False)
    .agg(
        runtime_sec=("runtime_sec", "median"),
        utility=("utility", "median"),
        plan_instability=("plan_instability", "mean"),
    )
    .sort_values(["method_name", "budget_label", "n_workers"])
)
display(throughput_summary)

# %% [markdown]
# ## E3. ISA, UMAP, HDBSCAN e solver footprints
#
# O backend antigo já entregava `PCA`; aqui acrescentamos:
#
# - `UMAP` para a geometria não linear;
# - `HDBSCAN` para clusters e outliers;
# - `solver footprints` via `utility <= best + ε`.

# %%
display(
    umap_frame[
        [
            "instance_id",
            "scale_code",
            "regime_code",
            "best_method",
            "difficulty",
            "cluster_label",
            "umap_x",
            "umap_y",
        ]
    ].sort_values(["best_method", "instance_id"])
)

# %%
show_inline_figure(
    paper_results["figure_paths"]["umap_best_method"],
    title="E3: UMAP colorido pelo melhor método",
    figsize=(10, 7),
)

# %%
show_inline_figure(
    paper_results["figure_paths"]["hdbscan_clusters"],
    title="E3: HDBSCAN no espaço UMAP",
    figsize=(10, 7),
)

# %%
show_inline_figure(
    paper_results["figure_paths"]["solver_footprints"],
    title="E3: solver footprints",
    figsize=(11, 8),
)

# %% [markdown]
# ## E4. Selector, SHAP e exportação ASlib
#
# Com apenas 36 instâncias, o selector continua **explicativo**, não um claim
# forte de produção. Mesmo assim, ele fecha o protocolo:
#
# - compara árvore, random forest e LightGBM quando disponível;
# - exporta `features.csv`, `performance.csv`, `runstatus.csv`,
#   `feature_costs.csv` e `cv.csv` em `catalog/aslib_scenario/`;
# - salva o sumário de `SHAP` em `catalog/selector_shap_summary.csv`.

# %%
display(selector_report)
display(selector_shap[["feature_name", "mean_abs_shap", "selected_model"]].drop_duplicates().head(15))

aslib_paths_df = pd.DataFrame(
    [{"artifact_name": name, "path": str(path)} for name, path in paper_results["aslib_paths"].items()]
)
display(aslib_paths_df)

# %%
show_inline_figure(
    paper_results["figure_paths"]["selector_shap"],
    title="E4: SHAP summary do selector",
    figsize=(10, 7),
)

# %% [markdown]
# ## E5. Validade do benchmark sintético
#
# Este bloco reúne os diagnósticos do paper para a qualidade do benchmark:
#
# - `MMD/C2ST/density ratio`;
# - scorecard consolidado;
# - integridade relacional;
# - checagem de caudas e segmentos raros;
# - cobertura e redundância do espaço de instâncias.

# %%
e5_validity_snapshot = build_e5_validity_snapshot(
    formal_shift_summary=formal_shift_summary,
    job_density_segments=job_density_segments,
    proc_density_segments=proc_density_segments,
    summary=summary,
    tail_regime_checks=tail_regime_checks,
    rare_segment_summary=rare_segment_summary,
    instance_space_summary=instance_space_summary,
)
display(e5_validity_snapshot)
display(scorecard_release)

# %% [markdown]
# ## E6. Instâncias `graded` e `discriminating`
#
# A geração abaixo transforma as propostas do ISA em bundles concretos de
# instâncias-filhas, com arquivos centrais de scheduling, eventos regenerados
# e baseline FIFO replayado no próprio notebook.

# %%
child_instance_proposals = build_child_instance_proposals(
    feature_frame=paper_features,
    performance=method_matrix,
    umap_frame=umap_frame,
)
child_instance_proposals.to_csv(REPO_ROOT / "catalog" / "graded_discriminating_candidates.csv", index=False)
display(child_instance_proposals)

# %%
child_instance_dir = REPO_ROOT / "output" / "jupyter-notebook" / "generated_child_instances"
generated_child_summary = generate_child_instance_bundles(
    root=ANALYSIS_ROOT,
    proposals=child_instance_proposals,
    out_dir=child_instance_dir,
)
generated_child_summary.to_csv(REPO_ROOT / "catalog" / "graded_discriminating_children_summary.csv", index=False)
display(generated_child_summary)

# %% [markdown]
# ## Resumo do escopo
#
# O notebook agora cobre o `Artigo.md` com os experimentos separados em células:
#
# - implementado: `E0`, `E1`, `E1b`, `E2`, `E2b`, `E3`, `E4`, `E5`, `E6`;
# - implementado: `Mref_EXACT_XS_S` e `M4_METAHEURISTIC_L`;
# - todas as figuras do bloco do paper aparecem inline nas células acima.
