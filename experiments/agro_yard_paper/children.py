
"""Generation of graded/discriminating child instances plus cache-aware notebook wrappers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .common import Any, np, time
from .pipeline import _build_job_order, _schedule_jobs_from_order, _summarize_schedule


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


def load_or_generate_child_instances(
    root: Path,
    proposals: pd.DataFrame,
    out_dir: Path,
    summary_path: Path,
) -> pd.DataFrame:
    if summary_path.exists():
        return pd.read_csv(summary_path)
    summary = generate_child_instance_bundles(root=root, proposals=proposals, out_dir=out_dir)
    summary.to_csv(summary_path, index=False)
    return summary
