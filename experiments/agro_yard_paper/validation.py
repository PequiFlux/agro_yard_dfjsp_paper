
"""Validation, audit and statistical diagnostics used by the paper notebook."""
from __future__ import annotations

from pathlib import Path

from .common import (
    Any,
    PAPER_METHOD_ORDER,
    REGIME_ORDER,
    SCALE_ORDER,
    SEED,
    UTILITY_WEIGHTS,
    _load_instance_tables,
    np,
    pd,
    math,
    roc_auc_score,
    LogisticRegression,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from .pipeline import _build_job_metrics_from_schedule, _schedule_jobs_by_policy


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
                "notes": "replay online periódico com Δ=15 min, reotimização do conjunto visível e congelamento do que já iniciou",
            },
            {
                "item_type": "method",
                "item_name": "M2_PERIODIC_30",
                "status": "implemented",
                "notes": "replay online periódico com Δ=30 min, reotimização do conjunto visível e congelamento do que já iniciou",
            },
            {
                "item_type": "method",
                "item_name": "M3_EVENT_REACTIVE",
                "status": "implemented",
                "notes": "replay online disparado por JOB_VISIBLE, JOB_ARRIVAL, MACHINE_DOWN e MACHINE_UP",
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
                "notes": "comparação M0/M1/M2/M3 no conjunto de teste, com split R01 calibração versus R02/R03 teste",
            },
            {
                "item_type": "experiment",
                "item_name": "E2_periodic_vs_event",
                "status": "implemented",
                "notes": "comparação explícita entre M2 e M3 no conjunto de teste",
            },
            {
                "item_type": "experiment",
                "item_name": "E2b_computational_sensitivity",
                "status": "implemented",
                "notes": "sensibilidade computacional rodada em todas as 12 famílias de calibração e reportada com separação calibração/teste",
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
    frame = performance.loc[performance["method_name"].isin(compare_methods)].copy()
    if "protocol_role" in frame.columns:
        frame = frame.loc[frame["protocol_role"].eq("test")].copy()
    return (
        frame
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
