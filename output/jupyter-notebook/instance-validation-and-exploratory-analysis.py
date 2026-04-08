# Generated from output/jupyter-notebook/instance-validation-and-exploratory-analysis.ipynb
# Run as a percent script in editors that support `# %%` cells, or as plain Python.

# %% [markdown]
# # Experiment: Instance Validation and Exploratory Analysis
#
# **Objetivo**
#
# Usar o próprio notebook como workspace interativo principal para gerar, validar e explorar o release oficial `v1.1.0-observed`, reaproveitando o backend consolidado em `tools/instance_analysis_repl.py`.
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
# O módulo `tools/instance_analysis_repl.py` funciona como backend compartilhado da análise, para evitar duas implementações diferentes do mesmo pipeline analítico.

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
from IPython.display import Markdown, display

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

import create_observed_noise_layer as observed_release_builder
import instance_analysis_repl as repl
import exact_solver_smoke as solver_smoke

observed_release_builder = importlib.reload(observed_release_builder)
repl = importlib.reload(repl)
solver_smoke = importlib.reload(solver_smoke)

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
