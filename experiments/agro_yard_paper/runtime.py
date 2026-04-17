
"""Runtime bootstrap and release loading helpers for the Agro Yard paper package."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import REPO_ROOT, ReleaseBundle, observed_release_builder, repl

DEFAULT_PIPELINE_CONFIG = {
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


def load_release_bundle(root: Path, artifact_dir: Path) -> ReleaseBundle:
    ctx = repl.load_context(root=root, artifact_dir=artifact_dir)
    return ReleaseBundle(
        root=root,
        artifact_dir=artifact_dir,
        ctx=ctx,
        summary=ctx["summary"],
        validation_observed=pd.read_csv(root / "catalog" / "validation_report_observed.csv"),
        validation_nominal_style=pd.read_csv(root / "catalog" / "validation_report.csv"),
        g2milp_contract=json.loads((root / "catalog" / "g2milp_generation_contract.json").read_text(encoding="utf-8")),
        params=ctx["params"].copy(),
        catalog=ctx["catalog"].copy(),
        family_summary=ctx["family_summary"].copy(),
        observed_noise_manifest=ctx["observed_noise_manifest"],
        manifest=ctx["manifest"],
        jobs=ctx["jobs"].copy(),
        jobs_enriched=ctx["jobs_enriched"].copy(),
        operations=ctx["operations"].copy(),
        eligible=ctx["eligible"].copy(),
        machines=ctx["machines"].copy(),
        precedences=ctx["precedences"].copy(),
        downtimes=ctx["downtimes"].copy(),
        events=ctx["events"].copy(),
        schedule=ctx["schedule"].copy(),
        job_metrics=ctx["job_metrics"].copy(),
        due_audit=ctx["due_audit"].copy(),
        proc_audit=ctx["proc_audit"].copy(),
        proc_audit_enriched=ctx["proc_audit_enriched"].copy(),
        congestion=ctx["congestion"].copy(),
        structural_report=ctx["structural_report"].copy(),
        event_report=ctx["event_report"].copy(),
        audit_reconciliation=ctx["audit_reconciliation"].copy(),
        regime_checks=ctx["regime_checks"].copy(),
        fifo_schema_report=ctx["fifo_schema_report"].copy(),
        release_consistency_report=ctx["release_consistency_report"].copy(),
        utilization=ctx["utilization"].copy(),
        instance_space_features=ctx["instance_space_features"].copy(),
        instance_space_pairs=ctx["instance_space_pairs"].copy(),
        instance_space_summary=ctx["instance_space_summary"].copy(),
        instance_space_knn_profile=ctx["instance_space_knn_profile"].copy(),
        instance_space_knn_regime_composition=ctx["instance_space_knn_regime_composition"].copy(),
        instance_space_knn_scale_composition=ctx["instance_space_knn_scale_composition"].copy(),
        diagnostics=ctx["diagnostics"].copy() if hasattr(ctx["diagnostics"], "copy") else dict(ctx["diagnostics"]),
        unload=ctx["unload"].copy(),
    )
