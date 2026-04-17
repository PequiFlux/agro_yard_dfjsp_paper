
"""Public API for the Agro Yard paper package used by the notebook."""
from __future__ import annotations

import importlib
import sys

from . import children, common, pipeline, plots, runtime, sensitivity, validation
from .children import build_child_instance_proposals, load_or_generate_child_instances
from .common import (
    METHOD_LABELS,
    METHOD_ORDER,
    PAPER_METHOD_ORDER,
    REGIME_ORDER,
    SCALE_ORDER,
    SEED,
    STAGE_ORDER,
    PaperPipelineResults,
    ReleaseBundle,
    repl,
    solver_smoke,
)
from .pipeline import load_or_run_full_pipeline
from .plots import (
    plot_e2_tradeoff,
    plot_hdbscan_clusters,
    plot_method_delta,
    plot_method_runtime,
    plot_performance_profile,
    plot_selector_shap,
    plot_solver_footprints,
    plot_umap_best_method,
)
from .runtime import DEFAULT_PIPELINE_CONFIG, load_release_bundle, prepare_analysis_root
from .sensitivity import load_or_run_compute_sensitivity
from .validation import (
    build_e0_audit_snapshot,
    build_e2_tradeoff_table,
    build_e5_validity_snapshot,
    build_experiment_coverage_table,
    build_fifo_replay_diff_table,
    build_performance_profile_frame,
    build_relational_consistency_reports,
    build_tail_and_segment_reports,
    run_domain_shift_experiment,
)

__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "METHOD_LABELS",
    "METHOD_ORDER",
    "PAPER_METHOD_ORDER",
    "PaperPipelineResults",
    "REGIME_ORDER",
    "ReleaseBundle",
    "SCALE_ORDER",
    "SEED",
    "STAGE_ORDER",
    "build_child_instance_proposals",
    "build_e0_audit_snapshot",
    "build_e2_tradeoff_table",
    "build_e5_validity_snapshot",
    "build_experiment_coverage_table",
    "build_fifo_replay_diff_table",
    "build_performance_profile_frame",
    "build_relational_consistency_reports",
    "build_tail_and_segment_reports",
    "load_or_generate_child_instances",
    "load_or_run_compute_sensitivity",
    "load_or_run_full_pipeline",
    "load_release_bundle",
    "plot_e2_tradeoff",
    "plot_hdbscan_clusters",
    "plot_method_delta",
    "plot_method_runtime",
    "plot_performance_profile",
    "plot_selector_shap",
    "plot_solver_footprints",
    "plot_umap_best_method",
    "prepare_analysis_root",
    "repl",
    "run_domain_shift_experiment",
    "solver_smoke",
]


def reload_package():
    module_names = [
        "experiments.agro_yard_paper.common",
        "experiments.agro_yard_paper.runtime",
        "experiments.agro_yard_paper.plots",
        "experiments.agro_yard_paper.pipeline",
        "experiments.agro_yard_paper.validation",
        "experiments.agro_yard_paper.sensitivity",
        "experiments.agro_yard_paper.children",
        __name__,
    ]
    for module_name in module_names[:-1]:
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)
    return importlib.reload(sys.modules[__name__])
