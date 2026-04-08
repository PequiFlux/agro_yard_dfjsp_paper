from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


PACKAGE_CHECKS = [
    ("gurobipy", "gurobipy"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("umap", "umap-learn"),
    ("hdbscan", "hdbscan"),
    ("shap", "shap"),
    ("jupyterlab", "jupyterlab"),
    ("notebook", "notebook"),
    ("ipykernel", "ipykernel"),
    ("nbclient", "nbclient"),
    ("ipywidgets", "ipywidgets"),
    ("psutil", "psutil"),
    ("joblib", "joblib"),
    ("tqdm", "tqdm"),
]


def assert_imports() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for module_name, package_name in PACKAGE_CHECKS:
        importlib.import_module(module_name)
        rows.append({"package": package_name, "status": "ok"})
    return rows


def assert_license_file() -> Path:
    license_path = Path(os.environ.get("GRB_LICENSE_FILE", "/licenses/gurobi.lic"))
    if not license_path.exists():
        raise FileNotFoundError(f"Gurobi license file not found: {license_path}")
    return license_path


def assert_gurobi_runtime() -> dict[str, object]:
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("paper_sanity")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(1, int(os.environ.get("GUROBI_CHECK_THREADS", "1")))

    x = model.addVar(lb=0.0, name="x")
    y = model.addVar(lb=0.0, name="y")
    model.addConstr(x + 2.0 * y >= 1.0, name="c0")
    model.setObjective(x + y, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Unexpected Gurobi sanity solve status: {model.Status}")

    return {
        "gurobi_version": ".".join(str(part) for part in gp.gurobi.version()),
        "objective": float(model.ObjVal),
        "status": int(model.Status),
        "threads": int(model.Params.Threads),
    }


def assert_repo_loader() -> dict[str, object]:
    workspace = Path("/workspace")
    if not workspace.exists():
        return {"workspace_present": False}

    sys.path.insert(0, str(workspace))
    from gurobi.load_instance import build_gurobi_views, load_instance  # type: ignore

    sample_dir = workspace / "instances" / "GO_XS_BALANCED_01"
    raw = load_instance(sample_dir)
    data = build_gurobi_views(raw)
    return {
        "workspace_present": True,
        "instance_id": data["params"]["instance_id"],
        "job_count": len(data["J"]),
        "machine_count": len(data["M"]),
        "eligible_key_count": len(data["ELIGIBLE_KEYS"]),
    }


def main() -> None:
    license_path = assert_license_file()
    imports = assert_imports()
    gurobi = assert_gurobi_runtime()
    loader = assert_repo_loader()

    payload = {
        "license_file": str(license_path),
        "imports": imports,
        "gurobi": gurobi,
        "repo_loader": loader,
        "gurobi_max_concurrent_models": int(os.environ.get("GUROBI_MAX_CONCURRENT_MODELS", "2")),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
