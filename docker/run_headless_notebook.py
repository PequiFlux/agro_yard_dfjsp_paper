from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


WORKSPACE = Path("/workspace")
TARGET_NOTEBOOK = WORKSPACE / os.environ.get(
    "NOTEBOOK_PATH",
    "output/jupyter-notebook/agro-yard-paper-benchmark-and-selection.ipynb",
)


def build_sanity_notebook() -> tuple[nbformat.NotebookNode, Path]:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# Paper Environment Sanity Check"),
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "from pathlib import Path",
                    "import json",
                    "import gurobipy as gp",
                    "import numpy as np",
                    "import pandas as pd",
                    "import scipy",
                    "import seaborn as sns",
                    "import sklearn",
                    "import lightgbm",
                    "import umap",
                    "import hdbscan",
                    "import shap",
                    "payload = {",
                    "    'workspace_exists': Path('/workspace').exists(),",
                    "    'gurobi_version': '.'.join(str(part) for part in gp.gurobi.version()),",
                    "    'numpy_version': np.__version__,",
                    "    'pandas_version': pd.__version__,",
                    "    'scipy_version': scipy.__version__,",
                    "    'seaborn_version': sns.__version__,",
                    "    'sklearn_version': sklearn.__version__,",
                    "    'lightgbm_version': lightgbm.__version__,",
                    "    'umap_module': umap.__name__,",
                    "    'hdbscan_module': hdbscan.__name__,",
                    "    'shap_version': shap.__version__,",
                    "}",
                    "print(json.dumps(payload, indent=2, sort_keys=True))",
                ]
            )
        ),
    ]
    return notebook, WORKSPACE / "tmp" / "jupyter-notebook" / "paper-environment-sanity.ipynb"


def main() -> None:
    out_dir = WORKSPACE / "tmp" / "jupyter-notebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout = int(os.environ.get("NOTEBOOK_EXECUTION_TIMEOUT", "1800"))
    kernel_name = os.environ.get("NOTEBOOK_KERNEL", "python3")

    if TARGET_NOTEBOOK.exists():
        notebook_path = TARGET_NOTEBOOK
        with notebook_path.open("r", encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
    else:
        notebook, notebook_path = build_sanity_notebook()

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    out_path = out_dir / f"{notebook_path.stem}.executed.ipynb"
    with out_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(f"Executed {notebook_path} -> {out_path}")


if __name__ == "__main__":
    main()
