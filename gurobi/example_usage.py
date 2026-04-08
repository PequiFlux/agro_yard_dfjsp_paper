#!/usr/bin/env python3
"""
Minimal example showing how to create Gurobi variables on top of the benchmark
data layout. This file intentionally stops before committing to a full exact
model formulation, so that you can plug it into either continuous-time MIP,
disjunctive formulations, or decomposition / rolling-horizon approaches.
"""
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from load_instance import load_instance, build_gurobi_views


def main() -> None:
    instance_dir = Path(__file__).resolve().parents[1] / "instances" / "GO_XS_BALANCED_01"
    raw = load_instance(instance_dir)
    data = build_gurobi_views(raw)

    model = gp.Model(data["params"]["instance_id"])

    # Assignment binary: x[j,o,m] = 1 if operation (j,o) uses machine m
    x = model.addVars(data["ELIGIBLE_KEYS"], vtype=GRB.BINARY, name="x")

    # Continuous start times for each operation
    s = model.addVars(data["OPS"], lb=0.0, name="s")

    # One completion variable per operation
    c = model.addVars(data["OPS"], lb=0.0, name="c")

    # Exactly one eligible machine per operation
    model.addConstrs(
        (gp.quicksum(x[j, o, m] for m in data["ELIGIBLE_BY_OP"][j, o]) == 1
         for (j, o) in data["OPS"]),
        name="assign_once"
    )

    # Completion = start + chosen processing time
    model.addConstrs(
        (
            c[j, o] == s[j, o] + gp.quicksum(data["PROC"][j, o, m] * x[j, o, m]
                                             for m in data["ELIGIBLE_BY_OP"][j, o])
            for (j, o) in data["OPS"]
        ),
        name="completion_def"
    )

    # Release / arrival constraint on the first operation
    model.addConstrs(
        (s[j, 1] >= data["RELEASE"][j] for j in data["J"]),
        name="release_first_stage"
    )

    # Linear precedence arcs
    model.addConstrs(
        (s[j, succ] >= c[j, pred]
         for (j, pred, succ) in data["PRED"]),
        name="precedence"
    )

    # This example intentionally does NOT add machine non-overlap constraints,
    # because those depend on the exact disjunctive formulation you want to test.
    # Examples:
    #   - pairwise sequencing binaries per machine
    #   - continuous-time disjunctive constraints with big-M
    #   - time-indexed MIP
    #   - rolling-horizon / event-driven re-optimization

    model.ModelSense = GRB.MINIMIZE
    model.setObjective(0.0)
    model.update()

    print(f"Model built for {data['params']['instance_id']}")
    print(f"  assignment vars : {len(x)}")
    print(f"  operation vars  : {len(s)} start + {len(c)} completion")
    print(f"  constraints     : {model.NumConstrs}")
    print("Next step: add your preferred machine non-overlap formulation and objective.")


if __name__ == "__main__":
    main()
