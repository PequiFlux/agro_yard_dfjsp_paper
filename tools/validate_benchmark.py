#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gurobi"))

from load_instance import load_instance, list_instances  # noqa: E402


def main() -> None:
    issues = []
    for instance_id in list_instances(ROOT):
        data = load_instance(ROOT / "instances" / instance_id)
        job_ids = {row["job_id"] for row in data["jobs"]}
        machine_ids = {row["machine_id"] for row in data["machines"]}

        op_count = defaultdict(int)
        for row in data["operations"]:
            op_count[row["job_id"]] += 1
        for job_id in job_ids:
            if op_count[job_id] != 4:
                issues.append(f"{instance_id}: job {job_id} does not have 4 operations")

        elig_count = defaultdict(int)
        for row in data["eligible_machines"]:
            elig_count[(row["job_id"], row["op_seq"])] += 1
            if row["machine_id"] not in machine_ids:
                issues.append(f"{instance_id}: eligible machine missing -> {row['machine_id']}")
            if row["proc_time_min"] <= 0:
                issues.append(f"{instance_id}: non-positive proc time -> {row}")
        for job_id in job_ids:
            for op_seq in (1, 2, 3, 4):
                if elig_count[(job_id, op_seq)] == 0:
                    issues.append(f"{instance_id}: no eligible machine for {(job_id, op_seq)}")

    print(json.dumps({"issue_count": len(issues), "issues": issues[:50]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
