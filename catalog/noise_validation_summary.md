# Noise validation summary

The v1.1-observed release passed structural validation on all 36 instances.

Diagnostics computed on the whole release:

- Due slack explained by priority class only: `R² = 0.4848` in v1.1-observed, versus `R² = 1.0000` in the nominal v1.0 release.
- UNLOAD processing time explained by load, machine ID, and moisture class: `R² = 0.4995` in v1.1-observed, versus `R² = 0.7540` in the nominal v1.0 release.

Behavioral sanity checks on FIFO summaries remain consistent at the family level:

- `avg_fifo_mean_flow_min`: for every scale, `balanced < peak < disrupted`.
- `avg_fifo_p95_flow_min`: for every scale, `balanced < peak < disrupted`.

This means the release is less deterministic while preserving the expected ordering of operational regimes.

Recommended validation blocks:

- Structural integrity: 4 operations per job, valid precedences, no operation without eligibility, positive and plausible times, and feasible FIFO baseline.
- Nominal-to-observed reconciliation: `jobs.csv::completion_due_min` must match `job_noise_audit.csv::completion_due_observed_min`, and `eligible_machines.csv::proc_time_min` must match `proc_noise_audit.csv::proc_time_observed_min`.
- Behavioral regime validation: `PEAK` and `DISRUPTED` must remain worse than `BALANCED` on mean flow and tail flow.
- Over-determinism diagnostics: the before/after `R²` values should decrease without collapsing the benchmark semantics.
