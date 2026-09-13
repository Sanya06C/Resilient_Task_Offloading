# QECO default power-time energy baseline

Status: complete. The approved evaluation convention was applied offline to the exact activity records from the frozen 1,000-episode baseline. D3QN was not loaded or rerun, no learning occurred, and task decisions, latency, and deadline outcomes are unchanged.

## Accounting convention

The result assumes `UE_COMP_ENERGY`, `UE_TRAN_ENERGY`, `UE_IDLE_ENERGY`, and `EDGE_COMP_ENERGY` are powers in watts and `DURATION=0.1` is seconds. Energy is measured in joules.

- Local UE computation: `P_local * duration * actual_active_fraction`.
- UE transmission: `P_tx * duration * actual_active_fraction`.
- Edge computation: `P_edge * duration * actual_active_fraction / active_UE_count`, allocating server power using QECO's scheduler share.
- UE idle: `P_idle * edge_wait_time`, counted once per UE and excluding concurrent local-computation or transmission activity.
- UE-only total: local computation + transmission + idle.
- Total-system energy: UE-only total + edge computation.

For partial slots, `actual_active_fraction = min(remaining_work, slot_capacity) / slot_capacity`. Concurrent services are treated as starting together at the beginning of a slot. These are task-attributable active-energy totals; background infrastructure energy, downlink result transmission, and sleep/wake transition energy are outside the present QECO model.

## Baseline results

Energy values are per-episode means, summed across all 20 UEs and both edge servers. The reported SD is variation across the 1,000 episodes.

| Component | Mean energy (J/episode) | Episode SD |
|---|---:|---:|
| Local UE computation | 169.593519 | 9.962507 |
| UE transmission | 263.794511 | 9.754260 |
| UE idle | 0.988235 | 0.107244 |
| Edge computation | 54.533841 | 1.962431 |
| UE-only total | 434.376265 | 16.298102 |
| Total-system energy | 488.910105 | 17.669273 |

The total-system composition is 34.69% local computation, 53.96% transmission, 0.20% UE idle, and 11.15% edge computation.

| Task metric | Result |
|---|---:|
| Arrived tasks | 601,153 |
| Completed tasks | 556,242 |
| Deadline violations | 44,911 (7.4708%) |
| Pooled completed-task latency | 0.560456 seconds |
| Locally routed | 216,685 (36.0449%) |
| Offloaded | 384,468 (63.9551%) |

## Energy-saving interpretation

Dividing component energy by the number of tasks routed through the corresponding path gives these descriptive averages:

- Local execution: 0.782673 J per locally routed task.
- Offloading, UE-only: 0.688699 J per offloaded task.
- Offloading, total system: 0.830541 J per offloaded task.

Under this baseline policy, offloading is associated with about 12.01% lower UE-side energy per routed task, while its UE-plus-edge energy is about 6.12% higher than local execution. This is not yet a causal crossover estimate: QECO selects different tasks for local execution and offloading, so the two groups can differ in task size, density, queue state, and failure outcome. The stress study must compare matched workloads or counterfactual local/offload execution under identical task traces before claiming the point at which offloading stops saving energy.

## Provenance and validation

- Source activity: `../default_legacy/episodes/*.npz`.
- `reaccount_baseline.py` verified every legacy metric against the original per-episode CSV before applying the new convention.
- All 1,000 source activity files are recorded by SHA-256 in `metadata.json`.
- `policy_runs=0` and `learning_calls=0` in the re-accounting metadata.
- The legacy energy arrays remain available in the new CSV under `legacy_*` columns for audit, but those values retain their original unvalidated simulator units and should not be numerically compared with joules.

Files: `episode_metrics.csv`, `summary.json`, and `metadata.json`.
