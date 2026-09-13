# QECO default accounting baseline

Status: completed; 1,000 episodes; frozen saved policy; energy formulas remain unvalidated.

Default physical configuration: 20 UEs, 2 edges, 100 arrival slots plus 10 drain slots, duration 0.1, arrival probability 0.3. Seed 20260913. Existing PerformanceMode/800 checkpoint evaluated greedily under the current mixed UE energy-state configuration. No training.

## Energy totals

All values below are means across episodes, summed across all 20 UEs and both edges. Units are unvalidated simulator units, not joules.

| Component | Mean per episode | Episode SD |
|---|---:|---:|
| Local UE computation | 1038.204692 | 71.549581 |
| UE transmission | 4270.447720 | 151.832008 |
| UE idle | 641.684029 | 63.873838 |
| Edge computation | 1809.475024 | 200.063498 |
| UE-only total | 5950.336441 | 231.026816 |
| Total system | 7759.811464 | 401.952267 |

## Task outcomes

- Arrivals: 601,153; completed: 556,242; deadline violations: 44,911 (7.4708%).
- Routed locally: 216,685 (36.0449%); offloaded: 384,468 (63.9551%).
- Actually started local computation: 216,685; transmission: 384,468; edge computation: 374,064.
- Local completions: 189,951; offloaded completions: 366,291.
- Pooled average completed-task latency: 0.560456 seconds (assuming duration in seconds).
- Mean of per-episode completed-task latency: 0.560348 seconds.
- Legacy stage drop counters: 39,765; missed failures compared with per-task accounting: 5,146.
- Arrivals whose diagnostic corrected finish predictors differed from the live policy fields: 601,153.

## Validation and interpretation

- Six unit tests passed, including original-vs-instrumented observation/outcome/reward-input equality, all-edge aggregation, conservation, deadline equality, queued expiry, and empty episodes.
- All 1,000 episode accounting checks passed; CSV totals and saved activity counts reconciled.
- Learned parameter hashes and checkpoint hashes are unchanged; learn_calls=0.
- main.py and Config.py are byte-identical to their backups; only the logging method changed in D3QN.py.
- The transmission predictor typo is corrected in diagnostic bookkeeping only. The original live fields and stale-density edge observation remain as policy inputs to preserve decisions.
- Successful latency excludes deadline failures. Energy totals include energy already consumed by failed tasks.
- These are reproducible legacy-formula totals, not a validated physical energy baseline. A physical energy-saving crossover claim remains blocked by the documented formula choice.
- The proposed power-time convention is described in ../../ENERGY_ACCOUNTING.md. reaccount_baseline.py can apply it to this exact saved activity after approval, without loading or rerunning D3QN.
- Episode SD is descriptive; this is one seeded workload sequence with retained policy LSTM history, not an independent-seed confidence interval.

Files: episode_metrics.csv, summary.json, metadata.json, episodes/*.npz, source_used/*.py, artifact_manifest.json.
