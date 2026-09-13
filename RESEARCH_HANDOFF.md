# Research handoff: industrial network-stress classification and QECO energy evaluation

**Handoff date:** 2026-09-13  
**QECO repository:** `D:\research\QECO`  
**Classifier review project:** `C:\Users\DELL\Downloads\iot_classifier_review`  
**Status:** classifier selection and one-time held-out evaluation complete; QECO accounting repaired; default power-time baseline complete; four-level network-stress experiment complete; first tested UE-energy crossover identified at Medium stress.

This document records what was inspected, changed, executed, validated, and learned. It also records what remains unresolved so future work does not accidentally invalidate the frozen model or overstate the results.

---

## 1. Scope and constraints followed

The work respected these constraints:

- No files were intentionally written under `C:\Windows\System32`.
- The classifier used the exact frozen training and validation parquet files. They were never rebuilt or modified.
- The held-out classifier dataset was opened exactly once, only after the classifier and threshold were locked.
- The classifier was not retrained, recalibrated, retuned, or threshold-adjusted after held-out evaluation.
- The QECO learned policy and reward function were not changed.
- QECO stress runs used the existing saved D3QN checkpoints with `epsilon=1` and zero learning calls.
- Network-stress experiments used the same task workload arrays and seed at every level.
- No packet-loss, jitter, retransmission, or unsupported radio parameters were invented.
- Ambiguous legacy energy formulas were retained for reward compatibility. A separate evaluation-only power-time accounting model was added and explicitly selected later.

The QECO Git worktree currently contains uncommitted changes and new experiment artifacts. Do not run `git reset --hard`, `git clean`, or overwrite these files without first archiving or committing them.

---

## 2. Frozen network-stress classifier phase

### 2.1 Inputs and frozen features

Development inputs:

- `C:\Users\DELL\Downloads\train_dataset.parquet`
- `C:\Users\DELL\Downloads\validation_dataset.parquet`

Frozen input hashes:

- Training SHA-256: `1e11477c951dddb399ed8d1d35e9c16805390b3212895bede1bc8982183b5c5b`
- Validation SHA-256: `1f5658adecc0b40201a7b0542a1d2e8110b2e1266656ecdd625e8aee740cb8fe`

The five model features remained unchanged:

1. `new_flows`
2. `unique_sources`
3. `unique_destinations`
4. `median_flow_duration_sec`
5. `estimated_packet_rate_per_sec`

The target remained `window_type`, with `ABNORMAL` as the positive class.

The combined training and validation files contain 195,806 windows:

- BENIGN: 191,008
- ABNORMAL: 4,798

The development sources are `a_day1`, `s_day1`, `tf_a`, and `tf_s`. Only `a_day1` and `s_day1` contain ABNORMAL windows. The two `tf_*` sources are BENIGN-only.

### 2.2 Candidate work completed

The original notebook and prior candidates were reviewed. The earlier models showed the following recurring trade-off:

- Logistic regression obtained useful abnormal recall but produced many false positives.
- Random forest obtained high accuracy mostly by favoring the majority BENIGN class and missed most abnormalities.
- XGBoost obtained high recall at the cost of extremely poor precision and accuracy.
- LR/XGBoost ensembles moved along the same precision-recall frontier rather than eliminating the trade-off.

The expanded experiment evaluated raw and transformed logistic regression, spline models, histogram boosting, XGBoost, Extra Trees, random forest, class weighting, source balancing, calibration, blends, and stacking. The saved report records 89 screened base configurations and 64 validation candidates, including post-processing and reference ensembles.

Temporal training folds were used for screening. Validation was treated as development data. The final selection rule was: among validation-F1 operating points, select a non-dominated candidate, then maximize F1, average precision, and balanced accuracy in that order.

### 2.3 Locked classifier

Selected artifact:

- Model ID: `refined_lr_11`
- File: `C:\Users\DELL\Downloads\iot_classifier_review\selected_model.joblib`
- Estimator: raw-feature `StandardScaler` plus logistic regression
- Logistic regression `C=10`
- Positive-class weighting exponent: `(N_negative/N_positive)^0.25`, approximately 2.67
- Within-class source-count weighting exponent: 2
- Locked threshold: `0.06908514746070128`
- Fit scope: frozen training dataset only

Source and timestamp were used only to construct training weights and temporal validation blocks. They were not passed to `predict_proba` as inference features.

### 2.4 Validation results

| Metric | Validation result |
|---|---:|
| Rows | 42,974 |
| Accuracy | 0.9557 |
| Precision | 0.4724 |
| Recall | 0.3264 |
| F1 | 0.3861 |
| Average precision / PR-AUC | 0.3325 |
| Balanced accuracy | 0.6551 |
| ROC-AUC | 0.7453 |
| Brier score | 0.03868 |
| Log loss | 0.16274 |
| Confusion matrix | `[[TN=40470, FP=669], [FN=1236, TP=599]]` |

This candidate improved all six requested pooled point estimates over the reproduced original notebook F1 ensemble. The improvement was not uniform across sources: aggressive-traffic recall decreased while stealth-traffic recall increased. Bootstrap intervals for some differences included zero, so this is best described as the strongest tested development candidate rather than a statistically confirmed universal improvement.

### 2.5 One-time held-out evaluation

Held-out input:

- `C:\Users\DELL\Downloads\heldout_dataset.parquet`
- Rows: 15,992
- `a_day2`: 4,710
- `s_day2`: 11,282

The evaluation record states `heldout_reads: 1`. The locked model and threshold were used without fitting or post-processing changes.

| Dataset | Accuracy | Precision | Recall | F1 | Average precision | Balanced accuracy | ROC-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pooled | 0.8407 | 0.6145 | 0.5784 | 0.5959 | 0.5928 | 0.7430 | 0.8323 | `[[11567,1178],[1369,1878]]` |
| `a_day2` | 0.8964 | 0.1257 | 0.6436 | 0.2104 | 0.4545 | 0.7727 | 0.8481 | `[[4157,452],[36,65]]` |
| `s_day2` | 0.8175 | 0.7141 | 0.5763 | 0.6378 | 0.6531 | 0.7435 | 0.8350 | `[[7410,726],[1333,1813]]` |

Generalization was judged mixed and not acceptable as a dependable high-precision detector across both sources. Pooled discrimination improved, but accuracy fell because held-out prevalence rose sharply. `a_day2` precision was only 12.57%, with 452 false positives for 101 actual abnormal windows, while `s_day2` was much stronger. No classifier tuning should occur after this held-out evaluation.

### 2.6 Classifier artifacts

Primary files under `C:\Users\DELL\Downloads\iot_classifier_review`:

- `REPORT.md`: full development and held-out report
- `selected_model.joblib`: locked classifier artifact
- `selection.json`: selected configuration, threshold, hashes, and validation metrics
- `heldout_evaluation.json`: one-time pooled and per-source held-out results
- `evaluate_heldout_once.py`: one-time held-out evaluator; it contains no fitting
- `classifier_review.ipynb`: consolidated research notebook
- `validation_results.csv`: complete validation candidate comparison
- `training_screen.csv`: temporal training-screen results
- `pareto.csv`: non-dominated operating points
- `matched_operating_points.csv`: candidate threshold comparisons
- `validation_predictions.npz`: saved validation predictions
- `validation_block_bootstrap.csv`: descriptive uncertainty results
- `candidate_specs.json`, `folds.json`, `shortlist.json`, and `verification.json`: reproducibility metadata
- `models/`, `oof/`, and `thresholds/`: saved fitted candidates and development outputs

---

## 3. QECO repository audit

Repository inspected: `D:\research\QECO`

Original main files:

- `Config.py`
- `MEC_Env.py`
- `D3QN.py`
- `main.py`

### 3.1 Default physical and simulation parameters

| Parameter | Default | Meaning in current code |
|---|---:|---|
| `N_UE` | 20 | Number of mobile/IoT devices |
| `N_EDGE` | 2 | Edge servers |
| `UE_COMP_CAP` | 2.6 | UE compute rate before slot scaling |
| `UE_TRAN_CAP` | 14 | UE transmission rate before slot scaling |
| `EDGE_COMP_CAP` | 42 | Edge compute rate before slot scaling |
| `UE_COMP_ENERGY` | 2 | Documented as UE computation power |
| `UE_TRAN_ENERGY` | 2.3 | Documented as UE transmission power |
| `UE_IDLE_ENERGY` | 0.1 | Documented as UE standby power |
| `EDGE_COMP_ENERGY` | 5 | Documented as edge computation power |
| `TASK_COMP_DENS` | 0.197, 0.297, 0.397 | Task computation densities |
| `TASK_MIN_SIZE`, `TASK_MAX_SIZE` | 1, 7 | Uniform task-size range |
| `TASK_ARRIVE_PROB` | 0.3 | Independent per-UE Bernoulli arrival probability |
| `MAX_DELAY` | 10 slots | Hard task deadline |
| `DURATION` | 0.1 | Slot duration |
| `N_TIME_SLOT` | 100 | Arrival slots per episode |
| `N_TIME` | 110 | Arrival slots plus deadline drain interval |
| `N_EPISODE` | 1000 | Default episode count |

The environment converts rates to per-slot capacities by multiplying by `DURATION`. Default per-slot capacities are therefore 0.26 for local computation, 1.4 for transmission, and 4.2 for edge computation.

### 3.2 Offloading and delay behavior

- Action 0 executes a whole task locally.
- Action 1 offloads to edge 0.
- Action 2 offloads to edge 1.
- `N_COMPONENT=1`, so active code does not split tasks.
- Local, transmission, and edge stages use separate queues.
- Transmission delay is produced by per-slot service at `UE_TRAN_CAP * DURATION`.
- Edge service is divided by the number of active UEs at that edge.
- Tasks unfinished at `MAX_DELAY` are marked as violations.
- The default observation dimension is six, calculated as `4 + N_EDGE`. An earlier assumption of five QECO state features was corrected.

The network model has fixed per-UE links. It has no shared radio bandwidth, channel/SNR model, packet loss, retransmission, or jitter. Network stress is therefore represented through an effective transmission service capacity rather than invented radio behavior.

### 3.3 Important original-code findings

1. **Energy aggregation bug in `D3QN.do_store_energy`.** The original loop overwrote edge and idle energy with the last nonzero element instead of summing all edge components.

2. **Transmission-time bookkeeping bug.** The predicted transmission finish was assigned to `t_ue_comp` rather than `t_ue_tran`. Both are policy observation fields. Correcting this live assignment would change future actions even with fixed weights.

3. **Stale-density edge observation.** The edge backlog update uses a task-density variable left over from a preceding loop. Correcting this would also change policy observations.

4. **Local processed-volume bug.** In a final partial slot, processed task size was divided by density a second time.

5. **Transmission-volume bug.** In a full transmission slot, the code recorded all remaining task data rather than the amount served by the current slot.

6. **Edge processed-volume bug.** The final partial edge slot divided remaining work by active UE count a second time.

7. **Incomplete deadline counters.** Stage-specific drop counters miss tasks that expire while waiting in queues. Per-task terminal state gives the correct total.

8. **Reward energy scope.** `QoE_Function` currently uses UE local-computation and transmission energy. Edge and idle energy are read but excluded from the active reward expression. This was preserved as requested.

9. **Ambiguous physical energy equations.** The legacy formulas mix capacity, task density, power, and duration inconsistently and cannot be interpreted as joules without correction.

---

## 4. QECO accounting layer implemented

### 4.1 Source changes

Modified files:

- `MEC_Env.py`: added optional observational accounting hooks and corrected confirmed processed/transmitted-volume bookkeeping bugs.
- `D3QN.py`: changed only `do_store_energy` so it sums all edge and idle components.

Unchanged files:

- `Config.py` is byte-identical to its saved original.
- `main.py` is byte-identical to its saved original.
- D3QN action selection, network architecture, learning, replay memory, reward, and checkpoint loading were not changed.

Original source copies are preserved under:

- `D:\research\QECO\accounting_baseline\original_sources`

New implementation files:

- `accounting.py`: episode accounting and power-time calculations
- `evaluate_baseline.py`: frozen-checkpoint baseline evaluator
- `reaccount_baseline.py`: offline re-accounting from saved activity records
- `test_accounting.py`: accounting regression and conservation tests
- `ENERGY_ACCOUNTING.md`: formula audit and assumptions

### 4.2 Metrics recorded per episode

The accounting layer records:

- Local UE computation energy
- UE transmission energy
- UE idle energy
- Edge computation energy
- UE-only total energy
- Total-system energy
- Completed-task average latency
- Average time to terminal outcome
- Deadline violations and fraction
- Actual task arrivals and completions
- Local and offloaded routing counts/fractions
- Local, transmission, and edge service-start counts
- Local and offloaded completion counts
- Transmission completion delay
- Actions, work volumes, activity fractions, task sizes, densities, and failure state
- Legacy energy arrays for audit

Energy spent before a task fails remains included. Successful-task latency excludes failures; time-to-terminal includes both success and failure.

### 4.3 Policy-preserving handling of timing bugs

The transmission-time assignment bug was confirmed. It was not corrected in the live QECO observation because that would change D3QN behavior. The accounting layer instead maintains corrected local and transmission finish predictions separately and records divergences. The stale-density edge observation remains documented and unchanged for the same reason.

All default-baseline task arrivals diverged from the corrected diagnostic timing fields, confirming that the bug is pervasive. A future live fix requires a new version of the environment and policy retraining; it cannot be mixed into the present frozen-policy comparison.

### 4.4 Original ambiguous energy formulas

Let:

- `d` = task computation density
- `c = UE_COMP_CAP * DURATION`
- `q = c / d`
- `r` = remaining task size
- `C = UE_TRAN_CAP * DURATION`
- `e = EDGE_COMP_CAP * DURATION`
- `m` = active UE count at an edge
- `dt = DURATION`

The original local full-slot expression is:

`q * (1**(-27) * q) = q^2`

Because `1**(-27)` equals one, this is not a small CPU capacitance coefficient. The configured `UE_COMP_ENERGY=2` is not used by the original local-energy calculation.

Other legacy formulas also have inconsistent units:

- TX energy uses `C * UE_TRAN_ENERGY` for every active slot, including a partial final slot.
- Edge energy mixes served data, configured edge power, and slot duration differently between full and partial slots.
- One idle-energy branch omits both configured idle power and duration.

Legacy totals were preserved and labelled `unvalidated_simulator_units`. They should not be presented as joules or used for a physical energy crossover.

### 4.5 Approved power-time evaluation model

The approved evaluation convention treats the four configured energy constants as powers in watts and `DURATION=0.1` as seconds.

For a service activity occupying fraction `f` of a slot:

`E = P * DURATION * f`

where:

`f = min(remaining_work, slot_capacity) / slot_capacity`

Component definitions:

- Local UE: `P_local * dt * f_local`
- Transmission: `P_tx * dt * f_tx`
- Edge: `P_edge * dt * f_edge / active_UE_count`
- UE idle: `P_idle` multiplied by the union of post-transmission edge-wait intervals, excluding concurrent UE computation/transmission activity
- UE-only total: local + transmission + idle
- Total-system energy: UE-only + edge

This is task-attributable active energy. QECO does not currently represent background device/server energy, downlink result transmission, sleep/wake transitions, cooling, or infrastructure energy.

### 4.6 Validation

Six unit tests pass:

1. Original versus instrumented observation, task outcome, and reward-input equality
2. Deadline equality and queued-expiry counting
3. Empty-episode behavior
4. All-edge energy aggregation
5. Processed/transmitted work conservation
6. Power-time bounds and reset behavior

The 1,000-episode baseline also verified:

- Policy trainable-variable hashes match before and after.
- Checkpoint hashes match before and after.
- `learn_calls=0`.
- Exactly 1,000 CSV episode rows and 1,000 detailed NPZ records exist.
- Arrivals equal local plus offloaded decisions.
- Arrivals equal completions plus violations.
- UE-only and total-system component sums reconcile.

---

## 5. Default QECO baseline

### 5.1 Protocol

- Existing `TrainedModel_20UE_2EN_PerformanceMode/800` checkpoints for all 20 UEs
- Greedy evaluation with `epsilon=1`
- No training or learning calls
- Default `Config.py` values
- Seed `20260913`
- 1,000 episodes
- 100 arrival slots plus 10 drain slots per episode
- LSTM history retained across episodes as in the original training loop
- Power-time accounting applied offline to the exact saved activity

### 5.2 Default results

Energy values are means per episode, summed over all 20 UEs and both edge servers.

| Metric | Result |
|---|---:|
| Local UE computation energy | 169.5935 J |
| UE transmission energy | 263.7945 J |
| UE idle energy | 0.9882 J |
| Edge computation energy | 54.5338 J |
| UE-only energy | 434.3763 J |
| Total-system energy | 488.9101 J |
| Arrived tasks | 601,153 |
| Completed tasks | 556,242 |
| Deadline violations | 44,911 / 7.4708% |
| Completed-task pooled latency | 0.5605 s |
| Locally routed | 216,685 / 36.0449% |
| Offloaded | 384,468 / 63.9551% |

The original stage counters reported only 39,765 drops and missed 5,146 failures found by per-task terminal accounting.

### 5.3 Baseline artifacts

- `accounting_baseline\default_legacy\BASELINE_REPORT.md`
- `accounting_baseline\default_legacy\episode_metrics.csv`
- `accounting_baseline\default_legacy\summary.json`
- `accounting_baseline\default_legacy\metadata.json`
- `accounting_baseline\default_legacy\episodes\*.npz`
- `accounting_baseline\default_legacy\artifact_manifest.json`
- `accounting_baseline\default_power_time_v1\BASELINE_REPORT.md`
- `accounting_baseline\default_power_time_v1\episode_metrics.csv`
- `accounting_baseline\default_power_time_v1\summary.json`
- `accounting_baseline\default_power_time_v1\metadata.json`
- `accounting_baseline\validation.json`

The `smoke_legacy` folder is a two-episode temporary validation run and is excluded from all final results.

---

## 6. Frozen industrial network-stress profiles

### 6.1 Data usage

Stress profiles were derived from the frozen training and validation datasets only. The held-out dataset was not reopened.

Only `a_day1` and `s_day1` were used for source-matched severity calibration because they contain both BENIGN and ABNORMAL windows. Their combined counts are:

- `a_day1`: 4,158 BENIGN and 3,264 ABNORMAL
- `s_day1`: 9,028 BENIGN and 1,534 ABNORMAL

The `tf_a` and `tf_s` BENIGN-only rows remain recorded in provenance but were excluded from the source-matched calibration. Including them would confound capture identity with stress because their packet-rate distributions differ sharply and there is no same-source ABNORMAL reference.

### 6.2 Stress score

Each feature was converted to a percentile relative to BENIGN windows from the same source. The composite score uses:

| Component | Weight |
|---|---:|
| Estimated packet rate | 0.40 |
| New flows | 0.15 |
| Unique sources | 0.10 |
| Unique destinations | 0.10 |
| Short-flow churn: `new_flows / max(duration, 0.001)` | 0.15 |
| Persistent-flow occupancy: `new_flows * duration` | 0.10 |

The duration interactions allow both rapid short-flow churn and long-lived flow occupancy to contribute. Duration alone was not forced into an unsupported one-direction interpretation.

ABNORMAL windows were split into score terciles separately within `a_day1` and `s_day1`, then combined. This gives every level representation from both traffic families rather than allowing the larger aggressive capture to define all cutoffs.

ABNORMAL tier counts:

| Source | Low | Medium | High |
|---|---:|---:|---:|
| `a_day1` | 1,088 | 1,088 | 1,088 |
| `s_day1` | 512 | 510 | 512 |

### 6.3 Offered-load and capacity mapping

The source-normalized offered-load multiplier uses a weighted geometric mean:

- Packet rate: 70%
- New flows: 15%
- Unique sources: 7.5%
- Unique destinations: 7.5%

Each feature is divided by its same-source BENIGN median. Ratios below one are clipped to one so abnormal traffic cannot create an artificial capacity increase.

Packet sizes, physical link bandwidth, and measured link utilization are absent. Therefore the raw multiplier is compressed conservatively:

`C_effective = 14 / sqrt(M)`

The mapping has a lower floor of 25% of default capacity, although none of the selected profiles reaches it. No packet loss is used.

| Profile | Representative load multiplier | Capacity factor | Effective QECO TX capacity |
|---|---:|---:|---:|
| Normal | 1.0000 | 1.0000 | 14.0000 |
| Low | 1.0000 | 1.0000 | 14.0000 |
| Medium | 2.7207 | 0.6063 | 8.4877 |
| High | 4.8964 | 0.4519 | 6.3269 |

The Low ABNORMAL group has lower representative packet/load pressure than the source-matched BENIGN reference. Low therefore uses the default capacity. Reducing capacity or assigning loss would invent evidence that the frozen features do not supply. This is an important finding: the frozen ABNORMAL label is not a monotonic congestion label.

### 6.4 Profile derivation artifacts

- `derive_network_stress_profiles.py`
- `network_stress_profiles\profiles.csv`
- `network_stress_profiles\abnormal_window_assignments.csv`
- `network_stress_profiles\derivation.json`

---

## 7. QECO network-stress experiments

### 7.1 Protocol

Normal, Low, Medium, and High use:

- The exact same 1,000 task-size and computation-density arrays saved by the baseline
- The same seed, `20260913`
- The same 20 pretrained checkpoints
- The same initial UE energy-state draw
- The same power-time accounting
- The same local and edge capacities
- The same deadlines and task-arrival workload
- Zero learning calls
- Unchanged D3QN weights and reward

The only changed physical environment parameter is effective UE transmission capacity. The policy is unchanged, but its actions may respond to resulting queue-state observations. Workload and checkpoint hashes were checked before and after each run.

Normal uses the completed default baseline. Low, Medium, and High were replayed with `evaluate_stress_level.py`. Low exactly reproduces Normal, validating the replay protocol.

### 7.2 Requested results

Energy is mean joules per episode across all UEs and edges. Completed latency includes successful tasks only. Time to terminal includes both completions and failures.

| Stress | UE-only energy | Total-system energy | TX energy | Completed latency | Time to terminal | Deadline violations | Offloading ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 434.3763 J | 488.9101 J | 263.7945 J | 0.5605 s | 0.5933 s | 44,911 / 7.47% | 63.96% |
| Low | 434.3763 J | 488.9101 J | 263.7945 J | 0.5605 s | 0.5933 s | 44,911 / 7.47% | 63.96% |
| Medium | 562.1667 J | 593.8707 J | 351.3879 J | 0.6388 s | 0.7247 s | 142,833 / 23.76% | 56.45% |
| High | 614.0728 J | 629.3264 J | 387.2401 J | 0.6347 s | 0.7695 s | 221,820 / 36.90% | 53.64% |

High completed-task latency is slightly lower than Medium because the successful set becomes highly selected. Only 43.86% of offloaded tasks finish at High, so slow tasks increasingly become deadline failures and leave the successful-latency denominator. Time to terminal and violation rate show the monotonic degradation.

Offloaded completion fractions:

- Normal/Low: 95.27%
- Medium: 68.72%
- High: 43.86%

### 7.3 Matched UE-energy crossover

A simple comparison between all local tasks and all offloaded tasks would be selection-biased because QECO routes different task sizes, densities, and queue states to each path. The crossover therefore uses the same tasks in numerator and denominator.

For each task selected for offloading:

- Numerator: its aggregate UE transmission plus UE idle energy under the stress run
- Denominator: power-time local-computation energy required for that same task size and density

The matched local energy formula is:

`E_local_counterfactual = P_local * duration * task_size / (UE_COMP_CAP * duration / density)`

| Stress | Actual offload UE J/task | Same tasks local J/task | Offload/local ratio | UE-energy result | Descriptive 95% block interval |
|---|---:|---:|---:|---:|---:|
| Normal | 0.6887 | 0.9599 | 0.7175 | 28.25% saving | 0.7168–0.7182 |
| Low | 0.6887 | 0.9599 | 0.7175 | 28.25% saving | 0.7168–0.7182 |
| Medium | 1.0363 | 0.9548 | 1.0854 | 8.54% increase | 1.0840–1.0869 |
| High | 1.2012 | 0.9492 | 1.2654 | 26.54% increase | 1.2632–1.2676 |

**Main result:** Medium is the first tested level where offloading stops saving UE energy. It corresponds to effective capacity 8.4877, or 60.63% of the default rate.

The exact numerical crossover is not 8.4877. The current profiles only establish that it lies somewhere between the tested Normal/Low capacity of 14 and Medium capacity of 8.4877. A finer matched capacity sweep is required to locate it.

The intervals are descriptive paired intervals from 5,000 resamples of 50 nonoverlapping 20-episode blocks. They are conditional on the frozen policy, chosen profile mapping, captured traffic, and one fixed workload sequence. They are not population confidence intervals for other factories or networks.

### 7.4 Stress artifacts

- `network_stress_profiles\REPORT.md`: complete stress report
- `network_stress_profiles\stress_results.csv`: final comparison table
- `network_stress_profiles\stress_results.json`: crossover definition, hashes, and results
- `network_stress_profiles\run_low`: complete Low per-episode and task records
- `network_stress_profiles\run_medium`: complete Medium per-episode and task records
- `network_stress_profiles\run_high`: complete High per-episode and task records
- `network_stress_profiles\run_low.log`, `run_medium.log`, `run_high.log`: execution logs
- `evaluate_stress_level.py`: frozen-policy stress runner
- `analyze_network_stress.py`: matched crossover and consistency analysis

`network_stress_profiles\smoke_medium` is a temporary two-episode run. It is excluded from final analysis. An automatic approval policy prevented its deletion; it can be archived or removed manually after checking the exact path.

---

## 8. Interpretation suitable for the research paper

The strongest defensible claims are:

1. The classifier provides useful pooled discrimination but does not generalize uniformly across industrial captures. Its one-time held-out performance must be reported per source.

2. QECO's original energy arrays contain bookkeeping and unit inconsistencies. The new accounting layer fixes confirmed logging/volume errors and provides explicit task-attributable power-time energy without changing the reward or learned policy.

3. Under the frozen default policy and task workload, transmission energy dominates active UE energy. Network degradation increases UE and total-system energy even though the policy reduces its offloading rate.

4. Under the selected conservative mapping, Medium is the first tested stress level where offloading consumes more UE energy than local computation for the same selected task sizes and densities.

5. High stress also causes severe deadline failure. A lower successful-task latency at High must not be interpreted as improved service because it is caused by survivor bias.

Claims that should not be made:

- Do not describe the classifier as universally reliable or high precision across all industrial sources.
- Do not call the legacy energy values joules.
- Do not state that ABNORMAL always means high network load.
- Do not claim measured packet loss, retransmission, or physical channel impairment.
- Do not describe 8.4877 as the exact physical crossover bandwidth.
- Do not claim causal local-versus-offload deadline equivalence from the analytic energy counterfactual.
- Do not claim that the current capacity mapping was directly measured from a physical 14-unit link. It is a conservative sensitivity mapping from relative observed load.

---

## 9. Known limitations requiring disclosure

### Classifier/data limitations

- Class prevalence and traffic-family composition differ strongly between training, validation, and held-out data.
- The held-out source-specific precision varies sharply.
- The frozen label represents any positive estimated abnormal packet contribution, not a directly measured network congestion endpoint.
- Packet-rate estimates themselves have capture-dependent error and heavy-tailed distributions.
- Completed-flow duration and packet totals may only be available after a flow ends; real-time deployment requires a feature-availability analysis.
- The existing split history and prior validation use make validation a development estimate rather than a fresh confirmatory test.

### QECO limitations

- Current networking has fixed per-UE capacity and no shared radio scheduler.
- The live transmission predictor writes to the wrong observation field.
- The edge-backlog observation uses stale density.
- Those observation issues remain to preserve the frozen policy.
- The checkpoint directory is named PerformanceMode, while current `Config.py` samples mixed UE energy states. This scenario/checkpoint distinction should be disclosed.
- The power-time constants are treated as watts based on code comments. They have not yet been calibrated against a specific industrial device or server.
- UE idle accounting excludes background idle power outside edge waiting.
- Downlink result-transfer energy is absent.
- Only one task-workload seed was used, although it contains 1,000 episodes.
- Episode block intervals do not capture uncertainty across independent networks, hardware, policies, or mapping choices.
- The task workload is QECO's iid Bernoulli/uniform generator, not an industrial task-arrival trace. The industrial dataset currently drives network-condition profiles only.

---

## 10. Recommended next steps

### Immediate next experiment: refine the crossover

Run a predeclared capacity grid between 14 and 8.4877 using the same frozen workload records and policy. A reasonable initial grid is:

`14, 13, 12, 11, 10, 9, 8.4877`

Then identify the adjacent capacities bracketing an offload/local ratio of one. If needed, perform a second, predeclared bisection inside that single interval. Do not tune the policy during this process.

The result should report both:

- Matched UE-energy ratio for the same offloaded task sizes/densities
- Offloaded completion fraction and deadline violations

Energy savings achieved only by failing tasks are not operational savings.

### Add explicit benchmark policies

For a stronger causal comparison, evaluate these fixed baselines on the same workload arrays:

- Always local
- Always edge 0
- Always edge 1
- Random edge/local with a fixed seed
- Frozen QECO

These are evaluation comparators, not modifications to the learned QECO weights. They would provide actual queueing, deadline, and energy outcomes rather than only an analytic local-energy counterfactual.

### Run mapping sensitivity analysis

The square-root compression is explicit but not uniquely determined by the dataset. Predeclare several plausible mappings, such as compression exponents 0.33, 0.5, and 1.0, or residual-capacity assumptions grounded in a documented link-utilization baseline. Report whether the crossover conclusion persists.

Do not introduce packet loss unless a trace or external measurement supplies it.

### Replicate workload uncertainty

After the fixed-seed matched study is preserved, run several independent QECO workload seeds while keeping the same profile capacities. Report across-seed means and intervals. Do not replace the current fixed-seed artifact; add this as a robustness study.

### Version the environment before fixing policy inputs

If the transmission predictor and stale-density observation are corrected:

1. Create a clearly versioned environment.
2. Add regression tests for the intended observation semantics.
3. Retrain D3QN from scratch in that corrected environment.
4. Do not compare corrected-policy results as though they used the original frozen policy.

### Calibrate physical energy

Before claiming factory-scale joule savings, document the actual UE CPU power, radio transmit power, idle power, edge power, slot duration, and rate/work units. Add downlink and background energy if they are material to the chosen system boundary.

---

## 11. Reproduction commands

Run commands from `D:\research\QECO`.

Accounting tests:

```powershell
.venv\Scripts\python.exe -B -m unittest test_accounting -v
```

Derive profiles from the frozen development files:

```powershell
python -B derive_network_stress_profiles.py
```

The derivation script intentionally refuses to overwrite an existing `network_stress_profiles` directory. Archive the existing results or change the output destination in a reviewed copy before reproducing it.

Example frozen-policy stress run:

```powershell
.venv\Scripts\python.exe -B evaluate_stress_level.py `
  --stress-level Medium `
  --transmission-capacity 8.48769868957287 `
  --output network_stress_profiles\new_run_medium
```

Analyze complete runs:

```powershell
python -B analyze_network_stress.py
```

The stress runner also refuses to overwrite its output directory. This prevents accidental destruction or mixing of experiment runs.

---

## 12. Current Git/workspace state

Tracked files modified:

- `D3QN.py`
- `MEC_Env.py`

New untracked implementation/artifact paths include:

- `ENERGY_ACCOUNTING.md`
- `accounting.py`
- `evaluate_baseline.py`
- `reaccount_baseline.py`
- `evaluate_stress_level.py`
- `derive_network_stress_profiles.py`
- `analyze_network_stress.py`
- `test_accounting.py`
- `accounting_baseline\`
- `network_stress_profiles\`
- `RESEARCH_HANDOFF.md`

Before committing, review storage size because the per-episode NPZ directories contain thousands of detailed records. Code, CSV, JSON, Markdown, and compact manifests belong in source control; large NPZ activity files may be better archived separately with their SHA-256 manifest.

---

## 13. Single-sentence current conclusion

Using frozen industrial-development features to define conservative effective-capacity profiles, a frozen QECO policy, identical 1,000-episode workloads, and corrected power-time accounting, offloading saves UE energy at Normal/Low but crosses to higher UE energy at Medium stress while deadline violations rise from 7.47% to 23.76%.

---

## 14. Stress-aware selective offloading phase (completed 2026-09-13)

### Purpose and constraints

A causal decision wrapper was added around the frozen D3QN. D3QN was not retrained, its weights and reward were not changed, and the stress mapping, workload arrays, deadlines, powers, and power-time accounting remained fixed. For every arriving task the wrapper estimates all three actions: local, edge 0, and edge 1.

The estimator reads only information available at the decision instant:

- task size and computation density;
- current local, transmission, and edge FIFO state;
- the current deadline horizon;
- current effective transmission capacity;
- configured resource capacities and powers needed for the physical estimate.

Known local and transmission backlog is simulated slot by slot. Transmissions already queued ahead of the candidate are propagated into the relevant edge forecast. Edge sharing uses the currently observed active-UE count as a fixed short-horizon forecast. No future task arrival, terminal outcome, accounting result, or future stress value is read. Every D3QN action in a slot is computed before any selector action, so simultaneous UEs inspect the same pre-step environment.

An infeasible action receives infinite constrained objective cost. The original action is retained when it is feasible and tied for minimum energy. Otherwise the lowest-energy feasible action is selected; when no action is predicted feasible, D3QN is retained. This is a constrained energy minimization, not a Medium/High stress shortcut.

### Added files

- `stress_aware_selector.py`: decision-time estimator and selection rule.
- `evaluate_selective_stress.py`: frozen-policy experiment runner with per-decision audits and integrity hashes.
- `analyze_selective_offloading.py`: paired aggregation, block-resampled intervals, forecast diagnostics, and Markdown/CSV/JSON reporting.
- `test_stress_aware_selector.py`: selector non-mutation, feasibility, queue, and radio-to-edge transition tests.
- `selective_offloading/REPORT.md`: concise research result.
- `selective_offloading/comparison.csv` and `comparison.json`: machine-readable comparisons and uncertainty.
- `selective_offloading/forecast_diagnostics.csv`: feasibility forecast diagnostics.
- `selective_offloading/run_{normal,low,medium,high}/`: 1,000-episode metrics, accounting details, decision audits, metadata, and summaries.

### Verification

- 11 accounting and selector tests pass.
- Each condition completed 1,000 episodes and 601,153 arrivals.
- All metadata report zero learning calls.
- Trainable policy hashes match before and after each run.
- Checkpoint files and frozen workload files retained their hashes.
- Normal and Low were run independently; their episode CSV and summary SHA-256 hashes are exactly identical because both retain capacity 14.
- The selector forecast never mutates the environment state in its unit test.
- A smoke-test defect was fixed before the full experiment: a transmitted forecast task initially entered the edge forecast with depleted remaining work. It now enters edge computation with its full CPU workload.

### Final paired results

|Stress|Method|UE J/episode|System J/episode|TX J/episode|Violations|Violation %|Terminal s|Offload %|Overrides|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Normal|Original|434.376|488.910|263.795|44,911|7.47|0.5933|63.96|0|
|Normal|Selective|397.618|455.805|246.729|41,102|6.84|0.6041|63.54|367,778|
|Low|Original|434.376|488.910|263.795|44,911|7.47|0.5933|63.96|0|
|Low|Selective|397.618|455.805|246.729|41,102|6.84|0.6041|63.54|367,778|
|Medium|Original|562.167|593.871|351.388|142,833|23.76|0.7247|56.45|0|
|Medium|Selective|522.685|554.444|246.872|77,999|12.97|0.6975|38.11|312,858|
|High|Original|614.073|629.326|387.240|221,820|36.90|0.7695|53.64|0|
|High|Selective|567.089|577.113|241.910|150,257|24.99|0.7775|27.08|287,855|

Paired whole-run UE-energy changes were -8.46% Normal/Low, -7.02% Medium, and -7.65% High. The paired 95% block-resampled intervals for the UE-energy change in J/episode exclude zero in every condition:

- Normal/Low: [-37.070, -36.455]
- Medium: [-39.813, -39.171]
- High: [-47.459, -46.524]

The time-to-terminal metric improves at Medium, rises by about 0.0080 seconds at High, and rises by about 0.0108 seconds at Normal/Low. This metric includes terminal failures at their deadline; it must be read with the separate and substantially improved violation rates.

### Matched-task energy finding

Matched savings compare observed TX plus idle UE energy for the exact selectively offloaded tasks with their power-time local-computation energy counterfactual:

- Normal/Low: +149,971.15 J, or +37.66% savings over 1,000 episodes.
- Medium: +12,759.04 J, or +4.91%. Positive savings are restored from Original QECO's -8.54%.
- High: -43,221.83 J, or -21.75%. Positive per-task offloading savings cannot be restored with the locked parameters.

The High result has an analytic explanation. Ignoring idle energy, TX becomes cheaper than local computation only when

`capacity > P_TX * C_UE / (P_LOCAL * density)`.

The thresholds for densities 0.197, 0.297, and 0.397 are 15.1777, 10.0673, and 7.5315. High capacity is 6.3269, below all three. TX energy alone therefore exceeds full local energy for every workload density. The wrapper still retains or chooses offloading when deadline feasibility requires it, as instructed; it does not force High to local execution.

### Current conclusion

The causal wrapper restores positive matched UE-energy savings at Medium while reducing whole-run UE energy, system energy, TX energy, and violations. At High it improves whole-run UE energy by 7.65%, system energy by 8.30%, and violations from 36.90% to 24.99%, but positive matched offloading savings are physically unavailable under the frozen power and capacity values. This establishes a defensible boundary rather than a failed tuning result.
