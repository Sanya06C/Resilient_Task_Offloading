# Frozen-data network-stress evaluation

## Result

The first tested level at which offloading stops saving UE energy is **Medium stress**, corresponding to an effective QECO transmission capacity of **8.4877** versus the default **14.0** (60.63% of default capacity).

For the exact tasks selected for offloading at Medium, measured UE transmission plus idle energy is 1.0363 J/task. Computing those same task sizes and densities locally would require 0.9548 J/task under the approved power-time model. The offload/local ratio is 1.0854, so offloading consumes 8.54% more UE energy. At High, the excess grows to 26.54%.

This identifies the first tested profile crossover, not an exact physical bandwidth threshold. The present levels bracket the crossover between 14.0 and 8.4877 capacity units. A finer capacity sweep would be needed to locate the numerical threshold inside that interval.

## Frozen-data profile derivation

Only the frozen `train_dataset.parquet` and `validation_dataset.parquet` files were read. Their SHA-256 hashes match the classifier study. The held-out dataset was not opened or used.

The combined development data contain 191,008 BENIGN and 4,798 ABNORMAL windows. Only `a_day1` and `s_day1` contain both classes. Stress calibration therefore uses source-matched BENIGN references from those two captures; the BENIGN-only `tf_a` and `tf_s` rows remain in the provenance counts but are excluded from severity calibration because their capture distributions cannot be normalized against same-source ABNORMAL windows.

Each window is ranked against BENIGN windows from its own source. The stress score weights are:

- Estimated packet rate: 40%.
- New flows: 15%.
- Unique sources: 10%.
- Unique destinations: 10%.
- Short-flow churn, `new_flows / max(duration, 0.001 s)`: 15%.
- Persistent-flow occupancy, `new_flows * duration`: 10%.

The two duration interactions allow short high-churn flows and long persistent flows to contribute without declaring either short or long duration inherently stressful. ABNORMAL windows are divided into source-specific score terciles, giving each stress level representation from both attack captures.

For mapping into QECO, a source-normalized offered-load multiplier gives 70% weight to packet rate, 15% to new flows, and 7.5% each to unique sources and destinations. Ratios below the same-source BENIGN median are clipped to one, so abnormal traffic cannot create artificial capacity gains. Because packet sizes, physical link rate, and measured link utilization are absent, the raw multiplier is conservatively compressed:

`effective_capacity = 14 / sqrt(offered_load_multiplier)`

The mapping has a lower bound of 25% of default capacity. That floor is not reached by these profiles. No packet-loss, retransmission, jitter, or unsupported radio value is introduced.

| Profile | Windows | Representative load multiplier | Capacity factor | Effective TX capacity |
|---|---:|---:|---:|---:|
| Normal | 13,186 source-matched BENIGN | 1.0000 | 1.0000 | 14.0000 |
| Low | 1,600 ABNORMAL | 1.0000 | 1.0000 | 14.0000 |
| Medium | 1,598 ABNORMAL | 2.7207 | 0.6063 | 8.4877 |
| High | 1,600 ABNORMAL | 4.8964 | 0.4519 | 6.3269 |

Low has the same QECO condition as Normal because its source-balanced packet/load indicators do not exceed the BENIGN reference. This is an evidence-based null degradation: the frozen ABNORMAL label is not itself a congestion measurement. Assigning packet loss or reducing capacity for this tier would invent unsupported network stress.

## Matched QECO results

All levels use the exact same 1,000 saved workload arrays and seed 20260913. Normal comes from the completed default baseline; Low, Medium, and High replay those exact arrays. Each run starts from the same checkpoint and zero LSTM history, retains history across episodes in the same way as the original runner, sets epsilon to one, and makes no learning calls. Hash checks confirm that saved workloads, checkpoints, and trainable weights remain unchanged.

Energy values are mean joules per episode across all 20 UEs and both edges. Latency is pooled across successfully completed tasks. Time to terminal includes both completions and deadline failures.

| Stress | UE-only energy | Total-system energy | TX energy | Completed latency (s) | Time to terminal (s) | Deadline violations | Offload ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | 434.3763 | 488.9101 | 263.7945 | 0.5605 | 0.5933 | 44,911 / 7.47% | 63.96% |
| Low | 434.3763 | 488.9101 | 263.7945 | 0.5605 | 0.5933 | 44,911 / 7.47% | 63.96% |
| Medium | 562.1667 | 593.8707 | 351.3879 | 0.6388 | 0.7247 | 142,833 / 23.76% | 56.45% |
| High | 614.0728 | 629.3264 | 387.2401 | 0.6347 | 0.7695 | 221,820 / 36.90% | 53.64% |

High's successful-task latency is slightly below Medium because only 43.86% of offloaded tasks complete at High, compared with 68.72% at Medium and 95.27% at Normal. Slow and difficult tasks increasingly become deadline violations and leave the successful-task latency denominator. Time to terminal and the violation rate show the monotonic degradation.

## Matched UE-energy crossover

For each profile, the numerator is aggregate UE transmission plus idle energy for tasks QECO selected for offloading. The denominator is power-time local-computation energy for those exact task sizes and computation densities:

`E_local_counterfactual = P_local * duration * task_size / (UE_COMP_CAP * duration / density)`

This matched calculation avoids comparing the offloaded group with a different set of tasks chosen for local execution. It does not simulate local queuing or claim equal deadline performance; it answers the energy-to-compute requirement for the same workloads. Energy consumed by offloaded tasks that later miss deadlines remains in the numerator.

| Stress | Actual offload UE J/task | Matched local J/task | Offload/local ratio | UE energy saving | 95% block interval for ratio | Offloaded completion |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 0.6887 | 0.9599 | 0.7175 | +28.25% | 0.7168–0.7182 | 95.27% |
| Low | 0.6887 | 0.9599 | 0.7175 | +28.25% | 0.7168–0.7182 | 95.27% |
| Medium | 1.0363 | 0.9548 | 1.0854 | −8.54% | 1.0840–1.0869 | 68.72% |
| High | 1.2012 | 0.9492 | 1.2654 | −26.54% | 1.2632–1.2676 | 43.86% |

The intervals are descriptive paired intervals from 5,000 resamples of 50 nonoverlapping 20-episode blocks. They are conditional on these captures, the chosen capacity mapping, the frozen policy, and this workload sequence. They are not confidence intervals for other industrial networks.

## Files

- `profiles.csv`: derived profile parameters and feature medians.
- `abnormal_window_assignments.csv`: every ABNORMAL development window and assigned tier.
- `derivation.json`: input hashes, source counts, score weights, mapping, and exclusions.
- `stress_results.csv` and `stress_results.json`: final comparison and crossover audit.
- `run_low`, `run_medium`, and `run_high`: per-episode metrics, detailed activity records, and frozen-policy metadata.
