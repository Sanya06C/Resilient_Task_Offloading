# QECO accounting baseline

This work preserves D3QN decisions, learned weights, observations, and the original reward energy arrays. No stress profiles are introduced. The evaluator does not import main.py because its module-level code deletes the models directory.

## Confirmed bookkeeping fixes

- D3QN.do_store_energy now sums every edge/idle component. Only the logging method changes.
- Local partial-slot processed data: remaining size, without dividing by density again.
- TX full-slot transmitted data: link capacity, not all remaining data.
- Edge partial-slot processed data: remaining size, without dividing by active UE count again.
- Episode totals sum all task/UE/edge dimensions, including energy spent on tasks that fail.
- Deadline failures come from per-task terminal outcomes, not stage counters that omit queued expiry.
- Task fractions use actual arrivals; an action of zero during an empty slot is not a local task.
- Completed-task latency excludes failed tasks. A separate time-to-terminal metric includes failures.
- Completion exactly at the deadline is a success; an unfinished task at that time is a violation.

## Timing issue and policy preservation

In MEC.step's transmission predictor, the result is assigned to t_ue_comp instead of t_ue_tran. These variables are policy inputs. Repairing the live assignment changes observations and potentially actions even with frozen weights. Accordingly, EpisodeAccounting maintains corrected local and TX finish predictors separately and counts divergences. The live fields stay unchanged for this baseline. Actual latency uses terminal timestamps, not either predicted field.

Another issue: the edge workload observation update uses ue_arrive_task_dens left over from the preceding loop. Correcting it would also change policy input, so it is documented rather than silently changed.

The actual observation dimension is 4 + N_EDGE = 6 under defaults.

## Original equations and unresolved physical interpretation

Let d be task density, c = UE_COMP_CAP * DURATION, q = c/d, r be remaining task data, C = UE_TRAN_CAP * DURATION, e = EDGE_COMP_CAP * DURATION, m the active UE count, and dt = DURATION.

- Local full-slot energy: q * (1**(-27) * q) = q^2.
- Local final-slot energy: (r/d) * (1**(-27) * q) = r*q/d.
- Since 1**(-27) equals 1, this is not a small hardware coefficient. The final-slot density division also differs from the full-slot data term. UE_COMP_ENERGY is not used. Merely changing 1 to 10 does not establish correct units or a defensible CPU model.
- TX energy each active slot: C * UE_TRAN_ENERGY, including the final partial slot. With Config describing that parameter as power, data times power is not energy.
- Edge full-slot energy per UE: (e/d) * EDGE_COMP_ENERGY * dt, although its data service is only e/(d*m).
- Edge final-slot energy: r * EDGE_COMP_ENERGY * dt.
- Idle full-slot energy: e/(d*m), with neither power nor time.
- Idle partial-slot energy: (r/m) * UE_IDLE_ENERGY, without time.
- Therefore legacy totals are explicitly labelled unvalidated simulator units, not joules. They cannot yet support a physical energy-saving crossover claim.

## Minimal proposed evaluation-only model (requires user's accounting choice)

Use the existing Config power constants, assuming watts and dt in seconds, and record actual service fractions f = min(remaining, capacity)/capacity.

- Local: P_local * dt * f_local.
- TX: P_tx * dt * f_tx.
- Edge: P_edge * dt * f_edge/m per UE, treating configured edge power as per-server active power and allocating it by the scheduler share. Unused share is not redistributed inside a slot, matching the existing scheduler.
- Idle: P_idle times the union of that UE's post-TX edge-wait intervals, excluding concurrent local/TX active time. Charge each UE only once, not once per outstanding task.
- Within a slot, concurrent service intervals start together; partial completion releases active power after the service fraction. End-to-end latency remains slot-based.
- UE total = local + TX + idle; system total = UE total + edge.
- This is task-attributable active energy. Background idle energy for all devices/servers, downlink return energy, and infrastructure energy are not represented.
- If the intended model is DVFS or a CPU capacitance model instead, device coefficient, CPU frequency, cycles/data units, and the intended equation are required. None is invented here.

Both versions retain the original arrays for reward compatibility. The proposed model is opt-in; the default accounting model is legacy_unvalidated until a choice is made.

## Baseline protocol

evaluate_baseline.py uses the existing PerformanceMode/800 checkpoint path for all 20 UEs, sets epsilon=1 exactly as the repository's commented evaluation block, calls no learn method, and verifies trainable-variable and checkpoint hashes before/after. It uses all current Config physical parameters, including the mixed UE energy-state distribution; this checkpoint/scenario distinction is explicit in metadata.

Config.N_EPISODE defaults to 1000. Workload generation follows main.train exactly, including the last MAX_DELAY empty slots. The runner records a seed and retains LSTM history between episodes as main.train does. The default main.py entry point itself trains, so it is intentionally not run for this frozen-policy baseline.

Outputs: per-episode CSV; per-task/resource NPZ files for audit or later approved re-accounting; metadata with code/checkpoint hashes and runtime; summary with pooled task rates and episode means/SD. Episode SD is descriptive, not an independent-seed confidence interval.

Run tests:
.venv/Scripts/python.exe -B -m unittest test_accounting -v

Run baseline:
.venv/Scripts/python.exe -B evaluate_baseline.py --output accounting_baseline/default_legacy --energy-model legacy_unvalidated

Use --energy-model power_time_v1 only after agreeing the proposed convention. Check metadata status=complete and trainable_weights_unchanged=true before treating a run as complete.
