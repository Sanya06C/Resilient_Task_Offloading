# Stress-Aware Selective Offloading Results

The learned QECO policy and its weights were kept frozen. At each task arrival, the selector compared local execution, edge 0, and edge 1 using the current task, known FIFO queues, deadline, and current transmission capacity. It simulated known backlog without future arrivals and held the observed edge sharing count constant for the short forecast. All D3QN actions were computed before any selector decision in a slot, so simultaneous UEs saw the same state.

## Primary comparison

|Stress|Method|UE J/episode|System J/episode|TX J/episode|Violations|Violation %|Terminal s|Offload %|Overrides|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Normal|Original|434.376|488.910|263.795|44,911|7.47|0.5933|63.96|0|
|Normal|Stress-aware|397.618|455.805|246.729|41,102|6.84|0.6041|63.54|367,778|
|Low|Original|434.376|488.910|263.795|44,911|7.47|0.5933|63.96|0|
|Low|Stress-aware|397.618|455.805|246.729|41,102|6.84|0.6041|63.54|367,778|
|Medium|Original|562.167|593.871|351.388|142,833|23.76|0.7247|56.45|0|
|Medium|Stress-aware|522.685|554.444|246.872|77,999|12.97|0.6975|38.11|312,858|
|High|Original|614.073|629.326|387.240|221,820|36.90|0.7695|53.64|0|
|High|Stress-aware|567.089|577.113|241.910|150,257|24.99|0.7775|27.08|287,855|

## Matched UE-energy result

This comparison uses the exact tasks selected for offloading in each run.

|Stress|Original matched saving %|Selective matched saving J|Selective matched saving %|UE energy change vs original %|
|---|---:|---:|---:|---:|
|Normal|28.25|149971.148|37.66|-8.46|
|Low|28.25|149971.148|37.66|-8.46|
|Medium|-8.54|12759.035|4.91|-7.02|
|High|-26.54|-43221.827|-21.75|-7.65|

## Override composition

|Stress|Total|Lower predicted energy|Original infeasible|Local to edge|Edge to local|Edge switch|
|---|---:|---:|---:|---:|---:|---:|
|Normal|367,778|335,870|31,908|117,697|133,666|116,415|
|Low|367,778|335,870|31,908|117,697|133,666|116,415|
|Medium|312,858|272,550|40,308|40,314|216,960|55,584|
|High|287,855|216,485|71,370|9,460|265,141|13,254|

## Main finding

- **Medium:** matched UE-energy savings are **positive** at 4.91%; total UE energy changes -7.02% relative to Original QECO.
- **High:** matched UE-energy savings are **negative** at -21.75%; total UE energy changes -7.65% relative to Original QECO.

## Physical break-even check

Before idle energy, offloading can save UE energy only when the effective capacity exceeds `P_TX * C_UE / (P_LOCAL * density)`. With the locked QECO powers and computation capacity:

|Computation density|Minimum capacity for TX energy below local energy|
|---:|---:|
|0.197|15.1777|
|0.297|10.0673|
|0.397|7.5315|

High stress uses capacity 6.3269, below all three thresholds. Thus no individual High-stress offload can have positive matched UE-energy savings under the locked power-time model; transmission energy alone already exceeds full local-computation energy. This is a constraint implied by the retained parameters, not a selector failure or a reason to alter the stress mapping after seeing outcomes.

## Interpretation limits

The forecasts are causal decision-time estimates, not an oracle. Future arrivals are unknown, and edge capacity sharing is held at its current observed count. The matched local benchmark is an energy counterfactual for selected tasks; it does not simulate a separate all-local queue trajectory. Report both matched savings and whole-run UE energy because they answer different questions.

The contemporaneous original D3QN action inside a stress-aware run can diverge from the action in the Original trajectory after earlier overrides change queue states. Overrides therefore count interventions against the frozen policy in the state it actually encounters.
