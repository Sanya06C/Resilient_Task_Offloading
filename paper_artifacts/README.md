# Paper-ready QECO result artifacts

All artifacts in this directory were generated from the frozen completed outputs in `selective_offloading/`. No policy, selector, stress profile, accounting rule, threshold, workload, or experiment setting was changed or rerun.

## Tables

- `final_results_table.csv` and `.tex`: Original QECO versus Stress-Aware QECO.
- `matched_offloading_savings_table.csv` and `.tex`: matched-task UE-energy savings.
- `override_transition_table.csv` and `.tex`: action-transition counts.

## Figures

- `figure_ue_energy_vs_stress.pdf` and `.png`
- `figure_deadline_violations_vs_stress.pdf` and `.png`
- `figure_transmission_energy_vs_stress.pdf` and `.png`
- `figure_offloading_ratio_vs_stress.pdf` and `.png`

PDF files are vector figures. PNG files are rendered at 400 dpi. Normal, Medium, and High results are shown; Low was omitted because its locked effective capacity and results are identical to Normal.

## Interpretation

`FINAL_FINDINGS.md` contains seven paper-ready findings. Matched savings compare observed UE transmission plus idle energy for the tasks selected for offloading with the power-time local-computation energy of those exact tasks. This counterfactual differs from whole-run UE energy, which also reflects routing and queue-trajectory changes.
