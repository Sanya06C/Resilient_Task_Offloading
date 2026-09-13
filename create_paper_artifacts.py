"""Create publication artifacts from completed QECO experiment outputs only."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "selective_offloading"
OUTPUT = ROOT / "paper_artifacts"
LEVELS = ["Normal", "Medium", "High"]
METHODS = {
    "Original QECO": "original",
    "Stress-Aware QECO": "selective",
}
COLORS = {"Original QECO": "#0072B2", "Stress-Aware QECO": "#D55E00"}
MARKERS = {"Original QECO": "s", "Stress-Aware QECO": "o"}


def save_figure(metric, ylabel, stem, percent=False):
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    x = np.arange(len(LEVELS))
    for method, prefix in METHODS.items():
        values = data[f"{prefix}_{metric}"].to_numpy()
        if percent:
            values = 100 * values
        ax.plot(x, values, color=COLORS[method], marker=MARKERS[method],
                linewidth=2.1, markersize=6.5, label=method)
    ax.set_xticks(x, LEVELS)
    ax.set_xlabel("Network stress level")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def latex_table(frame, path, formats):
    formatted = frame.copy()
    for column, formatter in formats.items():
        formatted[column] = formatted[column].map(formatter)
    path.write_text(formatted.to_latex(index=False, escape=True), encoding="utf-8")


OUTPUT.mkdir(exist_ok=True)
raw = pd.read_csv(SOURCE / "comparison.csv")
data = raw.set_index("stress_level").loc[LEVELS].reset_index()

result_rows = []
for _, row in data.iterrows():
    for method, prefix in METHODS.items():
        result_rows.append({
            "Stress": row.stress_level,
            "Method": method,
            "UE energy (J/episode)": row[f"{prefix}_ue_energy_j_per_episode"],
            "System energy (J/episode)": row[f"{prefix}_total_system_energy_j_per_episode"],
            "TX energy (J/episode)": row[f"{prefix}_tx_energy_j_per_episode"],
            "Deadline violations": int(row[f"{prefix}_deadline_violations"]),
            "Violation rate (%)": 100 * row[f"{prefix}_deadline_violation_fraction"],
            "Time-to-terminal (s)": row[f"{prefix}_time_to_terminal_seconds"],
            "Offloading ratio (%)": 100 * row[f"{prefix}_offloading_ratio"],
        })
results = pd.DataFrame(result_rows)
results.to_csv(OUTPUT / "final_results_table.csv", index=False)
latex_table(results, OUTPUT / "final_results_table.tex", {
    "UE energy (J/episode)": lambda x: f"{x:.3f}",
    "System energy (J/episode)": lambda x: f"{x:.3f}",
    "TX energy (J/episode)": lambda x: f"{x:.3f}",
    "Deadline violations": lambda x: f"{x:,}",
    "Violation rate (%)": lambda x: f"{x:.2f}",
    "Time-to-terminal (s)": lambda x: f"{x:.4f}",
    "Offloading ratio (%)": lambda x: f"{x:.2f}",
})

matched = pd.DataFrame({
    "Stress": data.stress_level,
    "Original matched saving (%)": 100 * data.original_matched_ue_energy_saving_fraction,
    "Stress-aware matched saving (J)": data.selective_matched_ue_energy_savings_j,
    "Stress-aware matched saving (%)": 100 * data.selective_matched_ue_energy_saving_fraction,
})
matched.to_csv(OUTPUT / "matched_offloading_savings_table.csv", index=False)
latex_table(matched, OUTPUT / "matched_offloading_savings_table.tex", {
    "Original matched saving (%)": lambda x: f"{x:.2f}",
    "Stress-aware matched saving (J)": lambda x: f"{x:.2f}",
    "Stress-aware matched saving (%)": lambda x: f"{x:.2f}",
})

# These counts were derived from the completed per-decision audit NPZ files.
# Keeping the derived table here avoids rerunning the policy or simulator.
transitions = pd.DataFrame([
    ["Normal", 117697, 133666, 67853, 48562],
    ["Medium", 40314, 216960, 34934, 20650],
    ["High", 9460, 265141, 9126, 4128],
], columns=["Stress", "Local->Edge", "Edge->Local", "Edge0->Edge1", "Edge1->Edge0"])
transitions.to_csv(OUTPUT / "override_transition_table.csv", index=False)
latex_table(transitions, OUTPUT / "override_transition_table.tex", {
    column: lambda x: f"{x:,}" for column in transitions.columns[1:]
})

save_figure("ue_energy_j_per_episode", "UE energy (J per episode)",
            "figure_ue_energy_vs_stress")
save_figure("deadline_violation_fraction", "Deadline violation rate (%)",
            "figure_deadline_violations_vs_stress", percent=True)
save_figure("tx_energy_j_per_episode", "Transmission energy (J per episode)",
            "figure_transmission_energy_vs_stress")
save_figure("offloading_ratio", "Offloading ratio (%)",
            "figure_offloading_ratio_vs_stress", percent=True)

findings = """# Final findings for the Results section

- Across 1,000 matched episodes per condition, Stress-Aware QECO reduced mean UE energy by 8.46% at Normal stress, 7.02% at Medium stress, and 7.65% at High stress relative to Original QECO.
- Total-system energy decreased by 6.77%, 6.64%, and 8.30% at Normal, Medium, and High stress, respectively; transmission energy decreased by 6.47%, 29.74%, and 37.53%.
- At Medium stress, the selector restored positive matched offloading energy savings: the same-task UE comparison improved from -8.54% for Original QECO to +4.91% for Stress-Aware QECO, equal to 12,759.04 J across the evaluated episodes.
- At High stress, the selector reduced overall UE energy and lowered the deadline violation rate from 36.90% to 24.99%, but matched per-task offloading savings remained negative at -21.75%.
- Positive High-stress per-task offloading savings were physically infeasible under the locked assumptions: the effective capacity of 6.3269 was below the transmission-only break-even capacities of 15.1777, 10.0673, and 7.5315 for the three task densities.
- The selector reduced the Medium-stress deadline violation rate from 23.76% to 12.97% while reducing time-to-terminal from 0.7247 s to 0.6975 s; at High stress, time-to-terminal increased slightly from 0.7695 s to 0.7775 s despite the large reduction in violations.
- Normal-stress offloading changed only from 63.96% to 63.54% because 116,415 overrides switched between edge servers and the 117,697 Local-to-Edge transitions substantially offset 133,666 Edge-to-Local transitions.
"""
(OUTPUT / "FINAL_FINDINGS.md").write_text(findings, encoding="utf-8")

readme = """# Paper-ready QECO result artifacts

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
"""
(OUTPUT / "README.md").write_text(readme, encoding="utf-8")

paper_results = """# Original QECO versus Stress-Aware QECO

All values use 1,000 matched episodes and the frozen methodology.

## Final results

| Stress | Method | UE energy (J/episode) | System energy (J/episode) | TX energy (J/episode) | Violations | Violation rate | Time-to-terminal (s) | Offloading ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | Original QECO | 434.376 | 488.910 | 263.795 | 44,911 | 7.47% | 0.5933 | 63.96% |
| Normal | Stress-Aware QECO | **397.618** | **455.805** | **246.729** | **41,102** | **6.84%** | 0.6041 | 63.54% |
| Medium | Original QECO | 562.167 | 593.871 | 351.388 | 142,833 | 23.76% | 0.7247 | 56.45% |
| Medium | Stress-Aware QECO | **522.685** | **554.444** | **246.872** | **77,999** | **12.97%** | **0.6975** | 38.11% |
| High | Original QECO | 614.073 | 629.326 | 387.240 | 221,820 | 36.90% | 0.7695 | 53.64% |
| High | Stress-Aware QECO | **567.089** | **577.113** | **241.910** | **150,257** | **24.99%** | 0.7775 | 27.08% |

## Matched offloading energy savings

| Stress | Original saving | Stress-aware saving | Stress-aware saving across episodes |
|---|---:|---:|---:|
| Normal | 28.25% | **37.66%** | 149,971.15 J |
| Medium | -8.54% | **+4.91%** | +12,759.04 J |
| High | -26.54% | **-21.75%** | -43,221.83 J |

Medium-stress positive savings were restored. At High stress, overall UE energy, system energy, transmission energy, and violation rate improved, but positive per-task offloading savings were physically infeasible under the locked power/capacity assumptions.

## Override transitions

| Stress | Local→Edge | Edge→Local | Edge 0→Edge 1 | Edge 1→Edge 0 |
|---|---:|---:|---:|---:|
| Normal | 117,697 | 133,666 | 67,853 | 48,562 |
| Medium | 40,314 | 216,960 | 34,934 | 20,650 |
| High | 9,460 | 265,141 | 9,126 | 4,128 |

## Figures

![UE energy versus network stress](figure_ue_energy_vs_stress.png)

**Figure 1.** Mean UE-only energy under Original and Stress-Aware QECO.

![Deadline violation rate versus network stress](figure_deadline_violations_vs_stress.png)

**Figure 2.** Pooled deadline violation rate under increasing network stress.

![Transmission energy versus network stress](figure_transmission_energy_vs_stress.png)

**Figure 3.** Mean UE transmission energy per episode.

![Offloading ratio versus network stress](figure_offloading_ratio_vs_stress.png)

**Figure 4.** Fraction of arriving tasks routed to either edge server.

## Results-section findings

See `FINAL_FINDINGS.md` for seven concise findings suitable for the paper.
"""
(OUTPUT / "PAPER_RESULTS.md").write_text(paper_results, encoding="utf-8")

print(f"Created {len(list(OUTPUT.iterdir()))} artifacts in {OUTPUT}")
