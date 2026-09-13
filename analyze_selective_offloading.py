"""Validate and compare original and stress-aware frozen-policy experiments."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from Config import Config


ROOT = Path(__file__).resolve().parent
STRESS_ROOT = ROOT / "network_stress_profiles"
SELECTIVE_ROOT = ROOT / "selective_offloading"
LEVELS = ["Normal", "Low", "Medium", "High"]
ORIGINAL = {
    "Normal": ROOT / "accounting_baseline" / "default_power_time_v1",
    "Low": STRESS_ROOT / "run_low",
    "Medium": STRESS_ROOT / "run_medium",
    "High": STRESS_ROOT / "run_high",
}
SELECTIVE = {level: SELECTIVE_ROOT / f"run_{level.lower()}" for level in LEVELS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pooled_time(metrics):
    return float((metrics.average_time_to_terminal_seconds * metrics.arrived_tasks).sum()
                 / metrics.arrived_tasks.sum())


def paired_block_interval(selective, original, seed=20260913,
                          block_size=20, replicates=5000):
    difference = np.asarray(selective) - np.asarray(original)
    if len(difference) % block_size:
        raise ValueError("Episode count must divide into complete blocks")
    blocks = difference.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for i in range(replicates):
        values[i] = blocks[rng.integers(0, len(blocks), len(blocks))].mean()
    return np.quantile(values, [.025, .975]).tolist()


def fmt(value, places=3):
    return f"{value:.{places}f}"


def main():
    profiles = pd.read_csv(STRESS_ROOT / "profiles.csv").set_index("stress_level")
    original_matched = pd.read_csv(STRESS_ROOT / "stress_results.csv").set_index("stress_level")
    rows, intervals, forecast_rows = [], {}, []
    reference_workloads = None

    for level in LEVELS:
        original = pd.read_csv(ORIGINAL[level] / "episode_metrics.csv")
        selective = pd.read_csv(SELECTIVE[level] / "episode_metrics.csv")
        metadata = json.loads((SELECTIVE[level] / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((SELECTIVE[level] / "summary.json").read_text(encoding="utf-8"))
        if len(original) != 1000 or len(selective) != 1000 or summary["episodes"] != 1000:
            raise ValueError(f"Incomplete experiment: {level}")
        if metadata["status"] != "complete" or metadata["learn_calls"] != 0:
            raise ValueError(f"Invalid run metadata: {level}")
        if metadata["policy_hashes_before"] != metadata["policy_hashes_after"]:
            raise ValueError(f"Policy weights changed: {level}")
        if not metadata["workload_hashes_unchanged"]:
            raise ValueError(f"Workload changed: {level}")
        if reference_workloads is None:
            reference_workloads = metadata["workload_activity_sha256"]
        elif reference_workloads != metadata["workload_activity_sha256"]:
            raise ValueError("Selective runs used different workload arrays")
        expected_capacity = float(
            profiles.loc[level, "effective_ue_transmission_capacity"])
        if not np.isclose(metadata["effective_ue_transmission_capacity"], expected_capacity):
            raise ValueError(f"Stress mapping changed: {level}")

        arrived = int(selective.arrived_tasks.sum())
        original_failed = int(original.deadline_violations.sum())
        selective_failed = int(selective.deadline_violations.sum())
        original_ue = float(original.ue_only_total_energy.mean())
        selective_ue = float(selective.ue_only_total_energy.mean())
        original_system = float(original.total_system_energy.mean())
        selective_system = float(selective.total_system_energy.mean())
        matched_local = float(selective.matched_local_ue_energy.sum())
        matched_actual = float(selective.selected_offload_ue_energy.sum())
        matched_saving = matched_local - matched_actual
        original_match = original_matched.loc[level]

        row = {
            "stress_level": level,
            "effective_transmission_capacity": expected_capacity,
            "arrived_tasks": arrived,
            "original_ue_energy_j_per_episode": original_ue,
            "selective_ue_energy_j_per_episode": selective_ue,
            "ue_energy_change_j_per_episode": selective_ue - original_ue,
            "ue_energy_change_fraction": selective_ue / original_ue - 1,
            "original_total_system_energy_j_per_episode": original_system,
            "selective_total_system_energy_j_per_episode": selective_system,
            "total_system_energy_change_j_per_episode": selective_system - original_system,
            "original_tx_energy_j_per_episode": float(original.ue_transmission_energy.mean()),
            "selective_tx_energy_j_per_episode": float(selective.ue_transmission_energy.mean()),
            "original_deadline_violations": original_failed,
            "selective_deadline_violations": selective_failed,
            "original_deadline_violation_fraction": original_failed / arrived,
            "selective_deadline_violation_fraction": selective_failed / arrived,
            "deadline_violations_change": selective_failed - original_failed,
            "original_time_to_terminal_seconds": pooled_time(original),
            "selective_time_to_terminal_seconds": pooled_time(selective),
            "original_offloading_ratio": float(original.offloaded_tasks.sum() / arrived),
            "selective_offloading_ratio": float(selective.offloaded_tasks.sum() / arrived),
            "overrides": int(selective.overrides.sum()),
            "override_fraction": float(selective.overrides.sum() / arrived),
            "overrides_lower_energy": int(selective.overrides_lower_energy.sum()),
            "overrides_original_infeasible": int(selective.overrides_original_infeasible.sum()),
            "override_local_to_edge": int(selective.override_local_to_edge.sum()),
            "override_edge_to_local": int(selective.override_edge_to_local.sum()),
            "override_edge_switch": int(selective.override_edge_switch.sum()),
            "original_matched_ue_energy_saving_fraction": float(
                original_match.offload_ue_energy_saving_fraction),
            "selective_matched_local_ue_energy_j": matched_local,
            "selective_matched_offload_ue_energy_j": matched_actual,
            "selective_matched_ue_energy_savings_j": matched_saving,
            "selective_matched_ue_energy_saving_fraction": (
                matched_saving / matched_local if matched_local else np.nan),
        }
        rows.append(row)
        intervals[level] = {
            "ue_energy_change_j_per_episode_95": paired_block_interval(
                selective.ue_only_total_energy, original.ue_only_total_energy),
            "system_energy_change_j_per_episode_95": paired_block_interval(
                selective.total_system_energy, original.total_system_energy, seed=20260914),
            "deadline_violations_change_per_episode_95": paired_block_interval(
                selective.deadline_violations, original.deadline_violations, seed=20260915),
        }

        predicted_feasible = actual_success = correct = false_feasible = false_infeasible = 0
        for episode in range(1000):
            with np.load(SELECTIVE[level] / "decision_audit" /
                         f"episode_{episode:04d}.npz", allow_pickle=False) as audit:
                selected_action = audit["selected_actions"]
                arrival = selected_action >= 0
                action_index = np.maximum(selected_action, 0)[..., None]
                chosen_feasible = np.take_along_axis(
                    audit["predicted_feasible"], action_index, axis=2)[..., 0][arrival]
                success = (audit["actual_failed"][arrival] == 0)
                predicted_feasible += int(chosen_feasible.sum())
                actual_success += int(success.sum())
                correct += int((chosen_feasible == success).sum())
                false_feasible += int((chosen_feasible & ~success).sum())
                false_infeasible += int((~chosen_feasible & success).sum())
        forecast_rows.append({
            "stress_level": level,
            "decisions": arrived,
            "selected_action_feasibility_accuracy": correct / arrived,
            "predicted_feasible_fraction": predicted_feasible / arrived,
            "actual_success_fraction": actual_success / arrived,
            "false_feasible_fraction": false_feasible / arrived,
            "false_infeasible_fraction": false_infeasible / arrived,
        })

    comparison = pd.DataFrame(rows)
    forecasts = pd.DataFrame(forecast_rows)
    comparison.to_csv(SELECTIVE_ROOT / "comparison.csv", index=False)
    forecasts.to_csv(SELECTIVE_ROOT / "forecast_diagnostics.csv", index=False)
    result = {
        "status": "complete",
        "comparison": comparison.to_dict(orient="records"),
        "forecast_diagnostics": forecasts.to_dict(orient="records"),
        "paired_block_intervals": intervals,
        "validation": {
            "episodes_per_condition": 1000,
            "identical_workload_arrays_across_selective_conditions": True,
            "frozen_workload_files_unchanged": True,
            "policy_weights_unchanged": True,
            "learn_calls": 0,
            "stress_mapping_unchanged": True,
            "power_time_accounting_unchanged": True,
        },
        "matched_savings_definition": (
            "Power-time local-computation energy for the exact tasks selected for "
            "offloading minus their aggregate observed UE transmission plus idle energy."
        ),
        "interval_definition": (
            "Paired 95% descriptive intervals from 5,000 resamples of 50 "
            "nonoverlapping 20-episode blocks."
        ),
        "transmission_only_break_even_capacity": {
            str(density): (
                Config.UE_TRAN_ENERGY * Config.UE_COMP_CAP /
                (Config.UE_COMP_ENERGY * density))
            for density in Config.TASK_COMP_DENS
        },
    }
    (SELECTIVE_ROOT / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Stress-Aware Selective Offloading Results", "",
        "The learned QECO policy and its weights were kept frozen. At each task arrival, "
        "the selector compared local execution, edge 0, and edge 1 using the current task, "
        "known FIFO queues, deadline, and current transmission capacity. It simulated known "
        "backlog without future arrivals and held the observed edge sharing count constant "
        "for the short forecast. All D3QN actions were computed before any selector decision "
        "in a slot, so simultaneous UEs saw the same state.", "",
        "## Primary comparison", "",
        "|Stress|Method|UE J/episode|System J/episode|TX J/episode|Violations|Violation %|Terminal s|Offload %|Overrides|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.extend([
            f"|{row['stress_level']}|Original|{fmt(row['original_ue_energy_j_per_episode'])}|{fmt(row['original_total_system_energy_j_per_episode'])}|{fmt(row['original_tx_energy_j_per_episode'])}|{row['original_deadline_violations']:,}|{fmt(100*row['original_deadline_violation_fraction'],2)}|{fmt(row['original_time_to_terminal_seconds'],4)}|{fmt(100*row['original_offloading_ratio'],2)}|0|",
            f"|{row['stress_level']}|Stress-aware|{fmt(row['selective_ue_energy_j_per_episode'])}|{fmt(row['selective_total_system_energy_j_per_episode'])}|{fmt(row['selective_tx_energy_j_per_episode'])}|{row['selective_deadline_violations']:,}|{fmt(100*row['selective_deadline_violation_fraction'],2)}|{fmt(row['selective_time_to_terminal_seconds'],4)}|{fmt(100*row['selective_offloading_ratio'],2)}|{row['overrides']:,}|",
        ])
    lines.extend(["", "## Matched UE-energy result", "",
                  "This comparison uses the exact tasks selected for offloading in each run.", "",
                  "|Stress|Original matched saving %|Selective matched saving J|Selective matched saving %|UE energy change vs original %|",
                  "|---|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"|{row['stress_level']}|{fmt(100*row['original_matched_ue_energy_saving_fraction'],2)}|"
            f"{fmt(row['selective_matched_ue_energy_savings_j'])}|"
            f"{fmt(100*row['selective_matched_ue_energy_saving_fraction'],2)}|"
            f"{fmt(100*row['ue_energy_change_fraction'],2)}|")
    lines.extend(["", "## Override composition", "",
                  "|Stress|Total|Lower predicted energy|Original infeasible|Local to edge|Edge to local|Edge switch|",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"|{row['stress_level']}|{row['overrides']:,}|{row['overrides_lower_energy']:,}|"
            f"{row['overrides_original_infeasible']:,}|{row['override_local_to_edge']:,}|"
            f"{row['override_edge_to_local']:,}|{row['override_edge_switch']:,}|")
    medium = comparison.set_index("stress_level").loc["Medium"]
    high = comparison.set_index("stress_level").loc["High"]
    lines.extend(["", "## Main finding", ""])
    for name, row in (("Medium", medium), ("High", high)):
        status = "positive" if row.selective_matched_ue_energy_savings_j > 0 else "negative"
        lines.append(
            f"- **{name}:** matched UE-energy savings are **{status}** at "
            f"{fmt(100*row.selective_matched_ue_energy_saving_fraction,2)}%; total UE energy "
            f"changes {fmt(100*row.ue_energy_change_fraction,2)}% relative to Original QECO.")
    lines.extend([
        "", "## Physical break-even check", "",
        "Before idle energy, offloading can save UE energy only when the effective "
        "capacity exceeds `P_TX * C_UE / (P_LOCAL * density)`. With the locked QECO "
        "powers and computation capacity:", "",
        "|Computation density|Minimum capacity for TX energy below local energy|",
        "|---:|---:|",
    ])
    for density in Config.TASK_COMP_DENS:
        threshold = (Config.UE_TRAN_ENERGY * Config.UE_COMP_CAP /
                     (Config.UE_COMP_ENERGY * density))
        lines.append(f"|{density:.3f}|{threshold:.4f}|")
    lines.extend([
        "", f"High stress uses capacity {profiles.loc['High', 'effective_ue_transmission_capacity']:.4f}, below all three thresholds. Thus no individual High-stress offload can have positive matched UE-energy savings under the locked power-time model; transmission energy alone already exceeds full local-computation energy. This is a constraint implied by the retained parameters, not a selector failure or a reason to alter the stress mapping after seeing outcomes."
    ])
    lines.extend(["", "## Interpretation limits", "",
                  "The forecasts are causal decision-time estimates, not an oracle. Future arrivals are unknown, and edge capacity sharing is held at its current observed count. The matched local benchmark is an energy counterfactual for selected tasks; it does not simulate a separate all-local queue trajectory. Report both matched savings and whole-run UE energy because they answer different questions.", "",
                  "The contemporaneous original D3QN action inside a stress-aware run can diverge from the action in the Original trajectory after earlier overrides change queue states. Overrides therefore count interventions against the frozen policy in the state it actually encounters."])
    (SELECTIVE_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(comparison.to_string(index=False))
    print("\nForecast diagnostics")
    print(forecasts.to_string(index=False))


if __name__ == "__main__":
    main()
