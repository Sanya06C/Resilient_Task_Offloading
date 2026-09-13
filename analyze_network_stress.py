"""Validate and summarize the matched QECO network-stress experiments."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT / "network_stress_profiles"
LEVELS = ["Normal", "Low", "Medium", "High"]
RUNS = {
    "Normal": ROOT / "accounting_baseline" / "default_power_time_v1",
    "Low": PROFILE_ROOT / "run_low",
    "Medium": PROFILE_ROOT / "run_medium",
    "High": PROFILE_ROOT / "run_high",
}
DETAILS = {
    "Normal": ROOT / "accounting_baseline" / "default_legacy" / "episodes",
    "Low": PROFILE_ROOT / "run_low" / "episodes",
    "Medium": PROFILE_ROOT / "run_medium" / "episodes",
    "High": PROFILE_ROOT / "run_high" / "episodes",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_bootstrap_ratio(actual, counterfactual, seed=20260913,
                          block_size=20, replicates=5000):
    if len(actual) % block_size:
        raise ValueError("Episode count must divide into complete blocks")
    a = actual.reshape(-1, block_size).sum(axis=1)
    c = counterfactual.reshape(-1, block_size).sum(axis=1)
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates)
    for i in range(replicates):
        take = rng.integers(0, len(a), len(a))
        ratios[i] = a[take].sum() / c[take].sum()
    return np.quantile(ratios, [0.025, 0.975]).tolist()


def main():
    profiles = pd.read_csv(PROFILE_ROOT / "profiles.csv").set_index("stress_level")
    rows = []
    workload_reference = None
    normal_csv = pd.read_csv(RUNS["Normal"] / "episode_metrics.csv")

    for level in LEVELS:
        summary = json.loads((RUNS[level] / "summary.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(RUNS[level] / "episode_metrics.csv")
        if len(metrics) != 1000 or summary["episodes"] != 1000:
            raise ValueError(f"Incomplete run: {level}")
        if level != "Normal":
            metadata = json.loads((RUNS[level] / "metadata.json").read_text(encoding="utf-8"))
            if metadata["status"] != "complete" or metadata["learn_calls"] != 0:
                raise ValueError(f"Invalid frozen-policy metadata: {level}")
            if metadata["policy_hashes_before"] != metadata["policy_hashes_after"]:
                raise ValueError(f"Policy weights changed: {level}")
            if workload_reference is None:
                workload_reference = metadata["workload_activity_sha256"]
            elif workload_reference != metadata["workload_activity_sha256"]:
                raise ValueError("Stress runs did not use identical workload files")

        offload_actual = (
            metrics.ue_transmission_energy.to_numpy() +
            metrics.ue_idle_energy.to_numpy())
        matched_local = np.zeros(1000)
        offloaded_count = np.zeros(1000)
        for episode in range(1000):
            path = DETAILS[level] / f"episode_{episode:04d}.npz"
            with np.load(path, allow_pickle=False) as saved:
                sizes = saved["sizes"]
                densities = saved["densities"]
                actions = saved["actions"]
                routed = (sizes > 0) & (actions > 0)
                # Under power-time accounting, full local execution of a task is:
                # P_local*dt*size/(UE_COMP_CAP*dt/density).
                task_local_energy = 2.0 * sizes * densities / 2.6
                matched_local[episode] = task_local_energy[routed].sum()
                offloaded_count[episode] = routed.sum()
        if not np.array_equal(offloaded_count, metrics.offloaded_tasks.to_numpy()):
            raise ValueError(f"Offload count mismatch: {level}")

        ratio = offload_actual.sum() / matched_local.sum()
        offloaded = int(summary["offloaded_tasks"]["sum"])
        row = {
            "stress_level": level,
            "effective_transmission_capacity": float(
                profiles.loc[level, "effective_ue_transmission_capacity"]),
            "capacity_factor": float(profiles.loc[level, "capacity_factor"]),
            "ue_only_energy_j_per_episode": summary["ue_only_total_energy"]["mean_per_episode"],
            "total_system_energy_j_per_episode": summary["total_system_energy"]["mean_per_episode"],
            "transmission_energy_j_per_episode": summary["ue_transmission_energy"]["mean_per_episode"],
            "latency_seconds": summary["pooled_completed_latency_seconds"],
            "time_to_terminal_seconds": float(
                (metrics.average_time_to_terminal_seconds * metrics.arrived_tasks).sum()
                / metrics.arrived_tasks.sum()),
            "deadline_violations": int(summary["deadline_violations"]["sum"]),
            "deadline_violation_fraction": summary["pooled_deadline_violation_fraction"],
            "offloaded_tasks": offloaded,
            "offloading_ratio": summary["pooled_offloaded_fraction"],
            "offloaded_completion_fraction": (
                summary["offloaded_completed_tasks"]["sum"] / offloaded),
            "actual_offload_ue_j_per_task": offload_actual.sum() / offloaded,
            "matched_local_ue_j_per_same_task": matched_local.sum() / offloaded,
            "offload_to_matched_local_energy_ratio": ratio,
            "offload_ue_energy_saving_fraction": 1 - ratio,
            "ratio_block_bootstrap_95_lower": block_bootstrap_ratio(
                offload_actual, matched_local)[0],
            "ratio_block_bootstrap_95_upper": block_bootstrap_ratio(
                offload_actual, matched_local)[1],
        }
        rows.append(row)

        if level == "Low":
            columns = [c for c in normal_csv.columns if c not in ("episode", "energy_model")]
            for column in columns:
                if pd.api.types.is_numeric_dtype(normal_csv[column]):
                    np.testing.assert_allclose(
                        normal_csv[column], metrics[column], rtol=1e-10, atol=1e-10,
                        err_msg=f"Low/default mismatch: {column}")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(PROFILE_ROOT / "stress_results.csv", index=False)
    crossed = comparison[comparison.offload_to_matched_local_energy_ratio >= 1]
    crossover = None if crossed.empty else crossed.iloc[0].stress_level
    result = {
        "status": "complete",
        "levels": LEVELS,
        "crossover_level": crossover,
        "crossover_definition": (
            "aggregate UE transmission plus idle energy for the tasks selected for "
            "offloading divided by power-time local-computation energy for those exact "
            "task sizes and densities; crossover is first ratio >= 1"),
        "uncertainty": (
            "Descriptive paired 95% interval from 5,000 resamples of 50 "
            "nonoverlapping 20-episode blocks. It is conditional on this policy, "
            "capture-derived mapping, and fixed workload sequence."),
        "normal_low_identical_condition_verified": True,
        "stress_run_workload_hashes_identical": True,
        "heldout_used": False,
        "policy_retrained_or_updated": False,
        "packet_loss_modeled": False,
        "input_hashes": json.loads(
            (PROFILE_ROOT / "derivation.json").read_text(encoding="utf-8"))["input_hashes"],
        "output_hashes": {
            "profiles.csv": sha256(PROFILE_ROOT / "profiles.csv"),
            "stress_results.csv": sha256(PROFILE_ROOT / "stress_results.csv"),
        },
        "results": comparison.to_dict(orient="records"),
    }
    (PROFILE_ROOT / "stress_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"\nFirst UE-energy crossover: {crossover}")


if __name__ == "__main__":
    main()
