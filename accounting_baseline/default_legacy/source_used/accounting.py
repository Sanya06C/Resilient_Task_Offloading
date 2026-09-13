"""Observational accounting. Never supplies observations, rewards, or actions."""
import csv
import json
from pathlib import Path
import numpy as np


class EpisodeAccounting:
    def __init__(self, energy_model="legacy_unvalidated"):
        if energy_model not in ("legacy_unvalidated", "power_time_v1"):
            raise ValueError(energy_model)
        self.energy_model = energy_model

    def reset(self, env):
        if env.n_cycle != 1 or env.n_component != 1:
            raise ValueError("Accounting currently supports the default one-cycle, whole-task model.")
        shape = (env.n_time, env.n_ue)
        self.actions = np.full(shape, -1, dtype=int)
        self.local_fraction = np.zeros(shape)
        self.tx_fraction = np.zeros(shape)
        self.edge_fraction = np.zeros((*shape, env.n_edge))
        self.edge_share = np.zeros_like(self.edge_fraction)
        self.volumes = {k: np.zeros(shape) for k in ("local", "tx", "edge")}
        self.task_active_seconds = {k: np.zeros(shape) for k in ("local", "tx", "edge")}
        self.predicted_local_finish = -np.ones(env.n_ue)
        self.predicted_tx_finish = -np.ones(env.n_ue)
        self.timing_mismatches = 0
        self.events = []

    def start_step(self, env, action):
        t = env.time_count
        arrived = env.arrive_task_size[t] > 0
        a = np.asarray(action, dtype=int)
        if np.any(np.asarray(action) != a) or np.any((a < 0) | (a >= env.n_actions)):
            raise ValueError("Invalid action")
        self.actions[t, arrived] = a[arrived]
        # Correct queue-finish bookkeeping is diagnostic only: do not feed it to D3QN.
        for u in np.flatnonzero(arrived):
            size, dens = env.arrive_task_size[t, u], env.arrive_task_dens[t, u]
            local = a[u] == 0
            self.predicted_local_finish[u] = min(
                max(self.predicted_local_finish[u] + 1, t)
                + np.ceil(size * local / (env.comp_cap_ue[u] / dens)) - 1,
                t + env.max_delay - 1)
            # For local actions, zero transfer leaves the existing TX finish unchanged.
            cap = env.tran_cap_ue[u, max(a[u] - 1, 0)]
            self.predicted_tx_finish[u] = min(
                max(self.predicted_tx_finish[u] + 1, t)
                + np.ceil(size * (not local) / cap) - 1,
                t + env.max_delay - 1)

    def service(self, env, kind, task, ue, capacity, edge=None, share=1.0):
        remaining = float(task["REMAIN"])
        capacity = float(capacity)
        if not np.isfinite(capacity) or capacity <= 0 or share <= 0:
            raise ValueError("Nonpositive/nonfinite service capacity or share")
        volume = min(remaining, capacity)
        fraction = volume / capacity
        t, arrival = env.time_count, int(task["TIME"])
        self.volumes[kind][arrival, ue] += volume
        self.task_active_seconds[kind][arrival, ue] += fraction * env.duration
        if kind == "local":
            self.local_fraction[t, ue] += fraction
        elif kind == "tx":
            self.tx_fraction[t, ue] += fraction
        else:
            self.edge_fraction[t, ue, edge] += fraction
            self.edge_share[t, ue, edge] = share
        self.events.append((t, arrival, ue, -1 if edge is None else edge,
                            kind, volume, fraction, share))

    def finish_step(self, env):
        arrived = env.arrive_task_size[env.time_count - 1] > 0
        mismatch = (env.t_ue_comp != self.predicted_local_finish) | (
                    env.t_ue_tran != self.predicted_tx_finish)
        self.timing_mismatches += int(np.count_nonzero(arrived & mismatch))

    def summarize(self, env, episode):
        arrived = env.arrive_task_size > 0
        if np.any(self.actions[arrived] < 0):
            raise ValueError("An arrived task has no recorded decision")
        delay = np.asarray(env.process_delay)
        if np.any(delay[arrived] <= 0):
            raise ValueError("Cannot summarize an episode with unresolved tasks")
        if np.any(arrived & (env.unfinish_task != 0) & (delay < env.max_delay)):
            raise ValueError("Episode ended before all task deadlines; do not count censoring as a violation")
        failed = arrived & ((env.unfinish_task != 0) | (delay > env.max_delay))
        success = arrived & ~failed
        local = arrived & (self.actions == 0)
        offload = arrived & (self.actions > 0)
        count = int(arrived.sum())
        mean = lambda x: float(np.mean(x)) if len(x) else None
        ratio = lambda n: float(n / count) if count else None
        legacy = {
            "local_ue_computation_energy": float(np.sum(env.ue_comp_energy)),
            "ue_transmission_energy": float(np.sum(env.ue_tran_energy)),
            "ue_idle_energy": float(np.sum(env.ue_idle_energy)),
            "edge_computation_energy": float(np.sum(env.edge_comp_energy)),
        }
        for name, value in legacy.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid legacy energy {name}: {value}")
        row = {"episode": int(episode), "energy_model": self.energy_model,
               "energy_units": "unvalidated_simulator_units",
               "arrived_tasks": count, "completed_tasks": int(success.sum()),
               "deadline_violations": int(failed.sum()),
               "deadline_violation_fraction": ratio(failed.sum()),
               "local_tasks": int(local.sum()), "offloaded_tasks": int(offload.sum()),
               "local_fraction": ratio(local.sum()), "offloaded_fraction": ratio(offload.sum()),
               "local_started_tasks": int(np.count_nonzero(self.volumes["local"] > 0)),
               "transmission_started_tasks": int(np.count_nonzero(self.volumes["tx"] > 0)),
               "edge_started_tasks": int(np.count_nonzero(self.volumes["edge"] > 0)),
               "local_completed_tasks": int((local & success).sum()),
               "offloaded_completed_tasks": int((offload & success).sum()),
               "average_latency_slots": mean(delay[success]),
               "average_latency_seconds": mean(delay[success] * env.duration),
               "average_time_to_terminal_slots": mean(delay[arrived]),
               "average_time_to_terminal_seconds": mean(delay[arrived] * env.duration),
               "legacy_drop_counter": int(env.drop_ue_count + env.drop_trans_count + env.drop_edge_count),
               "policy_timing_mismatch_arrivals": self.timing_mismatches}
        row.update(legacy)
        if self.energy_model == "power_time_v1":
            row.update(self.power_time_energy(env, arrived, success))
            row["energy_units"] = "joules_assuming_config_power_watts_duration_seconds"
        row["ue_only_total_energy"] = sum(row[k] for k in (
            "local_ue_computation_energy", "ue_transmission_energy", "ue_idle_energy"))
        row["total_system_energy"] = row["ue_only_total_energy"] + row["edge_computation_energy"]
        row.update({"legacy_" + k: v for k, v in legacy.items()})
        row["legacy_ue_only_total_energy"] = sum(legacy[k] for k in (
            "local_ue_computation_energy", "ue_transmission_energy", "ue_idle_energy"))
        row["legacy_total_system_energy"] = sum(legacy.values())
        for kind, volumes in self.volumes.items():
            if np.any(volumes > env.arrive_task_size + 1e-8) or np.any(volumes < 0):
                raise ValueError(f"Conservation failed: {kind}")
        np.testing.assert_allclose(self.volumes["local"][success & local], env.arrive_task_size[success & local], atol=1e-8)
        for kind in ("tx", "edge"):
            np.testing.assert_allclose(self.volumes[kind][success & offload], env.arrive_task_size[success & offload], atol=1e-8)
        for fraction in (self.local_fraction, self.tx_fraction, self.edge_fraction):
            if np.any(fraction > 1 + 1e-8):
                raise ValueError("Resource activity exceeds one slot")
        return row

    def power_time_energy(self, env, arrived, success):
        # Proposed convention: each slot's concurrent services start together.
        # A UE's idle interval is the union of post-TX edge-wait intervals,
        # excluding its CPU/radio active interval. Never charge once per task.
        waiting_end = np.zeros_like(self.local_fraction)
        for arrival, u in zip(*np.nonzero(arrived & (self.actions > 0))):
            tx_delay = env.process_delay_trans[arrival, u]
            if tx_delay <= 0:
                continue
            first = int(arrival + tx_delay)
            last = int(arrival + env.process_delay[arrival, u] - 1)
            for t in range(first, min(last + 1, env.n_time)):
                end = 1.0
                if success[arrival, u] and t == last:
                    e = self.actions[arrival, u] - 1
                    end = self.edge_fraction[t, u, e]
                waiting_end[t, u] = max(waiting_end[t, u], end)
        idle_fraction = np.maximum(
            waiting_end - np.maximum(self.local_fraction, self.tx_fraction), 0.0)
        allocated_edge_fraction = np.divide(
            self.edge_fraction, self.edge_share, out=np.zeros_like(self.edge_fraction),
            where=self.edge_share > 0)
        if np.any(allocated_edge_fraction.sum(axis=1) > 1 + 1e-8):
            raise ValueError("Edge shares exceed one server's active slot")
        return {
            "local_ue_computation_energy": float(self.local_fraction.sum() * env.duration * env.ue_p_comp),
            "ue_transmission_energy": float(self.tx_fraction.sum() * env.duration * env.ue_p_tran),
            "ue_idle_energy": float(idle_fraction.sum() * env.duration * env.ue_p_idle),
            "edge_computation_energy": float(allocated_edge_fraction.sum() * env.duration * env.edge_p_comp),
        }

    def save_details(self, env, folder, episode):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"episode_{episode:04d}.npz",
            sizes=env.arrive_task_size, densities=env.arrive_task_dens,
            actions=self.actions, delay=env.process_delay,
            failed=env.unfinish_task, transmission_delay=env.process_delay_trans,
            local_fraction=self.local_fraction, tx_fraction=self.tx_fraction,
            edge_fraction=self.edge_fraction, edge_share=self.edge_share,
            local_volume=self.volumes["local"], tx_volume=self.volumes["tx"],
            edge_volume=self.volumes["edge"],
            legacy_local_energy=env.ue_comp_energy, legacy_tx_energy=env.ue_tran_energy,
            legacy_idle_energy=env.ue_idle_energy, legacy_edge_energy=env.edge_comp_energy)


def append_episode(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate(rows):
    result = {"episodes": len(rows), "energy_model": rows[0]["energy_model"],
              "energy_units": rows[0]["energy_units"]}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "episode"]
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = {"mean_per_episode": float(np.mean(values)),
                       "sd_across_episodes": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "sum": float(np.sum(values))}
    n = sum(r["arrived_tasks"] for r in rows)
    successes = sum(r["completed_tasks"] for r in rows)
    result["pooled_deadline_violation_fraction"] = sum(r["deadline_violations"] for r in rows) / n if n else None
    result["pooled_offloaded_fraction"] = sum(r["offloaded_tasks"] for r in rows) / n if n else None
    result["pooled_completed_latency_seconds"] = sum(
        (r["average_latency_seconds"] or 0) * r["completed_tasks"] for r in rows) / successes if successes else None
    return result
