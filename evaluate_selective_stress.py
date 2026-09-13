"""Evaluate a decision-time energy selector around the frozen QECO policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import numpy as np
from Config import Config
from MEC_Env import MEC
from accounting import EpisodeAccounting, aggregate, append_episode
from stress_aware_selector import StressAwareSelector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_digest(policy):
    variables = policy.s.graph.get_collection("trainable_variables")
    h = hashlib.sha256()
    for variable, value in zip(variables, policy.sess.run(variables)):
        h.update(variable.name.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def audit_arrays(decisions, env):
    shape = (env.n_time, env.n_ue)
    original = np.full(shape, -1, dtype=np.int8)
    selected = np.full(shape, -1, dtype=np.int8)
    reason = np.full(shape, -1, dtype=np.int8)
    energy = np.full((*shape, env.n_actions), np.nan)
    feasible = np.zeros((*shape, env.n_actions), dtype=bool)
    delay = np.full((*shape, env.n_actions), -1, dtype=np.int8)
    reason_codes = {
        "retain_no_feasible_action": 0,
        "retain_original_minimum": 1,
        "override_original_infeasible": 2,
        "override_lower_ue_energy": 3,
    }
    for item in decisions:
        t, u = item["time"], item["ue"]
        original[t, u] = item["original_action"]
        selected[t, u] = item["selected_action"]
        reason[t, u] = reason_codes[item["reason"]]
        for estimate in item["estimates"]:
            a = estimate["action"]
            energy[t, u, a] = estimate["ue_energy"]
            feasible[t, u, a] = estimate["feasible"]
            delay[t, u, a] = estimate["predicted_delay_slots"]
    return original, selected, reason, energy, feasible, delay, reason_codes


def run(args):
    from D3QN import DuelingDoubleDeepQNetwork
    import tensorflow as tf

    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    if str(output).lower().startswith("c:\\windows\\system32"):
        raise ValueError("Refusing output under System32")
    workload_root = root / "accounting_baseline" / "default_legacy" / "episodes"
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    (output / "decision_audit").mkdir()
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME,
              Config.N_COMPONENT, Config.MAX_DELAY)
    env.tran_cap_ue[:] = args.transmission_capacity * env.duration
    env.accounting = EpisodeAccounting("power_time_v1")
    selector = StressAwareSelector()
    checkpoint_root = root / "TrainedModel_20UE_2EN_PerformanceMode" / "800"
    if (env.n_ue, env.n_edge) != (20, 2):
        raise ValueError("The saved checkpoint requires 20 UEs and 2 edges")

    workload_paths = [workload_root / f"episode_{i:04d}.npz"
                      for i in range(args.episodes)]
    if not all(path.exists() for path in workload_paths):
        raise FileNotFoundError("Frozen baseline workload record missing")
    workload_hashes = [digest(path) for path in workload_paths]
    checkpoint_hashes, prefixes = {}, []
    for ue in range(env.n_ue):
        prefix = checkpoint_root / f"{ue}_X_model" / "model.ckpt-800"
        for suffix in (".index", ".data-00000-of-00001"):
            path = Path(str(prefix) + suffix)
            checkpoint_hashes[str(path.relative_to(root))] = digest(path)
        prefixes.append(prefix)

    code_files = ("Config.py", "MEC_Env.py", "D3QN.py", "accounting.py",
                  "stress_aware_selector.py", "evaluate_selective_stress.py")
    metadata = {
        "status": "running",
        "stress_level": args.stress_level,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "energy_model": "power_time_v1",
        "effective_ue_transmission_capacity": args.transmission_capacity,
        "per_slot_transmission_capacity": args.transmission_capacity * env.duration,
        "policy": "PerformanceMode/800 checkpoint; epsilon=1; no learning",
        "selector": "decision-time feasible minimum predicted UE energy",
        "selector_inputs": [
            "arriving task size", "arriving computation density",
            "current local queue", "current transmission queue",
            "current edge queues", "deadline", "current effective transmission capacity"
        ],
        "forecast_scope": (
            "Known FIFO backlog is simulated without future arrivals. Edge service uses "
            "the current active-UE count as a frozen share forecast."
        ),
        "simultaneous_decisions": (
            "All original D3QN actions are obtained first; all selector decisions then "
            "use the same pre-step environment state."
        ),
        "override_rule": (
            "Retain the D3QN action if it is feasible and tied for minimum predicted "
            "UE energy; otherwise choose the feasible minimum. If none is predicted "
            "feasible, retain D3QN."
        ),
        "unchanged": [
            "D3QN checkpoints and weights", "D3QN observations and LSTM updates",
            "reward", "workload arrays", "deadlines", "power-time accounting",
            "stress profile mapping"
        ],
        "workloads": "exact episode arrays saved by default_legacy baseline",
        "workload_activity_sha256": workload_hashes,
        "checkpoint_hashes_before": checkpoint_hashes,
        "code_hashes_before": {name: digest(root / name) for name in code_files},
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "tensorflow": tf.__version__},
        "packet_loss": "not modeled",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    policies, rows = [], []
    started = time.perf_counter()
    try:
        for ue, prefix in enumerate(prefixes):
            policy = DuelingDoubleDeepQNetwork(
                env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                learning_rate=Config.LEARNING_RATE,
                reward_decay=Config.REWARD_DECAY,
                e_greedy=Config.E_GREEDY,
                replace_target_iter=Config.N_NETWORK_UPDATE,
                memory_size=Config.MEMORY_SIZE)
            policy.saver.restore(policy.sess, str(prefix))
            policy.epsilon = 1
            policies.append(policy)
        before = [policy_digest(policy) for policy in policies]
        metadata["policy_hashes_before"] = before

        for episode, workload_path in enumerate(workload_paths):
            with np.load(workload_path, allow_pickle=False) as saved:
                sizes = saved["sizes"].copy()
                densities = saved["densities"].copy()
            observation, lstm = env.reset(sizes, densities)
            selector.reset_episode()
            done = False
            while not done:
                original_actions = np.zeros(env.n_ue, dtype=int)
                for ue, policy in enumerate(policies):
                    obs = np.squeeze(observation[ue, :])
                    if np.sum(obs) != 0:
                        original_actions[ue] = policy.choose_action(obs)
                selected_actions = original_actions.copy()
                # Every UE sees the same pre-step state; one UE's selection cannot
                # leak into another UE's simultaneous decision.
                for ue in np.flatnonzero(sizes[env.time_count] > 0):
                    selected_actions[ue], _ = selector.select(
                        env, ue, original_actions[ue])
                observation, lstm, done = env.step(selected_actions)
                for ue, policy in enumerate(policies):
                    policy.update_lstm(lstm[ue, :])
                    policy.store_q_value.clear()

            row = env.accounting.summarize(env, episode)
            (original, selected, reason, predicted_energy,
             predicted_feasible, predicted_delay, reason_codes) = audit_arrays(
                selector.decisions, env)
            arrived = sizes > 0
            overridden = arrived & (original != selected)
            selected_offload = arrived & (selected > 0)
            matched_local = float(
                (Config.UE_COMP_ENERGY * sizes * densities / Config.UE_COMP_CAP)
                [selected_offload].sum())
            actual_offload_ue = row["ue_transmission_energy"] + row["ue_idle_energy"]
            row.update({
                "original_d3qn_local_tasks": int((arrived & (original == 0)).sum()),
                "original_d3qn_offloaded_tasks": int((arrived & (original > 0)).sum()),
                "overrides": int(overridden.sum()),
                "override_fraction": float(overridden.sum() / arrived.sum()),
                "overrides_lower_energy": int((reason == 3).sum()),
                "overrides_original_infeasible": int((reason == 2).sum()),
                "override_local_to_edge": int((overridden & (original == 0) & (selected > 0)).sum()),
                "override_edge_to_local": int((overridden & (original > 0) & (selected == 0)).sum()),
                "override_edge_switch": int((overridden & (original > 0) & (selected > 0)).sum()),
                "matched_local_ue_energy": matched_local,
                "selected_offload_ue_energy": actual_offload_ue,
                "matched_ue_energy_savings": matched_local - actual_offload_ue,
                "matched_ue_energy_saving_fraction": (
                    1 - actual_offload_ue / matched_local if matched_local else None),
            })
            rows.append(row)
            append_episode(output / "episode_metrics.csv", row)
            env.accounting.save_details(env, output / "episodes", episode)
            np.savez_compressed(
                output / "decision_audit" / f"episode_{episode:04d}.npz",
                original_actions=original, selected_actions=selected,
                reason_codes=reason, predicted_ue_energy=predicted_energy,
                predicted_feasible=predicted_feasible,
                predicted_delay_slots=predicted_delay,
                actual_delay_slots=env.process_delay,
                actual_failed=env.unfinish_task,
                reason_code_names=np.array(
                    [name for name, _ in sorted(reason_codes.items(), key=lambda x: x[1])]))
            if episode % 20 == 0 or episode == args.episodes - 1:
                print(
                    f"{args.stress_level} {episode + 1}/{args.episodes}: "
                    f"failures={row['deadline_violations']} "
                    f"ue_energy={row['ue_only_total_energy']:.6f} "
                    f"overrides={row['overrides']} "
                    f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

        after = [policy_digest(policy) for policy in policies]
        if before != after or any(policy.learn_step_counter != 0 for policy in policies):
            raise AssertionError("Frozen policy weights changed")
        if checkpoint_hashes != {
                name: digest(root / name) for name in checkpoint_hashes}:
            raise AssertionError("Checkpoint file changed")
        if workload_hashes != [digest(path) for path in workload_paths]:
            raise AssertionError("Frozen workload file changed")
        metadata.update(
            status="complete", episodes_completed=len(rows),
            elapsed_seconds=time.perf_counter() - started,
            policy_hashes_after=after, trainable_weights_unchanged=True,
            learn_calls=0, workload_hashes_unchanged=True,
            checkpoint_hashes_unchanged=True)
        (output / "summary.json").write_text(
            json.dumps(aggregate(rows), indent=2), encoding="utf-8")
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc),
                        episodes_completed=len(rows))
        raise
    finally:
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        for policy in policies:
            policy.sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-level", required=True,
                        choices=("Normal", "Low", "Medium", "High"))
    parser.add_argument("--transmission-capacity", required=True, type=float)
    parser.add_argument("--episodes", default=Config.N_EPISODE, type=int)
    parser.add_argument("--seed", default=20260913, type=int)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.transmission_capacity <= 0 or arguments.episodes <= 0:
        parser.error("Capacity and episode count must be positive")
    run(arguments)
