"""Evaluate one fixed QECO network condition on the frozen baseline workloads."""
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
from accounting import EpisodeAccounting, append_episode, aggregate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_digest(policy):
    variables = policy.s.graph.get_collection("trainable_variables")
    h = hashlib.sha256()
    for variable, value in zip(variables, policy.sess.run(variables)):
        h.update(variable.name.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def run(args):
    from D3QN import DuelingDoubleDeepQNetwork
    import tensorflow as tf

    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    workload_root = root / "accounting_baseline" / "default_legacy" / "episodes"
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME,
              Config.N_COMPONENT, Config.MAX_DELAY)
    env.tran_cap_ue[:] = args.transmission_capacity * env.duration
    env.accounting = EpisodeAccounting("power_time_v1")
    checkpoint_root = root / "TrainedModel_20UE_2EN_PerformanceMode" / "800"
    if (Config.N_UE, Config.N_EDGE) != (20, 2):
        raise ValueError("The saved checkpoint requires the default 20 UEs and 2 edges")

    workload_paths = [workload_root / f"episode_{i:04d}.npz"
                      for i in range(args.episodes)]
    if not all(path.exists() for path in workload_paths):
        raise FileNotFoundError("Frozen baseline workload record missing")
    workload_hashes = [digest(path) for path in workload_paths]
    checkpoint_hashes = {}
    prefixes = []
    for ue in range(env.n_ue):
        prefix = checkpoint_root / f"{ue}_X_model" / "model.ckpt-800"
        for suffix in (".index", ".data-00000-of-00001"):
            path = Path(str(prefix) + suffix)
            checkpoint_hashes[str(path.relative_to(root))] = digest(path)
        prefixes.append(prefix)

    metadata = {
        "status": "running",
        "stress_level": args.stress_level,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "energy_model": "power_time_v1",
        "effective_ue_transmission_capacity": args.transmission_capacity,
        "per_slot_transmission_capacity": args.transmission_capacity * env.duration,
        "default_ue_transmission_capacity": Config.UE_TRAN_CAP,
        "changed_environment_parameters": ["effective UE transmission capacity"],
        "unchanged_environment_parameters": {
            key: value for key, value in vars(Config).items() if key.isupper()
        },
        "ue_energy_state": list(map(float, env.ue_energy_state)),
        "policy": "PerformanceMode/800 checkpoint; epsilon=1; no learning",
        "policy_observations": "original QECO observations; policy timing bug retained",
        "workloads": "exact episode arrays saved by default_legacy baseline",
        "workload_activity_sha256": workload_hashes,
        "checkpoint_hashes_before": checkpoint_hashes,
        "code_hashes_before": {
            name: digest(root / name) for name in
            ("Config.py", "MEC_Env.py", "D3QN.py", "accounting.py",
             "evaluate_stress_level.py")
        },
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "tensorflow": tf.__version__},
        "packet_loss": "not modeled",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    policies = []
    rows = []
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
            done = False
            while not done:
                actions = np.zeros(env.n_ue)
                for ue, policy in enumerate(policies):
                    obs = np.squeeze(observation[ue, :])
                    if np.sum(obs) != 0:
                        actions[ue] = policy.choose_action(obs)
                observation, lstm, done = env.step(actions)
                for ue, policy in enumerate(policies):
                    policy.update_lstm(lstm[ue, :])
                    policy.store_q_value.clear()
            row = env.accounting.summarize(env, episode)
            rows.append(row)
            append_episode(output / "episode_metrics.csv", row)
            env.accounting.save_details(env, output / "episodes", episode)
            if episode % 20 == 0 or episode == args.episodes - 1:
                print(f"{args.stress_level} {episode + 1}/{args.episodes}: "
                      f"failures={row['deadline_violations']} "
                      f"ue_energy={row['ue_only_total_energy']:.6f} "
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
            learn_calls=0)
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
    args = parser.parse_args()
    if args.transmission_capacity <= 0 or args.episodes <= 0:
        parser.error("Capacity and episode count must be positive")
    run(args)
