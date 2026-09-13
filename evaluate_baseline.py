"""Evaluate the repository's saved policy; never train or import main.py."""
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
    for var, value in zip(variables, policy.sess.run(variables)):
        h.update(var.name.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def generate_workload(env):
    # Same draws, ordering, and drain interval as main.train.
    sizes = np.random.uniform(env.min_arrive_size, env.max_arrive_size,
                              size=[env.n_time, env.n_ue])
    sizes *= np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < env.task_arrive_prob
    sizes[-env.max_delay:, :] = 0
    densities = np.zeros([env.n_time, env.n_ue])
    for i in range(len(sizes)):
        for j in range(len(sizes[i])):
            if sizes[i, j] != 0:
                densities[i, j] = Config.TASK_COMP_DENS[np.random.randint(0, len(Config.TASK_COMP_DENS))]
    return sizes, densities


def run(args):
    from D3QN import DuelingDoubleDeepQNetwork
    import tensorflow as tf
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    if str(output).lower().startswith("c:\\windows\\system32"):
        raise ValueError("Refusing output under System32")
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)
    env.accounting = EpisodeAccounting(args.energy_model)
    checkpoint_root = root / "TrainedModel_20UE_2EN_PerformanceMode" / "800"
    if (Config.N_UE, Config.N_EDGE) != (20, 2):
        raise ValueError("Default saved checkpoint requires 20 UEs and 2 edges.")
    checkpoint_hashes = {}
    prefixes = []
    for ue in range(env.n_ue):
        prefix = checkpoint_root / f"{ue}_X_model" / "model.ckpt-800"
        for suffix in (".index", ".data-00000-of-00001"):
            path = Path(str(prefix) + suffix)
            checkpoint_hashes[str(path.relative_to(root))] = digest(path)
        prefixes.append(prefix)
    config = {k: v for k, v in vars(Config).items() if k.isupper()}
    metadata = {
        "status": "running", "seed": args.seed, "episodes_requested": args.episodes,
        "config": config, "energy_model": args.energy_model,
        "policy": "repository PerformanceMode/800 checkpoint, greedy epsilon=1, no learning",
        "scenario": "unchanged Config defaults including mixed UE_ENERGY_STATE",
        "checkpoint_hashes_before": checkpoint_hashes,
        "code_hashes_before": {name: digest(root / name) for name in
            ("Config.py", "MEC_Env.py", "D3QN.py", "main.py", "accounting.py", "evaluate_baseline.py")},
        "runtime": {"python": sys.version, "numpy": np.__version__, "tensorflow": tf.__version__,
                    "TF_ENABLE_ONEDNN_OPTS": os.environ["TF_ENABLE_ONEDNN_OPTS"],
                    "TF_NUM_INTRAOP_THREADS": os.environ["TF_NUM_INTRAOP_THREADS"],
                    "TF_NUM_INTEROP_THREADS": os.environ["TF_NUM_INTEROP_THREADS"]},
        "lstm_history": "retained across episodes exactly as main.train",
        "policy_observations": "original; corrected queue timing remains diagnostic only",
        "latency": "completed tasks only; failures reported separately; slots times Config.DURATION",
        "task_fractions": "routing decisions among actual arrivals; completion counts separate",
        "limitations": [
            "Legacy energy arrays are not validated physical energy.",
            "Default main.py trains; this runner uses its documented saved-policy evaluation settings.",
            "PerformanceMode checkpoint is evaluated under the current mixed-energy-state Config.",
            "Observed b_edge_comp calculation uses a stale density; unchanged because it affects policy input.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    policies, rows = [], []
    started = time.perf_counter()
    try:
        for ue, prefix in enumerate(prefixes):
            policy = DuelingDoubleDeepQNetwork(
                env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                learning_rate=Config.LEARNING_RATE, reward_decay=Config.REWARD_DECAY,
                e_greedy=Config.E_GREEDY, replace_target_iter=Config.N_NETWORK_UPDATE,
                memory_size=Config.MEMORY_SIZE)
            policy.saver.restore(policy.sess, str(prefix))
            policy.epsilon = 1
            policies.append(policy)
        before = [policy_digest(p) for p in policies]
        metadata["policy_hashes_before"] = before
        for episode in range(args.episodes):
            sizes, densities = generate_workload(env)
            obs, lstm = env.reset(sizes, densities)
            done = False
            while not done:
                actions = np.zeros(env.n_ue)
                for ue, policy in enumerate(policies):
                    observation = np.squeeze(obs[ue, :])
                    if np.sum(observation) != 0:
                        actions[ue] = policy.choose_action(observation)
                obs, lstm, done = env.step(actions)
                for ue, policy in enumerate(policies):
                    policy.update_lstm(lstm[ue, :])
                    # Diagnostic storage is unused by policy; keep memory bounded.
                    policy.store_q_value.clear()
            row = env.accounting.summarize(env, episode)
            rows.append(row)
            append_episode(output / "episode_metrics.csv", row)
            env.accounting.save_details(env, output / "episodes", episode)
            if episode % 10 == 0 or episode == args.episodes - 1:
                print(f"Episode {episode + 1}/{args.episodes}: tasks={row['arrived_tasks']} "
                      f"failures={row['deadline_violations']} "
                      f"system_energy={row['total_system_energy']:.6f} "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
        after = [policy_digest(p) for p in policies]
        if before != after or any(p.learn_step_counter != 0 for p in policies):
            raise AssertionError("Frozen policy changed")
        if checkpoint_hashes != {name: digest(root / name) for name in checkpoint_hashes}:
            raise AssertionError("Saved checkpoint changed")
        metadata.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                        episodes_completed=len(rows), policy_hashes_after=after,
                        trainable_weights_unchanged=True, learn_calls=0)
        (output / "summary.json").write_text(json.dumps(aggregate(rows), indent=2), encoding="utf-8")
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc), episodes_completed=len(rows))
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for policy in policies:
            policy.sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=Config.N_EPISODE)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", required=True)
    parser.add_argument("--energy-model", choices=("legacy_unvalidated", "power_time_v1"),
                        default="legacy_unvalidated")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    run(args)
