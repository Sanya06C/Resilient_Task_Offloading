"""
Train EH-aware D3QN policy on the chronological Train split of UCLM.
Enforces:
- Train split: first 40 days (56,726 rows, 2018-07-31 to 2018-09-08)
- 0.1-s physical integration with ZOH time mapping
- Exact tracking of gradient update steps
- SHA-256 checkpoint hashing for auditability
"""

from __future__ import annotations

import argparse
import csv
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

import numpy as np

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Config import Config
from eh_extension_v2.eh_mec_env_v2 import EHMECV2
from eh_extension_v2.battery_v2 import BatteryConfigV2
from eh_extension_v2.uclm_trace_v2 import load_uclm_trace_v2

DEFAULT_DATASET = Path(r"C:\tmp\energy-harvesting-dataset-audit\dataset.csv")
PROFILE_CSV = ROOT / "network_stress_profiles" / "profiles.csv"


def get_capacities() -> dict[str, float]:
    with PROFILE_CSV.open(newline="", encoding="utf-8") as handle:
        return {row["stress_level"]: float(row["effective_ue_transmission_capacity"])
                for row in csv.DictReader(handle)}


def generated_workload(rng: np.random.Generator, env: EHMECV2) -> tuple[np.ndarray, np.ndarray]:
    sizes = rng.uniform(env.min_arrive_size, env.max_arrive_size, size=(env.n_time, env.n_ue))
    sizes *= rng.random((env.n_time, env.n_ue)) < env.task_arrive_prob
    sizes[-env.max_delay:, :] = 0.0
    densities = np.zeros_like(sizes)
    densities[sizes > 0] = rng.choice(Config.TASK_COMP_DENS, size=int(np.count_nonzero(sizes)))
    return sizes, densities


def compute_reward(env: EHMECV2, ue: int, action: int) -> float:
    soc = env.batteries[ue].soc
    unmet = env.batteries[ue].cumulative_unmet_mah
    queue = max(env.t_ue_comp[ue] - env.time_count + 1, env.t_ue_tran[ue] - env.time_count + 1, 0)
    return float(-2.0 * (1.0 - soc) - 20.0 * unmet - 0.15 * queue - 0.01 * action)


def choose_action(policy, observation: np.ndarray) -> int:
    action = int(policy.choose_action(observation))
    if len(policy.store_q_value) > 500:
        policy.store_q_value.clear()
    return action


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def train_seed(seed: int, train_episodes: int = 200, trace_path: Path = DEFAULT_DATASET,
               output_base: Path = ROOT / "eh_extension_v2" / "checkpoints") -> dict:
    from D3QN import DuelingDoubleDeepQNetwork
    import tensorflow as tf

    print(f"=== Starting Training for Seed {seed} ({train_episodes} episodes) ===", flush=True)
    start_time = time.time()

    # Seed all generators
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Output directory
    seed_dir = output_base / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Build environment on Train split
    caps = get_capacities()
    env = EHMECV2(
        trace_path=trace_path,
        split="train",
        stress_level="Normal",
        harvest_enabled=True,
        persistent_battery=True,
        battery_config=BatteryConfigV2(capacity_mah=2000.0, initial_soc=0.50)
    )
    env.tran_cap_ue[:] = caps["Normal"] * env.duration

    # Initialize D3QN
    policy = DuelingDoubleDeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
        n_lstm_features=env.n_lstm_state,
        n_time=env.n_time,
        learning_rate=0.002,
        reward_decay=0.9,
        e_greedy=0.98,
        replace_target_iter=200,
        memory_size=2000,
        batch_size=32,
        e_greedy_increment=0.002,
        n_lstm_step=1
    )

    gradient_steps = 0
    episodes_logged = []

    for episode in range(train_episodes):
        sizes, densities = generated_workload(rng, env)
        observation, lstm = env.reset(sizes, densities)
        done = False

        while not done:
            actions = np.zeros(env.n_ue, dtype=int)
            active = np.any(observation != 0, axis=1)
            for ue in np.flatnonzero(active):
                actions[ue] = choose_action(policy, observation[ue])

            next_observation, _, done = env.step(actions)

            for ue in np.flatnonzero(active):
                r = compute_reward(env, ue, actions[ue])
                policy.store_transition(
                    observation[ue],
                    np.zeros(env.n_lstm_state),
                    actions[ue],
                    r,
                    next_observation[ue],
                    np.zeros(env.n_lstm_state)
                )

            if getattr(policy, "memory_counter", 0) > 64:
                policy.learn()
                gradient_steps += 1

            observation = next_observation

        if (episode + 1) % 25 == 0 or episode == train_episodes - 1:
            summary = env.eh_summary()
            episodes_logged.append({
                "episode": episode + 1,
                "epsilon": float(policy.epsilon),
                "gradient_steps": gradient_steps,
                "mean_final_soc": summary["mean_final_soc"],
                "minimum_soc": summary["minimum_soc"],
                "failures": summary["energy_induced_failures_total"],
            })
            print(f"[Seed {seed}] Ep {episode+1}/{train_episodes} | Steps: {gradient_steps} | "
                  f"Eps: {policy.epsilon:.3f} | SOC: {summary['mean_final_soc']:.3f} | "
                  f"Min SOC: {summary['minimum_soc']:.3f}", flush=True)

    # Save checkpoint
    ckpt_prefix = seed_dir / "eh_d3qn"
    policy.saver.save(policy.sess, str(ckpt_prefix))
    policy.sess.close()
    tf.compat.v1.reset_default_graph()

    elapsed = time.time() - start_time
    print(f"=== Seed {seed} Finished in {elapsed:.1f}s, Total Gradient Steps: {gradient_steps} ===", flush=True)

    # Compute SHA-256 hashes of saved files
    file_hashes = {}
    for f in seed_dir.glob("eh_d3qn*"):
        file_hashes[f.name] = sha256_file(f)

    meta = {
        "seed": seed,
        "train_episodes": train_episodes,
        "total_gradient_steps": gradient_steps,
        "elapsed_seconds": elapsed,
        "train_split_rows": 56726,
        "date_range": "2018-07-31 to 2018-09-08 (40 calendar days)",
        "final_mean_soc": env.eh_summary()["mean_final_soc"],
        "final_min_soc": env.eh_summary()["minimum_soc"],
        "checkpoint_hashes": file_hashes,
        "log_milestones": episodes_logged,
    }

    with open(seed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--trace", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "eh_extension_v2" / "checkpoints")
    args = parser.parse_args()

    print(f"Training 5 seeds: {args.seeds} on chronological Train split (56,726 rows)...", flush=True)
    all_meta = {}
    for seed in args.seeds:
        meta = train_seed(seed, train_episodes=args.episodes, trace_path=args.trace, output_base=args.output)
        all_meta[f"seed_{seed}"] = meta

    summary_file = args.output / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"All {len(args.seeds)} seeds trained successfully. Summary written to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
