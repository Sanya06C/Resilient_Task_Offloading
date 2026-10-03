import json
import math
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from Config import Config
from MEC_Env import MEC
from accounting import EpisodeAccounting
from stress_aware_selector import StressAwareSelector

ROOT = Path(__file__).resolve().parent
WORKLOAD_ROOT = ROOT / "accounting_baseline" / "default_legacy" / "episodes"
OUTPUT_FILE = ROOT / "publication" / "evidence" / "comparative_baselines_verified.json"

STRESS_LEVELS = {
    "Normal": 14.0,
    "Medium": 8.48769868957287,
    "High": 6.326878361484239,
}

def run_single_condition(task_tuple):
    policy_name, stress_name, capacity, n_episodes = task_tuple
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)
    env.tran_cap_ue[:] = capacity * env.duration
    env.accounting = EpisodeAccounting("power_time_v1")
    selector = StressAwareSelector() if policy_name == "Greedy-Rule" else None

    started = time.time()
    rows = []
    for ep in range(n_episodes):
        with np.load(WORKLOAD_ROOT / f"episode_{ep:04d}.npz", allow_pickle=False) as saved:
            sizes = saved["sizes"].copy()
            densities = saved["densities"].copy()

        obs, lstm = env.reset(sizes, densities)
        if selector: selector.reset_episode()
        done = False

        while not done:
            actions = np.zeros(env.n_ue, dtype=int)
            active = env.arrive_task_size[env.time_count] > 0
            for ue in np.flatnonzero(active):
                if policy_name == "Always-Local":
                    actions[ue] = 0
                elif policy_name == "Fixed-Edge-0":
                    actions[ue] = 1
                elif policy_name == "Greedy-Rule":
                    estimates = [selector.estimate_local(env, ue)]
                    estimates.extend(selector.estimate_edge(env, ue, edge) for edge in range(env.n_edge))
                    feas = [it for it in estimates if it["feasible"]]
                    if feas:
                        best_e = min(it["objective_energy"] for it in feas)
                        chosen = min((it for it in feas if it["objective_energy"] <= best_e + 1e-12), key=lambda x: x["action"])
                        actions[ue] = chosen["action"]
                    else:
                        best = min(estimates, key=lambda it: (it.get("predicted_delay_slots", 999), it["action"]))
                        actions[ue] = best["action"]
            obs, lstm, done = env.step(actions)

        row = env.accounting.summarize(env, ep)
        rows.append(row)
        if (ep + 1) % 500 == 0:
            print(f"[{policy_name} | {stress_name}] Episode {ep+1}/{n_episodes} ({time.time()-started:.1f}s)", flush=True)

    ue_energy = np.array([r["ue_only_total_energy"] for r in rows])
    sys_energy = np.array([r["total_system_energy"] for r in rows])
    tx_energy = np.array([r["ue_transmission_energy"] for r in rows])
    violations = np.array([r["deadline_violations"] for r in rows])
    violation_rates = np.array([r["deadline_violation_fraction"] for r in rows])
    t_terminal = np.array([r["average_time_to_terminal_seconds"] for r in rows])
    offloaded_frac = np.array([r["offloaded_fraction"] for r in rows])

    n = len(rows)
    t_crit = 1.962

    def stats(arr):
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        ci = float(t_crit * s / math.sqrt(n))
        return {"mean": m, "sd": s, "ci95": [m - ci, m + ci]}

    res = {
        "policy": policy_name,
        "stress_level": stress_name,
        "capacity": capacity,
        "episodes": n,
        "ue_energy": stats(ue_energy),
        "total_system_energy": stats(sys_energy),
        "transmission_energy": stats(tx_energy),
        "deadline_violations": stats(violations),
        "deadline_violation_rate": stats(violation_rates),
        "time_to_terminal_seconds": stats(t_terminal),
        "offloading_ratio": stats(offloaded_frac),
        "elapsed_seconds": time.time() - started,
        "raw_records": rows,
    }
    print(f"DONE: {policy_name} ({stress_name}) in {time.time()-started:.1f}s -> UE_E={res['ue_energy']['mean']:.3f} J, Viol={res['deadline_violations']['mean']:.1f}", flush=True)
    return (policy_name, stress_name, res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    args = parser.parse_args()
    tasks = []
    # Always-Local only needs to run once if we want, or run for each stress
    for pol in ["Always-Local", "Fixed-Edge-0", "Greedy-Rule"]:
        for stress, cap in STRESS_LEVELS.items():
            tasks.append((pol, stress, cap, args.episodes))

    print(f"Launching {len(tasks)} baseline evaluation tasks across 4 worker processes...")
    t0 = time.time()
    all_results = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for pol, stress, res in executor.map(run_single_condition, tasks):
            if pol not in all_results: all_results[pol] = {}
            all_results[pol][stress] = res

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll baselines completed in {time.time()-t0:.1f}s! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
