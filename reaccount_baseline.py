"""Apply an explicitly selected energy convention to saved baseline activity only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from accounting import EpisodeAccounting, append_episode, aggregate


def reaccount(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if str(output).lower().startswith("c:\\windows\\system32"):
        raise ValueError("Refusing output under System32")
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or not meta["trainable_weights_unchanged"]:
        raise ValueError("Source must be a completed frozen-policy baseline")
    with (source / "episode_metrics.csv").open(encoding="utf-8", newline="") as f:
        old_rows = list(csv.DictReader(f))
    if len(old_rows) != meta["episodes_completed"]:
        raise ValueError("Source row count disagrees with metadata")
    c = meta["config"]
    env = SimpleNamespace(n_time=c["N_TIME"], n_ue=c["N_UE"], n_edge=c["N_EDGE"],
        n_cycle=1, n_component=c["N_COMPONENT"], max_delay=c["MAX_DELAY"],
        duration=c["DURATION"], ue_p_comp=c["UE_COMP_ENERGY"],
        ue_p_tran=c["UE_TRAN_ENERGY"], ue_p_idle=c["UE_IDLE_ENERGY"],
        edge_p_comp=c["EDGE_COMP_ENERGY"])
    output.mkdir(parents=True, exist_ok=False)
    hashes, rows = {}, []
    for old in old_rows:
        episode = int(old["episode"])
        path = source / "episodes" / f"episode_{episode:04d}.npz"
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            mapping = {"arrive_task_size": "sizes", "arrive_task_dens": "densities",
                "process_delay": "delay", "unfinish_task": "failed",
                "process_delay_trans": "transmission_delay",
                "ue_comp_energy": "legacy_local_energy", "ue_tran_energy": "legacy_tx_energy",
                "ue_idle_energy": "legacy_idle_energy", "edge_comp_energy": "legacy_edge_energy"}
            for attr, key in mapping.items():
                setattr(env, attr, data[key])
            env.drop_ue_count = int(old["legacy_drop_counter"])
            env.drop_trans_count = env.drop_edge_count = 0
            audit = EpisodeAccounting("legacy_unvalidated")
            audit.reset(env)
            for attr in ("actions", "local_fraction", "tx_fraction", "edge_fraction", "edge_share"):
                setattr(audit, attr, data[attr])
            audit.volumes = {kind: data[key] for kind, key in
                            (("local", "local_volume"), ("tx", "tx_volume"), ("edge", "edge_volume"))}
            audit.timing_mismatches = int(old["policy_timing_mismatch_arrivals"])
            reference = audit.summarize(env, episode)
            for key, value in reference.items():
                if key not in old or value is None or isinstance(value, str):
                    continue
                np.testing.assert_allclose(value, float(old[key]), rtol=1e-10, atol=1e-8,
                                           err_msg=f"Source accounting mismatch episode {episode}: {key}")
            audit.energy_model = "power_time_v1"
            row = audit.summarize(env, episode)
            append_episode(output / "episode_metrics.csv", row)
            rows.append(row)
    (output / "summary.json").write_text(json.dumps(aggregate(rows), indent=2), encoding="utf-8")
    result = {"status": "complete", "operation": "offline accounting; no policy/model loaded",
        "source": str(source), "episodes_completed": len(rows),
        "energy_model": "power_time_v1", "energy_convention_explicitly_selected": True,
        "config": c, "source_activity_sha256": hashes,
        "source_metadata_sha256": hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
        "source_csv_sha256": hashlib.sha256((source / "episode_metrics.csv").read_bytes()).hexdigest(),
        "accounting_source_sha256": hashlib.sha256(Path(__file__).with_name("accounting.py").read_bytes()).hexdigest(),
        "policy_runs": 0, "learning_calls": 0,
        "assumptions": "ENERGY_ACCOUNTING.md: task-attributable power-time energy; concurrent slot starts; edge power allocated by scheduler share; idle counted once per UE excluding CPU/TX activity"}
    (output / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Re-accounted {len(rows)} saved episodes. No D3QN import or policy run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--energy-model", choices=["power_time_v1"], required=True,
                        help="Explicit selection of the documented proposed accounting convention.")
    args = parser.parse_args()
    reaccount(args.source, args.output)
