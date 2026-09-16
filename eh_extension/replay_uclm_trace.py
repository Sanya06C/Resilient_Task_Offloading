"""Create a reproducible battery-terminal replay summary from the public trace.

Usage:
    python -m eh_extension.replay_uclm_trace --trace C:\\path\\dataset.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .battery import BatteryAccount, BatteryConfig
from .uclm_trace import load_uclm_trace


def replay(trace_path: Path, output_path: Path, initial_soc: float) -> dict:
    trace = load_uclm_trace(trace_path)
    account = BatteryAccount(BatteryConfig(initial_soc=initial_soc))
    steps = [
        account.step_terminal_current(timestamp=timestamp, duration_s=float(seconds),
                                      battery_current_ma=float(current))
        for timestamp, seconds, current in zip(
            trace.timestamp, trace.sample_seconds, trace.battery_current_ma)
    ]
    summary = {
        "method": "UCLM observed battery-terminal-current replay only",
        "policy_coupled": False,
        "trace_path": str(trace_path.resolve()),
        "samples": len(steps),
        "initial_soc_assumption": initial_soc,
        "final_soc": account.soc,
        "total_observed_charge_mah": sum(step.observed_charge_mah for step in steps),
        "total_observed_discharge_mah": sum(step.observed_discharge_mah for step in steps),
        "battery_empty_steps": sum(step.unmet_discharge_mah > 0 for step in steps),
        "battery_full_clipping_mah": sum(step.clipped_charge_mah for step in steps),
        "quality_flags": trace.quality_flags(),
        "limitations": [
            "PV sc Current was not converted to harvested power.",
            "This replay must not be combined with QECO action energy.",
            "One-minute samples are not treated as measured 0.1-second dynamics.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eh_extension/results/uclm_replay_summary.json"))
    parser.add_argument("--initial-soc", type=float, default=0.50)
    args = parser.parse_args()
    print(json.dumps(replay(args.trace, args.output, args.initial_soc), indent=2))
