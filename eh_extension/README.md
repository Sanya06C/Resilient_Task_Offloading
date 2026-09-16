# Energy-harvesting extension boundary

This package is separate from the frozen QECO environment, D3QN checkpoints,
network-stress profiles, selector, and reported results. It does not retrain or
call the learned policy.

`uclm_trace.py` strictly reads the UCLM public trace. It preserves raw values
and reports quality flags. `battery.py` implements capacity-bounded coulomb
counting using the paper's documented sign convention: positive battery current
discharges the battery and negative current charges it.

Run a trace-only replay:

```powershell
python -m eh_extension.replay_uclm_trace `
  --trace "C:\path\to\dataset.csv" `
  --output "eh_extension\results\uclm_replay_summary.json"
```

## Scientific boundary

The UCLM `PV sc Current` is measured under panel short circuit. It is not panel
operating current and cannot be converted into harvested electrical power.
The observed battery current is already the net effect of the original UCLM
node's load and charge path. Combining it with QECO's local/TX energy would
double count load and fabricate a harvester model.

Accordingly, `BatteryAccount.step_terminal_current()` is valid for trace replay
only. `step_external_balance()` is reserved for a future policy-coupled study
after supplying an independently calibrated harvester charge-current profile,
target-hardware action currents, SOC initialization, and efficiencies.

The default capacity (2,000 mAh) comes from the UCLM platform description.
Default 50% initial SOC and unity efficiencies are explicitly neutral scenario
assumptions, not measured trace facts.
