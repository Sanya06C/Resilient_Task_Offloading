"""Explicit battery accounting, intentionally independent of QECO's frozen policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    """Scenario parameters.  Defaults are documented assumptions, not trace facts.

    Capacity is the 2,000 mAh nominal Li-Po capacity reported for the UCLM
    platform. Initial SOC (50%) and efficiency (100%) are neutral, configurable
    scenarios used only for trace replay; they must be replaced by hardware
    measurements before a policy-coupled evaluation is reported.
    """

    capacity_mah: float = 2000.0
    initial_soc: float = 0.50
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_mah <= 0:
            raise ValueError("capacity_mah must be positive")
        if not 0 <= self.initial_soc <= 1:
            raise ValueError("initial_soc must be in [0, 1]")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("efficiencies must be in (0, 1]")


@dataclass(frozen=True)
class BatteryStep:
    timestamp: object
    duration_s: float
    observed_charge_mah: float
    observed_discharge_mah: float
    soc_before: float
    soc_after: float
    clipped_charge_mah: float
    unmet_discharge_mah: float


class BatteryAccount:
    """Coulomb-count a measured terminal-current trace.

    This class accepts either measured net terminal current OR a separately
    supplied harvester/action-load model.  It must not combine UCLM net battery
    current with QECO action energy, because that would count the UCLM node load
    twice and attribute unmeasured solar generation to the panel-current field.
    """

    def __init__(self, config: BatteryConfig = BatteryConfig()) -> None:
        self.config = config
        self.stored_mah = config.capacity_mah * config.initial_soc

    @property
    def soc(self) -> float:
        return self.stored_mah / self.config.capacity_mah

    def step_terminal_current(self, *, timestamp: object, duration_s: float,
                              battery_current_ma: float) -> BatteryStep:
        """Apply a measured current; positive is discharge per UCLM convention."""
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        before = self.soc
        charge_mah = max(-battery_current_ma, 0.0) * duration_s / 3600.0
        discharge_mah = max(battery_current_ma, 0.0) * duration_s / 3600.0
        requested_charge = charge_mah * self.config.charge_efficiency
        requested_discharge = discharge_mah / self.config.discharge_efficiency
        after_charge = min(self.config.capacity_mah, self.stored_mah + requested_charge)
        clipped_charge = self.stored_mah + requested_charge - after_charge
        after = max(0.0, after_charge - requested_discharge)
        unmet = requested_discharge - (after_charge - after)
        self.stored_mah = after
        return BatteryStep(timestamp, duration_s, charge_mah, discharge_mah, before,
                           self.soc, clipped_charge, unmet)

    def step_external_balance(self, *, timestamp: object, duration_s: float,
                              harvest_charge_current_ma: float,
                              action_discharge_current_ma: float) -> BatteryStep:
        """Apply independently measured/modelled inputs for a future EH policy.

        Callers must provide a harvester charge current that is independent of
        the UCLM net battery-current column.
        """
        if harvest_charge_current_ma < 0 or action_discharge_current_ma < 0:
            raise ValueError("external balance currents must be non-negative")
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        before = self.soc
        charge_mah = harvest_charge_current_ma * duration_s / 3600.0
        discharge_mah = action_discharge_current_ma * duration_s / 3600.0
        requested_charge = charge_mah * self.config.charge_efficiency
        requested_discharge = discharge_mah / self.config.discharge_efficiency
        after_charge = min(self.config.capacity_mah, self.stored_mah + requested_charge)
        clipped_charge = self.stored_mah + requested_charge - after_charge
        after = max(0.0, after_charge - requested_discharge)
        unmet = requested_discharge - (after_charge - after)
        self.stored_mah = after
        return BatteryStep(timestamp, duration_s, charge_mah, discharge_mah, before,
                           self.soc, clipped_charge, unmet)
