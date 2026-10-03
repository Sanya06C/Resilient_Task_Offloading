"""Capacity-bounded Coulomb counting battery model operating at 0.1-s physical time slots.

Explicitly enforces:
1. Q_ch = I_harvest * dt / 3600 (mAh)
2. Q_dis_internal = (I_action * dt / 3600) / eta_dis (mAh)
3. I_action = (P_total / V) * 1000 (mA)
4. Energy Causality: An action cannot execute if stored usable energy is insufficient.
5. Persistent continuous horizon: Battery state carries forward across multi-episode runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BatteryConfigV2:
    """Configurable scenario parameters. Defaults are documented assumptions."""

    capacity_mah: float = 2000.0
    initial_soc: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    nominal_voltage_v: float = 3.7
    slot_duration_s: float = 0.1

    def __post_init__(self) -> None:
        if self.capacity_mah <= 0:
            raise ValueError("capacity_mah must be positive")
        if not 0 <= self.initial_soc <= 1:
            raise ValueError("initial_soc must be in [0, 1]")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("efficiencies must be in (0, 1]")
        if self.nominal_voltage_v <= 0:
            raise ValueError("nominal_voltage_v must be positive")
        if self.slot_duration_s <= 0:
            raise ValueError("slot_duration_s must be positive")


@dataclass(frozen=True)
class BatteryStepV2:
    slot_idx: int
    duration_s: float
    harvest_current_ma: float
    action_power_w: float
    voltage_v: float
    charge_raw_mah: float
    charge_effective_mah: float
    discharge_draw_mah: float
    discharge_internal_mah: float
    soc_before: float
    soc_after: float
    clipped_charge_mah: float
    unmet_discharge_mah: float


class BatteryAccountV2:
    """Physically consistent Coulomb-counting battery model."""

    def __init__(self, config: BatteryConfigV2 = BatteryConfigV2()) -> None:
        self.config = config
        self.stored_mah = config.capacity_mah * config.initial_soc
        self.min_soc_seen = self.soc

        # Cumulative lifetime metrics
        self.cumulative_harvested_raw_mah = 0.0
        self.cumulative_harvested_effective_mah = 0.0
        self.cumulative_discharged_draw_mah = 0.0
        self.cumulative_discharged_internal_mah = 0.0
        self.cumulative_clipped_mah = 0.0
        self.cumulative_unmet_mah = 0.0
        self.energy_induced_failures = 0

    @property
    def soc(self) -> float:
        return float(self.stored_mah / self.config.capacity_mah)

    def can_execute_action(self, power_w: float, duration_s: float | None = None, voltage_v: float | None = None) -> bool:
        """Energy causality check: Returns True if stored charge can supply required energy."""
        dt = self.config.slot_duration_s if duration_s is None else duration_s
        v = self.config.nominal_voltage_v if voltage_v is None else voltage_v
        i_action = (power_w / v) * 1000.0
        q_draw = i_action * (dt / 3600.0)
        q_internal_required = q_draw / self.config.discharge_efficiency
        return self.stored_mah >= q_internal_required

    def record_energy_failure(self) -> None:
        """Record a task failure caused by battery energy exhaustion."""
        self.energy_induced_failures += 1

    def step(self, *, slot_idx: int = 0, harvest_current_ma: float, action_power_w: float,
             duration_s: float | None = None, voltage_v: float | None = None) -> BatteryStepV2:
        """Integrate one physical time slot (default 0.1 s)."""
        dt = self.config.slot_duration_s if duration_s is None else duration_s
        v = self.config.nominal_voltage_v if voltage_v is None else voltage_v
        before_soc = self.soc

        # 1. Harvest charge calculation
        charge_raw = max(harvest_current_ma, 0.0) * (dt / 3600.0)
        charge_effective = charge_raw * self.config.charge_efficiency
        after_charge = min(self.config.capacity_mah, self.stored_mah + charge_effective)
        clipped = (self.stored_mah + charge_effective) - after_charge

        # 2. Discharge calculation
        i_action = (max(action_power_w, 0.0) / v) * 1000.0
        dis_draw = i_action * (dt / 3600.0)
        dis_internal = dis_draw / self.config.discharge_efficiency

        # 3. Update battery storage
        after_discharge = max(0.0, after_charge - dis_internal)
        unmet = dis_internal - (after_charge - after_discharge)
        self.stored_mah = after_discharge

        # 4. Update stats
        self.min_soc_seen = min(self.min_soc_seen, self.soc)
        self.cumulative_harvested_raw_mah += charge_raw
        self.cumulative_harvested_effective_mah += charge_effective
        self.cumulative_discharged_draw_mah += dis_draw
        self.cumulative_discharged_internal_mah += dis_internal
        self.cumulative_clipped_mah += clipped
        self.cumulative_unmet_mah += unmet

        return BatteryStepV2(
            slot_idx=slot_idx,
            duration_s=dt,
            harvest_current_ma=harvest_current_ma,
            action_power_w=action_power_w,
            voltage_v=v,
            charge_raw_mah=charge_raw,
            charge_effective_mah=charge_effective,
            discharge_draw_mah=dis_draw,
            discharge_internal_mah=dis_internal,
            soc_before=before_soc,
            soc_after=self.soc,
            clipped_charge_mah=clipped,
            unmet_discharge_mah=unmet,
        )

    def reset_to_soc(self, soc: float) -> None:
        """Explicitly reset battery state when an independent reset experiment is required."""
        self.stored_mah = self.config.capacity_mah * np.clip(soc, 0.0, 1.0)
        self.min_soc_seen = self.soc
