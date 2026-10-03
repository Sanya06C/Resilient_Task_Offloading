"""Corrected, time-aligned, physically consistent EH-MEC environment.

Key Scientific Corrections:
1. 0.1-second physical time slot integration (dt = 0.1 s).
2. Zero-Order Hold time mapping from UCLM 60-s trace (600 slots per UCLM minute).
3. Verified KCL harvest current: max(NodeCurrent + BatteryCurrent, 0.0).
4. Energy Causality: Rejects actions when battery energy is insufficient.
5. Persistent Battery State: Battery SOC and trace pointer persist continuously across task episodes.
6. Separate reporting of Per-UE and System-Wide totals in every summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from Config import Config
from MEC_Env import MEC
from accounting import EpisodeAccounting
from eh_extension_v2.battery_v2 import BatteryAccountV2, BatteryConfigV2
from eh_extension_v2.time_mapping import (
    BaseTimeMapping,
    ZeroOrderHoldMapping,
    LinearInterpolationMapping,
    ConservativeBatteryOnlyMapping,
    LegacyMacroMapping,
)
from eh_extension_v2.uclm_trace_v2 import UCLMTraceV2, load_uclm_trace_v2


class EHMECV2(MEC):
    """Corrected EH-aware MEC environment with physical 0.1-s Coulomb integration."""

    def __init__(
        self,
        num_ue: int = Config.N_UE,
        num_edge: int = Config.N_EDGE,
        num_time: int = Config.N_TIME,
        num_component: int = Config.N_COMPONENT,
        max_delay: int = Config.MAX_DELAY,
        *,
        trace: UCLMTraceV2 | None = None,
        trace_path: str | Path | None = None,
        split: Literal["train", "val", "test", "all"] = "test",
        time_mapping: BaseTimeMapping | None = None,
        battery_config: BatteryConfigV2 = BatteryConfigV2(),
        persistent_battery: bool = True,
        harvest_enabled: bool = True,
        p_local: float = Config.UE_COMP_ENERGY,
        p_tx: float = Config.UE_TRAN_ENERGY,
        p_idle: float = Config.UE_IDLE_ENERGY,
        p_edge: float = Config.EDGE_COMP_ENERGY,
        start_slot_offset: int = 0,
        stress_level: str | None = None,
        ue_tran_cap: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_ue, num_edge, num_time, num_component, max_delay)

        # Set stress capacity if provided
        if ue_tran_cap is not None:
            self.tran_cap_ue[:] = float(ue_tran_cap) * self.duration
        elif stress_level is not None:
            stress_caps = {
                "Normal": 14.0,
                "Low": 14.0,
                "Medium": 8.48769868957287,
                "High": 6.326878361484239,
            }
            if stress_level in stress_caps:
                self.tran_cap_ue[:] = stress_caps[stress_level] * self.duration


        # Power overrides for sensitivity sweeps
        self.ue_p_comp = float(p_local)
        self.ue_p_tran = float(p_tx)
        self.ue_p_idle = float(p_idle)
        self.edge_p_comp = float(p_edge)

        # Load trace if not supplied
        if trace is not None:
            self.trace = trace
        elif trace_path is not None:
            self.trace = load_uclm_trace_v2(trace_path, split=split)
        else:
            raise ValueError("Either trace or trace_path must be provided")

        # Time mapping setup
        if time_mapping is not None:
            self.time_mapping = time_mapping
        else:
            self.time_mapping = ZeroOrderHoldMapping(self.trace.source_current_ma, self.trace.battery_voltage_v)

        self.battery_config = battery_config
        self.persistent_battery = persistent_battery
        self.harvest_enabled = harvest_enabled
        self.global_slot = int(start_slot_offset)

        # 8-feature observation space setup
        self._base_n_features = self.n_features
        self.n_features = self._base_n_features + 2

        self.accounting = EpisodeAccounting("power_time_v1")
        self.batteries = [BatteryAccountV2(self.battery_config) for _ in range(self.n_ue)]
        self._episode_count = 0

    def _feature_tail(self) -> np.ndarray:
        """Return [SOC, harvest_availability] for all UEs at current slot."""
        h_ma = self.time_mapping.get_current_ma(self.global_slot) if self.harvest_enabled else 0.0
        h_norm = float(np.clip(h_ma / 500.0, 0.0, 1.0))
        socs = np.array([b.soc for b in self.batteries], dtype=float)
        return np.column_stack((socs, np.full(self.n_ue, h_norm, dtype=float)))

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        """Append the 2 EH features to active UE observations."""
        output = np.zeros((self.n_ue, self.n_features), dtype=float)
        active = np.any(observation != 0, axis=1)
        if np.any(active):
            output[active, :-2] = observation[active]
            output[active, -2:] = self._feature_tail()[active]
        return output

    def reset(self, arrive_task_size: np.ndarray, arrive_task_dens: np.ndarray):
        """Reset QECO queues and workload. Battery state persists unless persistent_battery is False."""
        self._episode_count += 1
        if not self.persistent_battery:
            self.batteries = [BatteryAccountV2(self.battery_config) for _ in range(self.n_ue)]

        # Reset base MEC queues outside observation allocation
        self.n_features = self._base_n_features
        try:
            observation, lstm = super().reset(arrive_task_size, arrive_task_dens)
        finally:
            self.n_features = self._base_n_features + 2

        return self._augment(observation), lstm

    def step(self, action):
        """Execute one 0.1-s step with energy causality and 0.1-s physical battery integration."""
        slot = self.time_count
        v = self.time_mapping.get_voltage_v(self.global_slot)
        h_ma = self.time_mapping.get_current_ma(self.global_slot) if self.harvest_enabled else 0.0

        # Enforce Energy Causality: Reject action if battery energy cannot supply it
        corrected_action = np.array(action, dtype=int)
        for ue in range(self.n_ue):
            act = corrected_action[ue]
            if act == 0:
                p_req = self.ue_p_comp
            elif act in (1, 2):
                p_req = self.ue_p_tran
            else:
                p_req = self.ue_p_idle

            if not self.batteries[ue].can_execute_action(p_req, duration_s=0.1, voltage_v=v):
                # Energy-induced task failure
                self.batteries[ue].record_energy_failure()
                corrected_action[ue] = 0  # Force local fallback / stall

        # Base MEC step
        self.n_features = self._base_n_features
        try:
            observation, lstm, done = super().step(corrected_action)
        finally:
            self.n_features = self._base_n_features + 2

        # Physical power-time attribution for slot
        local = self.accounting.local_fraction[slot]
        tx = self.accounting.tx_fraction[slot]
        idle = np.maximum(1.0 - np.maximum(local, tx), 0.0)
        watts = local * self.ue_p_comp + tx * self.ue_p_tran + idle * self.ue_p_idle

        # Integrate battery dynamics at dt = 0.1 s
        for ue, battery in enumerate(self.batteries):
            battery.step(
                slot_idx=self.global_slot,
                harvest_current_ma=h_ma,
                action_power_w=float(watts[ue]),
                duration_s=0.1,
                voltage_v=v,
            )

        self.global_slot += 1
        return self._augment(observation), lstm, done

    def eh_summary(self) -> dict:
        """Return comprehensive metrics reporting both Per-UE and System-Wide totals."""
        socs = [b.soc for b in self.batteries]
        min_socs = [b.min_soc_seen for b in self.batteries]
        h_raw = [b.cumulative_harvested_raw_mah for b in self.batteries]
        dis_draw = [b.cumulative_discharged_draw_mah for b in self.batteries]
        clipped = [b.cumulative_clipped_mah for b in self.batteries]
        unmet = [b.cumulative_unmet_mah for b in self.batteries]
        e_failures = [b.energy_induced_failures for b in self.batteries]

        return {
            "mean_final_soc": float(np.mean(socs)),
            "minimum_soc": float(np.min(min_socs)),
            "harvested_charge_mah_per_ue": float(np.mean(h_raw)),
            "harvested_charge_mah_total": float(np.sum(h_raw)),
            "discharged_charge_mah_per_ue": float(np.mean(dis_draw)),
            "discharged_charge_mah_total": float(np.sum(dis_draw)),
            "clipped_charge_mah_per_ue": float(np.mean(clipped)),
            "clipped_charge_mah_total": float(np.sum(clipped)),
            "unmet_discharge_mah_total": float(np.sum(unmet)),
            "energy_induced_failures_total": int(np.sum(e_failures)),
            "battery_depleted_ues": int(sum(soc <= 0.0001 for soc in socs)),
        }
