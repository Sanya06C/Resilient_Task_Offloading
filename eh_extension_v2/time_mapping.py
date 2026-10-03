"""Time-scale mapping methods between 60-second UCLM trace and 0.1-second QECO slots.

Primary Physical Method:
- ZeroOrderHoldMapping: Holds 1-minute UCLM sample constant over 600 QECO slots (0.1 s each).

Sensitivity Alternatives:
- LinearInterpolationMapping: Linearly interpolates across 600 intermediate 0.1-s slots.
- ConservativeBatteryOnlyMapping: Uses max(BatteryCurrent, 0) over 600 slots.
- LegacyMacroMapping: 1 slot = 1 minute (historical comparison only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

SLOTS_PER_UCLM_MINUTE = 600  # 60 s / 0.1 s = 600


class BaseTimeMapping(ABC):
    """Abstract interface for time mapping."""

    def __init__(self, trace_currents: np.ndarray, trace_voltages: np.ndarray | None = None) -> None:
        self.currents = np.asarray(trace_currents, dtype=float)
        self.voltages = np.asarray(trace_voltages, dtype=float) if trace_voltages is not None else None
        self.n_samples = len(self.currents)

    @abstractmethod
    def get_current_ma(self, slot_idx: int) -> float:
        """Return harvest current (mA) for 0.1-s slot_idx."""
        pass

    def get_voltage_v(self, slot_idx: int) -> float:
        """Return battery voltage (V) for 0.1-s slot_idx (default 3.7 V or trace)."""
        if self.voltages is None or len(self.voltages) == 0:
            return 3.7
        m = (slot_idx // SLOTS_PER_UCLM_MINUTE) % self.n_samples
        return max(float(self.voltages[m]), 3.3)


class ZeroOrderHoldMapping(BaseTimeMapping):
    """Primary: Sample-and-hold across 600 slots of 0.1 s."""

    def get_current_ma(self, slot_idx: int) -> float:
        m = (slot_idx // SLOTS_PER_UCLM_MINUTE) % self.n_samples
        return float(self.currents[m])


class LinearInterpolationMapping(BaseTimeMapping):
    """Sensitivity: Linearly interpolate between minute m and m+1."""

    def get_current_ma(self, slot_idx: int) -> float:
        m = (slot_idx // SLOTS_PER_UCLM_MINUTE) % self.n_samples
        m_next = (m + 1) % self.n_samples
        frac = (slot_idx % SLOTS_PER_UCLM_MINUTE) / float(SLOTS_PER_UCLM_MINUTE)
        return float((1.0 - frac) * self.currents[m] + frac * self.currents[m_next])


class ConservativeBatteryOnlyMapping(BaseTimeMapping):
    """Sensitivity: Observed charging current into battery only (max(BatteryCurrent, 0))."""

    def get_current_ma(self, slot_idx: int) -> float:
        m = (slot_idx // SLOTS_PER_UCLM_MINUTE) % self.n_samples
        return float(max(self.currents[m], 0.0))


class LegacyMacroMapping(BaseTimeMapping):
    """Historical comparison only: Advances 1 sample per decision slot."""

    def get_current_ma(self, slot_idx: int) -> float:
        m = slot_idx % self.n_samples
        return float(self.currents[m])

    def get_voltage_v(self, slot_idx: int) -> float:
        if self.voltages is None or len(self.voltages) == 0:
            return 3.7
        m = slot_idx % self.n_samples
        return max(float(self.voltages[m]), 3.3)
