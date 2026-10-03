"""Strict, physically consistent UCLM trace loader with chronological splits.

Implements verified Kirchhoff Current Law (KCL) source current:
    I_source(t) = max(NodeCurrent(t) + BatteryCurrent(t), 0.0)
Where:
- BatteryCurrent > 0 is charging flow into battery.
- BatteryCurrent < 0 is discharge flow out of battery.
- PV short-circuit current is preserved only as an observation proxy, never as power.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

_FIELDS = (
    "timestamp", "node_current_ma", "battery_current_ma", "pv_sc_current_ma",
    "wind_speed_mps", "temperature_c", "humidity_pct", "battery_voltage_v",
)

# Day-aligned chronological split boundaries (66 total calendar days)
TRAIN_ROWS = 56726  # 40 days (2018-07-31 to 2018-09-08)
VAL_ROWS = 18686    # 13 days (2018-09-09 to 2018-09-21)
TEST_ROWS = 18026   # 13 days (2018-09-22 to 2018-10-04)

# Frozen 1-hour rolling tercile thresholds derived strictly from Train split
TERCILE_LOW_THRESHOLD_MA = 0.01
TERCILE_HIGH_THRESHOLD_MA = 14.45


@dataclass(frozen=True)
class UCLMTraceV2:
    """Raw-value preserving UCLM trace with KCL source current and splits."""

    timestamp: np.ndarray
    node_current_ma: np.ndarray
    battery_current_ma: np.ndarray
    pv_sc_current_ma: np.ndarray
    wind_speed_mps: np.ndarray
    temperature_c: np.ndarray
    humidity_pct: np.ndarray
    battery_voltage_v: np.ndarray

    @property
    def sample_seconds(self) -> np.ndarray:
        seconds = np.diff(self.timestamp).astype("timedelta64[s]").astype(float)
        return np.append(seconds, float(np.median(seconds)))

    @property
    def source_current_ma(self) -> np.ndarray:
        """KCL source current entering the common node bus: max(Node + Battery, 0)."""
        return np.maximum(self.node_current_ma + self.battery_current_ma, 0.0)

    @property
    def battery_charge_current_ma(self) -> np.ndarray:
        """Current flowing into battery storage."""
        return np.maximum(self.battery_current_ma, 0.0)

    @property
    def battery_discharge_current_ma(self) -> np.ndarray:
        """Current flowing out of battery storage."""
        return np.maximum(-self.battery_current_ma, 0.0)

    def slice_split(self, split: Literal["train", "val", "test", "all"]) -> UCLMTraceV2:
        if split == "train":
            s = slice(0, TRAIN_ROWS)
        elif split == "val":
            s = slice(TRAIN_ROWS, TRAIN_ROWS + VAL_ROWS)
        elif split == "test":
            s = slice(TRAIN_ROWS + VAL_ROWS, len(self.timestamp))
        elif split == "all":
            s = slice(0, len(self.timestamp))
        else:
            raise ValueError(f"Unknown split: {split}")

        return UCLMTraceV2(
            timestamp=self.timestamp[s],
            node_current_ma=self.node_current_ma[s],
            battery_current_ma=self.battery_current_ma[s],
            pv_sc_current_ma=self.pv_sc_current_ma[s],
            wind_speed_mps=self.wind_speed_mps[s],
            temperature_c=self.temperature_c[s],
            humidity_pct=self.humidity_pct[s],
            battery_voltage_v=self.battery_voltage_v[s],
        )

    def classify_window_solar_tercile(self, start_idx: int, length_minutes: int = 60) -> str:
        """Classify a window into Low, Medium, or High solar using frozen Train thresholds."""
        end_idx = min(start_idx + length_minutes, len(self.timestamp))
        mean_current = float(np.mean(self.source_current_ma[start_idx:end_idx]))
        if mean_current <= TERCILE_LOW_THRESHOLD_MA:
            return "Low"
        elif mean_current <= TERCILE_HIGH_THRESHOLD_MA:
            return "Medium"
        else:
            return "High"


def _number(value: str) -> float:
    return float(value.strip().replace(",", "."))


def load_uclm_trace_v2(path: str | Path, split: Literal["train", "val", "test", "all"] = "all") -> UCLMTraceV2:
    path = Path(path)
    values = {field: [] for field in _FIELDS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if header is None or header[:9] != [
            "Date", "Hour", "Node ", "Battery Current", "PV sc Current",
            "Wind Speed", "Temperature", "Humidity", "Battery Voltage",
        ]:
            raise ValueError("Unexpected UCLM schema; source columns must be unchanged.")
        for line_number, row in enumerate(reader, start=2):
            if len(row) < 9:
                raise ValueError(f"Row {line_number} has fewer than nine fields.")
            try:
                values["timestamp"].append(np.datetime64(datetime.strptime(
                    f"{row[0].strip()} {row[1].strip()}", "%d/%m/%Y %H:%M:%S")))
                for field, value in zip(_FIELDS[1:], row[2:9]):
                    values[field].append(_number(value))
            except (TypeError, ValueError) as err:
                raise ValueError(f"Malformed source value at line {line_number}") from err

    full_trace = UCLMTraceV2(**{k: np.array(v) for k, v in values.items()})
    return full_trace.slice_split(split)
