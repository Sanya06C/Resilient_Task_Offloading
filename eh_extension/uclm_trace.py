"""Strict loader for the UCLM energy-harvesting CSV trace.

The public trace records a one-minute battery-terminal current.  Its panel
field is *short-circuit* current and is intentionally retained only as a
solar-availability observation; it is never treated as harvested power.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np


_FIELDS = (
    "timestamp", "node_current_ma", "battery_current_ma", "pv_sc_current_ma",
    "wind_speed_mps", "temperature_c", "humidity_pct", "battery_voltage_v",
)


@dataclass(frozen=True)
class UCLMTrace:
    """Raw-value-preserving UCLM trace plus non-destructive data-quality flags."""

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
        """Observed timestamp gaps; final value uses the median observed gap."""
        seconds = np.diff(self.timestamp).astype("timedelta64[s]").astype(float)
        return np.append(seconds, float(np.median(seconds)))

    @property
    def battery_charge_current_ma(self) -> np.ndarray:
        """Positive current flowing *into* storage, from the documented sign."""
        return np.maximum(-self.battery_current_ma, 0.0)

    @property
    def battery_discharge_current_ma(self) -> np.ndarray:
        """Positive current supplied by the battery, from the documented sign."""
        return np.maximum(self.battery_current_ma, 0.0)

    @property
    def battery_terminal_power_w(self) -> np.ndarray:
        """Positive means battery discharge; negative means battery charging."""
        return self.battery_voltage_v * self.battery_current_ma / 1000.0

    def quality_flags(self) -> dict[str, int]:
        """Counts flags without mutating, dropping, or repairing observations."""
        gaps = self.sample_seconds[:-1]
        return {
            "non_60_second_intervals": int(np.count_nonzero(gaps != 60.0)),
            "intervals_over_120_seconds": int(np.count_nonzero(gaps > 120.0)),
            "negative_pv_sc_current": int(np.count_nonzero(self.pv_sc_current_ma < 0.0)),
            "negative_wind_speed": int(np.count_nonzero(self.wind_speed_mps < 0.0)),
            "humidity_over_100_percent": int(np.count_nonzero(self.humidity_pct > 100.0)),
        }


def _number(value: str) -> float:
    return float(value.strip().replace(",", "."))


def load_uclm_trace(path: str | Path) -> UCLMTrace:
    """Load the source CSV, rejecting malformed rows instead of imputing them."""
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
                raise ValueError(f"Row {line_number} has fewer than nine source fields.")
            try:
                values["timestamp"].append(np.datetime64(datetime.strptime(
                    f"{row[0].strip()} {row[1].strip()}", "%d/%m/%Y %H:%M:%S")))
                for field, value in zip(_FIELDS[1:], row[2:9]):
                    values[field].append(_number(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Malformed source value at row {line_number}.") from error
    if not values["timestamp"]:
        raise ValueError("Trace has no data rows.")
    return UCLMTrace(**{field: np.asarray(value) for field, value in values.items()})
