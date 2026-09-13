"""Derive source-normalized QECO stress profiles from frozen development data.

Only train_dataset.parquet and validation_dataset.parquet are read. The held-out
dataset and classifier predictions are intentionally outside this analysis.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUTS = [
    Path(r"C:\Users\DELL\Downloads\train_dataset.parquet"),
    Path(r"C:\Users\DELL\Downloads\validation_dataset.parquet"),
]
EXPECTED_HASHES = {
    "train_dataset.parquet": "1e11477c951dddb399ed8d1d35e9c16805390b3212895bede1bc8982183b5c5b",
    "validation_dataset.parquet": "1f5658adecc0b40201a7b0542a1d2e8110b2e1266656ecdd625e8aee740cb8fe",
}
FEATURES = [
    "new_flows",
    "unique_sources",
    "unique_destinations",
    "median_flow_duration_sec",
    "estimated_packet_rate_per_sec",
]
COMPARABLE_SOURCES = ["a_day1", "s_day1"]
DEFAULT_CAPACITY = 14.0
OUTPUT = ROOT / "network_stress_profiles"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benign_percentile(reference, values):
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / len(reference)


def source_balanced_median(frame, column):
    medians = frame.groupby("source_dataset", observed=True)[column].median()
    return float(medians.median())


def main():
    OUTPUT.mkdir(exist_ok=False)
    hashes = {p.name: sha256(p) for p in INPUTS}
    if hashes != EXPECTED_HASHES:
        raise ValueError(f"Frozen input hash mismatch: {hashes}")
    parts = []
    for path in INPUTS:
        part = pd.read_parquet(path)
        part["frozen_split"] = path.stem.removesuffix("_dataset")
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    required = {"source_dataset", "window_type", *FEATURES}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[FEATURES].isna().any().any() or not np.isfinite(data[FEATURES]).all().all():
        raise ValueError("Nonfinite frozen feature value")

    comparable = data[data.source_dataset.isin(COMPARABLE_SOURCES)].copy()
    if set(comparable.window_type.unique()) != {"BENIGN", "ABNORMAL"}:
        raise ValueError("Comparable sources must contain both labels")
    duration_floor = float(comparable.loc[comparable.median_flow_duration_sec > 0,
                                          "median_flow_duration_sec"].min())
    comparable["short_flow_churn"] = comparable.new_flows / np.maximum(
        comparable.median_flow_duration_sec, duration_floor)
    comparable["persistent_flow_occupancy"] = (
        comparable.new_flows * comparable.median_flow_duration_sec)

    score_weights = {
        "estimated_packet_rate_per_sec": 0.40,
        "new_flows": 0.15,
        "unique_sources": 0.10,
        "unique_destinations": 0.10,
        "short_flow_churn": 0.15,
        "persistent_flow_occupancy": 0.10,
    }
    comparable["stress_score"] = 0.0
    ratio_features = [
        "estimated_packet_rate_per_sec", "new_flows",
        "unique_sources", "unique_destinations",
    ]
    ratio_weights = np.array([0.70, 0.15, 0.075, 0.075])
    comparable["offered_load_multiplier"] = np.nan
    reference = {}
    for source in COMPARABLE_SOURCES:
        src = comparable.source_dataset == source
        benign = src & comparable.window_type.eq("BENIGN")
        ref = {}
        for feature, weight in score_weights.items():
            comparable.loc[src, "stress_score"] += weight * benign_percentile(
                comparable.loc[benign, feature], comparable.loc[src, feature])
            ref[f"{feature}_benign_median"] = float(comparable.loc[benign, feature].median())
        ratios = []
        for feature in ratio_features:
            denominator = float(comparable.loc[benign, feature].median())
            if denominator <= 0:
                denominator = float(comparable.loc[benign & (comparable[feature] > 0), feature].median())
            ratios.append(np.maximum(comparable.loc[src, feature].to_numpy() / denominator, 1.0))
        ratios = np.column_stack(ratios)
        comparable.loc[src, "offered_load_multiplier"] = np.exp(
            np.log(ratios) @ ratio_weights)
        reference[source] = ref

    abnormal = comparable[comparable.window_type.eq("ABNORMAL")].copy()
    # Split within source so each level represents both traffic families instead
    # of allowing the larger aggressive capture to define pooled cut points.
    abnormal["stress_level"] = ""
    source_thresholds = {}
    for source in COMPARABLE_SOURCES:
        src = abnormal.source_dataset.eq(source)
        q1, q2 = abnormal.loc[src, "stress_score"].quantile([1 / 3, 2 / 3])
        source_thresholds[source] = {"q33": float(q1), "q67": float(q2)}
        abnormal.loc[src, "stress_level"] = pd.cut(
            abnormal.loc[src, "stress_score"],
            [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"],
            include_lowest=True).astype(str)

    normal = comparable[comparable.window_type.eq("BENIGN")].copy()
    normal["stress_level"] = "Normal"
    profiled = pd.concat([normal, abnormal], ignore_index=True)
    rows = []
    for level in ["Normal", "Low", "Medium", "High"]:
        group = profiled[profiled.stress_level.eq(level)]
        row = {
            "stress_level": level,
            "n_windows": int(len(group)),
            "source_balanced_stress_score": source_balanced_median(group, "stress_score"),
            "source_balanced_offered_load_multiplier": (
                1.0 if level == "Normal" else
                source_balanced_median(group, "offered_load_multiplier")),
        }
        for feature in FEATURES + ["short_flow_churn", "persistent_flow_occupancy"]:
            row[f"source_balanced_median_{feature}"] = source_balanced_median(group, feature)
        rows.append(row)
    profiles = pd.DataFrame(rows)
    # Packet-derived load cannot be converted directly to radio occupancy because
    # packet sizes/link utilization are absent. Square-root compression is an
    # explicit conservative sensitivity mapping; the 25% floor avoids asserting
    # unsupported link collapse.
    multipliers = np.maximum.accumulate(
        profiles.source_balanced_offered_load_multiplier.to_numpy())
    profiles["monotone_offered_load_multiplier"] = multipliers
    profiles["capacity_factor"] = np.maximum(1 / np.sqrt(multipliers), 0.25)
    profiles.loc[profiles.stress_level.eq("Normal"), "capacity_factor"] = 1.0
    profiles["effective_ue_transmission_capacity"] = DEFAULT_CAPACITY * profiles.capacity_factor
    if not np.all(np.diff(profiles.capacity_factor) <= 1e-12):
        raise AssertionError("Capacity factors must decrease monotonically")
    profiles.to_csv(OUTPUT / "profiles.csv", index=False)
    abnormal[["source_dataset", "frozen_split", "stress_level", "stress_score",
              "offered_load_multiplier", *FEATURES]].to_csv(
                  OUTPUT / "abnormal_window_assignments.csv", index=False)
    provenance = {
        "input_hashes": hashes,
        "heldout_used": False,
        "rows_total": int(len(data)),
        "label_counts": data.window_type.value_counts().to_dict(),
        "source_label_counts": {
            f"{source}:{label}": int(count)
            for (source, label), count in data.groupby(["source_dataset", "window_type"]).size().items()
        },
        "comparable_sources": COMPARABLE_SOURCES,
        "reason_for_comparable_sources": (
            "Only a_day1 and s_day1 contain both BENIGN and ABNORMAL windows. "
            "Source-matched BENIGN references avoid treating capture identity as severity."),
        "duration_floor_seconds": duration_floor,
        "score_weights": score_weights,
        "offered_load_features": dict(zip(ratio_features, ratio_weights.tolist())),
        "source_abnormal_tercile_thresholds": source_thresholds,
        "capacity_mapping": (
            "C_eff = 14 / sqrt(M), bounded below at 0.25*14; M is the monotone "
            "source-balanced geometric load multiplier. Square-root compression is "
            "a conservative sensitivity assumption because packet sizes and physical "
            "link utilization are unavailable."),
        "packet_loss": "not modeled or invented",
    }
    (OUTPUT / "derivation.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(profiles.to_string(index=False))


if __name__ == "__main__":
    main()
