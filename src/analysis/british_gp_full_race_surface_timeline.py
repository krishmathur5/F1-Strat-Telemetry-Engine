
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "processed" / "multi_race"
    / "runs" / "20261124_175647"
    / "multi_race_integrated.csv"
)

df = pd.read_csv(INPUT)

data = df[
    df["event_name"].eq("British Grand Prix")
].copy()

data["is_valid_racing_lap"] = (
    data["is_valid_racing_lap"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({"true": True, "false": False})
)

for col in ["lap_time_s", "LapNumber", "TyreLife", "TrackTemp"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\n1. FULL RACE COMPOUND AND WEATHER TIMELINE")
print("=" * 110)

timeline = (
    data.groupby("LapNumber")
    .agg(
        records=("Driver", "size"),
        valid_laps=("is_valid_racing_lap", "sum"),
        rainfall_records=("Rainfall", "sum"),
        median_lap_time=("lap_time_s", "median"),
        median_track_temp=("TrackTemp", "median"),
        compounds=("Compound", lambda x: ", ".join(
            f"{k}:{v}" for k, v in
            x.value_counts().sort_index().items()
        )),
    )
)

print(timeline.round(2).to_string())

print("\n2. SLICK OBSERVATIONS BY RACE LAP")
print("=" * 110)

slick = data[
    data["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
].copy()

slick_summary = (
    slick.groupby("LapNumber")
    .agg(
        slick_records=("Driver", "size"),
        valid_slick_laps=("is_valid_racing_lap", "sum"),
        rainfall_slick_records=("Rainfall", "sum"),
        median_slick_lap_time=("lap_time_s", "median"),
    )
)

print(slick_summary.round(2).to_string())

print("\n3. COMPOUND SWITCHES BY DRIVER")
print("=" * 110)

for driver, group in data.groupby("Driver"):
    group = group.sort_values("LapNumber")

    changes = group[
        group["Compound"].ne(group["Compound"].shift())
    ]

    transitions = [
        f"lap {int(row.LapNumber)}: {row.Compound}"
        for row in changes.itertuples()
    ]

    print(f"{driver:5s}: " + " | ".join(transitions))

print("\n4. WEATHER FLAG COUNTS IN CURRENT DRY-SLICK POPULATION")
print("=" * 110)

valid_slick = slick[
    slick["is_valid_racing_lap"].eq(True)
].copy()

print(
    pd.crosstab(
        valid_slick["event_name"],
        valid_slick["Rainfall"],
        margins=True
    ).to_string()
)

print("\n5. BRITISH GP SLICK LAP COUNTS BY RAINFALL")
print("=" * 110)

print(
    valid_slick.groupby("Rainfall")
    .agg(
        observations=("lap_time_s", "size"),
        drivers=("Driver", "nunique"),
        median_lap_time=("lap_time_s", "median"),
        mean_lap_time=("lap_time_s", "mean"),
    )
    .round(3)
    .to_string()
)

print("\nFULL RACE TIMELINE COMPLETE")
