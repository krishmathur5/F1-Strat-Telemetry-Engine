
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "regression_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading integrated dataset...")

df = pd.read_csv(DATA_PATH)

# Recreate the A1 Strict Slick population
df["is_valid_racing_lap"] = (
    df["is_valid_racing_lap"]
    .astype(str).str.lower().eq("true")
)

df["rain_observed"] = (
    df["Rainfall"]
    .astype(str).str.lower().eq("true")
)

df["lap_time_s"] = pd.to_numeric(
    df["lap_time_s"], errors="coerce"
)

df["LapNumber"] = pd.to_numeric(
    df["LapNumber"], errors="coerce"
)

df["TrackTemp"] = pd.to_numeric(
    df["TrackTemp"], errors="coerce"
)

df["TyreLife"] = pd.to_numeric(
    df["TyreLife"], errors="coerce"
)

# Apply the original A1 population filters
df = df[
    df["is_valid_racing_lap"]
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & df["lap_time_s"].notna()
    & (df["lap_time_s"] > 0)
    & df["TyreLife"].notna()
].copy()

# Focus on laps 15–30
phase = df[
    df["LapNumber"].between(15, 30)
].copy()

# Race-phase summary
summary = (
    phase.groupby("LapNumber")
    .agg(
        driver_count=("Driver", "nunique"),
        observation_count=("Driver", "count"),
        median_lap_time=("lap_time_s", "median"),
        mean_lap_time=("lap_time_s", "mean"),
        fastest_lap=("lap_time_s", "min"),
        slowest_lap=("lap_time_s", "max"),
        rainfall_observations=("rain_observed", "sum"),
        mean_track_temp=("TrackTemp", "mean"),
        median_tyre_life=("TyreLife", "median"),
    )
    .reset_index()
)

summary["rainfall_share"] = (
    summary["rainfall_observations"]
    / summary["observation_count"]
)

print("\nRACE PHASE SUMMARY: LAPS 15–30")
print("=" * 90)

print(summary.round(3).to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "race_phase_summary_laps_15_30.csv",
    index=False
)

# Driver-level detail for the key laps
key_laps = phase[
    phase["LapNumber"].isin([18, 19, 20, 25, 26, 27, 28])
].copy()

detail_columns = [
    "Driver",
    "LapNumber",
    "Compound",
    "TyreLife",
    "lap_time_s",
    "rain_observed",
    "TrackTemp",
    "TrackStatus",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "PitInTime",
    "PitOutTime",
]

detail_columns = [
    c for c in detail_columns if c in key_laps.columns
]

key_laps = key_laps.sort_values(
    ["LapNumber", "lap_time_s"]
)

print("\nDRIVER-LEVEL DETAIL: KEY LAPS")
print("=" * 90)

print(
    key_laps[detail_columns]
    .to_string(index=False)
)

key_laps[detail_columns].to_csv(
    OUTPUT_DIR / "race_phase_driver_detail.csv",
    index=False
)

print("\nEXPORTED FILES")
print("=" * 90)

print(
    OUTPUT_DIR / "race_phase_summary_laps_15_30.csv"
)

print(
    OUTPUT_DIR / "race_phase_driver_detail.csv"
)

print("\nRace-phase investigation completed.")
