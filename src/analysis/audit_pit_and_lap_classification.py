
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


def parse_bool(series):
    """Convert common boolean representations into True/False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


print("Loading original FIA-integrated dataset...")

df = pd.read_csv(DATA_PATH)

# Convert lap number into numeric format
df["LapNumber"] = pd.to_numeric(
    df["LapNumber"], errors="coerce"
)

# Convert lap time
df["lap_time_s"] = pd.to_numeric(
    df["lap_time_s"], errors="coerce"
)

# Convert weather variables
for col in [
    "TrackTemp",
    "WeatherTimeDifferenceSeconds",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

# Standardise relevant boolean columns
boolean_columns = [
    "is_valid_racing_lap",
    "is_pit_in_lap",
    "is_pit_out_lap",
    "is_pit_transition",
    "is_deleted",
    "is_accurate",
    "has_valid_lap_time",
    "has_weather_match",
    "Rainfall",
]

for col in boolean_columns:
    if col in df.columns:
        df[col] = parse_bool(df[col])

# Keep all original records for laps 17–30.
# Do not apply the A1 regression filters here.
phase = df[
    df["LapNumber"].between(17, 30)
].copy()

print("\nDATASET AUDIT")
print("=" * 90)

print(f"Original dataset records: {len(df)}")
print(f"Records in laps 17–30: {len(phase)}")
print(
    "Unique drivers in laps 17–30:",
    phase["Driver"].nunique()
)

# --------------------------------------------------
# 1. LAP-BY-LAP CLASSIFICATION SUMMARY
# --------------------------------------------------

summary = (
    phase.groupby("LapNumber")
    .agg(
        total_records=("Driver", "count"),
        unique_drivers=("Driver", "nunique"),
        valid_racing_laps=("is_valid_racing_lap", "sum"),
        pit_in_laps=("is_pit_in_lap", "sum"),
        pit_out_laps=("is_pit_out_lap", "sum"),
        pit_transition_laps=("is_pit_transition", "sum"),
        deleted_laps=("is_deleted", "sum"),
        accurate_laps=("is_accurate", "sum"),
        weather_matched=("has_weather_match", "sum"),
        rainfall_records=("Rainfall", "sum"),
        median_lap_time=("lap_time_s", "median"),
        mean_track_temp=("TrackTemp", "mean"),
        mean_weather_time_difference=(
            "WeatherTimeDifferenceSeconds", "mean"
        ),
    )
    .reset_index()
)

print("\nLAP-BY-LAP CLASSIFICATION")
print("=" * 120)

print(summary.round(3).to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "pit_lap_classification_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. EXCLUSION REASON COUNTS
# --------------------------------------------------

print("\nEXCLUSION REASON COUNTS")
print("=" * 90)

if "exclusion_reason" in phase.columns:
    reasons = (
        phase["exclusion_reason"]
        .fillna("No exclusion reason")
        .value_counts()
    )

    print(reasons.to_string())

    reasons.rename_axis("exclusion_reason").reset_index(
        name="record_count"
    ).to_csv(
        OUTPUT_DIR / "pit_lap_exclusion_reason_counts.csv",
        index=False
    )

# --------------------------------------------------
# 3. PIT-RELATED RECORDS
# --------------------------------------------------

pit_mask = (
    phase["is_pit_in_lap"]
    | phase["is_pit_out_lap"]
    | phase["is_pit_transition"]
)

pit_records = phase[pit_mask].copy()

pit_columns = [
    "Driver",
    "LapNumber",
    "Compound",
    "TyreLife",
    "lap_time_s",
    "PitInTime",
    "PitOutTime",
    "is_pit_in_lap",
    "is_pit_out_lap",
    "is_pit_transition",
    "is_valid_racing_lap",
    "is_deleted",
    "exclusion_reason",
]

pit_columns = [
    c for c in pit_columns if c in pit_records.columns
]

print("\nPIT-RELATED RECORDS")
print("=" * 100)

if len(pit_records) > 0:
    print(
        pit_records[pit_columns]
        .sort_values(["LapNumber", "Driver"])
        .to_string(index=False)
    )
else:
    print("No pit-related records found in laps 17–30.")

pit_records[pit_columns].to_csv(
    OUTPUT_DIR / "pit_related_records_laps_17_30.csv",
    index=False
)

# --------------------------------------------------
# 4. ALL RECORDS WITH CLASSIFICATION FLAGS
# --------------------------------------------------

detail_columns = [
    "Driver",
    "LapNumber",
    "Compound",
    "TyreLife",
    "lap_time_s",
    "Rainfall",
    "TrackTemp",
    "TrackStatus",
    "is_valid_racing_lap",
    "is_pit_in_lap",
    "is_pit_out_lap",
    "is_pit_transition",
    "is_deleted",
    "is_accurate",
    "has_weather_match",
    "WeatherTimeDifferenceSeconds",
    "exclusion_reason",
]

detail_columns = [
    c for c in detail_columns if c in phase.columns
]

phase_detail = phase[
    detail_columns
].sort_values(["LapNumber", "Driver"])

phase_detail.to_csv(
    OUTPUT_DIR / "all_lap_classifications_17_30.csv",
    index=False
)

# --------------------------------------------------
# 5. VALID LAP COUNTS BY LAP
# --------------------------------------------------

print("\nVALID RACING LAP COUNTS")
print("=" * 90)

valid_counts = (
    phase.groupby("LapNumber")
    .agg(
        total_records=("Driver", "count"),
        valid_laps=("is_valid_racing_lap", "sum"),
    )
    .reset_index()
)

valid_counts["excluded_records"] = (
    valid_counts["total_records"]
    - valid_counts["valid_laps"]
)

print(valid_counts.to_string(index=False))

print("\nEXPORTED FILES")
print("=" * 90)

for filename in [
    "pit_lap_classification_summary.csv",
    "pit_lap_exclusion_reason_counts.csv",
    "pit_related_records_laps_17_30.csv",
    "all_lap_classifications_17_30.csv",
]:
    print(OUTPUT_DIR / filename)

print("\nPit and lap classification audit completed.")
