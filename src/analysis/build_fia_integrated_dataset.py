
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

LAP_WEATHER_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_laps_weather.csv"
)

DELETED_LAPS_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_deleted_laps.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_PATH = (
    OUTPUT_DIR / "2024_british_gp_fia_integrated.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR / "2024_british_gp_lap_quality_audit.csv"
)

print("Building FIA-integrated lap dataset...")

# --------------------------------------------------
# 2. Load source datasets
# --------------------------------------------------

laps = pd.read_csv(LAP_WEATHER_PATH)
deleted = pd.read_csv(DELETED_LAPS_PATH)

print(f"Lap-weather records: {len(laps)}")
print(f"Officially deleted records: {len(deleted)}")

# --------------------------------------------------
# 3. Prepare unique matching keys
# --------------------------------------------------

laps["LapNumber"] = pd.to_numeric(
    laps["LapNumber"], errors="coerce"
)

deleted["LapNumber"] = pd.to_numeric(
    deleted["LapNumber"], errors="coerce"
)

laps = laps.dropna(
    subset=["Driver", "LapNumber"]
).copy()

deleted = deleted.dropna(
    subset=["Driver", "LapNumber"]
).copy()

laps["LapNumber"] = laps["LapNumber"].astype(int)
deleted["LapNumber"] = deleted["LapNumber"].astype(int)

keys = ["Driver", "LapNumber"]

if laps.duplicated(keys).any():
    raise ValueError(
        "Duplicate driver/lap keys found in lap-weather dataset."
    )

if deleted.duplicated(keys).any():
    raise ValueError(
        "Duplicate driver/lap keys found in deletion dataset."
    )

# --------------------------------------------------
# 4. Integrate official deletion metadata
# --------------------------------------------------

deleted_lookup = deleted[
    keys + ["Deleted", "DeletedReason"]
].copy()

# Remove the empty deletion fields from the original export
laps = laps.drop(
    columns=["Deleted", "DeletedReason"],
    errors="ignore"
)

df = laps.merge(
    deleted_lookup,
    on=keys,
    how="left",
    validate="one_to_one"
)

# Records not present in the official deletion export
# are marked as not deleted in this retrieved dataset.
df["is_deleted"] = (
    df["Deleted"].eq(True)
).fillna(False)

df["deletion_reason"] = df["DeletedReason"]

# --------------------------------------------------
# 5. Create separate data-quality flags
# --------------------------------------------------

df["is_accurate"] = (
    df["IsAccurate"]
    .astype("string")
    .str.lower()
    .map({"true": True, "false": False})
)

df["is_pit_in_lap"] = df["PitInTime"].notna()

df["is_pit_out_lap"] = df["PitOutTime"].notna()

df["is_pit_transition"] = (
    df["is_pit_in_lap"]
    | df["is_pit_out_lap"]
)

# --------------------------------------------------
# 6. Validate lap time and weather availability
# --------------------------------------------------

lap_time_seconds = pd.to_timedelta(
    df["LapTime"],
    errors="coerce"
).dt.total_seconds()

df["lap_time_s"] = lap_time_seconds

df["has_valid_lap_time"] = (
    df["lap_time_s"].notna()
    & (df["lap_time_s"] > 0)
)

df["has_weather_match"] = (
    df["WeatherSeconds"].notna()
)

# --------------------------------------------------
# 7. Determine eligibility for pace analysis
# --------------------------------------------------

df["is_valid_racing_lap"] = (
    df["is_accurate"].eq(True)
    & ~df["is_deleted"]
    & ~df["is_pit_transition"]
    & df["has_valid_lap_time"]
    & df["has_weather_match"]
)

# --------------------------------------------------
# 8. Generate exclusion reasons
# --------------------------------------------------

def get_exclusion_reasons(row):
    reasons = []

    if row["is_accurate"] is not True:
        reasons.append("FastF1 timing accuracy check failed")

    if row["is_deleted"]:
        reasons.append("Official lap-time deletion")

    if row["is_pit_transition"]:
        reasons.append("Pit-in/out lap")

    if not row["has_valid_lap_time"]:
        reasons.append("Invalid or missing lap time")

    if not row["has_weather_match"]:
        reasons.append("Weather observation unmatched")

    return "; ".join(reasons)


df["exclusion_reason"] = df.apply(
    get_exclusion_reasons,
    axis=1
)

# --------------------------------------------------
# 9. Sort records
# --------------------------------------------------

df = df.sort_values(
    ["Driver", "LapNumber"]
).reset_index(drop=True)

# --------------------------------------------------
# 10. Generate audit summary
# --------------------------------------------------

total = len(df)

print("\nFIA-INTEGRATED DATA QUALITY SUMMARY")
print("=" * 75)

print(f"Total records: {total}")
print(f"Officially deleted: {df['is_deleted'].sum()}")
print(f"Timing-inaccurate: {(df['is_accurate'] == False).sum()}")
print(f"Pit-in laps: {df['is_pit_in_lap'].sum()}")
print(f"Pit-out laps: {df['is_pit_out_lap'].sum()}")
print(f"Weather matched: {df['has_weather_match'].sum()}")
print(f"Valid racing laps: {df['is_valid_racing_lap'].sum()}")
print(f"Not eligible for pace analysis: {(~df['is_valid_racing_lap']).sum()}")

print("\nOFFICIAL DELETION REASONS")
print("=" * 75)

print(
    df.loc[df["is_deleted"], [
        "Driver",
        "LapNumber",
        "lap_time_s",
        "deletion_reason"
    ]].to_string(index=False)
)

print("\nPACE-ANALYSIS LAP COUNTS BY DRIVER")
print("=" * 75)

print(
    df[df["is_valid_racing_lap"]]
    .groupby("Driver")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

# --------------------------------------------------
# 11. Export integrated dataset and audit
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

df[~df["is_valid_racing_lap"]].to_csv(
    AUDIT_PATH,
    index=False
)

print("\nIntegration completed successfully.")
print(f"Integrated dataset: {OUTPUT_PATH}")
print(f"Quality audit: {AUDIT_PATH}")
