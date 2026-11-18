
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2024_british_gp_laps.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "2024_british_gp_laps_processed.csv"
)

# --------------------------------------------------
# 2. LOAD RAW DATA
# --------------------------------------------------

print("Loading raw Formula 1 data...")

laps = pd.read_csv(RAW_FILE)

print("Raw records:", len(laps))

# --------------------------------------------------
# 3. SELECT REQUIRED COLUMNS
# --------------------------------------------------

column_mapping = {
    "Driver": "driver_code",
    "DriverNumber": "driver_number",
    "LapNumber": "lap_number",
    "Stint": "stint_number",
    "LapTime": "lap_time",
    "Sector1Time": "sector1_time",
    "Sector2Time": "sector2_time",
    "Sector3Time": "sector3_time",
    "Compound": "tyre_compound",
    "TyreLife": "tyre_life",
    "FreshTyre": "fresh_tyre",
    "Team": "team",
    "SpeedI1": "speed_i1",
    "SpeedI2": "speed_i2",
    "SpeedFL": "speed_fl",
    "SpeedST": "speed_st",
    "Position": "position",
    "TrackStatus": "track_status",
    "IsPersonalBest": "is_personal_best",
    "IsAccurate": "is_accurate",
}

# Retain only columns available in the source
available_mapping = {
    source: destination
    for source, destination in column_mapping.items()
    if source in laps.columns
}

processed = laps[list(available_mapping.keys())].copy()

processed.rename(
    columns=available_mapping,
    inplace=True
)

# --------------------------------------------------
# 4. CONVERT LAP TIMES INTO SECONDS
# --------------------------------------------------

time_columns = [
    "lap_time",
    "sector1_time",
    "sector2_time",
    "sector3_time",
]

for column in time_columns:
    if column in processed.columns:
        processed[column] = (
            pd.to_timedelta(
                processed[column],
                errors="coerce"
            )
            .dt.total_seconds()
        )

# --------------------------------------------------
# 5. STANDARDISE DATA TYPES
# --------------------------------------------------

integer_columns = [
    "driver_number",
    "lap_number",
    "stint_number",
    "tyre_life",
    "position",
]

for column in integer_columns:
    if column in processed.columns:
        processed[column] = pd.to_numeric(
            processed[column],
            errors="coerce"
        ).astype("Int64")

speed_columns = [
    "speed_i1",
    "speed_i2",
    "speed_fl",
    "speed_st",
]

for column in speed_columns:
    if column in processed.columns:
        processed[column] = pd.to_numeric(
            processed[column],
            errors="coerce"
        )

# --------------------------------------------------
# 6. VALIDATE RECORDS
# --------------------------------------------------

print("\n--- VALIDATION ---")

print("Processed records:", len(processed))

print(
    "Missing lap numbers:",
    processed["lap_number"].isna().sum()
)

print(
    "Missing driver codes:",
    processed["driver_code"].isna().sum()
)

print(
    "Missing lap times:",
    processed["lap_time"].isna().sum()
)

print(
    "Duplicate driver/lap combinations:",
    processed.duplicated(
        subset=["driver_code", "lap_number"]
    ).sum()
)

# --------------------------------------------------
# 7. SAVE PROCESSED DATA
# --------------------------------------------------

processed.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTRANSFORMATION SUCCESSFUL")
print("Output:", OUTPUT_FILE)
print("Total columns:", len(processed.columns))
print("Total records:", len(processed))
