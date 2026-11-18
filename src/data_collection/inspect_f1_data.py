
from pathlib import Path
import pandas as pd

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate raw dataset
CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2024_british_gp_laps.csv"
)

# Load dataset
print("Loading Formula 1 dataset...")

laps = pd.read_csv(CSV_FILE)

# 1. Dataset dimensions
print("\n--- DATASET OVERVIEW ---")
print("Rows:", laps.shape[0])
print("Columns:", laps.shape[1])

# 2. Column names
print("\n--- AVAILABLE COLUMNS ---")
for column in laps.columns:
    print(column)

# 3. Data types
print("\n--- DATA TYPES ---")
print(laps.dtypes)

# 4. Missing values
print("\n--- MISSING VALUES ---")
missing = pd.DataFrame({
    "Missing_Count": laps.isnull().sum(),
    "Missing_Percentage": (
        laps.isnull().mean() * 100
    ).round(2)
})

print(missing.to_string())

# 5. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print("Completely duplicated rows:", laps.duplicated().sum())

if {"Driver", "LapNumber"}.issubset(laps.columns):
    print(
        "Repeated driver/lap combinations:",
        laps.duplicated(
            subset=["Driver", "LapNumber"]
        ).sum()
    )

# 6. Lap time inspection
if "LapTime" in laps.columns:
    lap_times = pd.to_timedelta(
        laps["LapTime"],
        errors="coerce"
    )

    print("\n--- LAP TIME STATISTICS ---")
    print(
        "Valid lap times:",
        lap_times.notna().sum()
    )

    print(
        "Average lap time (seconds):",
        round(lap_times.dt.total_seconds().mean(), 3)
    )

# 7. Preview records
print("\n--- FIRST 10 RECORDS ---")
print(laps.head(10).to_string(index=False))

print("\nDATA INSPECTION COMPLETE!")
