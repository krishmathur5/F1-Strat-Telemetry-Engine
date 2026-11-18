
from pathlib import Path

import fastf1
import pandas as pd

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Create a local cache directory
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Enable FastF1 caching
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Define the race session
YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "R"

print("Connecting to FastF1...")
print(f"Loading {YEAR} {GRAND_PRIX} - Race")

# Retrieve the session
session = fastf1.get_session(
    YEAR,
    GRAND_PRIX,
    SESSION_TYPE
)

# Load lap data without telemetry for now
session.load(
    telemetry=False,
    weather=False,
    messages=False
)

# Extract lap data
laps = session.laps

# Display session information
print("\nSESSION INFORMATION")
print("-------------------")
print("Event:", session.event["EventName"])
print("Circuit:", session.event["Location"])
print("Date:", session.date)
print("Total laps recorded:", len(laps))

# Display driver information
print("\nDRIVERS")
print("-------")

drivers = sorted(laps["Driver"].dropna().unique())

print("Number of drivers:", len(drivers))
print("Driver codes:", drivers)

# Display lap data
print("\nLAP DATA PREVIEW")
print("----------------")

columns = [
    "Driver",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST"
]

available_columns = [
    col for col in columns if col in laps.columns
]

print(laps[available_columns].head(10).to_string(index=False))

# Export a local raw-data snapshot
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "2024_british_gp_laps.csv"

laps.to_csv(OUTPUT_FILE, index=False)

print("\nDATA COLLECTION SUCCESSFUL")
print("Saved to:", OUTPUT_FILE)
