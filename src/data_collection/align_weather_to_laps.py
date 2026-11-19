
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

LAP_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_laps.csv"
WEATHER_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_weather.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "2024_british_gp_laps_weather.csv"

print("Starting weather-to-lap alignment...")

# --------------------------------------------------
# 2. Load source datasets
# --------------------------------------------------

laps = pd.read_csv(LAP_PATH)
weather = pd.read_csv(WEATHER_PATH)

print(f"Lap records: {len(laps)}")
print(f"Weather observations: {len(weather)}")

# --------------------------------------------------
# 3. Convert session timestamps
# --------------------------------------------------

laps["LapStartTime"] = pd.to_timedelta(
    laps["LapStartTime"],
    errors="coerce"
)

laps["Time"] = pd.to_timedelta(
    laps["Time"],
    errors="coerce"
)

weather["Time"] = pd.to_timedelta(
    weather["Time"],
    errors="coerce"
)

# Remove rows without usable timestamps
laps = laps.dropna(
    subset=["LapStartTime", "Time"]
).copy()

weather = weather.dropna(
    subset=["Time"]
).copy()

# --------------------------------------------------
# 4. Calculate lap midpoint
# --------------------------------------------------

laps["LapStartSeconds"] = (
    laps["LapStartTime"].dt.total_seconds()
)

laps["LapEndSeconds"] = (
    laps["Time"].dt.total_seconds()
)

laps["LapMidpointSeconds"] = (
    laps["LapStartSeconds"] +
    laps["LapEndSeconds"]
) / 2

weather["WeatherSeconds"] = (
    weather["Time"].dt.total_seconds()
)

# --------------------------------------------------
# 5. Prepare weather columns
# --------------------------------------------------

weather = weather.rename(
    columns={
        "Time": "WeatherTime",
        "WeatherSeconds": "WeatherSeconds"
    }
)

weather_columns = [
    "WeatherTime",
    "WeatherSeconds",
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindDirection",
    "WindSpeed"
]

weather = weather[weather_columns].copy()

# --------------------------------------------------
# 6. Match each lap to nearest weather observation
# --------------------------------------------------

laps = laps.sort_values("LapMidpointSeconds")
weather = weather.sort_values("WeatherSeconds")

aligned = pd.merge_asof(
    laps,
    weather,
    left_on="LapMidpointSeconds",
    right_on="WeatherSeconds",
    direction="nearest",
    tolerance=45
)

# --------------------------------------------------
# 7. Calculate alignment diagnostics
# --------------------------------------------------

aligned["WeatherTimeDifferenceSeconds"] = (
    aligned["LapMidpointSeconds"] -
    aligned["WeatherSeconds"]
).abs()

matched = aligned["WeatherSeconds"].notna().sum()
unmatched = aligned["WeatherSeconds"].isna().sum()

print("\nWEATHER ALIGNMENT SUMMARY")
print("=" * 65)

print(f"Lap records retained: {len(aligned)}")
print(f"Weather matches: {matched}")
print(f"Unmatched laps: {unmatched}")
print(
    f"Match percentage: "
    f"{matched / len(aligned) * 100:.2f}%"
)

if matched:
    print(
        "Average timestamp difference: "
        f"{aligned['WeatherTimeDifferenceSeconds'].mean():.2f} seconds"
    )

    print(
        "Maximum timestamp difference: "
        f"{aligned['WeatherTimeDifferenceSeconds'].max():.2f} seconds"
    )

# --------------------------------------------------
# 8. Inspect weather and lap quality
# --------------------------------------------------

print("\nWEATHER MATCH COUNTS BY RAINFALL")
print("=" * 65)

print(
    aligned["Rainfall"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nLAP ACCURACY COUNTS")
print("=" * 65)

print(
    aligned["IsAccurate"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nTRACK STATUS COUNTS")
print("=" * 65)

print(
    aligned["TrackStatus"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

# --------------------------------------------------
# 9. Preview aligned data
# --------------------------------------------------

preview_columns = [
    "Driver",
    "LapNumber",
    "LapTime",
    "TrackStatus",
    "IsAccurate",
    "LapMidpointSeconds",
    "AirTemp",
    "TrackTemp",
    "Rainfall",
    "WindSpeed",
    "WeatherTimeDifferenceSeconds"
]

print("\nALIGNED DATA PREVIEW")
print("=" * 100)

print(
    aligned[preview_columns]
    .head(15)
    .to_string(index=False)
)

# --------------------------------------------------
# 10. Export
# --------------------------------------------------

aligned.to_csv(OUTPUT_PATH, index=False)

print("\nWeather alignment completed.")
print(f"Output saved: {OUTPUT_PATH}")
