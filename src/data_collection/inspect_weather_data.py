
from pathlib import Path
import fastf1

# Project directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Enable existing FastF1 cache
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

print("Loading 2024 British GP weather data...")

# Load the race session
session = fastf1.get_session(
    2024,
    "British Grand Prix",
    "R"
)

session.load(
    laps=False,
    telemetry=False,
    weather=True
)

# Retrieve weather observations
weather = session.weather_data.copy()

print("\nWEATHER DATA PREVIEW")
print("=" * 75)

print(weather.head(10).to_string(index=False))

print("\nWEATHER DATA COLUMNS")
print("=" * 75)

print(weather.columns.tolist())

print("\nDATA TYPES")
print("=" * 75)

print(weather.dtypes)

print("\nMISSING VALUES")
print("=" * 75)

print(weather.isna().sum())

print("\nWEATHER OBSERVATION COUNT")
print("=" * 75)

print(len(weather))

# Export original weather observations
output_path = (
    ROOT_DIR
    / "data"
    / "raw"
    / "2024_british_gp_weather.csv"
)

weather.to_csv(output_path, index=False)

print("\nWeather extraction completed.")
print(f"Export saved: {output_path}")
