
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed" /
    "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" /
    "regression_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "raw_weather_timeline_laps_42_45.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Integrated dataset records: {len(df)}")


# ============================================================
# 2. NORMALISE WEATHER FIELDS
# ============================================================

def to_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


df["Rainfall"] = to_bool(df["Rainfall"])

df["WeatherSeconds"] = pd.to_numeric(
    df["WeatherSeconds"],
    errors="coerce"
)

weather_columns = [
    "WeatherSeconds",
    "WeatherTime",
    "Rainfall",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "WindDirection",
]

available_columns = [
    col for col in weather_columns if col in df.columns
]


# ============================================================
# 3. EXTRACT UNIQUE WEATHER TIMESTAMPS
# ============================================================

weather = df[available_columns].dropna(
    subset=["WeatherSeconds"]
).copy()

# Group by timestamp to identify whether multiple integrated
# records disagree about the weather at the same timestamp.
aggregation = {}

for col in available_columns:
    if col != "WeatherSeconds":
        aggregation[col] = (
            "first" if col == "WeatherTime"
            else "nunique" if col == "Rainfall"
            else "median"
        )

timeline = (
    weather.groupby("WeatherSeconds")
    .agg(aggregation)
    .reset_index()
    .sort_values("WeatherSeconds")
)

timeline = timeline.rename(columns={
    "Rainfall": "rainfall_unique_values"
})

# A timestamp should have one rainfall classification.
timeline["rainfall_conflict"] = (
    timeline["rainfall_unique_values"] > 1
)


# ============================================================
# 4. FILTER THE RELEVANT WEATHER WINDOW
# ============================================================

# The audit identified weather timestamps roughly between
# 7566 and 7926 seconds for laps 42-45.
window = timeline[
    timeline["WeatherSeconds"].between(7440, 8040)
].copy()

window["seconds_since_previous_sample"] = (
    window["WeatherSeconds"].diff()
)


# ============================================================
# 5. EXPORT
# ============================================================

window.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\nRAW WEATHER TIMELINE")
print("=" * 120)

print(
    window.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nTIMELINE QUALITY CHECK")
print("=" * 80)

print(f"Unique weather timestamps: {len(window)}")

print(
    "Timestamps with conflicting rainfall classifications: "
    f"{int(window['rainfall_conflict'].sum())}"
)

print(
    "Rainfall classifications represented: "
    f"{sorted(window['rainfall_unique_values'].unique().tolist())}"
)

print(
    "Median interval between weather samples: "
    f"{window['seconds_since_previous_sample'].median():.2f} seconds"
)

print("\nEXPORTED FILE")
print(OUTPUT_PATH)

print("\nRaw weather timeline inspection completed.")
