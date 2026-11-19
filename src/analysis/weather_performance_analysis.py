
from pathlib import Path

import pandas as pd
import plotly.express as px


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_summary.csv"
)

COMPOUND_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_by_compound.csv"
)

BOX_PLOT_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_lap_times.html"
)

TEMP_PLOT_PATH = (
    OUTPUT_DIR / "2024_british_gp_track_temperature.html"
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def parse_bool(series):
    """Convert common CSV boolean representations to True/False."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })
    )


# ==================================================
# 3. LOAD DATA
# ==================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Records loaded: {len(df)}")


# ==================================================
# 4. CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "Driver",
    "LapNumber",
    "LapTime",
    "lap_time_s",
    "Compound",
    "TyreLife",
    "Rainfall",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "is_deleted",
    "is_pit_transition",
    "is_accurate",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==================================================
# 5. NORMALISE DATA TYPES
# ==================================================

boolean_columns = [
    "is_deleted",
    "is_pit_transition",
    "is_accurate",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap"
]

for column in boolean_columns:
    df[column] = parse_bool(df[column])

numeric_columns = [
    "LapNumber",
    "lap_time_s",
    "TyreLife",
    "TrackTemp",
    "AirTemp",
    "Humidity"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

df["Rainfall"] = parse_bool(df["Rainfall"])

df["rain_condition"] = df["Rainfall"].map({
    True: "Rain observed",
    False: "No rain observed"
}).fillna("Unknown")

df["Compound"] = (
    df["Compound"]
    .fillna("Unknown")
    .astype(str)
    .str.upper()
)


# ==================================================
# 6. CREATE TWO ANALYSIS POPULATIONS
# ==================================================

# Strict population: use the existing conservative flag.
df["strict_eligible"] = (
    df["is_valid_racing_lap"].eq(True)
)

# FIA-clean population:
# Exclude officially deleted laps and pit transitions.
# Retain timing-inaccurate laps for comparison.
df["fia_clean_eligible"] = (
    ~df["is_deleted"].fillna(False)
    & ~df["is_pit_transition"].fillna(False)
    & df["has_valid_lap_time"].fillna(False)
    & df["has_weather_match"].fillna(False)
)

print("\nANALYSIS POPULATION COUNTS")
print("=" * 60)

print(
    "Strict population:",
    df["strict_eligible"].sum()
)

print(
    "FIA-clean population:",
    df["fia_clean_eligible"].sum()
)


# ==================================================
# 7. PREPARE ANALYSIS DATA
# ==================================================

strict = df[df["strict_eligible"]].copy()
strict["population"] = "Strict"

fia_clean = df[df["fia_clean_eligible"]].copy()
fia_clean["population"] = "FIA-clean"

analysis = pd.concat(
    [strict, fia_clean],
    ignore_index=True
)

analysis = analysis[
    analysis["rain_condition"] != "Unknown"
].copy()

if analysis.empty:
    raise ValueError(
        "No records available for weather analysis."
    )


# ==================================================
# 8. WEATHER PERFORMANCE SUMMARY
# ==================================================

summary = (
    analysis
    .groupby(
        ["population", "rain_condition"],
        observed=True
    )
    .agg(
        lap_count=("lap_time_s", "count"),
        average_lap_time_s=("lap_time_s", "mean"),
        median_lap_time_s=("lap_time_s", "median"),
        fastest_lap_time_s=("lap_time_s", "min"),
        slowest_lap_time_s=("lap_time_s", "max"),
        average_track_temp_c=("TrackTemp", "mean"),
        average_air_temp_c=("AirTemp", "mean"),
        average_humidity_pct=("Humidity", "mean")
    )
    .reset_index()
)

print("\nWEATHER PERFORMANCE SUMMARY")
print("=" * 80)

print(summary.round(3).to_string(index=False))

summary.to_csv(SUMMARY_PATH, index=False)


# ==================================================
# 9. WEATHER PERFORMANCE BY TYRE COMPOUND
# ==================================================

compound_summary = (
    analysis
    .groupby(
        ["population", "rain_condition", "Compound"],
        observed=True
    )
    .agg(
        lap_count=("lap_time_s", "count"),
        average_lap_time_s=("lap_time_s", "mean"),
        median_lap_time_s=("lap_time_s", "median"),
        average_tyre_age=("TyreLife", "mean"),
        average_track_temp_c=("TrackTemp", "mean")
    )
    .reset_index()
)

print("\nPERFORMANCE BY TYRE COMPOUND")
print("=" * 80)

print(compound_summary.round(3).to_string(index=False))

compound_summary.to_csv(
    COMPOUND_PATH,
    index=False
)


# ==================================================
# 10. LAP-TIME DISTRIBUTION PLOT
# ==================================================

fig = px.box(
    analysis,
    x="rain_condition",
    y="lap_time_s",
    color="rain_condition",
    facet_col="population",
    points="outliers",
    hover_data=[
        "Driver",
        "LapNumber",
        "Compound",
        "TyreLife",
        "TrackTemp"
    ],
    title=(
        "2024 British GP — Lap-Time Distribution "
        "by Rainfall Classification"
    ),
    labels={
        "rain_condition": "Nearest weather observation",
        "lap_time_s": "Lap time (seconds)",
        "population": "Analysis population"
    }
)

fig.update_layout(
    template="plotly_white",
    showlegend=False
)

fig.write_html(BOX_PLOT_PATH)


# ==================================================
# 11. TRACK TEMPERATURE VS LAP TIME
# ==================================================

temp_data = strict.dropna(
    subset=["TrackTemp", "lap_time_s"]
).copy()

fig_temp = px.scatter(
    temp_data,
    x="TrackTemp",
    y="lap_time_s",
    color="rain_condition",
    symbol="Compound",
    hover_data=[
        "Driver",
        "LapNumber",
        "TyreLife"
    ],
    title=(
        "2024 British GP — Track Temperature "
        "vs Lap Time (Strict Population)"
    ),
    labels={
        "TrackTemp": "Track temperature (°C)",
        "lap_time_s": "Lap time (seconds)",
        "rain_condition": "Nearest weather observation"
    }
)

fig_temp.update_layout(
    template="plotly_white"
)

fig_temp.write_html(TEMP_PLOT_PATH)


# ==================================================
# 12. FINAL REPORT
# ==================================================

print("\nOUTPUT FILES")
print("=" * 60)

print(f"Weather summary: {SUMMARY_PATH}")
print(f"Compound summary: {COMPOUND_PATH}")
print(f"Lap-time plot: {BOX_PLOT_PATH}")
print(f"Track-temperature plot: {TEMP_PLOT_PATH}")

print("\nWeather performance analysis completed.")
