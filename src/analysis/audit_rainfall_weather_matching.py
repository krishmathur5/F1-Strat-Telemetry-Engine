
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

LAP_OUTPUT = OUTPUT_DIR / "rainfall_weather_matching_by_lap.csv"
DETAIL_OUTPUT = OUTPUT_DIR / "rainfall_weather_matching_detail.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original records: {len(df)}")


# ============================================================
# NORMALISE TYPES
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


for col in ["is_valid_racing_lap", "Rainfall"]:
    if col in df.columns:
        df[col] = to_bool(df[col])

if "rain_observed" not in df.columns:
    df["rain_observed"] = df["Rainfall"]

df["rain_observed"] = to_bool(df["rain_observed"])

numeric_columns = [
    "LapNumber",
    "lap_time_s",
    "TyreLife",
    "LapStartSeconds",
    "LapEndSeconds",
    "LapMidpointSeconds",
    "WeatherSeconds",
    "WeatherTimeDifferenceSeconds",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Compound"] = df["Compound"].astype(str).str.upper()
df["Driver"] = df["Driver"].astype(str)


# ============================================================
# RECONSTRUCT A1 STRICT SLICK POPULATION
# ============================================================

df = df[
    df["is_valid_racing_lap"]
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & (df["lap_time_s"] > 0)
].copy()

df = df.dropna(
    subset=[
        "LapNumber",
        "lap_time_s",
        "TyreLife",
        "Driver",
        "Compound",
        "rain_observed",
    ]
).copy()

if len(df) != 658:
    raise ValueError(
        f"Expected 658 observations, found {len(df)}"
    )


# ============================================================
# SELECT RELEVANT LAP WINDOW
# ============================================================

# Inspect the initial wet transition and the later rainfall window.
df = df[
    df["LapNumber"].between(17, 27)
    | df["LapNumber"].between(39, 46)
].copy()


# ============================================================
# OBSERVATION-LEVEL WEATHER MATCHING DETAIL
# ============================================================

detail_columns = [
    "Driver",
    "LapNumber",
    "Compound",
    "rain_observed",
    "Rainfall",
    "LapStartSeconds",
    "LapEndSeconds",
    "LapMidpointSeconds",
    "WeatherSeconds",
    "WeatherTimeDifferenceSeconds",
]

available_detail_columns = [
    col for col in detail_columns if col in df.columns
]

detail = df[available_detail_columns].copy()

detail = detail.sort_values(
    ["LapNumber", "Driver"]
)

detail.to_csv(DETAIL_OUTPUT, index=False)


# ============================================================
# SUMMARY BY LAP AND RAINFALL STATUS
# ============================================================

summary_rows = []

for lap, group in df.groupby("LapNumber"):

    wet = group[group["rain_observed"]]
    dry = group[~group["rain_observed"]]

    def safe_median(series):
        return series.median() if series.notna().any() else None

    summary_rows.append({
        "LapNumber": int(lap),
        "observations": len(group),
        "wet_observations": len(wet),
        "dry_observations": len(dry),
        "unique_weather_timestamps": (
            group["WeatherSeconds"].nunique()
            if "WeatherSeconds" in group.columns else None
        ),
        "wet_weather_timestamps": (
            wet["WeatherSeconds"].nunique()
            if "WeatherSeconds" in wet.columns else None
        ),
        "dry_weather_timestamps": (
            dry["WeatherSeconds"].nunique()
            if "WeatherSeconds" in dry.columns else None
        ),
        "wet_median_time_difference_s": (
            safe_median(
                wet["WeatherTimeDifferenceSeconds"]
            )
            if "WeatherTimeDifferenceSeconds" in wet.columns
            else None
        ),
        "dry_median_time_difference_s": (
            safe_median(
                dry["WeatherTimeDifferenceSeconds"]
            )
            if "WeatherTimeDifferenceSeconds" in dry.columns
            else None
        ),
        "wet_median_weather_seconds": (
            safe_median(wet["WeatherSeconds"])
            if "WeatherSeconds" in wet.columns else None
        ),
        "dry_median_weather_seconds": (
            safe_median(dry["WeatherSeconds"])
            if "WeatherSeconds" in dry.columns else None
        ),
    })


lap_summary = pd.DataFrame(summary_rows)

lap_summary.to_csv(LAP_OUTPUT, index=False)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nWEATHER MATCHING SUMMARY BY LAP")
print("=" * 120)

print(
    lap_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nOBSERVATION-LEVEL WEATHER MATCHING DETAIL")
print("=" * 120)

print(
    detail.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nEXPORTED FILES")
print(LAP_OUTPUT)
print(DETAIL_OUTPUT)

print("\nWeather timestamp matching audit completed.")
