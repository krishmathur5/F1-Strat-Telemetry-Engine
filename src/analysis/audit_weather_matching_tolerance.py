
from pathlib import Path

import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

LAP_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_laps.csv"
WEATHER_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_weather.csv"
INTEGRATED_PATH = ROOT_DIR / "data" / "processed" / "2024_british_gp_fia_integrated.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "regression_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "weather_matching_tolerance_audit.csv"

CURRENT_TOLERANCE = 45


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading datasets...")

laps = pd.read_csv(LAP_PATH)
weather = pd.read_csv(WEATHER_PATH)
integrated = pd.read_csv(INTEGRATED_PATH)

print(f"Raw laps: {len(laps)}")
print(f"Weather observations: {len(weather)}")
print(f"Integrated records: {len(integrated)}")


# ==================================================
# 3. RECONSTRUCT A1 KEYS
# ==================================================

def parse_bool(series):

    if pd.api.types.is_bool_dtype(series):
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


integrated["is_valid_racing_lap"] = parse_bool(
    integrated["is_valid_racing_lap"]
)

for col in ["LapNumber", "lap_time_s", "TyreLife"]:
    integrated[col] = pd.to_numeric(
        integrated[col],
        errors="coerce"
    )

integrated["Compound"] = (
    integrated["Compound"]
    .fillna("Unknown")
    .astype(str)
    .str.upper()
)

a1 = integrated[
    integrated["is_valid_racing_lap"].eq(True)
    & integrated["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
].copy()

a1 = a1.dropna(
    subset=[
        "Driver",
        "LapNumber",
        "lap_time_s",
        "Rainfall",
        "TrackTemp",
        "TyreLife",
    ]
)

a1 = a1[
    (a1["lap_time_s"] > 0)
    & (a1["TyreLife"] > 0)
].copy()

a1_keys = a1[["Driver", "LapNumber"]].copy()

print(f"\nA1 population: {len(a1)}")


# ==================================================
# 4. PREPARE LAP MIDPOINTS
# ==================================================

for col in ["LapStartTime", "Time"]:
    laps[col] = pd.to_timedelta(
        laps[col],
        errors="coerce"
    )

weather["Time"] = pd.to_timedelta(
    weather["Time"],
    errors="coerce"
)

laps = laps.dropna(
    subset=["Driver", "LapNumber", "LapStartTime", "Time"]
).copy()

laps["LapNumber"] = pd.to_numeric(
    laps["LapNumber"],
    errors="coerce"
)

laps = laps.dropna(
    subset=["LapNumber"]
).copy()

laps["LapMidpointSeconds"] = (
    laps["LapStartTime"].dt.total_seconds()
    + laps["Time"].dt.total_seconds()
) / 2

weather = weather.dropna(
    subset=["Time"]
).copy()

weather["WeatherSeconds"] = (
    weather["Time"].dt.total_seconds()
)

a1_laps = laps.merge(
    a1_keys,
    on=["Driver", "LapNumber"],
    how="inner",
    validate="one_to_one"
)

weather_times = (
    weather["WeatherSeconds"]
    .sort_values()
    .to_numpy()
)

print(f"Raw laps matched to A1: {len(a1_laps)}")
print(f"Weather timestamps: {len(weather_times)}")


# ==================================================
# 5. FIND NEAREST WEATHER DISTANCE
# ==================================================

rows = []

for _, lap in a1_laps.iterrows():

    midpoint = lap["LapMidpointSeconds"]

    distances = abs(weather_times - midpoint)

    nearest_index = distances.argmin()
    nearest_distance = float(distances[nearest_index])

    rows.append(
        {
            "Driver": lap["Driver"],
            "LapNumber": lap["LapNumber"],
            "LapMidpointSeconds": midpoint,
            "NearestWeatherSeconds": weather_times[nearest_index],
            "NearestWeatherDistanceSeconds": nearest_distance,
            "MatchedWithin45Seconds": (
                nearest_distance <= CURRENT_TOLERANCE
            ),
        }
    )

distance_detail = pd.DataFrame(rows)


# ==================================================
# 6. TOLERANCE COVERAGE
# ==================================================

print("\nTOLERANCE COVERAGE")
print("=" * 65)

tolerances = [15, 30, 45, 60, 75, 90]

summary_rows = []

for tolerance in tolerances:

    matched = (
        distance_detail["NearestWeatherDistanceSeconds"]
        <= tolerance
    )

    count = int(matched.sum())

    summary_rows.append(
        {
            "tolerance_seconds": tolerance,
            "matched_observations": count,
            "unmatched_observations": len(distance_detail) - count,
            "coverage_percent": count / len(distance_detail) * 100,
        }
    )

    print(
        f"{tolerance:>3}s: "
        f"{count}/{len(distance_detail)} matched "
        f"({count / len(distance_detail) * 100:.2f}%)"
    )


# ==================================================
# 7. UNMATCHED DISTANCE DISTRIBUTION
# ==================================================

unmatched = distance_detail[
    ~distance_detail["MatchedWithin45Seconds"]
].copy()

print("\nOBSERVATIONS UNMATCHED AT 45 SECONDS")
print("=" * 65)

print(f"Unmatched count: {len(unmatched)}")

if len(unmatched):

    print(
        "Minimum nearest distance: "
        f"{unmatched['NearestWeatherDistanceSeconds'].min():.2f}s"
    )

    print(
        "Median nearest distance: "
        f"{unmatched['NearestWeatherDistanceSeconds'].median():.2f}s"
    )

    print(
        "Maximum nearest distance: "
        f"{unmatched['NearestWeatherDistanceSeconds'].max():.2f}s"
    )

    print("\nDistance bands:")

    bins = [45, 50, 60, 75, 90, 120, float("inf")]

    labels = [
        "45-50s",
        "50-60s",
        "60-75s",
        "75-90s",
        "90-120s",
        "Over 120s",
    ]

    unmatched["DistanceBand"] = pd.cut(
        unmatched["NearestWeatherDistanceSeconds"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=False,
    )

    print(
        unmatched["DistanceBand"]
        .value_counts(sort=False)
        .to_string()
    )


# ==================================================
# 8. EXPORT
# ==================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSummary saved: {OUTPUT_PATH}")

detail_path = (
    OUTPUT_DIR / "weather_matching_tolerance_detail.csv"
)

distance_detail.to_csv(
    detail_path,
    index=False
)

print(f"Detail saved: {detail_path}")

print("\nTolerance audit completed.")
