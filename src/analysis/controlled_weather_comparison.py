
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

DRIVER_SUMMARY_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_by_driver.csv"
)

GROUP_SUMMARY_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_controlled_groups.csv"
)

MATCHED_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_matched_comparison.csv"
)

CHART_PATH = (
    OUTPUT_DIR / "2024_british_gp_weather_controlled_comparison.html"
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def parse_bool(series):
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
# 4. VALIDATE REQUIRED COLUMNS
# ==================================================

required_columns = [
    "Driver",
    "LapNumber",
    "lap_time_s",
    "Compound",
    "TyreLife",
    "Rainfall",
    "TrackTemp",
    "is_deleted",
    "is_pit_transition",
    "is_accurate",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ==================================================
# 5. NORMALISE DATA TYPES
# ==================================================

boolean_columns = [
    "is_deleted",
    "is_pit_transition",
    "is_accurate",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap",
    "Rainfall"
]

for col in boolean_columns:
    df[col] = parse_bool(df[col])

for col in ["lap_time_s", "TyreLife", "TrackTemp", "LapNumber"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Compound"] = (
    df["Compound"]
    .fillna("Unknown")
    .astype(str)
    .str.upper()
)

df = df.dropna(
    subset=["Driver", "lap_time_s", "TyreLife", "Rainfall"]
).copy()


# ==================================================
# 6. CREATE TYRE-AGE BANDS
# ==================================================

df["tyre_age_band"] = pd.cut(
    df["TyreLife"],
    bins=[0, 5, 10, 15, 20, 100],
    labels=[
        "1-5 laps",
        "6-10 laps",
        "11-15 laps",
        "16-20 laps",
        "21+ laps"
    ],
    include_lowest=True
)

df["rain_condition"] = df["Rainfall"].map({
    True: "Rain observed",
    False: "No rain observed"
})


# ==================================================
# 7. CREATE ANALYSIS POPULATIONS
# ==================================================

df["strict_eligible"] = (
    df["is_valid_racing_lap"].eq(True)
)

df["fia_clean_eligible"] = (
    ~df["is_deleted"].fillna(False)
    & ~df["is_pit_transition"].fillna(False)
    & df["has_valid_lap_time"].fillna(False)
    & df["has_weather_match"].fillna(False)
)

strict = df[df["strict_eligible"]].copy()
strict["population"] = "Strict"

fia_clean = df[df["fia_clean_eligible"]].copy()
fia_clean["population"] = "FIA-clean"

analysis = pd.concat(
    [strict, fia_clean],
    ignore_index=True
)

analysis = analysis.dropna(
    subset=["tyre_age_band", "rain_condition"]
)

print("\nPOPULATION COUNTS")
print("=" * 65)

print(f"Strict: {len(strict)}")
print(f"FIA-clean: {len(fia_clean)}")


# ==================================================
# 8. DRIVER-LEVEL COMPARISON
# ==================================================

driver_summary = (
    analysis
    .groupby(
        ["population", "Driver", "rain_condition"],
        observed=True
    )
    .agg(
        lap_count=("lap_time_s", "count"),
        average_lap_time_s=("lap_time_s", "mean"),
        median_lap_time_s=("lap_time_s", "median"),
        average_track_temp_c=("TrackTemp", "mean")
    )
    .reset_index()
)

print("\nDRIVER-LEVEL WEATHER COMPARISON")
print("=" * 85)

print(
    driver_summary.round(3).to_string(index=False)
)

driver_summary.to_csv(
    DRIVER_SUMMARY_PATH,
    index=False
)


# ==================================================
# 9. CONTROLLED GROUP COMPARISON
# ==================================================

group_columns = [
    "population",
    "Driver",
    "Compound",
    "tyre_age_band",
    "rain_condition"
]

group_summary = (
    analysis
    .groupby(group_columns, observed=True)
    .agg(
        lap_count=("lap_time_s", "count"),
        average_lap_time_s=("lap_time_s", "mean"),
        median_lap_time_s=("lap_time_s", "median"),
        average_track_temp_c=("TrackTemp", "mean"),
        average_tyre_age=("TyreLife", "mean")
    )
    .reset_index()
)

print("\nCONTROLLED GROUP SUMMARY")
print("=" * 85)

print(
    group_summary.to_string(index=False)
)

group_summary.to_csv(
    GROUP_SUMMARY_PATH,
    index=False
)


# ==================================================
# 10. MATCH DRY AND RAIN GROUPS
# ==================================================

# A matched group must have the same:
# - Driver
# - Tyre compound
# - Tyre-age band
#
# It must also contain both rainfall classifications.

matched = (
    group_summary
    .pivot(
        index=[
            "population",
            "Driver",
            "Compound",
            "tyre_age_band"
        ],
        columns="rain_condition",
        values=[
            "lap_count",
            "average_lap_time_s",
            "average_track_temp_c"
        ]
    )
)

matched.columns = [
    f"{metric}_{condition.lower().replace(' ', '_')}"
    for metric, condition in matched.columns
]

matched = matched.reset_index()

required_matches = [
    "lap_count_no_rain_observed",
    "lap_count_rain_observed",
    "average_lap_time_s_no_rain_observed",
    "average_lap_time_s_rain_observed"
]

matched = matched.dropna(
    subset=required_matches
).copy()

# Require at least five laps in each condition.
matched = matched[
    (matched["lap_count_no_rain_observed"] >= 5)
    & (matched["lap_count_rain_observed"] >= 5)
].copy()

matched["rain_minus_dry_seconds"] = (
    matched["average_lap_time_s_rain_observed"]
    - matched["average_lap_time_s_no_rain_observed"]
)

matched["track_temp_difference_c"] = (
    matched["average_track_temp_c_rain_observed"]
    - matched["average_track_temp_c_no_rain_observed"]
)

print("\nMATCHED DRY VS RAIN COMPARISONS")
print("=" * 100)

if matched.empty:
    print(
        "No groups met the minimum sample requirement "
        "in both rainfall conditions."
    )
else:
    print(
        matched.round(3).to_string(index=False)
    )

matched.to_csv(
    MATCHED_PATH,
    index=False
)


# ==================================================
# 11. MATCHED COMPARISON SUMMARY
# ==================================================

print("\nMATCHED COMPARISON COUNTS")
print("=" * 65)

if matched.empty:
    print("No matched groups available.")
else:
    print(
        matched.groupby("population").agg(
            matched_groups=("Driver", "count"),
            average_lap_time_difference_s=(
                "rain_minus_dry_seconds", "mean"
            ),
            median_lap_time_difference_s=(
                "rain_minus_dry_seconds", "median"
            )
        ).round(3).to_string()
    )


# ==================================================
# 12. VISUALISE MATCHED COMPARISONS
# ==================================================

if not matched.empty:

    fig = px.bar(
        matched,
        x="Driver",
        y="rain_minus_dry_seconds",
        color="Compound",
        facet_col="population",
        hover_data=[
            "tyre_age_band",
            "lap_count_no_rain_observed",
            "lap_count_rain_observed",
            "track_temp_difference_c"
        ],
        title=(
            "Matched Rain vs Dry Lap-Time Difference "
            "by Driver, Compound and Tyre Age"
        ),
        labels={
            "Driver": "Driver",
            "rain_minus_dry_seconds": (
                "Rain minus dry average lap time (seconds)"
            ),
            "Compound": "Tyre compound"
        }
    )

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(CHART_PATH)


# ==================================================
# 13. FINAL REPORT
# ==================================================

print("\nOUTPUT FILES")
print("=" * 65)

print(f"Driver summary: {DRIVER_SUMMARY_PATH}")
print(f"Controlled groups: {GROUP_SUMMARY_PATH}")
print(f"Matched comparisons: {MATCHED_PATH}")

if not matched.empty:
    print(f"Interactive chart: {CHART_PATH}")

print("\nControlled weather comparison completed.")
