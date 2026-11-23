
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

LAP_PATH = (
    ROOT_DIR / "data" / "raw"
    / "2024_british_gp_laps.csv"
)

WEATHER_PATH = (
    ROOT_DIR / "data" / "raw"
    / "2024_british_gp_weather.csv"
)

OUTPUT_DIR = (
    ROOT_DIR / "data" / "processed"
    / "regression_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    OUTPUT_DIR / "weather_matching_method_comparison.csv"
)

DETAIL_PATH = (
    OUTPUT_DIR / "weather_matching_method_detail.csv"
)


# --------------------------------------------------
# 2. Load original source datasets
# --------------------------------------------------

print("Loading original lap and weather datasets...")

laps = pd.read_csv(LAP_PATH)
weather = pd.read_csv(WEATHER_PATH)

print(f"Lap records: {len(laps)}")
print(f"Weather observations: {len(weather)}")


# --------------------------------------------------
# 3. Prepare timestamps
# --------------------------------------------------

laps["LapStartTime"] = pd.to_timedelta(
    laps["LapStartTime"], errors="coerce"
)

laps["Time"] = pd.to_timedelta(
    laps["Time"], errors="coerce"
)

weather["Time"] = pd.to_timedelta(
    weather["Time"], errors="coerce"
)

laps = laps.dropna(
    subset=["LapStartTime", "Time"]
).copy()

weather = weather.dropna(
    subset=["Time"]
).copy()

laps["LapStartSeconds"] = (
    laps["LapStartTime"].dt.total_seconds()
)

laps["LapEndSeconds"] = (
    laps["Time"].dt.total_seconds()
)

laps["LapMidpointSeconds"] = (
    laps["LapStartSeconds"]
    + laps["LapEndSeconds"]
) / 2

weather["WeatherSeconds"] = (
    weather["Time"].dt.total_seconds()
)

# Preserve original row identity for comparisons.
laps = laps.reset_index(drop=True)
laps["_row_id"] = laps.index


# --------------------------------------------------
# 4. Prepare weather observations
# --------------------------------------------------

weather = weather[
    ["WeatherSeconds", "Rainfall"]
].copy()

weather["Rainfall"] = (
    weather["Rainfall"]
    .astype("boolean")
)

weather = weather.sort_values(
    "WeatherSeconds"
)


# --------------------------------------------------
# 5. Match using three methods
# --------------------------------------------------

def match_weather(direction):

    left = laps[
        [
            "_row_id",
            "Driver",
            "LapNumber",
            "LapMidpointSeconds"
        ]
    ].copy()

    left = left.sort_values(
        "LapMidpointSeconds"
    )

    matched = pd.merge_asof(
        left,
        weather,
        left_on="LapMidpointSeconds",
        right_on="WeatherSeconds",
        direction=direction,
        tolerance=45
    )

    matched["MatchDifferenceSeconds"] = (
        matched["LapMidpointSeconds"]
        - matched["WeatherSeconds"]
    ).abs()

    matched = matched.rename(
        columns={
            "WeatherSeconds": f"{direction}_WeatherSeconds",
            "Rainfall": f"{direction}_Rainfall",
            "MatchDifferenceSeconds":
                f"{direction}_MatchDifferenceSeconds"
        }
    )

    return matched.set_index("_row_id")


print("\nRunning weather matching comparisons...")

nearest = match_weather("nearest")
backward = match_weather("backward")
forward = match_weather("forward")


# --------------------------------------------------
# 6. Combine comparison results
# --------------------------------------------------

detail = laps[
    [
        "_row_id",
        "Driver",
        "LapNumber",
        "LapMidpointSeconds"
    ]
].set_index("_row_id")

detail = detail.join(nearest[
    [
        "nearest_WeatherSeconds",
        "nearest_Rainfall",
        "nearest_MatchDifferenceSeconds"
    ]
])

detail = detail.join(backward[
    [
        "backward_WeatherSeconds",
        "backward_Rainfall",
        "backward_MatchDifferenceSeconds"
    ]
])

detail = detail.join(forward[
    [
        "forward_WeatherSeconds",
        "forward_Rainfall",
        "forward_MatchDifferenceSeconds"
    ]
])

# Compare rainfall classifications against nearest.
detail["backward_classification_changed"] = (
    detail["backward_Rainfall"].notna()
    & detail["nearest_Rainfall"].notna()
    & (
        detail["backward_Rainfall"]
        != detail["nearest_Rainfall"]
    )
)

detail["forward_classification_changed"] = (
    detail["forward_Rainfall"].notna()
    & detail["nearest_Rainfall"].notna()
    & (
        detail["forward_Rainfall"]
        != detail["nearest_Rainfall"]
    )
)


# --------------------------------------------------
# 7. Generate summary statistics
# --------------------------------------------------

summary_rows = []

for population_name, population in [
    ("All laps", detail),
    (
        "Laps 42-45",
        detail.loc[
            pd.to_numeric(
                detail["LapNumber"], errors="coerce"
            ).between(42, 45)
        ]
    )
]:

    for method in ["nearest", "backward", "forward"]:

        rainfall_col = f"{method}_Rainfall"
        difference_col = (
            f"{method}_MatchDifferenceSeconds"
        )

        matched = population[rainfall_col].notna()

        summary_rows.append({
            "Population": population_name,
            "Method": method,
            "Total laps": len(population),
            "Matched laps": int(matched.sum()),
            "Unmatched laps": int((~matched).sum()),
            "Match percentage": (
                matched.mean() * 100
                if len(population) else 0
            ),
            "Wet laps": int(
                population[rainfall_col].eq(True).sum()
            ),
            "Dry laps": int(
                population[rainfall_col].eq(False).sum()
            ),
            "Mean match difference seconds": (
                population.loc[
                    matched, difference_col
                ].mean()
            )
        })


summary = pd.DataFrame(summary_rows)


# --------------------------------------------------
# 8. Classification changes
# --------------------------------------------------

change_rows = []

for population_name, population in [
    ("All laps", detail),
    (
        "Laps 42-45",
        detail.loc[
            pd.to_numeric(
                detail["LapNumber"], errors="coerce"
            ).between(42, 45)
        ]
    )
]:

    for method in ["backward", "forward"]:

        change_col = (
            f"{method}_classification_changed"
        )

        comparable = (
            population["nearest_Rainfall"].notna()
            & population[f"{method}_Rainfall"].notna()
        )

        changed = population.loc[
            comparable, change_col
        ]

        change_rows.append({
            "Population": population_name,
            "Comparison": f"{method} vs nearest",
            "Comparable laps": int(comparable.sum()),
            "Changed classifications": int(changed.sum()),
            "Percentage changed": (
                changed.mean() * 100
                if len(changed) else 0
            )
        })

changes = pd.DataFrame(change_rows)


# --------------------------------------------------
# 9. Export results
# --------------------------------------------------

summary.to_csv(SUMMARY_PATH, index=False)
detail.reset_index().to_csv(DETAIL_PATH, index=False)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\nMATCHING METHOD SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

print("\nRAINFALL CLASSIFICATION CHANGES")
print("=" * 100)
print(changes.to_string(index=False))

print("\nLAPS 42-45 — METHOD COMPARISON")
print("=" * 100)

comparison_columns = [
    "Driver",
    "LapNumber",
    "LapMidpointSeconds",
    "nearest_Rainfall",
    "backward_Rainfall",
    "forward_Rainfall",
    "nearest_MatchDifferenceSeconds",
    "backward_MatchDifferenceSeconds",
    "forward_MatchDifferenceSeconds"
]

print(
    detail.loc[
        pd.to_numeric(
            detail["LapNumber"], errors="coerce"
        ).between(42, 45),
        comparison_columns
    ].to_string(index=False)
)

print("\nEXPORTED FILES")
print("=" * 100)
print(SUMMARY_PATH)
print(DETAIL_PATH)

print("\nWeather matching sensitivity analysis completed.")
