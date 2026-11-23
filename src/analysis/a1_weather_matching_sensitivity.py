
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

LAP_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_laps.csv"
WEATHER_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_weather.csv"
INTEGRATED_PATH = ROOT_DIR / "data" / "processed" / "2024_british_gp_fia_integrated.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "regression_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "a1_weather_matching_sensitivity_summary.csv"
DETAIL_PATH = OUTPUT_DIR / "a1_weather_matching_sensitivity_detail.csv"

TOLERANCE_SECONDS = 45

METHODS = ["nearest", "backward", "forward"]

FORMULA = (
    "lap_time_s ~ rain_observed "
    "+ TrackTemp "
    "+ TyreLife "
    "+ LapNumber "
    "+ I(LapNumber ** 2) "
    "+ C(Driver) "
    "+ C(Compound)"
)


# ==================================================
# 2. BOOLEAN PARSING
# ==================================================

def parse_bool(series):
    """Convert common CSV boolean representations to True/False."""

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


# ==================================================
# 3. LOAD SOURCE DATA
# ==================================================

print("Loading source datasets...")

laps = pd.read_csv(LAP_PATH)
weather = pd.read_csv(WEATHER_PATH)
integrated = pd.read_csv(INTEGRATED_PATH)

print(f"Raw lap records: {len(laps)}")
print(f"Raw weather observations: {len(weather)}")
print(f"Integrated records: {len(integrated)}")


# ==================================================
# 4. RECONSTRUCT EXACT A1 POPULATION
# ==================================================

print("\nReconstructing A1 strict slick population...")

integrated["is_valid_racing_lap"] = parse_bool(
    integrated["is_valid_racing_lap"]
)

for col in ["LapNumber", "lap_time_s", "TyreLife", "TrackTemp"]:
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

integrated["Rainfall"] = parse_bool(
    integrated["Rainfall"]
)

strict = integrated[
    integrated["is_valid_racing_lap"].eq(True)
].copy()

a1 = strict[
    strict["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
].copy()

required = [
    "Driver",
    "Compound",
    "lap_time_s",
    "Rainfall",
    "TrackTemp",
    "TyreLife",
    "LapNumber",
]

a1 = a1.dropna(subset=required).copy()

a1 = a1[
    (a1["lap_time_s"] > 0)
    & (a1["TyreLife"] > 0)
].copy()

a1["Driver"] = a1["Driver"].astype(str)
a1["Compound"] = a1["Compound"].astype(str)
a1["rain_observed"] = a1["Rainfall"].astype(float)

print(f"A1 reconstructed observations: {len(a1)}")

if len(a1) != 658:
    raise ValueError(
        f"Expected 658 A1 observations, reconstructed {len(a1)}. "
        "Stop and investigate before fitting models."
    )

if a1.duplicated(["Driver", "LapNumber"]).any():
    raise ValueError(
        "Duplicate Driver/LapNumber keys found in A1 population."
    )


# ==================================================
# 5. PREPARE RAW LAP AND WEATHER TIMESTAMPS
# ==================================================

print("\nPreparing raw timestamps...")

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

weather = weather.dropna(
    subset=["Time"]
).copy()

laps["LapNumber"] = pd.to_numeric(
    laps["LapNumber"],
    errors="coerce"
)

laps = laps.dropna(
    subset=["LapNumber"]
).copy()

laps["LapStartSeconds"] = (
    laps["LapStartTime"].dt.total_seconds()
)

laps["LapEndSeconds"] = (
    laps["Time"].dt.total_seconds()
)

laps["LapMidpointSeconds"] = (
    laps["LapStartSeconds"] + laps["LapEndSeconds"]
) / 2

weather["WeatherSeconds"] = (
    weather["Time"].dt.total_seconds()
)

weather["Rainfall"] = parse_bool(
    weather["Rainfall"]
)

weather["TrackTemp"] = pd.to_numeric(
    weather["TrackTemp"],
    errors="coerce"
)

weather = weather.rename(
    columns={"Time": "WeatherTime"}
)

weather_columns = [
    "WeatherTime",
    "WeatherSeconds",
    "Rainfall",
    "TrackTemp",
]

weather = weather[weather_columns].copy()

# Only raw laps corresponding to the A1 keys are required.
a1_keys = a1[["Driver", "LapNumber"]].copy()

a1_raw_laps = laps.merge(
    a1_keys,
    on=["Driver", "LapNumber"],
    how="inner",
    validate="one_to_one"
)

print(f"Raw laps matched to A1 keys: {len(a1_raw_laps)}")

if len(a1_raw_laps) != 658:
    raise ValueError(
        "Could not map all 658 A1 records to unique raw laps. "
        "Investigate key alignment before continuing."
    )


# ==================================================
# 6. MATCH WEATHER USING THREE METHODS
# ==================================================

print("\nApplying weather matching methods...")

method_results = []

for method in METHODS:

    left = a1_raw_laps.sort_values(
        "LapMidpointSeconds"
    ).copy()

    right = weather.sort_values(
        "WeatherSeconds"
    ).copy()

    matched = pd.merge_asof(
        left,
        right,
        left_on="LapMidpointSeconds",
        right_on="WeatherSeconds",
        direction=method,
        tolerance=TOLERANCE_SECONDS,
    )

    matched["WeatherTimeDifferenceSeconds"] = (
        matched["LapMidpointSeconds"]
        - matched["WeatherSeconds"]
    ).abs()

    matched["Driver"] = matched["Driver"].astype(str)

    matched["LapNumber"] = pd.to_numeric(
        matched["LapNumber"],
        errors="coerce"
    )

    matched = matched[
        [
            "Driver",
            "LapNumber",
            "WeatherTime",
            "WeatherSeconds",
            "Rainfall",
            "TrackTemp",
            "WeatherTimeDifferenceSeconds",
        ]
    ].copy()

    matched = matched.rename(
        columns={
            "Rainfall": f"Rainfall_{method}",
            "TrackTemp": f"TrackTemp_{method}",
            "WeatherTime": f"WeatherTime_{method}",
            "WeatherSeconds": f"WeatherSeconds_{method}",
            "WeatherTimeDifferenceSeconds":
                f"MatchDifference_{method}",
        }
    )

    method_results.append(matched)

    match_count = matched[f"WeatherSeconds_{method}"].notna().sum()

    print(
        f"{method}: {match_count}/658 matched "
        f"({match_count / 658 * 100:.2f}%)"
    )


# ==================================================
# 7. MERGE MATCHING RESULTS
# ==================================================

print("\nCombining matching results...")

detail = a1[
    [
        "Driver",
        "LapNumber",
        "lap_time_s",
        "Compound",
        "TyreLife",
        "rain_observed",
        "TrackTemp",
    ]
].copy()

for result in method_results:

    detail = detail.merge(
        result,
        on=["Driver", "LapNumber"],
        how="left",
        validate="one_to_one"
    )

detail = detail.sort_values(
    ["Driver", "LapNumber"]
).reset_index(drop=True)


# ==================================================
# 8. IDENTIFY COMMON MATCHED SAMPLE
# ==================================================

match_columns = [
    f"WeatherSeconds_{method}"
    for method in METHODS
]

detail["common_matched"] = (
    detail[match_columns].notna().all(axis=1)
)

common = detail[
    detail["common_matched"]
].copy()

print("\nMATCH COVERAGE")
print("=" * 65)

print(f"Original A1 population: {len(detail)}")
print(f"Common matched population: {len(common)}")
print(f"Excluded from common sample: {len(detail) - len(common)}")


# ==================================================
# 9. FIT REGRESSION: COMMON AND METHOD-SPECIFIC SAMPLES
# ==================================================

summary_rows = []

for method in METHODS:

    rain_col = f"Rainfall_{method}"
    temp_col = f"TrackTemp_{method}"
    match_col = f"WeatherSeconds_{method}"

    for sample_type in ["common_matched", "method_specific"]:

        if sample_type == "common_matched":
            model_data = detail[
                detail["common_matched"]
            ].copy()
        else:
            model_data = detail[
                detail[match_col].notna()
            ].copy()

        model_data["rain_observed"] = (
            parse_bool(model_data[rain_col]).astype(float)
        )

        model_data["TrackTemp"] = pd.to_numeric(
            model_data[temp_col],
            errors="coerce"
        )

        model_data = model_data.dropna(
            subset=[
                "rain_observed",
                "TrackTemp",
                "lap_time_s",
                "TyreLife",
                "LapNumber",
                "Driver",
                "Compound",
            ]
        ).copy()

        model_data = model_data[
            (model_data["lap_time_s"] > 0)
            & (model_data["TyreLife"] > 0)
        ].copy()

        if model_data["rain_observed"].nunique() < 2:
            print(
                f"\n{method} / {sample_type}: "
                "skipped; only one rainfall category."
            )
            continue

        model = smf.ols(
            formula=FORMULA,
            data=model_data
        ).fit()

        robust = model.get_robustcov_results(
            cov_type="cluster",
            groups=model_data["Driver"].to_numpy(),
            use_correction=True,
            df_correction=True
        )

        param_names = list(model.params.index)
        rain_index = param_names.index("rain_observed")

        ci = robust.conf_int()

        result = {
            "method": method,
            "sample": sample_type,
            "sample_count": len(model_data),
            "driver_count": model_data["Driver"].nunique(),
            "dry_laps": int(
                (model_data["rain_observed"] == 0).sum()
            ),
            "wet_laps": int(
                (model_data["rain_observed"] == 1).sum()
            ),
            "r_squared": model.rsquared,
            "rain_coefficient_seconds": robust.params[rain_index],
            "rain_clustered_se": robust.bse[rain_index],
            "rain_p_value": robust.pvalues[rain_index],
            "rain_ci_lower": ci[rain_index, 0],
            "rain_ci_upper": ci[rain_index, 1],
            "mean_match_difference_seconds":
                model_data[
                    f"MatchDifference_{method}"
                ].mean(),
        }

        summary_rows.append(result)

        print(f"\n{method.upper()} / {sample_type.upper()}")
        print("-" * 65)
        print(f"Observations: {len(model_data)}")
        print(f"Drivers: {model_data['Driver'].nunique()}")
        print(f"Dry laps: {result['dry_laps']}")
        print(f"Wet laps: {result['wet_laps']}")
        print(
            f"Rain coefficient: "
            f"{robust.params[rain_index]:.6f}"
        )
        print(
            f"Clustered SE: {robust.bse[rain_index]:.6f}"
        )
        print(
            f"p-value: {robust.pvalues[rain_index]:.6g}"
        )
        print(
            f"95% CI: [{ci[rain_index, 0]:.6f}, "
            f"{ci[rain_index, 1]:.6f}]"
        )



# ==================================================
# 10. CLASSIFICATION COMPARISON
# ==================================================

print("\nRAINFALL CLASSIFICATION COMPARISON")
print("=" * 65)

for method in ["backward", "forward"]:

    comparison = detail[
        detail["common_matched"]
    ].copy()

    nearest_col = "Rainfall_nearest"
    alternative_col = f"Rainfall_{method}"

    changed = (
        parse_bool(comparison[nearest_col])
        != parse_bool(comparison[alternative_col])
    )

    changed_count = int(changed.sum())

    print(
        f"{method} vs nearest: "
        f"{changed_count}/{len(comparison)} changed "
        f"({changed_count / len(comparison) * 100:.2f}%)"
    )


# ==================================================
# 11. AUDIT MATCHED VS UNMATCHED SAMPLE COMPOSITION
# ==================================================

print("\nSAMPLE-SELECTION AUDIT")
print("=" * 65)

selection_rows = []
compound_rows = []

for method in METHODS:

    match_col = f"WeatherSeconds_{method}"

    detail[f"matched_{method}"] = (
        detail[match_col].notna()
    )

    for status, group in detail.groupby(
        f"matched_{method}"
    ):

        status_label = (
            "Matched" if status else "Unmatched"
        )

        selection_rows.append(
            {
                "method": method,
                "match_status": status_label,
                "observations": len(group),
                "drivers": group["Driver"].nunique(),
                "mean_lap_number": group["LapNumber"].mean(),
                "median_lap_number": group["LapNumber"].median(),
                "mean_lap_time": group["lap_time_s"].mean(),
                "mean_tyre_life": group["TyreLife"].mean(),
                "mean_original_track_temp": group["TrackTemp"].mean(),
                "original_wet_laps": int(
                    (group["rain_observed"] == 1).sum()
                ),
                "original_dry_laps": int(
                    (group["rain_observed"] == 0).sum()
                ),
                "original_rain_share": group["rain_observed"].mean(),
            }
        )

        compound_counts = (
            group["Compound"]
            .value_counts()
            .to_dict()
        )

        for compound, count in compound_counts.items():
            compound_rows.append(
                {
                    "method": method,
                    "match_status": status_label,
                    "compound": compound,
                    "observations": count,
                }
            )


selection_summary = pd.DataFrame(selection_rows)
compound_summary = pd.DataFrame(compound_rows)

selection_path = (
    OUTPUT_DIR / "a1_weather_matching_sample_selection.csv"
)

compound_path = (
    OUTPUT_DIR / "a1_weather_matching_compound_selection.csv"
)

selection_summary.to_csv(
    selection_path,
    index=False
)

compound_summary.to_csv(
    compound_path,
    index=False
)

print("\nMATCHED VS UNMATCHED CHARACTERISTICS")
print(selection_summary.round(3).to_string(index=False))

print("\nCOMPOUND COMPOSITION")
print(compound_summary.to_string(index=False))

print(f"\nSelection audit saved: {selection_path}")
print(f"Compound audit saved: {compound_path}")

