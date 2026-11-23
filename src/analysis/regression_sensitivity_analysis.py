
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

OUTPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_regression_sensitivity.csv"
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

boolean_columns = [
    "Rainfall",
    "is_deleted",
    "is_pit_transition",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap"
]

for col in boolean_columns:
    df[col] = parse_bool(df[col])

numeric_columns = [
    "LapNumber",
    "lap_time_s",
    "TyreLife",
    "TrackTemp"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df["Compound"] = (
    df["Compound"]
    .fillna("Unknown")
    .astype(str)
    .str.upper()
)

df["rain_observed"] = df["Rainfall"].astype(float)


# ==================================================
# 4. RECREATE ORIGINAL POPULATIONS
# ==================================================

strict = df[
    df["is_valid_racing_lap"].eq(True)
].copy()

fia_clean = df[
    ~df["is_deleted"].fillna(False)
    & ~df["is_pit_transition"].fillna(False)
    & df["has_valid_lap_time"].fillna(False)
    & df["has_weather_match"].fillna(False)
].copy()


# ==================================================
# 5. MODEL DEFINITIONS
# ==================================================

model_definitions = [
    ("A1_Strict_Slick", strict, True),
    ("A2_Strict_All", strict, False),
    ("B1_FIA_Clean_Slick", fia_clean, True),
    ("B2_FIA_Clean_All", fia_clean, False)
]

formula = (
    "lap_time_s ~ rain_observed "
    "+ TrackTemp "
    "+ TyreLife "
    "+ LapNumber "
    "+ I(LapNumber ** 2) "
    "+ C(Driver) "
    "+ C(Compound)"
)

results = []


# ==================================================
# 6. FIT AND COMPARE COVARIANCE ESTIMATORS
# ==================================================

for model_name, population, slick_only in model_definitions:

    print(f"\nProcessing {model_name}...")

    model_data = population.copy()

    if slick_only:
        model_data = model_data[
            model_data["Compound"].isin(
                ["HARD", "MEDIUM", "SOFT"]
            )
        ].copy()

    required = [
        "Driver",
        "Compound",
        "lap_time_s",
        "rain_observed",
        "TrackTemp",
        "TyreLife",
        "LapNumber"
    ]

    model_data = model_data.dropna(
        subset=required
    ).copy()

    model_data = model_data[
        (model_data["lap_time_s"] > 0)
        & (model_data["TyreLife"] > 0)
    ].copy()

    model_data["Driver"] = (
        model_data["Driver"].astype(str)
    )

    if len(model_data) < 50:
        print("Insufficient observations. Skipping.")
        continue

    if model_data["rain_observed"].nunique() < 2:
        print("Insufficient rainfall variation. Skipping.")
        continue

    model = smf.ols(
        formula=formula,
        data=model_data
    ).fit()

    # Driver-clustered standard errors
    clustered = model.get_robustcov_results(
        cov_type="cluster",
        groups=model_data["Driver"].to_numpy(),
        use_correction=True,
        df_correction=True,
        use_t=True
    )

    # HC3 heteroskedasticity-robust standard errors
    hc3 = model.get_robustcov_results(
        cov_type="HC3",
        use_t=True
    )

    rain_index = list(model.params.index).index(
        "rain_observed"
    )

    for method_name, robust in [
        ("Driver_Clustered", clustered),
        ("HC3", hc3)
    ]:

        ci = robust.conf_int()

        results.append({
            "model": model_name,
            "method": method_name,
            "sample_count": len(model_data),
            "driver_clusters": model_data["Driver"].nunique(),
            "rain_coefficient": robust.params[rain_index],
            "standard_error": robust.bse[rain_index],
            "t_statistic": robust.tvalues[rain_index],
            "p_value": robust.pvalues[rain_index],
            "ci_lower": ci[rain_index, 0],
            "ci_upper": ci[rain_index, 1]
        })

    print(
        f"Observations: {len(model_data)}"
    )

    print(
        f"Driver clusters: {model_data['Driver'].nunique()}"
    )


# ==================================================
# 7. EXPORT RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSENSITIVITY COMPARISON")
print("=" * 90)

print(
    results_df.round(6).to_string(index=False)
)

print("\nExported to:")
print(OUTPUT_PATH)

print("\nSensitivity analysis completed.")
